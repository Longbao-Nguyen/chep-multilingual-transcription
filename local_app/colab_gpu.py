"""Explicit, short Google Colab T4 GPU probe."""
from __future__ import annotations

import json
import logging
import threading
import uuid
from pathlib import Path
from typing import Any

from .colab_cli import ColabCLI, HELPER_AUTH_PREAMBLE, safe_error
from .config import atomic_json

log = logging.getLogger(__name__)

_PROBE_GPU_HELPER = HELPER_AUTH_PREAMBLE + r"""
import json, sys, uuid
from colab_cli.client import Accelerator, ColabRequestError, TooManyAssignmentsError, Variant
from colab_cli.runtime import ColabRuntime
from colab_cli.utils import get_status_code

payload = json.loads(sys.stdin.read())
email, client = _load_colab_session(payload['fallback_token'])
pending = payload.get('pending_endpoint')
if pending:
    try:
        client.unassign(pending)
    except Exception:
        pass

try:
    ccu = client.get_consumption_user_info()
    quota = {
        'balance': f'{ccu.paid_compute_units_balance:.2f}',
        'rate': f'{ccu.consumption_rate_hourly:.2f}',
        'active': int(ccu.assignments_count),
    }
except Exception:
    quota = None

endpoint = None
gpu_name = None
status = 'ready'
message = ''
try:
    res = client.assign(uuid.uuid4(), variant=Variant.GPU, accelerator=Accelerator.T4)
    endpoint = res.endpoint
    info = res.runtime_proxy_info
    runtime = ColabRuntime(info.url, info.token)
    try:
        outputs = runtime.execute_code(
            "import subprocess; p = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True); print(p.stdout.strip())",
            timeout=60,
        )
        gpu_name = ''.join(o.get('text', '') for o in outputs if o.get('output_type') == 'stream').strip()
    finally:
        try:
            runtime.stop(shutdown_kernel=True)
        except Exception:
            pass
except TooManyAssignmentsError:
    status = 'quota_exhausted'
    message = 'Colab GPU capacity or active-session limit reached. Stop unused runtimes or try again later.'
except ColabRequestError as exc:
    if get_status_code(exc) == 400:
        status = 'quota_exhausted'
        message = 'This Google Colab account currently has no T4 GPU allocation available.'
    else:
        raise
finally:
    cleaned = True
    if endpoint:
        try:
            client.unassign(endpoint)
        except Exception:
            cleaned = False

print('CHEP_COLAB_JSON:' + json.dumps({
    'ok': status == 'ready',
    'status': status,
    'email': email,
    'quota': quota,
    'gpu_name': gpu_name,
    'endpoint': endpoint,
    'cleaned': cleaned,
    'message': message,
}), flush=True)
"""

_CLEANUP_PROBE_HELPER = HELPER_AUTH_PREAMBLE + r"""
import json, sys
payload = json.loads(sys.stdin.read())
email, client = _load_colab_session(payload['fallback_token'])
client.unassign(payload['endpoint'])
print('CHEP_COLAB_JSON:' + json.dumps({'ok': True, 'email': email}), flush=True)
"""


class ColabGpuCheck:
    """Explicit Colab T4 GPU probe with compute-unit telemetry and strict cleanup."""

    def __init__(self, root: Path, cli: ColabCLI) -> None:
        self.root, self.cli = root, cli
        self.pending_path = root / 'colab_gpu_check_cleanup.json'
        self.lock = threading.RLock()
        self.thread: threading.Thread | None = None
        self.stop = threading.Event()
        self.state: dict[str, Any] = {'status': 'unchecked', 'username': None, 'message': '', 'quota': None}

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            state = dict(self.state)
        cli_snap = self.cli.snapshot()
        if state['username'] != cli_snap.get('username') or cli_snap.get('status') != 'ready':
            return {'status': 'unchecked', 'username': None, 'message': '', 'quota': None}
        if state['status'] == 'ready' and self.pending_path.is_file():
            state['status'] = 'cleanup_pending'
        return state

    def start(self) -> None:
        cli_snap = self.cli.snapshot()
        if cli_snap.get('status') != 'ready':
            raise ValueError('Authenticate Colab CLI before checking GPU access.')
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise ValueError('Colab GPU check is already running.')
            username = cli_snap['username']
            if self.state['status'] == 'ready' and self.state['username'] == username and not self.pending_path.is_file():
                return
            self.state = {
                'status': 'checking',
                'username': username,
                'message': 'Checking Google Colab T4 GPU availability. This may take a minute...',
                'quota': None,
            }
            self.thread = threading.Thread(target=self._check, args=(username,), daemon=True)
            self.thread.start()

    def _check(self, username: str) -> None:
        try:
            if self.cli.identity() != username:
                raise ValueError('Google Colab account changed. Authenticate the CLI and check again.')
            pending_endpoint = None
            if self.pending_path.is_file():
                saved = json.loads(self.pending_path.read_text(encoding='utf-8'))
                if saved.get('owner') != username:
                    raise ValueError('Sign in as ' + str(saved.get('owner')) + ' to release the previous Colab probe.')
                pending_endpoint = saved.get('endpoint')

            res = self.cli.run_helper(
                _PROBE_GPU_HELPER,
                {'pending_endpoint': pending_endpoint},
                timeout=180,
                stopped=self.stop.is_set,
            )
            if pending_endpoint and res.get('cleaned', True):
                self.pending_path.unlink(missing_ok=True)
            endpoint = res.get('endpoint')
            if endpoint and not res.get('cleaned', True):
                atomic_json(self.pending_path, {'owner': username, 'endpoint': endpoint})

            quota = res.get('quota')
            if res.get('status') == 'quota_exhausted':
                self._set('quota_exhausted', username, res.get('message') or 'Colab GPU quota exhausted.', quota)
                return

            gpu_name = res.get('gpu_name') or ''
            if gpu_name:
                suffix = '' if res.get('cleaned', True) else ' Temporary runtime cleanup is pending; retry Check GPU condition.'
                self._set('ready', username, f'Colab {gpu_name} confirmed.{suffix}', quota)
            else:
                self._set('quota_exhausted', username, 'Colab did not assign a CUDA GPU runtime right now.', quota)
        except Exception as exc:
            log.warning('Colab GPU check failed: %s', type(exc).__name__)
            msg = str(exc) if isinstance(exc, ValueError) else safe_error(str(exc))
            self._set('error', username, msg)

    def _set(self, status: str, username: str | None, message: str, quota: dict[str, Any] | None = None) -> None:
        with self.lock:
            self.state = {'status': status, 'username': username, 'message': message, 'quota': quota}

    def close(self) -> None:
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=5)
