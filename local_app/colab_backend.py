"""Durable remote transcription via Google Colab CLI."""
from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
import uuid
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .catalog import resolve_artifact
from .colab_cli import ColabCLI, ColabStopped, HELPER_AUTH_PREAMBLE
from .colab_script import build_worker_script
from .kaggle_artifacts import sha, validate_model_profile
from .kaggle_errors import CleanupError

MAX_REMOTE_BYTES = 2 * 1024**3
UPLOAD_CHUNK_BYTES = 6 * 1024 * 1024
log = logging.getLogger(__name__)


def _refresh_session_helper() -> str:
    return HELPER_AUTH_PREAMBLE + r"""
import json, sys, uuid
from colab_cli.client import Accelerator, Variant
from colab_cli.state import SessionState

payload = json.loads(sys.stdin.read())
email, client = _load_colab_session(payload['fallback_token'])
target_ep = payload.get('endpoint')
allow_create = payload.get('allow_create', True)

assignments = {a.endpoint: a for a in client.list_assignments()}
if target_ep and target_ep in assignments:
    info = assignments[target_ep].runtime_proxy_info
    session = {
        'owner': email,
        'endpoint': target_ep,
        'url': info.url,
        'token': info.token,
        'expires_at': info.expires_at().isoformat(),
        'reused': True,
    }
elif target_ep and not allow_create:
    raise RuntimeError('The Google Colab runtime disconnected. Retry to run on a new Colab GPU session.')
else:
    res = client.assign(uuid.uuid4(), variant=Variant.GPU, accelerator=Accelerator.T4)
    info = res.runtime_proxy_info
    session = {
        'owner': email,
        'endpoint': res.endpoint,
        'url': info.url,
        'token': info.token,
        'expires_at': info.expires_at().isoformat(),
        'reused': False,
    }

print('CHEP_COLAB_JSON:' + json.dumps(session), flush=True)
"""


def _upload_file_helper() -> str:
    return r"""
import json, sys
from colab_cli.contents import ContentsClient
from colab_cli.state import SessionState

payload = json.loads(sys.stdin.read())
s = SessionState(name='chep', token=payload['token'], url=payload['url'], endpoint=payload['endpoint'])
client = ContentsClient(s)
client.upload(payload['local_path'], payload['remote_path'])
print('CHEP_COLAB_JSON:' + json.dumps({'ok': True}), flush=True)
"""


def _exec_helper() -> str:
    return r"""
import json, sys
from colab_cli.runtime import ColabRuntime

payload = json.loads(sys.stdin.read())
runtime = ColabRuntime(payload['url'], payload['token'])
try:
    outputs = runtime.execute_code(payload['code'], timeout=float(payload.get('timeout', 60)))
finally:
    runtime.stop(shutdown_kernel=False)
text = ''.join(o.get('text', '') for o in outputs if o.get('output_type') == 'stream')
errors = [o.get('evalue', '') for o in outputs if o.get('output_type') == 'error']
if errors:
    raise RuntimeError('; '.join(errors))
print('CHEP_COLAB_JSON:' + json.dumps({'ok': True, 'stdout': text}), flush=True)
"""


def _poll_status_helper() -> str:
    return r"""
import json, sys, tempfile
from pathlib import Path
from colab_cli.contents import ContentsClient
from colab_cli.state import SessionState

payload = json.loads(sys.stdin.read())
s = SessionState(name='chep', token=payload['token'], url=payload['url'], endpoint=payload['endpoint'])
client = ContentsClient(s)
status_data = None
with tempfile.TemporaryDirectory() as td:
    dst = Path(td) / 'status.json'
    try:
        client.download(payload['status_path'], str(dst))
        status_data = json.loads(dst.read_text(encoding='utf-8'))
    except FileNotFoundError:
        status_data = None
print('CHEP_COLAB_JSON:' + json.dumps({'ok': True, 'status': status_data}), flush=True)
"""


def _download_results_helper() -> str:
    return r"""
import json, sys
from pathlib import Path
from colab_cli.contents import ContentsClient
from colab_cli.state import SessionState

payload = json.loads(sys.stdin.read())
s = SessionState(name='chep', token=payload['token'], url=payload['url'], endpoint=payload['endpoint'])
client = ContentsClient(s)
out_dir = Path(payload['local_dir'])
out_dir.mkdir(parents=True, exist_ok=True)
for pair in payload['files']:
    client.download(pair['remote'], str(out_dir / pair['name']))
print('CHEP_COLAB_JSON:' + json.dumps({'ok': True}), flush=True)
"""


def _unassign_helper() -> str:
    return HELPER_AUTH_PREAMBLE + r"""
import json, sys
from colab_cli.client import ColabRequestError
from colab_cli.runtime import ColabRuntime
from colab_cli.utils import get_status_code

payload = json.loads(sys.stdin.read())
email, client = _load_colab_session(payload['fallback_token'])
ep = payload['endpoint']
assignments = {a.endpoint: a for a in client.list_assignments()}
if ep not in assignments:
    print('CHEP_COLAB_JSON:' + json.dumps({'ok': True, 'missing': True}), flush=True)
else:
    info = assignments[ep].runtime_proxy_info
    try:
        ColabRuntime(info.url, info.token).stop(shutdown_kernel=True)
    except Exception:
        pass
    try:
        client.unassign(ep)
    except ColabRequestError as exc:
        if get_status_code(exc) != 404:
            raise
    print('CHEP_COLAB_JSON:' + json.dumps({'ok': True, 'missing': False}), flush=True)
"""


def validate_colab_result(directory: Path, job_id: str, source_hash: str) -> dict[str, Any]:
    """Validate result.json metadata, GPU provider identity, and output artifact checksums."""
    manifest = directory / 'result.json'
    if not manifest.is_file() or manifest.stat().st_size > 1024**2:
        raise ValueError('Google Colab did not return a valid result manifest.')
    result = json.loads(manifest.read_text(encoding='utf-8'))
    if (
        result.get('schema_version') != 1
        or result.get('job_id') != job_id
        or result.get('source_sha256') != source_hash
    ):
        raise ValueError('Google Colab result does not match this recording.')
    if result.get('ok') is not True:
        raise ValueError('Google Colab transcription failed: ' + str(result.get('error', 'Unknown remote error.'))[:1000])
    profile = result.get('profile', {})
    if profile.get('device') != 'cuda' or profile.get('provider') != 'colab' or not profile.get('device_name'):
        raise ValueError('Google Colab did not confirm a CUDA GPU profile.')
    for ext in ('txt', 'srt'):
        path = directory / (job_id + '.' + ext)
        meta = result.get('files', {}).get(ext, {})
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != meta.get('size')
            or sha(path) != meta.get('sha256')
        ):
            raise ValueError('Google Colab transcript integrity check failed.')
    return result


class ColabBackend:
    """Manage Google Colab T4 sessions, chunked audio upload, polling, and cleanup."""

    def __init__(self, store: Any) -> None:
        self.store = store
        self.cli = ColabCLI(store.root)
        self._session_lock = threading.RLock()
        self._warm: dict[str, Any] | None = None

    def save(self, job_id: str, remote: dict[str, Any]) -> None:
        self.store.update(job_id, remote=json.dumps(remote))

    def _session_expiring(self, session: dict[str, Any]) -> bool:
        exp = session.get('expires_at')
        if not exp:
            return True
        try:
            dt = datetime.fromisoformat(exp)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return (dt - datetime.now(timezone.utc)).total_seconds() < 300
        except ValueError:
            return True

    def _ensure_session(
        self,
        owner: str,
        preferred_endpoint: str | None = None,
        stopped: Any = None,
        *,
        allow_create: bool = True,
    ) -> dict[str, Any]:
        with self._session_lock:
            if (
                self._warm
                and self._warm.get('owner') == owner
                and not self._session_expiring(self._warm)
                and (not preferred_endpoint or self._warm.get('endpoint') == preferred_endpoint)
            ):
                return {**self._warm, 'reused': True}
            endpoint_hint = preferred_endpoint or (
                self._warm.get('endpoint') if self._warm and self._warm.get('owner') == owner else None
            )
            session = self.cli.run_helper(
                _refresh_session_helper(),
                {'endpoint': endpoint_hint, 'allow_create': allow_create},
                timeout=180,
                stopped=stopped,
            )
            if session.get('owner') != owner:
                raise ValueError('Signed-in Google account changed. Restart with the matching account.')
            self._warm = dict(session)
            return dict(session)

    def release_session(self, endpoint: str | None = None) -> None:
        with self._session_lock:
            target = endpoint or (self._warm.get('endpoint') if self._warm else None)
            if self._warm and (endpoint is None or self._warm.get('endpoint') == target):
                self._warm = None
        if not target:
            return
        try:
            self.cli.run_helper(_unassign_helper(), {'endpoint': target}, timeout=90)
        except Exception:
            log.warning('Failed releasing Colab endpoint %s', target)

    def _has_queued_colab_jobs(self) -> bool:
        for item in self.store.list():
            if item.get('status') != 'QUEUED':
                continue
            try:
                if json.loads(item['config']).get('device_preference') == 'colab':
                    return True
            except Exception:
                pass
        return False

    def cleanup(self, job_id: str) -> None:
        job = self.store.get(job_id, include_removed=True)
        if not job:
            return
        remote = json.loads(job.get('remote') or '{}')
        if not remote or remote.get('cleaned'):
            return
        endpoint = remote.get('endpoint')
        try:
            if endpoint:
                owner = self.cli.identity()
                if remote.get('owner') and owner != remote['owner']:
                    raise CleanupError('authentication', 'Sign in as ' + remote['owner'] + ' to remove this Colab session.')
                with self._session_lock:
                    keep_warm = self._warm and self._warm.get('endpoint') == endpoint and self._has_queued_colab_jobs()
                    session_info = dict(self._warm) if keep_warm else None
                if session_info:
                    self._remove_remote_job_dir(session_info, job_id)
                else:
                    self.cli.run_helper(_unassign_helper(), {'endpoint': endpoint}, timeout=90)
                    with self._session_lock:
                        if self._warm and self._warm.get('endpoint') == endpoint:
                            self._warm = None
            remote.update(cleaned=True, cleanup_error=None, cleanup_kind=None)
        except Exception as exc:
            remote['cleanup_error'] = str(exc)
            remote['cleanup_kind'] = getattr(
                exc,
                'kind',
                'timeout' if 'timed out' in str(exc).lower() else 'local_cli' if isinstance(exc, OSError) else 'unexpected',
            )
            log.warning('Colab cleanup pending for job %s: %s', job_id, type(exc).__name__)
            raise
        finally:
            self.save(job_id, remote)

    def _remove_remote_job_dir(self, session: dict[str, Any], job_id: str) -> None:
        if not re.fullmatch(r'[a-f0-9]{32}', job_id):
            return
        self.cli.run_helper(
            _exec_helper(),
            {
                'url': session['url'],
                'token': session['token'],
                'endpoint': session['endpoint'],
                'code': f"import shutil; shutil.rmtree('/content/chep-job/{job_id}', ignore_errors=True)",
                'timeout': 30,
            },
            timeout=60,
        )

    def _upload_chunked(
        self,
        session: dict[str, Any],
        source: Path,
        job_id: str,
        work_dir: Path,
        stopped: Callable[[], bool],
    ) -> tuple[dict[str, Any], int]:
        remote_parts = f'/content/chep-job/{job_id}/parts'
        self.cli.run_helper(
            _exec_helper(),
            {
                'url': session['url'],
                'token': session['token'],
                'endpoint': session['endpoint'],
                'code': f"import pathlib; pathlib.Path({remote_parts!r}).mkdir(parents=True, exist_ok=True)",
                'timeout': 30,
            },
            timeout=60,
            stopped=stopped,
        )
        total_size = source.stat().st_size
        n_chunks = max(1, math.ceil(total_size / UPLOAD_CHUNK_BYTES))
        temp_chunk = work_dir / 'upload_part.bin'
        owner = session['owner']
        try:
            with source.open('rb') as stream:
                for idx in range(n_chunks):
                    if stopped():
                        raise ColabStopped()
                    session = self._ensure_session(owner, session['endpoint'], stopped=stopped, allow_create=False)
                    chunk_bytes = stream.read(UPLOAD_CHUNK_BYTES)
                    temp_chunk.write_bytes(chunk_bytes)
                    self.cli.run_helper(
                        _upload_file_helper(),
                        {
                            'url': session['url'],
                            'token': session['token'],
                            'endpoint': session['endpoint'],
                            'local_path': str(temp_chunk),
                            'remote_path': f'{remote_parts}/part_{idx:04d}.bin',
                        },
                        timeout=300,
                        stopped=stopped,
                    )
                    self.store.update(job_id, progress=min(0.14, 0.03 + 0.11 * (idx + 1) / n_chunks))
        finally:
            temp_chunk.unlink(missing_ok=True)
        return session, n_chunks

    def run(self, job: dict[str, Any], stopped: Callable[[], bool], guard: Any = None) -> None:
        guard = guard or nullcontext()
        job_id = job['id']
        config = json.loads(job['config'])
        config['model_config'] = resolve_artifact(config['language'], config['model_config'])
        if not config['model_config'].get('sweet_spot'):
            raise ValueError('This model does not have a validated Colab GPU profile.')
        source = self.store.root / 'sources' / job_id
        if source.stat().st_size > MAX_REMOTE_BYTES:
            raise ValueError('Online GPU supports recordings up to 2 GB. Split this recording or use Local processing.')

        owner = self.cli.identity()
        remote = json.loads(job.get('remote') or '{}')
        if remote and remote.get('owner') and remote['owner'] != owner:
            raise ValueError('Sign in as ' + remote['owner'] + ' to resume this recording.')
        if remote.get('cleaned'):
            remote = {}
        if remote.get('stage') == 'failed':
            self.cleanup(job_id)
            remote = {}

        source_sha256 = remote.get('source_sha256') or sha(source)
        work_dir = self.store.root / 'work' / job_id
        work_dir.mkdir(parents=True, exist_ok=True)

        self.store.update(job_id, status='WAITING_GPU', progress=0.02)
        session = self._ensure_session(owner, remote.get('endpoint'), stopped=stopped)
        reused = bool(remote.get('endpoint') and session.get('endpoint') == remote.get('endpoint') and remote.get('submitted'))

        remote = {
            'provider': 'colab',
            'owner': owner,
            'endpoint': session['endpoint'],
            'source_sha256': source_sha256,
            'submitted': reused,
            'stage': 'waiting',
            'cleaned': False,
        }
        self.save(job_id, remote)

        if not reused:
            try:
                if stopped():
                    raise ColabStopped()
                self.store.update(job_id, status='UPLOADING', progress=0.03)
                session, n_chunks = self._upload_chunked(session, source, job_id, work_dir, stopped)
                script_path = work_dir / 'run_colab_job.py'
                script_path.write_text(build_worker_script(config, job_id, source_sha256, n_chunks), encoding='utf-8')
                session = self._ensure_session(owner, session['endpoint'], stopped=stopped, allow_create=False)
                remote_script = f'/content/chep-job/{job_id}/run_colab_job.py'
                self.cli.run_helper(
                    _upload_file_helper(),
                    {
                        'url': session['url'],
                        'token': session['token'],
                        'endpoint': session['endpoint'],
                        'local_path': str(script_path),
                        'remote_path': remote_script,
                    },
                    timeout=120,
                    stopped=stopped,
                )
                launch_code = (
                    f"import pathlib, subprocess, sys\n"
                    f"log_f = open('/content/chep-job/{job_id}/worker.log', 'w', encoding='utf-8')\n"
                    f"subprocess.Popen([sys.executable, {remote_script!r}], stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True)\n"
                )
                self.cli.run_helper(
                    _exec_helper(),
                    {
                        'url': session['url'],
                        'token': session['token'],
                        'endpoint': session['endpoint'],
                        'code': launch_code,
                        'timeout': 60,
                    },
                    timeout=90,
                    stopped=stopped,
                )
                remote.update(submitted=True, stage='running')
                self.save(job_id, remote)
            except ColabStopped:
                raise
            except Exception:
                remote['stage'] = 'failed'
                self.save(job_id, remote)
                raise

        try:
            self.store.update(job_id, status='REMOTE_RUNNING', progress=0.16)
            status_path = f'/content/chep-job/{job_id}/status.json'
            deadline = time.monotonic() + 14400
            while True:
                if stopped():
                    raise ColabStopped()
                if time.monotonic() > deadline:
                    raise RuntimeError('Google Colab transcription timed out after 4 hours.')
                session = self._ensure_session(owner, session['endpoint'], stopped=stopped, allow_create=False)
                poll = self.cli.run_helper(
                    _poll_status_helper(),
                    {
                        'url': session['url'],
                        'token': session['token'],
                        'endpoint': session['endpoint'],
                        'status_path': status_path,
                    },
                    timeout=90,
                    stopped=stopped,
                )
                st = poll.get('status') or {}
                if st.get('done'):
                    if not st.get('ok'):
                        raise RuntimeError('Google Colab transcription failed: ' + str(st.get('error', 'Unknown error.')))
                    break
                if st:
                    updates: dict[str, Any] = {'progress': float(st.get('progress', 0.16))}
                    if st.get('duration') is not None:
                        updates['duration'] = float(st['duration'])
                    if st.get('processed') is not None:
                        updates['processed'] = float(st['processed'])
                    self.store.update(job_id, **updates)
                for _ in range(10):
                    if stopped():
                        raise ColabStopped()
                    time.sleep(0.3)

            self.store.update(job_id, status='DOWNLOADING', progress=0.96)
            results_dir = work_dir / 'results'
            session = self._ensure_session(owner, session['endpoint'], stopped=stopped, allow_create=False)
            self.cli.run_helper(
                _download_results_helper(),
                {
                    'url': session['url'],
                    'token': session['token'],
                    'endpoint': session['endpoint'],
                    'local_dir': str(results_dir),
                    'files': [
                        {'remote': f'/content/chep-job/{job_id}/result.json', 'name': 'result.json'},
                        {'remote': f'/content/chep-job/{job_id}/{job_id}.txt', 'name': f'{job_id}.txt'},
                        {'remote': f'/content/chep-job/{job_id}/{job_id}.srt', 'name': f'{job_id}.srt'},
                    ],
                },
                timeout=180,
                stopped=stopped,
            )
            result = validate_colab_result(results_dir, job_id, source_sha256)
            validate_model_profile(result, config)
        except ColabStopped:
            raise
        except Exception:
            remote['stage'] = 'failed'
            self.save(job_id, remote)
            raise

        if stopped():
            raise ColabStopped()
        with guard:
            if stopped():
                raise ColabStopped()
            outputs_dir = self.store.root / 'outputs'
            for ext in ('txt', 'srt'):
                (results_dir / f'{job_id}.{ext}').replace(outputs_dir / f'{job_id}.{ext}')
            self.store.update(
                job_id,
                status='COMPLETED',
                progress=1.0,
                duration=result.get('duration'),
                processed=result.get('duration') or 0.0,
                profile=json.dumps(result['profile']),
            )
        source.unlink(missing_ok=True)
        try:
            self.cleanup(job_id)
        except Exception:
            pass
