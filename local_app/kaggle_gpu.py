"""Explicit, short Kaggle GPU probe. Quota alone does not prove GPU access."""
from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid

from .config import atomic_json
from .kaggle_backend import kernel_status, cleanup_response
from .kaggle_cli import safe_error

log = logging.getLogger(__name__)


class KaggleGpuCheck:
    def __init__(self, root, cli):
        self.root, self.cli = root, cli
        self.pending_path = root / 'gpu_check_cleanup.json'
        self.lock = threading.RLock()
        self.thread = None
        self.stop = threading.Event()
        self.state = {'status': 'unchecked', 'username': None, 'message': '', 'quota': None}

    def snapshot(self):
        with self.lock:
            state = dict(self.state)
        if state['username'] != self.cli.snapshot().get('username') or self.cli.snapshot().get('status') != 'ready':
            return {'status': 'unchecked', 'username': None, 'message': '', 'quota': None}
        if state['status'] == 'ready' and self.pending_path.is_file():
            state['status'] = 'cleanup_pending'
        return state

    def start(self):
        if self.cli.snapshot().get('status') != 'ready':
            raise ValueError('Authenticate Kaggle CLI before checking GPU access.')
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise ValueError('Kaggle GPU check is already running.')
            username = self.cli.snapshot()['username']
            if self.state['status'] == 'ready' and self.state['username'] == username and not self.pending_path.is_file():
                return
            self.state = {'status': 'checking', 'username': username,
                          'message': 'Checking Kaggle GPU access. This may take a few minutes.', 'quota': None}
            self.thread = threading.Thread(target=self._check, args=(username,), daemon=True)
            self.thread.start()

    def _check(self, username):
        ref = None
        try:
            if self.cli.identity() != username:
                raise ValueError('Kaggle account changed. Authenticate the CLI and check again.')
            if self.pending_path.is_file():
                saved = json.loads(self.pending_path.read_text(encoding='utf-8'))
                if saved.get('owner') != username:
                    raise ValueError('Sign in as ' + str(saved.get('owner')) + ' to remove a previous GPU test notebook.')
                self._delete_probe(saved['ref'], username)
            output, _ = self.cli.run(['quota', '--format', 'json'])
            rows = json.loads(output)
            gpu = next((row for row in rows if row.get('resource') == 'GPU'), None)
            if not gpu:
                self._set('verification_required', username, 'Kaggle did not report GPU access. Verify your phone number in ')
                return
            if gpu.get('remaining', '').startswith('0.00'):
                self._set('quota_exhausted', username, 'Your Kaggle GPU quota is exhausted. Check again after it refreshes.', gpu)
                return
            nonce = uuid.uuid4().hex[:24]
            ref = username + '/chep-gpu-check-' + nonce
            atomic_json(self.pending_path, {'owner': username, 'ref': ref})
            folder = self.root / 'work' / ('gpu-check-' + nonce)
            folder.mkdir(parents=True, exist_ok=True)
            code = "import json, subprocess\ntry:\n p = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True)\n gpu = p.stdout.strip() if p.returncode == 0 else None\nexcept FileNotFoundError:\n gpu = None\nwith open('/kaggle/working/gpu-check.json','w') as f: json.dump({'gpu': gpu}, f)\n"
            atomic_json(folder / 'run.ipynb', {'cells': [{'cell_type': 'code', 'execution_count': None,
                'metadata': {}, 'outputs': [], 'source': code.splitlines(keepends=True)}],
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
                'nbformat': 4, 'nbformat_minor': 5})
            atomic_json(folder / 'kernel-metadata.json', {'id': ref, 'title': ref.split('/')[1],
                'code_file': 'run.ipynb', 'language': 'python', 'kernel_type': 'notebook',
                'is_private': True, 'enable_gpu': True, 'enable_internet': False,
                'dataset_sources': [], 'competition_sources': [], 'kernel_sources': [], 'model_sources': []})
            output, code = self.cli.run(['kernels', 'push', '-p', str(folder), '--accelerator', 'NvidiaTeslaT4', '-t', '300'], timeout=120, check=False)
            if code:
                if re.search(r'\b403\b|verif|phone', output, re.I):
                    self._set('verification_required', username, 'Kaggle has not enabled GPU access for this account. Verify your phone number in ')
                else:
                    self._set('error', username, safe_error(output))
                return
            for _ in range(120):
                if self.stop.is_set():
                    return
                output, _ = self.cli.run(['kernels', 'status', ref])
                status = kernel_status(output)
                if status == 'COMPLETE':
                    break
                if status and any(word in status for word in ('ERROR', 'CANCEL', 'FAILED')):
                    self._set('verification_required', username, 'Kaggle could not start the GPU test. Verify your phone number in ')
                    return
                time.sleep(5)
            else:
                self._set('error', username, 'Kaggle GPU test timed out. Check again later.')
                return
            output_dir = folder / 'output'
            output_dir.mkdir(exist_ok=True)
            self.cli.run(['kernels', 'output', ref, '-p', str(output_dir), '--file-pattern', '^gpu-check.json$', '--force'])
            result = output_dir / 'gpu-check.json'
            device = json.loads(result.read_text(encoding='utf-8')).get('gpu') if result.is_file() else None
            if device and 'T4' in device:
                self._set('ready', username, 'Kaggle T4 GPU confirmed.', gpu)
            else:
                self._set('verification_required', username, 'Kaggle did not assign a T4 GPU. Verify your phone number in ')
        except Exception as exc:
            log.warning('Kaggle GPU check failed: %s', type(exc).__name__)
            self._set('error', username, str(exc) if isinstance(exc, ValueError) else 'Could not confirm Kaggle GPU access. Check your connection and retry.')
        finally:
            if ref:
                try:
                    self._delete_probe(ref, username)
                except Exception:
                    log.exception('Temporary Kaggle GPU probe cleanup pending for %s', ref)
                    with self.lock:
                        self.state['message'] += ' Temporary test notebook cleanup is pending; click Check GPU condition to retry.'

    def _delete_probe(self, ref, username):
        if not re.fullmatch(re.escape(username) + r'/chep-gpu-check-[a-f0-9]{24}', ref):
            raise ValueError('Invalid GPU test notebook identifier.')
        output, code = self.cli.run(['kernels', 'delete', ref, '-y'], check=False)
        if code and cleanup_response(output, 'kernels', ref) != 'missing':
            raise RuntimeError(safe_error(output))
        self.pending_path.unlink(missing_ok=True)

    def _set(self, status, username, message, quota=None):
        with self.lock:
            self.state = {'status': status, 'username': username, 'message': message, 'quota': quota}

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=5)
