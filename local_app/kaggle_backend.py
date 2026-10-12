"""Durable remote transcription; only explicitly selected Kaggle jobs leave this machine."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from contextlib import nullcontext

from .catalog import resolve_artifact
from .config import atomic_json
from .kaggle_cli import KaggleCLI, KaggleStopped
from .kaggle_notebook import notebook

REMOTE_ACTIVE = ('UPLOADING', 'WAITING_GPU', 'REMOTE_RUNNING', 'DOWNLOADING')
MAX_REMOTE_BYTES = 2 * 1024**3


def kernel_status(output):
    match = re.search(r'has status "(?:KernelWorkerStatus\.)?([A-Z_]+)"', output)
    return match.group(1) if match else None


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_result(directory, job_id, source_hash):
    manifest = directory / 'result.json'
    if not manifest.is_file() or manifest.stat().st_size > 1024**2:
        raise ValueError('Kaggle did not return a valid result manifest.')
    result = json.loads(manifest.read_text(encoding='utf-8'))
    if result.get('schema_version') != 1 or result.get('job_id') != job_id or result.get('source_sha256') != source_hash:
        raise ValueError('Kaggle result does not match this recording.')
    if result.get('ok') is not True:
        raise ValueError('Kaggle transcription failed: ' + str(result.get('error', 'Unknown remote error.'))[:1000])
    profile = result.get('profile', {})
    if profile.get('device') != 'cuda' or profile.get('provider') != 'kaggle' or 'T4' not in profile.get('device_name', ''):
        raise ValueError('Kaggle did not confirm the requested GPU profile.')
    for ext in ('txt', 'srt'):
        path = directory / (job_id + '.' + ext)
        meta = result.get('files', {}).get(ext, {})
        if not path.is_file() or path.is_symlink() or path.stat().st_size != meta.get('size') or sha(path) != meta.get('sha256'):
            raise ValueError('Kaggle transcript integrity check failed.')
    return result


class KaggleBackend:
    def __init__(self, store):
        self.store = store
        self.cli = KaggleCLI(store.root)

    def save(self, job_id, remote):
        self.store.update(job_id, remote=json.dumps(remote))

    def cleanup(self, job_id):
        job = self.store.get(job_id)
        remote = json.loads(job.get('remote') or '{}')
        if not remote or remote.get('cleaned'):
            return
        if self.cli.identity() != remote['owner']:
            raise ValueError('Sign in as ' + remote['owner'] + ' to remove this remote session.')
        try:
            # Delete only resources recorded for this job, never enumerate/delete user resources.
            for kind, key in (('kernels', 'kernel'), ('datasets', 'dataset')):
                if not remote.get(key + '_attempted'):
                    continue
                ref = remote[key]
                if not re.fullmatch(re.escape(remote['owner']) + r'/chep-(?:run|input)-[a-f0-9]{24}', ref):
                    raise ValueError('Invalid Chep remote resource identifier.')
                output, code = self.cli.run([kind, 'delete', ref, '-y'], check=False)
                if code and '404' not in output and 'not found' not in output.lower():
                    raise RuntimeError('Could not remove all Kaggle files. Open your Kaggle account or retry cleanup.')
            remote.update(cleaned=True, cleanup_error=None)
        except Exception as exc:
            remote['cleanup_error'] = str(exc)
            raise
        finally:
            self.save(job_id, remote)

    def run(self, job, stopped, guard=None):
        guard = guard or nullcontext()
        job_id = job['id']
        config = json.loads(job['config'])
        config['model_config'] = resolve_artifact(config['language'], config['model_config'])
        if not config['model_config'].get('sweet_spot'):
            raise ValueError('This model does not have a validated Kaggle GPU profile.')
        source = self.store.root / 'sources' / job_id
        if source.stat().st_size > MAX_REMOTE_BYTES:
            raise ValueError('Online GPU supports recordings up to 2 GB. Split this recording or use Local processing.')
        owner = self.cli.identity()
        remote = json.loads(job.get('remote') or '{}')
        if remote and remote['owner'] != owner:
            raise ValueError('Sign in as ' + remote['owner'] + ' to resume this recording.')
        if remote.get('cleaned'):
            remote = {}
        if remote.get('stage') == 'failed':
            self.cleanup(job_id)
            remote = {}
        if not remote:
            nonce = uuid.uuid4().hex[:24]
            remote = {'owner': owner, 'dataset': owner + '/chep-input-' + nonce,
                      'kernel': owner + '/chep-run-' + nonce, 'stage': 'created',
                      'source_sha256': sha(source), 'cleaned': False}
            # Persist IDs before any network mutation, including ambiguous timeouts.
            self.save(job_id, remote)
        work = self.store.root / 'work' / job_id
        work.mkdir(exist_ok=True)
        inputs, kernel, outputs = (work / name for name in ('input', 'kernel', 'results'))
        for path in (inputs, kernel, outputs):
            path.mkdir(exist_ok=True)

        def stage(status, progress):
            with guard:
                if stopped():
                    raise KaggleStopped()
                self.store.update(job_id, status=status, progress=progress)

        def run(args, timeout=120, check=True):
            return self.cli.run(args, timeout=timeout, stopped=stopped, check=check)

        def wait(seconds):
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                if stopped():
                    raise KaggleStopped()
                time.sleep(.3)

        if remote['stage'] == 'created':
            stage('UPLOADING', .02)
            # On restart, check the deterministic ID before attempting creation again.
            output, code = run(['datasets', 'status', remote['dataset']], check=False)
            if code:
                audio = inputs / (job_id + '.audio')
                if not audio.exists():
                    try:
                        os.link(source, audio)
                    except OSError:
                        shutil.copyfile(source, audio)
                # A private dataset with 'other' license grants no new open license to user recordings.
                atomic_json(inputs / 'dataset-metadata.json', {'id': remote['dataset'],
                    'title': remote['dataset'].split('/')[1], 'licenses': [{'name': 'other'}]})
                remote['dataset_attempted'] = True
                self.save(job_id, remote)
                run(['datasets', 'create', '-p', str(inputs)], timeout=3600)
            remote['stage'] = 'dataset'
            self.save(job_id, remote)
        if remote['stage'] == 'dataset':
            stage('UPLOADING', .10)
            for attempt in range(120):
                output, code = run(['datasets', 'status', remote['dataset']], check=False)
                if not code and output.strip().lower() == 'ready':
                    break
                if not code and output.strip().lower() in ('error', 'failed'):
                    remote['stage'] = 'failed'
                    self.save(job_id, remote)
                    raise ValueError('Kaggle could not prepare this dataset. Retry to upload a new private dataset.')
                wait(5)
            else:
                raise ValueError('Kaggle is still preparing the dataset. Retry to resume.')
            atomic_json(kernel / 'run.ipynb', notebook(config, job_id, remote['source_sha256']))
            atomic_json(kernel / 'kernel-metadata.json', {'id': remote['kernel'],
                'title': remote['kernel'].split('/')[1], 'code_file': 'run.ipynb', 'language': 'python',
                'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True, 'enable_internet': True,
                'dataset_sources': [remote['dataset']], 'competition_sources': [], 'kernel_sources': [], 'model_sources': []})
            remote['stage'] = 'submitting'
            self.save(job_id, remote)
        if remote['stage'] == 'submitting':
            stage('WAITING_GPU', .15)
            output, code = run(['kernels', 'status', remote['kernel']], check=False)
            if code or not kernel_status(output):
                # No blind automatic resubmission after timeout: saved stage is checked on Retry.
                remote['kernel_attempted'] = True
                self.save(job_id, remote)
                run(['kernels', 'push', '-p', str(kernel), '--accelerator', 'NvidiaTeslaT4', '-t', '14400'], timeout=180)
            remote['stage'] = 'submitted'
            remote['submitted_at'] = time.time()
            self.save(job_id, remote)
        stage('WAITING_GPU', .20)
        failures = 0
        while True:
            try:
                output, _ = run(['kernels', 'status', remote['kernel']])
                failures = 0
            except RuntimeError:
                failures += 1
                if failures >= 5:
                    raise
                wait(10)
                continue
            status = kernel_status(output)
            if not status:
                raise ValueError('Kaggle returned an unknown session status. Retry to reconnect.')
            if status == 'COMPLETE':
                break
            if any(word in status for word in ('ERROR', 'CANCEL', 'FAILED')):
                remote['stage'] = 'failed'
                self.save(job_id, remote)
                raise ValueError('Kaggle stopped this notebook. Check the session in your Kaggle account and Retry.')
            stage('REMOTE_RUNNING' if status == 'RUNNING' else 'WAITING_GPU', .35 if status == 'RUNNING' else .20)
            if time.time() - remote.get('submitted_at', time.time()) > 5 * 3600:
                raise ValueError('Kaggle session is taking longer than expected. Retry to reconnect.')
            wait(10)
        stage('DOWNLOADING', .90)
        run(['kernels', 'output', remote['kernel'], '-p', str(outputs), '--file-pattern',
             rf'^(result\.json|{job_id}\.(txt|srt))$', '--force'], timeout=600)
        try:
            result = validate_result(outputs, job_id, remote['source_sha256'])
            if result['profile'].get('language') != config['language'] or result['profile'].get('revision') != config['model_config']['revision']:
                raise ValueError('Kaggle returned a different model or language.')
        except ValueError:
            remote['stage'] = 'failed'
            self.save(job_id, remote)
            raise
        with guard:
            if stopped():
                raise KaggleStopped()
            for ext in ('txt', 'srt'):
                os.replace(outputs / (job_id + '.' + ext), self.store.root / 'outputs' / (job_id + '.' + ext))
            self.store.update(job_id, status='COMPLETED', progress=1, duration=result['duration'],
                              processed=result['duration'], profile=json.dumps(result['profile']), error=None)
        remote['stage'] = 'imported'
        self.save(job_id, remote)
        source.unlink(missing_ok=True)
        try:
            self.cleanup(job_id)
        except Exception:
            pass  # Result remains available; cleanup failure stays visible and retryable.
