"""Durable remote transcription; only explicitly selected Kaggle jobs leave this machine."""
from __future__ import annotations

import hashlib
import json
import logging
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
from .kaggle_merge import DUAL_THRESHOLD_SECONDS, merge_segments, shard_ranges
from .kaggle_notebook import notebook

REMOTE_ACTIVE = ('UPLOADING', 'WAITING_GPU', 'REMOTE_RUNNING', 'DOWNLOADING')
MAX_REMOTE_BYTES = 2 * 1024**3
FIRST_LOG_TIMEOUT_SECONDS = 120
FIRST_LOG_RECONNECT_GRACE_SECONDS = 10
MAX_FIRST_LOG_RETRIES = 2
log = logging.getLogger(__name__)


def cleanup_response(output, kind, ref):
    """Classify a failed deletion without accepting unrelated failures as 404s."""
    lower = output.lower()
    if re.search(r'\b(401|403|429|5\d\d)\b', lower):
        return 'error'
    if re.search(r'\b404\b', lower):
        return 'missing'
    label = 'notebook' if kind == 'kernels' else 'dataset'
    if 'not found' in lower and (label in lower or kind[:-1] in lower or ref.lower() in lower):
        return 'missing'
    return 'error'


class CleanupError(ValueError):
    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind


def cleanup_failure(output):
    if re.search(r'\b401\b|unauthenticated', output, re.I):
        return CleanupError('authentication', 'Kaggle authentication expired. Reauthenticate the CLI; Chep will retry cleanup automatically.')
    if re.search(r'\b403\b|forbidden|permission denied', output, re.I):
        return CleanupError('permission', 'Kaggle denied deletion. Check the signed-in account; Chep will retry cleanup automatically.')
    if re.search(r'\b429\b|too many requests|rate limit', output, re.I):
        return CleanupError('rate_limit', 'Kaggle is rate limiting requests. Chep will retry cleanup automatically.')
    if re.search(r'\b5\d\d\b', output):
        return CleanupError('server', 'Kaggle server error. Chep will retry cleanup automatically.')
    return CleanupError('unexpected', 'Kaggle could not remove this resource. Check your account; Chep will retry cleanup automatically.')


def kernel_status(output):
    match = re.search(r'has status "(?:KernelWorkerStatus\.)?([A-Z_]+)"', output)
    return match.group(1) if match else None


def silent_startup_overdue(running_since, monitor, now=None):
    current = time.time() if now is None else now
    return (not monitor.seen.is_set() and
            current - running_since >= FIRST_LOG_TIMEOUT_SECONDS and
            current - monitor.started_at >= FIRST_LOG_RECONNECT_GRACE_SECONDS)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_result(directory, job_id, source_hash, *, stem=None, part_index=None,
                    span=None, source_duration=None, model_dataset=None):
    manifest = directory / 'result.json'
    if not manifest.is_file() or manifest.stat().st_size > 1024**2:
        raise ValueError('Kaggle did not return a valid result manifest.')
    result = json.loads(manifest.read_text(encoding='utf-8'))
    if result.get('schema_version') != 1 or result.get('job_id') != job_id or result.get('source_sha256') != source_hash:
        raise ValueError('Kaggle result does not match this recording.')
    if result.get('ok') is not True:
        raise ValueError('Kaggle transcription failed: ' + str(result.get('error', 'Unknown remote error.'))[:1000])
    if part_index is not None and result.get('part_index') != part_index:
        raise ValueError('Kaggle shard index mismatch.')
    if span is not None and result.get('span') != list(span):
        raise ValueError('Kaggle shard range mismatch.')
    if source_duration is not None:
        actual = result.get('source_duration')
        if not isinstance(actual, (int, float)) or abs(actual - source_duration) > .1:
            raise ValueError('Kaggle decoded a different recording duration.')
    if model_dataset is not None and result.get('model_dataset') != model_dataset:
        raise ValueError('Kaggle returned a different public model Dataset.')
    profile = result.get('profile', {})
    if profile.get('device') != 'cuda' or profile.get('provider') != 'kaggle' or 'T4' not in profile.get('device_name', ''):
        raise ValueError('Kaggle did not confirm the requested GPU profile.')
    for ext in (('txt', 'srt', 'segments.json') if span is not None else ('txt', 'srt')):
        path = directory / ((stem or job_id) + '.' + ext)
        meta = result.get('files', {}).get(ext, {})
        if not path.is_file() or path.is_symlink() or path.stat().st_size != meta.get('size') or sha(path) != meta.get('sha256'):
            raise ValueError('Kaggle transcript integrity check failed.')
    return result


def validate_model_profile(result, config):
    profile = result['profile']
    model = config['model_config']
    expected = model['sweet_spot']
    if (profile.get('language') != config['language'] or
            profile.get('revision') != model['revision'] or
            profile.get('compute_type') != expected['compute_type'] or
            profile.get('concurrency') != expected['concurrency']):
        raise ValueError('Kaggle returned a different model, language, or GPU profile.')


class KaggleBackend:
    def __init__(self, store):
        self.store = store
        self.cli = KaggleCLI(store.root)

    def save(self, job_id, remote):
        self.store.update(job_id, remote=json.dumps(remote))

    def _restart_silent_kernel(self, job_id, remote, entry, folder, run, index=None):
        """Replace only one RUNNING notebook after 120 seconds without a live log."""
        if entry.get('startup_retries', 0) >= MAX_FIRST_LOG_RETRIES:
            raise ValueError('A Kaggle notebook repeatedly started without returning logs. '
                             'Automatic retries are exhausted; retry the job later.')
        old_ref = entry['ref']
        suffix = f'-{index}' if index is not None else ''
        if not re.fullmatch(re.escape(remote['owner']) + r'/chep-run-[a-f0-9]{24}' +
                            re.escape(suffix), old_ref):
            raise CleanupError('metadata', 'Invalid Chep notebook ID; automatic restart was stopped.')
        output, code = run(['kernels', 'delete', old_ref, '-y'], check=False)
        if code and cleanup_response(output, 'kernels', old_ref) != 'missing':
            raise cleanup_failure(output)
        new_ref = remote['owner'] + '/chep-run-' + uuid.uuid4().hex[:24] + suffix
        metadata_path = folder / 'kernel-metadata.json'
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        metadata.update(id=new_ref, title=new_ref.split('/')[1])
        atomic_json(metadata_path, metadata)
        remote.setdefault('retired_kernels', []).append({
            'ref': old_ref, 'attempted': True, 'cleaned': True})
        remote.setdefault('startup_retry_history', []).append({
            'old_ref': old_ref, 'new_ref': new_ref, 'part_index': index,
            'running_since': entry.get('running_since'), 'restarted_at': time.time()})
        entry.update(ref=new_ref, attempted=False, cleaned=False, submitted=False,
                     complete=False, running_since=None, first_log_seen_at=None,
                     startup_retries=entry.get('startup_retries', 0) + 1)
        if index is None:
            remote.update(kernel=new_ref, kernel_attempted=False, kernel_cleaned=False,
                          running_since=None, first_log_seen_at=None,
                          startup_retries=entry['startup_retries'], submitted_at=time.time())
        self.save(job_id, remote)  # Persist the new owned ID before submission.
        entry['attempted'] = True
        if index is None:
            remote['kernel_attempted'] = True
        self.save(job_id, remote)
        run(['kernels', 'push', '-p', str(folder), '--accelerator', 'NvidiaTeslaT4',
             '-t', '14400'], timeout=180)
        entry['submitted'] = True
        self.save(job_id, remote)
        log.warning('Restarted Kaggle notebook without a first log for job %s, shard %s',
                    job_id, index if index is not None else 'single')

    def cleanup(self, job_id):
        job = self.store.get(job_id, include_removed=True)
        if not job:
            return
        remote = json.loads(job.get('remote') or '{}')
        if not remote or remote.get('cleaned'):
            return
        try:
            kernels = (remote.get('kernels', []) if remote.get('dual') else [{
                'ref': remote.get('kernel'), 'attempted': remote.get('kernel_attempted'),
                'cleaned': remote.get('kernel_cleaned')}]) + remote.get('retired_kernels', [])
            pending_kernels = [entry for entry in kernels if entry.get('attempted') and not entry.get('cleaned')]
            pending_dataset = remote.get('dataset_attempted') and not remote.get('dataset_cleaned')
            if (pending_kernels or pending_dataset) and self.cli.identity() != remote['owner']:
                raise CleanupError('authentication', 'Sign in as ' + remote['owner'] + ' to remove this remote session.')
            # Delete only resources recorded for this job, never enumerate/delete user resources.
            for entry in pending_kernels:
                ref = entry['ref']
                pattern = (re.escape(remote['owner']) + r'/chep-run-[a-f0-9]{24}' +
                           (r'-[01]' if remote.get('dual') else ''))
                if not re.fullmatch(pattern, ref):
                    raise CleanupError('metadata', 'Invalid Chep remote resource identifier. Cleanup requires the original resource IDs.')
                output, code = self.cli.run(['kernels', 'delete', ref, '-y'], check=False)
                if code and cleanup_response(output, 'kernels', ref) != 'missing':
                    raise cleanup_failure(output)
                entry['cleaned'] = True
                if not remote.get('dual') and ref == remote.get('kernel'):
                    remote['kernel_cleaned'] = True
                self.save(job_id, remote)
                log.info('Kaggle kernel cleanup complete for job %s (%s)', job_id, 'missing' if code else 'deleted')
            if pending_dataset:
                ref = remote['dataset']
                if not re.fullmatch(re.escape(remote['owner']) + r'/chep-input-[a-f0-9]{24}', ref):
                    raise CleanupError('metadata', 'Invalid Chep remote Dataset identifier.')
                output, code = self.cli.run(['datasets', 'delete', ref, '-y'], check=False)
                if code and cleanup_response(output, 'datasets', ref) != 'missing':
                    raise cleanup_failure(output)
                remote['dataset_cleaned'] = True
                self.save(job_id, remote)
                log.info('Kaggle Dataset cleanup complete for job %s (%s)', job_id, 'missing' if code else 'deleted')
            remote.update(cleaned=True, cleanup_error=None, cleanup_kind=None)
        except Exception as exc:
            remote['cleanup_error'] = str(exc)
            remote['cleanup_kind'] = getattr(exc, 'kind', 'timeout' if 'timed out' in str(exc).lower() else 'local_cli' if isinstance(exc, OSError) else 'unexpected')
            log.warning('Kaggle cleanup pending for job %s: %s', job_id, type(exc).__name__)
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
        model_dataset = config['model_config'].get('kaggle_dataset')
        if not model_dataset:
            raise ValueError('This model has no pinned public Kaggle Dataset.')
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
        if remote.get('model_dataset'):
            # A resumed session keeps the exact public input version it started with.
            model_dataset = remote['model_dataset']
            config['model_config']['kaggle_dataset'] = model_dataset
        if not remote:
            from .worker import decoded_duration
            duration = decoded_duration(source)
            nonce = uuid.uuid4().hex[:24]
            remote = {'owner': owner, 'dataset': owner + '/chep-input-' + nonce,
                      'kernel': owner + '/chep-run-' + nonce, 'stage': 'created',
                      'source_sha256': sha(source), 'cleaned': False,
                      'source_duration': duration, 'dual': duration >= DUAL_THRESHOLD_SECONDS,
                      'model_dataset': model_dataset}
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
            if remote.get('dual'):
                remote['spans'] = shard_ranges(remote['source_duration'])
                remote['kernels'] = [{'ref': remote['kernel'] + f'-{index}',
                                      'attempted': False, 'cleaned': False, 'submitted': False}
                                     for index in range(2)]
                for index, entry in enumerate(remote['kernels']):
                    folder = kernel / str(index)
                    folder.mkdir(exist_ok=True)
                    atomic_json(folder / 'run.ipynb', notebook(config, job_id, remote['source_sha256'],
                        part_index=index, span=remote['spans'][index], source_duration=remote['source_duration']))
                    atomic_json(folder / 'kernel-metadata.json', {'id': entry['ref'],
                        'title': entry['ref'].split('/')[1], 'code_file': 'run.ipynb', 'language': 'python',
                        'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True, 'enable_internet': True,
                        'machine_shape': 'NvidiaTeslaT4', 'dataset_sources': [remote['dataset'], model_dataset],
                        'competition_sources': [], 'kernel_sources': [], 'model_sources': []})
                remote['stage'] = 'dual_submitting'
                self.save(job_id, remote)
            else:
                if not remote.get('model_dataset'):
                    remote['model_dataset'] = model_dataset
                atomic_json(kernel / 'run.ipynb', notebook(config, job_id, remote['source_sha256']))
                atomic_json(kernel / 'kernel-metadata.json', {'id': remote['kernel'],
                    'title': remote['kernel'].split('/')[1], 'code_file': 'run.ipynb', 'language': 'python',
                    'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True, 'enable_internet': True,
                    'dataset_sources': [remote['dataset'], model_dataset], 'competition_sources': [], 'kernel_sources': [], 'model_sources': []})
                remote['stage'] = 'submitting'
                self.save(job_id, remote)
        if remote.get('dual'):
            try:
                return self._run_dual(job_id, config, remote, source, work, kernel, outputs, stopped, guard)
            except KaggleStopped:
                raise
            except (ValueError, KeyError, json.JSONDecodeError):
                remote['stage'] = 'failed'
                self.save(job_id, remote)
                try:
                    self.cleanup(job_id)
                except Exception:
                    pass  # Retry retains exact owned IDs for another cleanup attempt.
                raise
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
        monitor = None
        try:
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
                if status == 'RUNNING' and not remote.get('first_log_seen_at'):
                    if not remote.get('running_since'):
                        remote['running_since'] = time.time()
                        self.save(job_id, remote)
                    if monitor is None:
                        monitor = self.cli.watch_first_log(remote['kernel'])
                    if monitor.seen.is_set():
                        remote['first_log_seen_at'] = monitor.first_log_at or time.time()
                        self.save(job_id, remote)
                        monitor.close()
                        monitor = None
                    elif silent_startup_overdue(remote['running_since'], monitor):
                        monitor.close()
                        monitor = None
                        entry = {'ref': remote['kernel'], 'attempted': remote.get('kernel_attempted'),
                                 'cleaned': remote.get('kernel_cleaned'),
                                 'running_since': remote['running_since'],
                                 'startup_retries': remote.get('startup_retries', 0)}
                        self._restart_silent_kernel(job_id, remote, entry, kernel, run)
                        stage('WAITING_GPU', .20)
                        continue
                stage('REMOTE_RUNNING' if status == 'RUNNING' else 'WAITING_GPU',
                      .35 if status == 'RUNNING' else .20)
                if time.time() - remote.get('submitted_at', time.time()) > 5 * 3600:
                    raise ValueError('Kaggle session is taking longer than expected. Retry to reconnect.')
                wait(10)
        finally:
            if monitor is not None:
                monitor.close()
        stage('DOWNLOADING', .90)
        run(['kernels', 'output', remote['kernel'], '-p', str(outputs), '--file-pattern',
             rf'^(result\.json|{job_id}\.(txt|srt))$', '--force'], timeout=600)
        try:
            result = validate_result(outputs, job_id, remote['source_sha256'],
                                     model_dataset=remote.get('model_dataset'))
            validate_model_profile(result, config)
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

    def _run_dual(self, job_id, config, remote, source, work, kernel, outputs, stopped, guard):
        from .worker import stamp

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

        if remote['stage'] == 'dual_submitting':
            stage('WAITING_GPU', .15)
            for index, entry in enumerate(remote['kernels']):
                if entry.get('submitted'):
                    continue
                if entry.get('attempted'):
                    output, code = run(['kernels', 'status', entry['ref']], check=False)
                    if code or not kernel_status(output):
                        raise ValueError('Kaggle submission status is uncertain. Retry after checking cleanup.')
                else:
                    entry['attempted'] = True
                    self.save(job_id, remote)
                    run(['kernels', 'push', '-p', str(kernel / str(index)),
                         '--accelerator', 'NvidiaTeslaT4', '-t', '14400'], timeout=180)
                entry['submitted'] = True
                self.save(job_id, remote)
            remote['stage'] = 'dual_submitted'
            remote['submitted_at'] = time.time()
            self.save(job_id, remote)

        if remote['stage'] != 'dual_submitted':
            raise ValueError('Invalid dual-notebook session state.')
        pending = {index for index, entry in enumerate(remote['kernels']) if not entry.get('complete')}
        failures = 0
        monitors = {}
        try:
            while pending:
                any_running = False
                for index in sorted(pending):
                    entry = remote['kernels'][index]
                    try:
                        output, _ = run(['kernels', 'status', entry['ref']])
                        failures = 0
                    except RuntimeError:
                        failures += 1
                        if failures >= 5:
                            raise
                        wait(10)
                        continue
                    status = kernel_status(output)
                    if status == 'COMPLETE':
                        if index in monitors:
                            monitors.pop(index).close()
                        entry['complete'] = True
                        pending.remove(index)
                        self.save(job_id, remote)
                    elif status and any(word in status for word in ('ERROR', 'CANCEL', 'FAILED')):
                        remote['stage'] = 'failed'
                        self.save(job_id, remote)
                        try:
                            self.cleanup(job_id)
                        except Exception:
                            pass  # Exact owned resources remain recorded for Retry cleanup.
                        raise ValueError('One Kaggle shard failed. Retry the complete job after cleanup.')
                    elif status == 'RUNNING':
                        any_running = True
                        if not entry.get('first_log_seen_at'):
                            if not entry.get('running_since'):
                                entry['running_since'] = time.time()
                                self.save(job_id, remote)
                            if index not in monitors:
                                monitors[index] = self.cli.watch_first_log(entry['ref'])
                            monitor = monitors[index]
                            if monitor.seen.is_set():
                                entry['first_log_seen_at'] = monitor.first_log_at or time.time()
                                self.save(job_id, remote)
                                monitors.pop(index).close()
                            elif silent_startup_overdue(entry['running_since'], monitor):
                                monitors.pop(index).close()
                                self._restart_silent_kernel(job_id, remote, entry,
                                    kernel / str(index), run, index=index)
                    elif status is None:
                        raise ValueError('Kaggle returned an unknown shard status. Retry to reconnect.')
                if pending:
                    stage('REMOTE_RUNNING' if any_running else 'WAITING_GPU', .35 if any_running else .20)
                    if time.time() - remote.get('submitted_at', time.time()) > 5 * 3600:
                        raise ValueError('Kaggle shards are taking longer than expected. Retry to reconnect.')
                    wait(10)
        finally:
            for monitor in monitors.values():
                monitor.close()

        stage('DOWNLOADING', .90)
        results = []
        shards = []
        for index, entry in enumerate(remote['kernels']):
            folder = outputs / str(index)
            folder.mkdir(exist_ok=True)
            stem = f'{job_id}-{index}'
            run(['kernels', 'output', entry['ref'], '-p', str(folder), '--file-pattern',
                 rf'^(result\.json|{stem}\.(txt|srt|segments\.json))$', '--force'], timeout=600)
            result = validate_result(folder, job_id, remote['source_sha256'], stem=stem,
                part_index=index, span=remote['spans'][index],
                source_duration=remote['source_duration'], model_dataset=remote['model_dataset'])
            validate_model_profile(result, config)
            segment_path = folder / (stem + '.segments.json')
            if segment_path.stat().st_size > 64 * 1024**2:
                raise ValueError('Kaggle shard segment file is too large.')
            shards.append(json.loads(segment_path.read_text(encoding='utf-8')))
            results.append(result)
        segments, audit = merge_segments(shards, remote['spans'])
        atomic_json(work / 'seam-audit.json', {'overlap_seconds': 5, 'removed': audit})
        text = ''.join(f"[{stamp(s['start'])} - {stamp(s['end'])}] {s['text']}\n" for s in segments)
        srt = ''.join(f"{number}\n{stamp(s['start'], True)} --> {stamp(s['end'], True)}\n{s['text']}\n\n"
                      for number, s in enumerate(segments, 1))
        with guard:
            if stopped():
                raise KaggleStopped()
            for extension, content in (('txt', text), ('srt', srt)):
                temporary = outputs / (job_id + '.' + extension + '.part')
                temporary.write_text(content, encoding='utf-8')
                os.replace(temporary, self.store.root / 'outputs' / (job_id + '.' + extension))
            profile = {**results[0]['profile'], 'dual_notebook': True,
                       'overlap_seconds': 5, 'model_dataset': remote['model_dataset']}
            self.store.update(job_id, status='COMPLETED', progress=1,
                              duration=remote['source_duration'], processed=remote['source_duration'],
                              profile=json.dumps(profile), error=None)
        remote['stage'] = 'imported'
        self.save(job_id, remote)
        source.unlink(missing_ok=True)
        try:
            self.cleanup(job_id)
        except Exception:
            pass
