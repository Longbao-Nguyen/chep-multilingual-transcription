"""Durable remote transcription; only explicitly selected Kaggle jobs leave this machine."""
from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

from .catalog import resolve_artifact
from .config import atomic_json
from .kaggle_artifacts import (
    format_merged_transcripts,
    prepare_audio_dataset_dir,
    sha,
    validate_model_profile,
    validate_result,
    write_dual_kernel_bundles,
    write_single_kernel_bundle,
)
from .kaggle_cli import KaggleCLI, KaggleStopped
from .kaggle_errors import CleanupError, cleanup_failure, cleanup_response
from .kaggle_merge import DUAL_THRESHOLD_SECONDS, merge_segments, shard_ranges
from .kaggle_polling import (
    DEFAULT_FIRST_LOG_RECONNECT_GRACE_SECONDS,
    DEFAULT_FIRST_LOG_TIMEOUT_SECONDS,
    MAX_FIRST_LOG_RETRIES,
    kernel_status,
    silent_startup_overdue as _silent_startup_overdue,
    wait_cancellable,
    wait_for_dataset_ready,
)

REMOTE_ACTIVE = ('UPLOADING', 'WAITING_GPU', 'REMOTE_RUNNING', 'DOWNLOADING')
MAX_REMOTE_BYTES = 2 * 1024**3
FIRST_LOG_TIMEOUT_SECONDS = DEFAULT_FIRST_LOG_TIMEOUT_SECONDS
FIRST_LOG_RECONNECT_GRACE_SECONDS = DEFAULT_FIRST_LOG_RECONNECT_GRACE_SECONDS
log = logging.getLogger(__name__)

__all__ = [
    'REMOTE_ACTIVE',
    'MAX_REMOTE_BYTES',
    'FIRST_LOG_TIMEOUT_SECONDS',
    'FIRST_LOG_RECONNECT_GRACE_SECONDS',
    'MAX_FIRST_LOG_RETRIES',
    'CleanupError',
    'cleanup_response',
    'cleanup_failure',
    'kernel_status',
    'silent_startup_overdue',
    'sha',
    'validate_result',
    'validate_model_profile',
    'KaggleBackend',
]


def silent_startup_overdue(running_since: float, monitor: Any, now: float | None = None) -> bool:
    """Delegate silent-startup detection using this module's configurable thresholds."""
    return _silent_startup_overdue(
        running_since,
        monitor,
        now,
        timeout_seconds=FIRST_LOG_TIMEOUT_SECONDS,
        grace_seconds=FIRST_LOG_RECONNECT_GRACE_SECONDS,
    )


class KaggleBackend:
    """High-level orchestrator for durable single- and dual-notebook Kaggle jobs."""

    def __init__(self, store: Any) -> None:
        self.store = store
        self.cli = KaggleCLI(store.root)

    def save(self, job_id: str, remote: dict[str, Any]) -> None:
        self.store.update(job_id, remote=json.dumps(remote))

    def _restart_silent_kernel(
        self,
        job_id: str,
        remote: dict[str, Any],
        entry: dict[str, Any],
        folder: Path,
        run: Callable[..., tuple[str, int]],
        index: int | None = None,
    ) -> None:
        """Replace only one RUNNING notebook after 120 seconds without a live log."""
        if entry.get('startup_retries', 0) >= MAX_FIRST_LOG_RETRIES:
            raise ValueError(
                'A Kaggle notebook repeatedly started without returning logs. '
                'Automatic retries are exhausted; retry the job later.'
            )
        old_ref = entry['ref']
        suffix = f'-{index}' if index is not None else ''
        if not re.fullmatch(re.escape(remote['owner']) + r'/chep-run-[a-f0-9]{24}' + re.escape(suffix), old_ref):
            raise CleanupError('metadata', 'Invalid Chep notebook ID; automatic restart was stopped.')
        output, code = run(['kernels', 'delete', old_ref, '-y'], check=False)
        if code and cleanup_response(output, 'kernels', old_ref) != 'missing':
            raise cleanup_failure(output)
        new_ref = remote['owner'] + '/chep-run-' + uuid.uuid4().hex[:24] + suffix
        metadata_path = folder / 'kernel-metadata.json'
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        metadata.update(id=new_ref, title=new_ref.split('/')[1])
        atomic_json(metadata_path, metadata)
        remote.setdefault('retired_kernels', []).append({'ref': old_ref, 'attempted': True, 'cleaned': True})
        remote.setdefault('startup_retry_history', []).append(
            {
                'old_ref': old_ref,
                'new_ref': new_ref,
                'part_index': index,
                'running_since': entry.get('running_since'),
                'restarted_at': time.time(),
            }
        )
        entry.update(
            ref=new_ref,
            attempted=False,
            cleaned=False,
            submitted=False,
            complete=False,
            running_since=None,
            first_log_seen_at=None,
            startup_retries=entry.get('startup_retries', 0) + 1,
        )
        if index is None:
            remote.update(
                kernel=new_ref,
                kernel_attempted=False,
                kernel_cleaned=False,
                running_since=None,
                first_log_seen_at=None,
                startup_retries=entry['startup_retries'],
                submitted_at=time.time(),
            )
        self.save(job_id, remote)  # Persist the new owned ID before submission.
        entry['attempted'] = True
        if index is None:
            remote['kernel_attempted'] = True
        self.save(job_id, remote)
        run(['kernels', 'push', '-p', str(folder), '--accelerator', 'NvidiaTeslaT4', '-t', '14400'], timeout=180)
        entry['submitted'] = True
        self.save(job_id, remote)
        log.warning(
            'Restarted Kaggle notebook without a first log for job %s, shard %s',
            job_id,
            index if index is not None else 'single',
        )

    def cleanup(self, job_id: str) -> None:
        job = self.store.get(job_id, include_removed=True)
        if not job:
            return
        remote = json.loads(job.get('remote') or '{}')
        if not remote or remote.get('cleaned'):
            return
        try:
            kernels = (
                remote.get('kernels', [])
                if remote.get('dual')
                else [
                    {
                        'ref': remote.get('kernel'),
                        'attempted': remote.get('kernel_attempted'),
                        'cleaned': remote.get('kernel_cleaned'),
                    }
                ]
            ) + remote.get('retired_kernels', [])
            pending_kernels = [entry for entry in kernels if entry.get('attempted') and not entry.get('cleaned')]
            pending_dataset = remote.get('dataset_attempted') and not remote.get('dataset_cleaned')
            if (pending_kernels or pending_dataset) and self.cli.identity() != remote['owner']:
                raise CleanupError('authentication', 'Sign in as ' + remote['owner'] + ' to remove this remote session.')
            # Delete only resources recorded for this job, never enumerate/delete user resources.
            for entry in pending_kernels:
                ref = entry['ref']
                pattern = (
                    re.escape(remote['owner'])
                    + r'/chep-run-[a-f0-9]{24}'
                    + (r'-[01]' if remote.get('dual') else '')
                )
                if not re.fullmatch(pattern, ref):
                    raise CleanupError(
                        'metadata',
                        'Invalid Chep remote resource identifier. Cleanup requires the original resource IDs.',
                    )
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
            remote['cleanup_kind'] = getattr(
                exc,
                'kind',
                'timeout' if 'timed out' in str(exc).lower() else 'local_cli' if isinstance(exc, OSError) else 'unexpected',
            )
            log.warning('Kaggle cleanup pending for job %s: %s', job_id, type(exc).__name__)
            raise
        finally:
            self.save(job_id, remote)

    def run(self, job: dict[str, Any], stopped: Callable[[], bool], guard: Any = None) -> None:
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
            remote = {
                'owner': owner,
                'dataset': owner + '/chep-input-' + nonce,
                'kernel': owner + '/chep-run-' + nonce,
                'stage': 'created',
                'source_sha256': sha(source),
                'cleaned': False,
                'source_duration': duration,
                'dual': duration >= DUAL_THRESHOLD_SECONDS,
                'model_dataset': model_dataset,
            }
            # Persist IDs before any network mutation, including ambiguous timeouts.
            self.save(job_id, remote)

        work = self.store.root / 'work' / job_id
        work.mkdir(exist_ok=True)
        inputs, kernel, outputs = (work / name for name in ('input', 'kernel', 'results'))
        for path in (inputs, kernel, outputs):
            path.mkdir(exist_ok=True)

        def stage(status: str, progress: float) -> None:
            with guard:
                if stopped():
                    raise KaggleStopped()
                self.store.update(job_id, status=status, progress=progress)

        def run_cli(args: list[str], timeout: int = 120, check: bool = True) -> tuple[str, int]:
            return self.cli.run(args, timeout=timeout, stopped=stopped, check=check)

        def wait(seconds: float) -> None:
            wait_cancellable(seconds, stopped)

        self._prepare_remote_inputs_and_bundles(
            job_id, config, remote, source, inputs, kernel, model_dataset, stage, run_cli, wait
        )
        if remote.get('dual'):
            try:
                self._run_dual(job_id, config, remote, source, work, kernel, outputs, stopped, guard)
                return
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
            output, code = run_cli(['kernels', 'status', remote['kernel']], check=False)
            if code or not kernel_status(output):
                # No blind automatic resubmission after timeout: saved stage is checked on Retry.
                remote['kernel_attempted'] = True
                self.save(job_id, remote)
                run_cli(['kernels', 'push', '-p', str(kernel), '--accelerator', 'NvidiaTeslaT4', '-t', '14400'], timeout=180)
            remote['stage'] = 'submitted'
            remote['submitted_at'] = time.time()
            self.save(job_id, remote)

        self._poll_single_kernel(job_id, remote, kernel, stage, run_cli, wait)
        self._download_and_import_single(job_id, config, remote, source, outputs, guard, stopped, run_cli)

    def _prepare_remote_inputs_and_bundles(
        self,
        job_id: str,
        config: dict[str, Any],
        remote: dict[str, Any],
        source: Path,
        inputs: Path,
        kernel: Path,
        model_dataset: str,
        stage: Callable[[str, float], None],
        run_cli: Callable[..., tuple[str, int]],
        wait: Callable[[float], None],
    ) -> None:
        if remote['stage'] == 'created':
            stage('UPLOADING', .02)
            # On restart, check the deterministic ID before attempting creation again.
            _, code = run_cli(['datasets', 'status', remote['dataset']], check=False)
            if code:
                prepare_audio_dataset_dir(inputs, source, job_id, remote['dataset'])
                remote['dataset_attempted'] = True
                self.save(job_id, remote)
                run_cli(['datasets', 'create', '-p', str(inputs)], timeout=3600)
            remote['stage'] = 'dataset'
            self.save(job_id, remote)

        if remote['stage'] == 'dataset':
            stage('UPLOADING', .10)
            dataset_state = wait_for_dataset_ready(remote['dataset'], run_cli, wait)
            if dataset_state == 'failed':
                remote['stage'] = 'failed'
                self.save(job_id, remote)
                raise ValueError('Kaggle could not prepare this dataset. Retry to upload a new private dataset.')
            if dataset_state != 'ready':
                raise ValueError('Kaggle is still preparing the dataset. Retry to resume.')
            if remote.get('dual'):
                remote['spans'] = shard_ranges(remote['source_duration'])
                remote['kernels'] = [
                    {'ref': remote['kernel'] + f'-{index}', 'attempted': False, 'cleaned': False, 'submitted': False}
                    for index in range(2)
                ]
                write_dual_kernel_bundles(kernel, config, job_id, remote, model_dataset)
                remote['stage'] = 'dual_submitting'
                self.save(job_id, remote)
            else:
                if not remote.get('model_dataset'):
                    remote['model_dataset'] = model_dataset
                write_single_kernel_bundle(kernel, config, job_id, remote, model_dataset)
                remote['stage'] = 'submitting'
                self.save(job_id, remote)

    def _poll_single_kernel(
        self,
        job_id: str,
        remote: dict[str, Any],
        kernel: Path,
        stage: Callable[[str, float], None],
        run_cli: Callable[..., tuple[str, int]],
        wait: Callable[[float], None],
    ) -> None:
        stage('WAITING_GPU', .20)
        failures = 0
        monitor = None
        try:
            while True:
                try:
                    output, _ = run_cli(['kernels', 'status', remote['kernel']])
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
                        entry = {
                            'ref': remote['kernel'],
                            'attempted': remote.get('kernel_attempted'),
                            'cleaned': remote.get('kernel_cleaned'),
                            'running_since': remote['running_since'],
                            'startup_retries': remote.get('startup_retries', 0),
                        }
                        self._restart_silent_kernel(job_id, remote, entry, kernel, run_cli)
                        stage('WAITING_GPU', .20)
                        continue
                stage('REMOTE_RUNNING' if status == 'RUNNING' else 'WAITING_GPU', .35 if status == 'RUNNING' else .20)
                if time.time() - remote.get('submitted_at', time.time()) > 5 * 3600:
                    raise ValueError('Kaggle session is taking longer than expected. Retry to reconnect.')
                wait(10)
        finally:
            if monitor is not None:
                monitor.close()

    def _download_and_import_single(
        self,
        job_id: str,
        config: dict[str, Any],
        remote: dict[str, Any],
        source: Path,
        outputs: Path,
        guard: Any,
        stopped: Callable[[], bool],
        run_cli: Callable[..., tuple[str, int]],
    ) -> None:
        with guard:
            if stopped():
                raise KaggleStopped()
            self.store.update(job_id, status='DOWNLOADING', progress=.90)
        run_cli(
            [
                'kernels',
                'output',
                remote['kernel'],
                '-p',
                str(outputs),
                '--file-pattern',
                rf'^(result\.json|{job_id}\.(txt|srt))$',
                '--force',
            ],
            timeout=600,
        )
        try:
            result = validate_result(
                outputs,
                job_id,
                remote['source_sha256'],
                model_dataset=remote.get('model_dataset'),
            )
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
            self.store.update(
                job_id,
                status='COMPLETED',
                progress=1,
                duration=result['duration'],
                processed=result['duration'],
                profile=json.dumps(result['profile']),
                error=None,
            )
        remote['stage'] = 'imported'
        self.save(job_id, remote)
        source.unlink(missing_ok=True)
        try:
            self.cleanup(job_id)
        except Exception:
            pass  # Result remains available; cleanup failure stays visible and retryable.

    def _run_dual(
        self,
        job_id: str,
        config: dict[str, Any],
        remote: dict[str, Any],
        source: Path,
        work: Path,
        kernel: Path,
        outputs: Path,
        stopped: Callable[[], bool],
        guard: Any,
    ) -> None:
        def stage(status: str, progress: float) -> None:
            with guard:
                if stopped():
                    raise KaggleStopped()
                self.store.update(job_id, status=status, progress=progress)

        def run_cli(args: list[str], timeout: int = 120, check: bool = True) -> tuple[str, int]:
            return self.cli.run(args, timeout=timeout, stopped=stopped, check=check)

        def wait(seconds: float) -> None:
            wait_cancellable(seconds, stopped)

        if remote['stage'] == 'dual_submitting':
            stage('WAITING_GPU', .15)
            for index, entry in enumerate(remote['kernels']):
                if entry.get('submitted'):
                    continue
                if entry.get('attempted'):
                    output, code = run_cli(['kernels', 'status', entry['ref']], check=False)
                    if code or not kernel_status(output):
                        raise ValueError('Kaggle submission status is uncertain. Retry after checking cleanup.')
                else:
                    entry['attempted'] = True
                    self.save(job_id, remote)
                    run_cli(
                        [
                            'kernels',
                            'push',
                            '-p',
                            str(kernel / str(index)),
                            '--accelerator',
                            'NvidiaTeslaT4',
                            '-t',
                            '14400',
                        ],
                        timeout=180,
                    )
                entry['submitted'] = True
                self.save(job_id, remote)
            remote['stage'] = 'dual_submitted'
            remote['submitted_at'] = time.time()
            self.save(job_id, remote)

        if remote['stage'] != 'dual_submitted':
            raise ValueError('Invalid dual-notebook session state.')

        self._poll_dual_kernels(job_id, remote, kernel, stage, run_cli, wait)
        self._download_and_import_dual(job_id, config, remote, source, work, outputs, stage, guard, stopped, run_cli)

    def _poll_dual_kernels(
        self,
        job_id: str,
        remote: dict[str, Any],
        kernel: Path,
        stage: Callable[[str, float], None],
        run_cli: Callable[..., tuple[str, int]],
        wait: Callable[[float], None],
    ) -> None:
        pending = {index for index, entry in enumerate(remote['kernels']) if not entry.get('complete')}
        failures = 0
        monitors: dict[int, Any] = {}
        try:
            while pending:
                any_running = False
                for index in sorted(pending):
                    entry = remote['kernels'][index]
                    try:
                        output, _ = run_cli(['kernels', 'status', entry['ref']])
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
                                self._restart_silent_kernel(
                                    job_id, remote, entry, kernel / str(index), run_cli, index=index
                                )
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

    def _download_and_import_dual(
        self,
        job_id: str,
        config: dict[str, Any],
        remote: dict[str, Any],
        source: Path,
        work: Path,
        outputs: Path,
        stage: Callable[[str, float], None],
        guard: Any,
        stopped: Callable[[], bool],
        run_cli: Callable[..., tuple[str, int]],
    ) -> None:
        stage('DOWNLOADING', .90)
        results = []
        shards = []
        for index, entry in enumerate(remote['kernels']):
            folder = outputs / str(index)
            folder.mkdir(exist_ok=True)
            stem = f'{job_id}-{index}'
            run_cli(
                [
                    'kernels',
                    'output',
                    entry['ref'],
                    '-p',
                    str(folder),
                    '--file-pattern',
                    rf'^(result\.json|{stem}\.(txt|srt|segments\.json))$',
                    '--force',
                ],
                timeout=600,
            )
            result = validate_result(
                folder,
                job_id,
                remote['source_sha256'],
                stem=stem,
                part_index=index,
                span=remote['spans'][index],
                source_duration=remote['source_duration'],
                model_dataset=remote['model_dataset'],
            )
            validate_model_profile(result, config)
            segment_path = folder / (stem + '.segments.json')
            if segment_path.stat().st_size > 64 * 1024**2:
                raise ValueError('Kaggle shard segment file is too large.')
            shards.append(json.loads(segment_path.read_text(encoding='utf-8')))
            results.append(result)
        segments, audit = merge_segments(shards, remote['spans'])
        atomic_json(work / 'seam-audit.json', {'overlap_seconds': 5, 'removed': audit})
        text, srt = format_merged_transcripts(segments)
        with guard:
            if stopped():
                raise KaggleStopped()
            for extension, content in (('txt', text), ('srt', srt)):
                temporary = outputs / (job_id + '.' + extension + '.part')
                temporary.write_text(content, encoding='utf-8')
                os.replace(temporary, self.store.root / 'outputs' / (job_id + '.' + extension))
            profile = {
                **results[0]['profile'],
                'dual_notebook': True,
                'overlap_seconds': 5,
                'model_dataset': remote['model_dataset'],
            }
            self.store.update(
                job_id,
                status='COMPLETED',
                progress=1,
                duration=remote['source_duration'],
                processed=remote['source_duration'],
                profile=json.dumps(profile),
                error=None,
            )
        remote['stage'] = 'imported'
        self.save(job_id, remote)
        source.unlink(missing_ok=True)
        try:
            self.cleanup(job_id)
        except Exception:
            pass
