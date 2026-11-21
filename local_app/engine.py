"""Queue, execution dispatch, and lifecycle coordination for Chep."""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

from .catalog import is_validated, selection
from .cleanup_manager import CleanupManager
from .components import Components
from .hardware import detect
from .kaggle_backend import KaggleBackend, MAX_REMOTE_BYTES
from .kaggle_cli import KaggleStopped
from .kaggle_gpu import KaggleGpuCheck
from .local_runner import run_local_job
from .models import ModelRegistry
from .processes import WorkerJob
from .runtime import RuntimeResolver
from .store import ACTIVE, Store

log = logging.getLogger(__name__)


class Engine:
    """Coordinate local and online transcription queues, runtime checks, and cleanup."""

    def __init__(self, store: Store, models: ModelRegistry) -> None:
        self.store, self.models = store, models
        self.lock = threading.RLock()
        self.state = 'PAUSED'
        self.active: str | None = None
        self.process: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None
        self.closing = False
        self.worker_job = WorkerJob()
        self.hardware = detect('cpu')
        self.components = Components(self.store.root, self.models, lambda: self.runtime.invalidate())
        self.runtime = RuntimeResolver(self.store.root, self.components)
        self.kaggle = KaggleBackend(self.store)
        self.kaggle_gpu = KaggleGpuCheck(self.store.root, self.kaggle.cli)
        self.cleanup_manager = CleanupManager(self.store, self.kaggle, lambda job_id: self._source(job_id))
        self.cleanup_lock = self.cleanup_manager.lock
        self.store.recover()

    @property
    def cleanup_thread(self) -> threading.Thread | None:
        return self.cleanup_manager.thread

    def selected(self) -> dict[str, Any]:
        settings = self.store.settings()
        return selection(settings['language'], settings['device'])

    def configuration(self) -> tuple[dict[str, Any], dict[str, Any]]:
        with self.lock:
            selected = self.selected()
            manager = self.models.get(selected['language'])
            if selected['device_preference'] == 'kaggle':
                if self.kaggle.cli.installed and self.kaggle.cli.snapshot()['status'] == 'idle':
                    self.kaggle.cli.action('check')
                return (
                    manager.snapshot(),
                    {
                        'status': 'online',
                        'profile': None,
                        'attempts': [],
                        'message': 'Kaggle will attach the public model Dataset. No local model or GPU runtime is required.',
                    },
                )
            if self.state == 'PAUSED' and not self.closing:
                self.runtime.ensure(selected, manager)
            return manager.snapshot(), self.runtime.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            pending = self.store.pending_cleanup()

            def owner(job: dict[str, Any]) -> str | None:
                try:
                    return json.loads(job['remote'] or '{}').get('owner')
                except (TypeError, ValueError):
                    return None

            return {
                'state': self.state,
                'active_id': self.active,
                'jobs': self.store.list(),
                'cleanup_pending': [
                    {
                        'id': job['id'],
                        'name': job['name'],
                        'owner': owner(job),
                        'errors': (json.loads(job['cleanup'] or '{}')).get('errors', {}),
                    }
                    for job in pending
                ],
            }

    def start_cleanup_recovery(self) -> None:
        self.cleanup_manager.start_recovery()

    def start(self) -> None:
        with self.lock:
            if self.closing:
                raise ValueError('The application is closing.')
            if self.state in ('RUNNING', 'PAUSING') or (self.thread and self.thread.is_alive()):
                raise ValueError('The queue is running or stopping.')
            if self.runtime.thread and self.runtime.thread.is_alive():
                raise ValueError('Wait for the configuration check to finish.')
            queued = [job for job in self.store.list() if job['status'] == 'QUEUED']
            if not queued:
                raise ValueError('There are no queued files.')
            for job in queued:
                config = json.loads(job['config'])
                if not is_validated(config):
                    raise ValueError(f"{config['model_config']['label']} is awaiting model validation.")
                if config['device_preference'] == 'kaggle':
                    if not self.kaggle.cli.installed:
                        raise ValueError('Download Kaggle CLI before starting online transcription.')
                    if self.kaggle.cli.snapshot()['status'] != 'ready':
                        raise ValueError('Connect your Kaggle account before starting online transcription.')
                    if job['size'] > MAX_REMOTE_BYTES:
                        raise ValueError(
                            'Online GPU supports files up to 2 GB. Split this file or add it with Local processing selected.'
                        )
                    continue
                if self.models.get(config['language'], config['model_config']).snapshot()['status'] != 'ready':
                    raise ValueError(
                        f"Download and verify the {config['model_config']['label']} model required by {job['name']} first."
                    )
            self.state = 'RUNNING'
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    def pause(self) -> None:
        with self.lock:
            self.state = 'PAUSING' if self.active else 'PAUSED'

    def cancel(self, job_id: str) -> None:
        with self.lock:
            job = self.store.get(job_id)
            if not job:
                raise KeyError(job_id)
            if job['status'] == 'QUEUED':
                self.store.update(job_id, status='CANCELLED')
                try:
                    self._source(job_id).unlink(missing_ok=True)
                except OSError:
                    log.exception('Source cleanup pending after cancellation of job %s', job_id)
                if job.get('remote'):
                    try:
                        self.kaggle.cleanup(job_id)
                    except Exception:
                        log.exception('Kaggle cleanup pending after cancellation of job %s', job_id)
            elif self.active == job_id and job['status'] in ACTIVE:
                self.state = 'PAUSING'
                self.store.update(job_id, status='CANCELLED')
                if self.process and self.process.poll() is None:
                    self.process.terminate()
            else:
                raise ValueError('This file is no longer queued or running.')

    def retry(self, job_id: str) -> None:
        with self.lock:
            job = self.store.get(job_id)
            if not job:
                raise KeyError(job_id)
            if job['status'] not in ('FAILED', 'INTERRUPTED') or not self._source(job_id).is_file():
                raise ValueError('This file cannot be retried. Add the original file again.')
            self.store.update(job_id, status='QUEUED', error=None, progress=0, processed=0, profile=None, attempts=None)

    def delete(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            job = self.store.get(job_id)
            if job:
                if self.active == job_id or job['status'] in ACTIVE:
                    raise ValueError('Cancel this file and wait for its worker to stop before deleting it.')
                self.store.mark_removed(job_id)
        return self.cleanup_manager.cleanup_removed(job_id)

    def retry_cleanup(self, job_id: str) -> dict[str, Any]:
        return self.cleanup_manager.retry_cleanup(job_id)

    def _source(self, job_id: str) -> Path:
        return self.store.root / 'sources' / job_id

    def _loop(self) -> None:
        try:
            while True:
                with self.lock:
                    if self.closing or self.state != 'RUNNING':
                        break
                    job = self.store.claim()
                    if not job:
                        break
                    self.active = job['id']
                self._job(job)
                with self.lock:
                    self.active = None
        finally:
            with self.lock:
                self.active = None
                self.process = None
                self.state = 'PAUSED'

    def _job(self, job: dict[str, Any]) -> None:
        if json.loads(job['config'])['device_preference'] == 'kaggle':
            self._online_job(job)
            return
        run_local_job(
            job=job,
            store=self.store,
            lock=self.lock,
            components=self.components,
            models=self.models,
            runtime=self.runtime,
            worker_job=self.worker_job,
            selected_fn=self.selected,
            source_path_fn=lambda job_id: self._source(job_id),
            popen_fn=lambda *args, **kwargs: subprocess.Popen(*args, **kwargs),
            is_closing_fn=lambda: self.closing,
            set_process_fn=lambda proc: setattr(self, 'process', proc),
            set_hardware_fn=lambda hardware: setattr(self, 'hardware', hardware),
        )

    def _online_job(self, job: dict[str, Any]) -> None:
        job_id = job['id']

        def stopped() -> bool:
            return self.closing or self.store.get(job_id)['status'] == 'CANCELLED'

        try:
            self.kaggle.run(job, stopped, self.lock)
        except KaggleStopped:
            pass
        except Exception as error:
            if not stopped():
                self.store.update(job_id, status='FAILED', error=str(error))
        finally:
            current = self.store.get(job_id)
            if current['status'] == 'CANCELLED' and not self.closing:
                try:
                    self.kaggle.cleanup(job_id)
                except Exception:
                    pass
                self._source(job_id).unlink(missing_ok=True)
            if current['status'] in ('COMPLETED', 'CANCELLED'):
                work = self.store.root / 'work' / job_id
                if work.resolve().parent == (self.store.root / 'work').resolve() and not work.is_symlink():
                    shutil.rmtree(work, ignore_errors=True)

    def close(self) -> None:
        with self.lock:
            self.closing = True
            self.state = 'PAUSING'
            if self.process and self.process.poll() is None:
                self.process.terminate()
        self.cleanup_manager.stop.set()
        self.cleanup_manager.wake.set()
        self.runtime.close()
        self.components.close()
        if self.thread:
            self.thread.join(timeout=15)
        self.kaggle_gpu.close()
        self.kaggle.cli.close()
        self.cleanup_manager.close()
        self.store.recover()
        self.worker_job.close()
