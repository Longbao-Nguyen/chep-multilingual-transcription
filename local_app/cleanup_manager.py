"""Deferred and synchronous cleanup orchestration for deleted queue records."""
from __future__ import annotations

import json
import logging
import shutil
import threading
from pathlib import Path
from typing import Any, Callable

from .colab_backend import ColabBackend
from .kaggle_backend import KaggleBackend
from .store import Store

log = logging.getLogger(__name__)


class CleanupManager:
    """Handle local/remote cleanup and background retry recovery for removed jobs."""

    def __init__(
        self,
        store: Store,
        kaggle: KaggleBackend,
        source_path_fn: Callable[[str], Path],
        colab: ColabBackend | None = None,
    ) -> None:
        self.store = store
        self.kaggle = kaggle
        self.colab = colab
        self._source_path = source_path_fn
        self.lock = threading.RLock()
        self.thread: threading.Thread | None = None
        self.stop = threading.Event()
        self.wake = threading.Event()

    def start_recovery(self) -> None:
        if self.thread and self.thread.is_alive():
            self.wake.set()
            return

        def recover() -> None:
            while not self.stop.is_set():
                for job in self.store.pending_cleanup():
                    if self.stop.is_set():
                        return
                    try:
                        self.retry_cleanup(job['id'])
                    except Exception:
                        log.exception('Deferred cleanup retry failed for job %s', job['id'])
                self.wake.wait(60)
                self.wake.clear()

        self.thread = threading.Thread(target=recover, daemon=True)
        self.thread.start()

    def _cleanup_result(self, job_id: str) -> dict[str, Any]:
        job = self.store.get(job_id, include_removed=True)
        if not job:
            return {'removed': True, 'cleanup_pending': False}
        if not job['removed']:
            raise ValueError('This job has not been removed from the queue.')
        cleanup = json.loads(job['cleanup'] or '{}')
        return {
            'removed': True,
            'cleanup_pending': not all(cleanup.get(key) for key in ('source', 'outputs', 'work', 'remote')),
            'errors': cleanup.get('errors', {}),
        }

    def retry_cleanup(self, job_id: str) -> dict[str, Any]:
        return self.cleanup_removed(job_id)

    def cleanup_removed(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            return self._cleanup_removed_locked(job_id)

    def _cleanup_removed_locked(self, job_id: str) -> dict[str, Any]:
        job = self.store.get(job_id, include_removed=True)
        if not job:
            return {'removed': True, 'cleanup_pending': False}
        if not job['removed']:
            raise ValueError('Remove the job before retrying deferred cleanup.')
        cleanup = json.loads(job['cleanup'])

        def attempt(key: str, operation: Callable[[], None]) -> None:
            if cleanup[key]:
                return
            try:
                operation()
            except OSError as exc:
                if key == 'remote':
                    cleanup['errors'][key] = 'Remote cleanup could not start. Check the CLI installation and retry.'
                    log.warning('Remote cleanup pending for job %s: %s', job_id, type(exc).__name__)
                else:
                    cleanup['errors'][key] = (
                        f'Local file cleanup failed ({type(exc).__name__}). Check data folder permissions and retry.'
                    )
                    log.warning('Local %s cleanup pending for job %s: %s', key, job_id, type(exc).__name__)
            except Exception as exc:
                cleanup['errors'][key] = str(exc) if key == 'remote' else 'Cleanup failed. Retry or check the logs.'
                log.exception('%s cleanup pending for job %s', key, job_id)
            else:
                cleanup[key] = True
                cleanup['errors'].pop(key, None)
            finally:
                self.store.update(job_id, cleanup=json.dumps(cleanup))

        attempt('source', lambda: self._source_path(job_id).unlink(missing_ok=True))

        def remove_outputs() -> None:
            for ext in ('.txt', '.srt', '.txt.part', '.srt.part'):
                (self.store.root / 'outputs' / job_id).with_suffix(ext).unlink(missing_ok=True)

        attempt('outputs', remove_outputs)

        def remove_work() -> None:
            work = self.store.root / 'work' / job_id
            if work.is_symlink():
                work.unlink(missing_ok=True)
            elif work.exists():
                shutil.rmtree(work)

        attempt('work', remove_work)

        def remove_remote() -> None:
            try:
                remote = json.loads(job.get('remote') or '{}')
            except (TypeError, ValueError):
                remote = {}
            if remote.get('provider') == 'colab' and self.colab is not None:
                self.colab.cleanup(job_id)
            else:
                self.kaggle.cleanup(job_id)

        attempt('remote', remove_remote)
        result = self._cleanup_result(job_id)
        if not result['cleanup_pending']:
            self.store.delete(job_id)
            log.info('Removed job %s and completed resource cleanup', job_id)
        return result

    def close(self) -> None:
        self.stop.set()
        self.wake.set()
        if self.thread:
            self.thread.join(timeout=5)
