from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from .config import ROOT, atomic_json
from .hardware import candidates, detect
from .catalog import selection, is_validated, resolve_artifact
from .runtime import RuntimeResolver
from .store import ACTIVE, Store
from .processes import WorkerJob
from .components import Components
from .kaggle_backend import KaggleBackend, MAX_REMOTE_BYTES
from .kaggle_cli import KaggleStopped


class Engine:
    def __init__(self, store: Store, models):
        self.store, self.models = store, models
        self.lock = threading.RLock()
        self.state = "PAUSED"
        self.active = None
        self.process = None
        self.thread = None
        self.closing = False
        self.worker_job = WorkerJob()
        self.hardware = detect('cpu')
        self.components = Components(self.store.root, self.models, lambda: self.runtime.invalidate())
        self.runtime = RuntimeResolver(self.store.root, self.components)
        self.kaggle = KaggleBackend(self.store)
        self.store.recover()

    def selected(self):
        settings = self.store.settings()
        return selection(settings['language'], settings['device'])

    def configuration(self):
        with self.lock:
            selected = self.selected()
            manager = self.models.get(selected['language'])
            if selected['device_preference'] == 'kaggle':
                if self.kaggle.cli.installed and self.kaggle.cli.snapshot()['status'] == 'idle':
                    self.kaggle.cli.action('check')
                return manager.snapshot(), {'status': 'online', 'profile': None, 'attempts': [],
                    'message': 'The selected model will download on Kaggle. No local model or GPU runtime is required.'}
            if self.state == 'PAUSED' and not self.closing:
                self.runtime.ensure(selected, manager)
            return manager.snapshot(), self.runtime.snapshot()

    def snapshot(self):
        with self.lock:
            return {"state": self.state, "active_id": self.active, "jobs": self.store.list()}

    def start(self):
        with self.lock:
            if self.closing:
                raise ValueError("The application is closing.")
            if self.state in ("RUNNING", "PAUSING") or (self.thread and self.thread.is_alive()):
                raise ValueError("The queue is running or stopping.")
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
                        raise ValueError('Online GPU supports files up to 2 GB. Split this file or add it with Local processing selected.')
                    continue
                if self.models.get(config['language'], config['model_config']).snapshot()['status'] != 'ready':
                    raise ValueError(f"Download and verify the {config['model_config']['label']} model required by {job['name']} first.")
            self.state = "RUNNING"
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    def pause(self):
        with self.lock:
            self.state = "PAUSING" if self.active else "PAUSED"

    def cancel(self, job_id):
        with self.lock:
            job = self.store.get(job_id)
            if not job:
                raise KeyError(job_id)
            if job["status"] == "QUEUED":
                if job.get('remote'):
                    self.kaggle.cleanup(job_id)
                self.store.update(job_id, status="CANCELLED")
                self._source(job_id).unlink(missing_ok=True)
            elif self.active == job_id and job["status"] in ACTIVE:
                self.state = "PAUSING"
                self.store.update(job_id, status="CANCELLED")
                if self.process and self.process.poll() is None:
                    self.process.terminate()
            else:
                raise ValueError("This file is no longer queued or running.")

    def retry(self, job_id):
        with self.lock:
            job = self.store.get(job_id)
            if not job:
                raise KeyError(job_id)
            if job["status"] not in ("FAILED", "INTERRUPTED") or not self._source(job_id).is_file():
                raise ValueError("This file cannot be retried. Add the original file again.")
            self.store.update(job_id, status="QUEUED", error=None, progress=0, processed=0, profile=None, attempts=None)

    def delete(self, job_id):
        with self.lock:
            job = self.store.get(job_id)
            if not job:
                raise KeyError(job_id)
            if self.active == job_id or job["status"] in ACTIVE:
                raise ValueError("Cancel this file and wait for its worker to stop before deleting it.")
            if job.get('remote'):
                self.kaggle.cleanup(job_id)
            self._source(job_id).unlink(missing_ok=True)
            for ext in (".txt", ".srt", ".txt.part", ".srt.part"):
                (self.store.root / "outputs" / job_id).with_suffix(ext).unlink(missing_ok=True)
            self.store.delete(job_id)

    def _source(self, job_id):
        return self.store.root / "sources" / job_id

    def _loop(self):
        try:
            while True:
                with self.lock:
                    if self.closing or self.state != "RUNNING":
                        break
                    job = self.store.claim()
                    if not job:
                        break
                    self.active = job["id"]
                self._job(job)
                with self.lock:
                    self.active = None
        finally:
            with self.lock:
                self.active = None
                self.process = None
                self.state = "PAUSED"

    def _job(self, job):
        if json.loads(job['config'])['device_preference'] == 'kaggle':
            self._online_job(job)
            return
        job_id = job["id"]
        work = self.store.root / "work" / job_id
        output = self.store.root / "outputs" / job_id
        work.mkdir(exist_ok=True)
        try:
            config = json.loads(job['config'])
            self.hardware = self.components.hardware(config['device_preference'])
            profiles = list(candidates(config['model_config'], self.hardware, config['device_preference']))
            manager = self.models.get(config['language'], config['model_config'])
            config = {**config, 'model_config': resolve_artifact(config['language'], config['model_config'])}
            if manager.snapshot()['status'] != 'ready':
                raise RuntimeError('The model required by this file is not available. Download / repair it and retry.')
            attempts = []
            self.store.update(job_id, attempts=None)
            for index, current in enumerate(profiles):
                with self.lock:
                    if self.closing or self.store.get(job_id)["status"] == "CANCELLED":
                        return
                    self.store.update(job_id, profile=None, progress=0, processed=0,
                                      status="LOADING", error=None)
                requested = current.json()
                if attempts:
                    requested['reason'] = ('GPU fallback to CPU. ' if current.device == 'cpu' else 'Trying the next configuration. ') + attempts[-1]['error']
                spec = {"profile": requested, "model": str(manager.path), "source": str(self._source(job_id)),
                        "work": str(work), "output": str(output), 'config': config}
                atomic_json(work / "spec.json", spec)
                done, failure, gpu_failure, validated = False, None, False, False
                with (self.store.root / "logs" / f"{job_id}.log").open("a", encoding="utf-8") as log:
                    with self.lock:
                        if self.closing or self.store.get(job_id)["status"] == "CANCELLED":
                            return
                        self.process = subprocess.Popen([sys.executable, "-m", "local_app.worker", str(work / "spec.json")],
                            cwd=ROOT, stdout=subprocess.PIPE, stderr=log, text=True, encoding="utf-8",
                            env={**os.environ, 'HF_HUB_OFFLINE':'1', **({'CUDA_VISIBLE_DEVICES':''} if current.device == 'cpu' else self.components.environment())},
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                        process = self.process
                        self.worker_job.attach(process)
                    for line in process.stdout:
                        try:
                            event = json.loads(line)
                        except ValueError:
                            continue
                        with self.lock:
                            if self.closing or self.store.get(job_id)["status"] == "CANCELLED":
                                continue
                            if event.get("event") == "progress":
                                self.store.update(job_id, **{k: v for k, v in event.items() if k != "event"})
                            elif event.get('event') == 'validated':
                                validated = True
                                self.store.update(job_id, profile=json.dumps(event['profile']))
                                selected = self.selected()
                                if (config['language'], config['device_preference'], config['model_config']['revision']) == (selected['language'], selected['device_preference'], selected['model_config']['revision']):
                                    self.runtime.confirm(selected, event['profile'], self.hardware)
                            elif event.get("event") == "done":
                                done = True
                            elif event.get("event") == "error":
                                failure, gpu_failure = event["message"], event["gpu_failure"]
                    return_code = process.wait()
                    process.stdout.close()
                    with self.lock:
                        self.process = None
                with self.lock:
                    if self.closing or self.store.get(job_id)["status"] == "CANCELLED":
                        return
                    if return_code == 0 and done and validated:
                        for extension in (".txt", ".srt"):
                            os.replace(output.with_suffix(extension + ".part"), output.with_suffix(extension))
                        self.store.update(job_id, status="COMPLETED", progress=1, error=None)
                        self._source(job_id).unlink(missing_ok=True)
                        return
                # Native CUDA crashes may terminate the process before emitting an error.
                if not failure and return_code != 0 and current.device == "cuda":
                    gpu_failure = True
                attempts.append({'profile': requested, 'error': failure or f'Worker exited with code {return_code}.'})
                self.store.update(job_id, attempts=json.dumps(attempts))
                if (gpu_failure or not validated) and index + 1 < len(profiles):
                    continue
                raise RuntimeError(failure or f"Worker stopped unexpectedly (exit {return_code}). See the logs folder.")
            raise RuntimeError('No execution profile is available for this model.')
        except Exception as error:
            with self.lock:
                if not self.closing and self.store.get(job_id)["status"] != "CANCELLED":
                    self.store.update(job_id, status="FAILED", error=str(error))
        finally:
            shutil.rmtree(work, ignore_errors=True)
            for extension in (".txt.part", ".srt.part"):
                output.with_suffix(extension).unlink(missing_ok=True)
            with self.lock:
                if self.store.get(job_id)["status"] == "CANCELLED":
                    self._source(job_id).unlink(missing_ok=True)

    def _online_job(self, job):
        job_id = job['id']
        def stopped():
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

    def close(self):
        with self.lock:
            self.closing = True
            self.state = "PAUSING"
            if self.process and self.process.poll() is None:
                self.process.terminate()
        self.runtime.close()
        self.components.close()
        if self.thread:
            self.thread.join(timeout=15)
        self.kaggle.cli.close()
        self.store.recover()
        self.worker_job.close()
