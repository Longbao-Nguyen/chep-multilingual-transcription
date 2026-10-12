"""Resolve runtime profiles in isolated, disposable processes."""
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from .config import ROOT, atomic_json
from .hardware import candidates, detect
from .processes import WorkerJob

class RuntimeResolver:
    def __init__(self, root, components=None):
        self.root = root
        self.components = components
        self.lock = threading.RLock()
        self.worker_job = WorkerJob()
        self.thread = None
        self.process = None
        self.generation = 0
        self.closing = False
        self.key = None
        self.state = {'status':'waiting_model', 'profile':None, 'message':'Download the selected model first.', 'attempts':[]}

    def snapshot(self):
        with self.lock:
            return {**self.state}

    def confirm(self, selected, actual, hardware):
        with self.lock:
            self.key = json.dumps(selected, sort_keys=True)
            self.state = {'status':'ready', 'profile':actual, 'hardware':hardware,
                          'message':'Configuration verified by the current job.', 'attempts':[]}

    def invalidate(self):
        with self.lock:
            self.generation += 1
            self.key = None
            if self.process and self.process.poll() is None:
                self.process.terminate()
            self.state = {'status':'checking','profile':None,'message':'Checking configuration...','attempts':[]}

    def ensure(self, selected, manager):
        key = json.dumps(selected, sort_keys=True)
        with self.lock:
            if self.closing:
                return
            if self.thread and self.thread.is_alive():
                return
            cfg = selected['model_config']
            if manager.snapshot()['status'] != 'ready':
                self.state = {'status':'waiting_model','profile':None,'message':'Download and verify the selected model first.','attempts':[]}
                return
            if cfg['validation_status'] != 'validated':
                self.state = {'status':'blocked','profile':None,'message':'This model is awaiting compatibility validation. Install a newer app release when available.','attempts':[]}
                return
            if self.key == key:
                return
            self.key = key
            self.state = {'status':'checking','profile':None,'message':'Checking configuration...','attempts':[]}
            generation = self.generation
            self.thread = threading.Thread(target=self._resolve, args=(selected, manager, generation), daemon=True)
            self.thread.start()

    def _resolve(self, selected, manager, generation):
        hardware = self.components.hardware(selected['device_preference']) if self.components else detect(selected['device_preference'])
        failures = []
        chosen = None
        try:
            for profile in candidates(selected['model_config'], hardware, selected['device_preference']):
                with self.lock:
                    if self.closing or generation != self.generation:
                        return
                try:
                    actual = self.probe(selected, manager.path, profile.json(), generation)
                    actual['reason'] = ('GPU configuration failed; using CPU. ' if profile.device == 'cpu' and failures and selected['device_preference'] == 'auto' else '') + profile.reason
                    if failures:
                        actual['reason'] += ' Previous attempt: ' + failures[-1]['error']
                    chosen = actual
                    break
                except Exception as error:
                    failures.append({'profile':profile.json(),'error':str(error)})
            with self.lock:
                if generation == self.generation and not self.closing:
                    self.state = {'status':'ready' if chosen else 'error','profile':chosen,'hardware':hardware,
                        'message':'Configuration verified.' if chosen else 'No configuration could run. Free memory, repair the model, or update the GPU driver and retry.',
                        'attempts':failures}
        except Exception as error:
            with self.lock:
                if generation == self.generation:
                    self.state = {'status':'error','profile':None,'message':str(error),'attempts':failures}

    def probe(self, selected, path, profile, generation):
        spec_path = self.root/'work'/f'probe-{generation}.json'
        atomic_json(spec_path, {'mode':'probe','model':str(path),'profile':profile,'config':selected})
        env = {**os.environ, 'HF_HUB_OFFLINE':'1'}
        if profile['device'] == 'cpu':
            env['CUDA_VISIBLE_DEVICES'] = ''
        elif self.components:
            env.update(self.components.environment())
        with (self.root/'logs'/'runtime-check.log').open('a',encoding='utf-8') as log:
            with self.lock:
                if self.closing or generation != self.generation:
                    raise RuntimeError('Configuration check cancelled.')
                self.process = subprocess.Popen([sys.executable,'-m','local_app.worker',str(spec_path)],cwd=ROOT,
                    stdout=subprocess.PIPE,stderr=log,text=True,encoding='utf-8',env=env,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                process = self.process
                self.worker_job.attach(process)
            try:
                output,_ = process.communicate(timeout=300)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                raise RuntimeError('Configuration check exceeded five minutes.')
            finally:
                process.wait()
                with self.lock:
                    self.process = None
                spec_path.unlink(missing_ok=True)
        actual = None
        error = None
        for line in output.splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get('event') == 'validated':
                actual = event['profile']
            elif event.get('event') == 'error':
                error = event['message']
        if process.returncode or not actual:
            raise RuntimeError(error or f'Worker could not initialize this configuration (exit {process.returncode}).')
        return actual

    def close(self):
        with self.lock:
            self.closing = True
            if self.process and self.process.poll() is None:
                self.process.terminate()
        if self.thread:
            self.thread.join(timeout=15)
        self.worker_job.close()
