"""Persistent, serial queue for explicitly selected optional downloads."""
from __future__ import annotations
import json
import threading
import time
from .catalog import LANGUAGES
from .config import atomic_json
from .gpu_components import CATALOG, PROVIDERS, discover
from .hardware import detect

class Components:
    def __init__(self, root, models, changed=lambda:None):
        self.root,self.models,self.changed=root,models,changed
        self.lock=threading.RLock()
        self.stop=threading.Event()
        self.wake=threading.Event()
        self.closing=False
        self.thread=None
        self.active=None
        self.refresh_requested=True
        self.gpu={'compatible':False,'provider':None,'reason':'Checking GPU compatibility...','adapters':[],
                  'status':'checking','download_bytes':0,'eligible_indices':[]}
        self.provider=None
        self.path=root/'download_queue.json'
        try: self.state=json.loads(self.path.read_text(encoding='utf-8'))
        except (OSError,ValueError): self.state={'tasks':[],'paused':False,'consumed_setups':[]}
        for task in self.state['tasks']:
            if task['status'] in ('downloading','verifying','installing'): task['status']='queued'
        initial=root/'initial_setup.json'
        if initial.is_file():
            try:
                choices=json.loads(initial.read_text(encoding='utf-8-sig'))
                identity=choices['setup_id']
                if choices.get('schema_version')==1 and identity not in self.state['consumed_setups']:
                    for component in choices.get('components',[]): self._enqueue(component)
                    self.state['consumed_setups'].append(identity)
                    self.save()
            except (ValueError,KeyError,TypeError):
                # A malformed preset must never prevent the app from starting.
                self.state['setup_error']='Optional component choices could not be read. Select downloads in the app.'

    def save(self): atomic_json(self.path,self.state)

    def start(self):
        if self.thread is None:
            self.thread=threading.Thread(target=self._loop,daemon=True)
            self.thread.start()

    def _enqueue(self, component):
        if component!='gpu:auto' and component not in ['model:'+c for c in LANGUAGES]:
            raise ValueError('Unknown optional component.')
        previous=next((t for t in self.state['tasks'] if t['id']==component),None)
        if previous and previous['status'] not in ('error','cancelled','complete'): return
        task={'id':component,'label':'GPU Runtime' if component=='gpu:auto' else LANGUAGES[component[6:]]['label']+' Transcription Model',
              'status':'queued','progress':0,'message':'Waiting to download.','error':None}
        if previous: self.state['tasks'].remove(previous)
        self.state['tasks'].append(task)

    def enqueue(self, component):
        with self.lock:
            if component=='gpu:auto' and (not self.gpu['compatible'] or self.gpu['status']=='checking'):
                raise ValueError(self.gpu['reason'])
            self._enqueue(component)
            self.save(); self.wake.set()

    def control(self, action, component=None):
        with self.lock:
            if action in ('pause','resume'):
                self.state['paused']=action=='pause'
                if action=='pause': self.stop.set()
            elif action=='cancel':
                for task in self.state['tasks']:
                    if task['id']==component and task['status']!='complete':
                        task.update(status='cancelled',message='Download cancelled.',error=None)
                if component==self.active: self.stop.set()
            else: raise ValueError('Unknown download action.')
            self.save(); self.wake.set()

    def refresh(self):
        with self.lock:
            self.refresh_requested=True
            self.gpu.update(status='checking',reason='Checking GPU compatibility...')
            self.wake.set()

    def snapshot(self):
        with self.lock:
            return {'gpu':dict(self.gpu),'paused':self.state['paused'],'tasks':[dict(t) for t in self.state['tasks']],
                    'setup_error':self.state.get('setup_error')}

    def hardware(self, preference):
        hardware=detect('cpu')
        with self.lock: gpu=dict(self.gpu)
        if preference!='cpu':
            if gpu['status']=='ready' and gpu['compatible']:
                hardware=detect('auto')
                hardware['gpus']=[g for g in hardware['gpus'] if g['index'] in gpu['eligible_indices']]
            else: hardware['warning']=gpu['reason']
        return hardware

    def environment(self):
        with self.lock:
            return self.provider.environment() if self.provider and self.gpu['status']=='ready' else {}

    def _check(self):
        discovered=discover()
        provider=PROVIDERS[discovered['provider']](self.root) if discovered['compatible'] else None
        ready=provider.ready() if provider else False
        status='ready' if ready else 'missing' if provider else 'unavailable'
        reason=discovered['reason']
        if provider and not ready: reason='A compatible GPU is available. Download the GPU runtime to enable acceleration. Using CPU until it is ready.'
        with self.lock:
            self.provider=provider
            self.gpu={**discovered,'status':status,'reason':reason,
                      'download_bytes':sum(f['size'] for f in provider.config['files']) if provider else 0}
        self.changed()

    def _loop(self):
        while not self.closing:
            with self.lock:
                refresh=self.refresh_requested
                self.refresh_requested=False
            if refresh:
                try: self._check()
                except Exception:
                    with self.lock:
                        self.gpu.update(compatible=False,status='unavailable',reason='GPU runtime verification failed. Using CPU. Check compatibility again.')
            with self.lock:
                task=next((t for t in self.state['tasks'] if t['status']=='queued'),None) if not self.state['paused'] else None
                if task:
                    self.active=task['id']; self.stop.clear()
                    task.update(status='downloading',error=None); self.save()
            if task is None:
                self.wake.wait(1); self.wake.clear(); continue
            def update(**changes):
                with self.lock:
                    if task['status']!='cancelled': task.update(changes)
            try:
                if task['id']=='gpu:auto':
                    with self.lock: provider=self.provider
                    if not provider: raise ValueError(self.gpu['reason'])
                    success=provider.ready() or provider.install(self.stop,update)
                    if success: self._check()
                else:
                    manager=self.models.get(task['id'][6:])
                    # Startup verification can already be running; finish it before a repair request.
                    while manager.thread and manager.thread.is_alive() and not self.stop.wait(.1):
                        update(**{k:v for k,v in manager.snapshot().items() if k in ('status','progress','message','error')})
                    if not self.stop.is_set(): manager.start()
                    while manager.thread and manager.thread.is_alive() and not self.stop.wait(.1):
                        update(**{k:v for k,v in manager.snapshot().items() if k in ('status','progress','message','error')})
                    if self.stop.is_set():
                        manager.stop.set()
                        if manager.thread: manager.thread.join(35)
                        if manager.thread and manager.thread.is_alive(): raise RuntimeError('Download is still stopping. Restart the app before retrying.')
                        manager.stop.clear()
                        if manager.snapshot()['status']!='ready':
                            manager.update(status='missing',message='Download stopped. Resume or retry from Optional Components.',error=None)
                    result=manager.snapshot()
                    success=result['status']=='ready'
                    if not success and not self.stop.is_set(): raise RuntimeError(result.get('error') or 'Model download did not finish.')
                with self.lock:
                    if task['status']!='cancelled':
                        task.update(status='complete' if success else 'queued',progress=1 if success else task['progress'],
                                    message='Ready.' if success else 'Download paused.')
            except Exception as error:
                with self.lock:
                    if task['status']!='cancelled': task.update(status='error',error=str(error),message='Download could not finish. Retry when ready.')
            finally:
                with self.lock: self.active=None; self.save()

    def close(self):
        self.closing=True; self.stop.set(); self.wake.set()
        if self.thread: self.thread.join(40)
