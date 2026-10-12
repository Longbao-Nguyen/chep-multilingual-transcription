from __future__ import annotations
import hashlib
import os
import shutil
import threading
import urllib.request
from pathlib import Path
from .catalog import model_config, resolve_artifact

def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

class ModelManager:
    def __init__(self, root: Path, override: Path | None = None, language='fr', config=None):
        self.config = resolve_artifact(language, config or model_config(language))
        self.language = language
        self.path = override or root / 'models' / self.config['revision']
        if not override and not (self.path / 'model.bin').is_file():
            for revision in self.config.get('legacy_ct2_revisions', []):
                if not revision.isalnum():
                    raise ValueError('Invalid legacy model revision.')
                legacy = root / 'models' / revision
                if all((legacy / name).is_file() for name in self.config['files']):
                    self.path = legacy
                    break
        self.source = self.path
        self.path.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.thread = None
        self.stop = threading.Event()
        self.state = {'status': 'missing', 'progress': 0, 'message': 'Download this model to get started.', 'error': None}
        if (self.path / 'model.bin').is_file():
            self.start(download=False)

    def snapshot(self):
        with self.lock:
            return {**self.state, 'language': self.language, 'path': str(self.path), 'revision': self.config['revision'],
                    'download_bytes': sum(m['size'] for m in self.config['files'].values())}

    def update(self, **changes):
        with self.lock:
            self.state.update(changes)

    def start(self, download=True):
        with self.lock:
            if self.stop.is_set() or (self.thread and self.thread.is_alive()) or self.state['status'] == 'ready':
                return
            self.state.update(status='downloading' if download else 'verifying', error=None, progress=0)
            self.thread = threading.Thread(target=self._run, args=(download,), daemon=True)
            self.thread.start()

    def _run(self, download):
        try:
            files = self.config['files']
            self.source.mkdir(exist_ok=True)
            if download:
                missing = sum(m['size'] for name,m in files.items() if not (self.source/name).is_file())
                if shutil.disk_usage(self.path).free < missing + 512*1024**2:
                    raise RuntimeError('Not enough free disk space to download and prepare this model.')
            total = sum(m['size'] for m in files.values())
            completed = 0
            for name, meta in files.items():
                if self.stop.is_set():
                    return
                target = self.source/name
                self.update(status='verifying', message=f'Verifying {name}...')
                valid = target.is_file() and target.stat().st_size == meta['size'] and sha256(target) == meta['sha256']
                if not valid:
                    if not download:
                        raise RuntimeError('Model files are missing or damaged. Choose Download / repair model.')
                    self._download(name, meta, completed, total)
                completed += meta['size']
                self.update(progress=completed/total)
            if self.stop.is_set():
                return
            self.update(status='ready', progress=1, message='Model files verified.')
        except Exception as error:
            if not self.stop.is_set():
                self.update(status='error', error=str(error), message='Model preparation could not finish.')

    def _download(self, name, meta, completed, total):
        target = self.source/name
        part = self.source/(name+'.part')
        size = meta['size']
        for attempt in range(4):
            if self.stop.is_set():
                return
            offset = part.stat().st_size if part.exists() else 0
            if offset > size:
                part.unlink()
                offset = 0
            try:
                if offset < size:
                    headers = {'User-Agent': 'Chep/0.4'}
                    if offset:
                        headers['Range'] = f'bytes={offset}-'
                    with urllib.request.urlopen(urllib.request.Request(meta['url'], headers=headers), timeout=30) as response:
                        if offset and response.status != 206:
                            offset = 0
                        if response.status == 206 and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                            raise RuntimeError('Invalid download resume response.')
                        with part.open('ab' if offset else 'wb') as stream:
                            while block := response.read(1024**2):
                                if self.stop.is_set():
                                    return
                                stream.write(block)
                                offset += len(block)
                                if offset > size:
                                    raise RuntimeError('Downloaded model exceeds its expected size.')
                                self.update(status='downloading', progress=(completed+offset)/total,
                                    message=f'Downloading {name} / {offset/1024**2:.0f} / {size/1024**2:.0f} MB')
                self.update(status='verifying', message=f'Verifying {name}...')
                if part.stat().st_size != size or sha256(part) != meta['sha256']:
                    part.unlink()
                    raise RuntimeError('Model SHA-256 verification failed. Please retry the download.')
                os.replace(part,target)
                return
            except Exception:
                if attempt == 3:
                    raise
                self.stop.wait(2**attempt)

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=35)

class ModelRegistry:
    def __init__(self, root, override=None):
        self.root, self.override = root, override
        self.lock = threading.RLock()
        self.managers = {}

    def get(self, language, config=None):
        config = resolve_artifact(language, config or model_config(language))
        key = (language, config['revision'])
        with self.lock:
            if key not in self.managers:
                self.managers[key] = ModelManager(self.root, self.override if language == 'fr' else None, language, config)
            return self.managers[key]

    def close(self):
        for manager in list(self.managers.values()):
            manager.close()
