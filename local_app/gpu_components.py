"""Optional GPU providers. Discovery never initializes an inference device."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import subprocess
import zipfile
from .config import ROOT, atomic_json
from .models import ModelManager, sha256

CATALOG = json.loads((ROOT/'config/gpu_runtimes.json').read_text(encoding='utf-8'))['providers']

def discover():
    fallback={'compatible':False,'provider':None,'adapters':[],'eligible_indices':[],
              'reason':'GPU acceleration is unavailable on this platform. Using CPU.'}
    if os.name != 'nt':
        return fallback
    try:
        result=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',
                              str(ROOT/'config/gpu-detect.ps1')],capture_output=True,text=True,encoding='utf-8',errors='replace',
                              timeout=20,creationflags=subprocess.CREATE_NO_WINDOW)
        result.check_returncode()
        data=json.loads(result.stdout)
        if data.get('provider') not in PROVIDERS:
            data.update(compatible=False,provider=None)
        return data
    except Exception:
        return {**fallback,'reason':'GPU compatibility could not be checked. Using CPU. Check the graphics driver and try again.'}

class GPUProvider:
    """An adapter must supply install, readiness and worker activation for its backend."""
    id: str
    def __init__(self, root):
        self.root=Path(root)
        self.config=CATALOG[self.id]
        self.path=self.root/'runtimes'/(self.id+'-'+self.config['version'])

    def ready(self):
        marker=self.path/'installed.json'
        try:
            data=json.loads(marker.read_text())
            if data['version'] != self.config['version'] or not data['files']:
                return False
            for name,digest in data['files'].items():
                validate_member(name)
                if not (self.path/name).is_file() or sha256(self.path/name)!=digest:
                    return False
            return all((self.path/p).is_dir() for p in self.config['dll_dirs'])
        except (OSError,ValueError,KeyError):
            return False

    def environment(self):
        return {'CHEP_GPU_RUNTIME':str(self.path),'CHEP_GPU_PROVIDER':self.id}

def validate_member(name):
    path=PurePosixPath(name)
    if path.is_absolute() or PureWindowsPath(name).drive or '..' in path.parts or '\\' in name:
        raise ValueError('Unsafe runtime archive path.')

class CudaProvider(GPUProvider):
    id='cuda'
    def install(self, stop, update):
        cache=self.root/'downloads'/self.id
        cache.mkdir(parents=True,exist_ok=True)
        files={f['name']:f for f in self.config['files']}
        # Reuse the checksum-verified, resumable downloader; no pip or global installation.
        downloader=ModelManager(self.root,override=cache,config={'revision':self.config['version'],'requires_conversion':False,'files':files})
        downloader.stop=stop
        downloader.update=lambda **changes:update(**changes)
        total=sum(f['size'] for f in files.values())
        if shutil.disk_usage(cache).free < total + 3*1024**3:
            raise RuntimeError('Not enough disk space for the GPU runtime download and installation.')
        done=0
        for name,meta in files.items():
            if stop.is_set(): return False
            target=cache/name
            if not target.is_file() or target.stat().st_size!=meta['size'] or sha256(target)!=meta['sha256']:
                downloader._download(name,meta,done,total)
            if stop.is_set(): return False
            done+=meta['size']
        staging=self.path.with_name(self.path.name+'.installing')
        if staging.exists():
            if staging.is_symlink() or staging.resolve().parent!=self.path.parent.resolve():
                raise RuntimeError('Unexpected runtime staging path.')
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        hashes={}
        update(status='installing',progress=1,message='Installing GPU runtime...')
        try:
            for name in files:
                with zipfile.ZipFile(cache/name) as archive:
                    for item in archive.infolist():
                        validate_member(item.filename)
                        if stop.is_set(): return False
                        # Keep executable runtime libraries and their license notices, not headers/import libraries.
                        if item.is_dir() or not (item.filename.lower().endswith('.dll') or 'license' in item.filename.lower()):
                            continue
                        if item.external_attr >> 16 & 0o170000 == 0o120000:
                            raise ValueError('Runtime archive contains a symlink.')
                        target=staging/item.filename
                        target.parent.mkdir(parents=True,exist_ok=True)
                        with archive.open(item) as source,target.open('wb') as destination:
                            shutil.copyfileobj(source,destination,1024**2)
                        hashes[item.filename]=sha256(target)
            if not hashes or not all((staging/d).is_dir() for d in self.config['dll_dirs']):
                raise RuntimeError('GPU runtime archive is incomplete.')
            atomic_json(staging/'installed.json',{'version':self.config['version'],'files':hashes})
            if self.path.exists():
                if self.path.is_symlink() or self.path.resolve().parent!=staging.parent.resolve():
                    raise RuntimeError('Unexpected installed runtime path.')
                shutil.rmtree(self.path)
            os.replace(staging,self.path)
            for name in files: (cache/name).unlink(missing_ok=True)
            return True
        finally:
            if staging.exists() and not staging.is_symlink() and staging.resolve().parent==self.path.parent.resolve():
                shutil.rmtree(staging)

PROVIDERS={'cuda':CudaProvider}
