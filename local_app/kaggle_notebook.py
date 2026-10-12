"""Build a private, self-contained Kaggle worker from the same inference code as Local."""
import json
from .config import ROOT


def notebook(config, job_id, source_sha256):
    files = {name: (ROOT / 'local_app' / name).read_text(encoding='utf-8')
             for name in ('__init__.py', 'worker.py', 'config.py', 'hardware.py')}
    bootstrap = '''import os, sys, subprocess
from pathlib import Path
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'faster-whisper==1.2.1', 'ctranslate2==4.8.2'], check=True)
root = Path('/tmp/chep-worker')
(root / 'local_app').mkdir(parents=True, exist_ok=True)
'''
    bootstrap += f'files = {files!r}\n'
    bootstrap += "for name, content in files.items():\n    (root / 'local_app' / name).write_text(content, encoding='utf-8')\n"
    runner = '''import hashlib, json, os, subprocess, sys, traceback
from pathlib import Path
from huggingface_hub import snapshot_download
from local_app import worker

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

out = Path('/kaggle/working')
result = {'schema_version': 1, 'job_id': JOB_ID, 'source_sha256': SOURCE_SHA256, 'ok': False}
try:
    # Fail explicitly if Kaggle assigns another accelerator; this policy was measured on T4.
    gpu = subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=name', '--format=csv,noheader'], text=True).strip()
    if 'T4' not in gpu:
        raise RuntimeError('The requested T4 GPU was not assigned: ' + gpu)
    candidates = list(Path('/kaggle/input').rglob(JOB_ID + '.audio'))
    if len(candidates) != 1 or sha(candidates[0]) != SOURCE_SHA256:
        raise RuntimeError('Input audio checksum mismatch.')
    cfg = CONFIG['model_config']
    model = Path(snapshot_download(cfg['ct2_model'] or cfg['model'], revision=cfg['revision'],
                                  allow_patterns=list(cfg['files']), cache_dir='/tmp/chep-models'))
    for name, meta in cfg['files'].items():
        if (model/name).stat().st_size != meta['size'] or sha(model/name) != meta['sha256']:
            raise RuntimeError('Model integrity check failed: ' + name)
    profile = {**cfg['sweet_spot'], 'device': 'cuda', 'device_index': 0, 'cpu_threads': 4,
               'device_name': gpu, 'reason': 'Kaggle T4; validated model-specific profile.', 'provider': 'kaggle'}
    work = Path('/tmp/chep-audio')
    work.mkdir(exist_ok=True)
    def emit(event, **values):
        if event == 'validated':
            result['profile'] = values['profile']
        if event == 'done':
            result['duration'] = values['duration']
        print(json.dumps({'event': event, **values}), flush=True)
    worker.emit = emit
    worker.run({'config': CONFIG, 'profile': profile, 'model': str(model), 'source': str(candidates[0]),
                'work': str(work), 'output': str(out / JOB_ID)})
    result['files'] = {}
    for ext in ('txt', 'srt'):
        path = out / (JOB_ID + '.' + ext)
        os.replace(str(path) + '.part', path)
        result['files'][ext] = {'sha256': sha(path), 'size': path.stat().st_size}
    result['ok'] = True
except Exception as exc:
    result['error'] = str(exc)
    traceback.print_exc()
finally:
    (out / 'result.json').write_text(json.dumps(result), encoding='utf-8')
'''
    runner = f'CONFIG = {config!r}\nJOB_ID = {job_id!r}\nSOURCE_SHA256 = {source_sha256!r}\n' + runner
    bootstrap += f"(root / 'run.py').write_text({runner!r}, encoding='utf-8')\n"
    bootstrap += "subprocess.run([sys.executable, str(root / 'run.py')], cwd=root, check=True)\n"
    return {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
        'cells': [{'id': 'chep-transcribe', 'cell_type': 'code', 'metadata': {},
                   'source': bootstrap.splitlines(True), 'outputs': [], 'execution_count': None}]}
