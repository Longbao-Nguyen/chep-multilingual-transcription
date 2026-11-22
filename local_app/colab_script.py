"""Build a self-contained remote Colab worker script from the local inference code."""
from __future__ import annotations

from pathlib import Path

from .config import ROOT


def build_worker_script(
    config: dict,
    job_id: str,
    source_sha256: str,
    n_chunks: int,
    *,
    source_duration: float | None = None,
) -> str:
    """Return standalone Python source executed inside the allocated Colab VM."""
    files = {
        name: (ROOT / 'local_app' / name).read_text(encoding='utf-8')
        for name in ('__init__.py', 'worker.py', 'config.py', 'hardware.py')
    }
    runner = f"""import ctypes, hashlib, importlib.metadata, json, os, shutil, site, subprocess, sys, time, traceback, urllib.error, urllib.request
from pathlib import Path

CONFIG = {config!r}
JOB_ID = {job_id!r}
SOURCE_SHA256 = {source_sha256!r}
N_CHUNKS = {n_chunks!r}
SOURCE_DURATION = {source_duration!r}
LOCAL_APP_FILES = {files!r}

JOB_DIR = Path('/content/chep-job') / JOB_ID
WORK_ROOT = Path('/tmp/chep-worker')
MODELS_ROOT = Path('/content/chep-models')
STATUS_PATH = JOB_DIR / 'status.json'
RESULT_PATH = JOB_DIR / 'result.json'


def write_status(payload):
    tmp = STATUS_PATH.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    os.replace(tmp, STATUS_PATH)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ensure_packages():
    required = {{'faster-whisper': '1.2.1', 'ctranslate2': '4.8.2'}}
    missing = []
    for pkg, ver in required.items():
        try:
            if importlib.metadata.version(pkg) != ver:
                missing.append(f'{{pkg}}=={{ver}}')
        except importlib.metadata.PackageNotFoundError:
            missing.append(f'{{pkg}}=={{ver}}')
    if not missing:
        return
    uv = shutil.which('uv')
    if uv:
        subprocess.run([uv, 'pip', 'install', '--system', '-q', *missing], check=True)
    else:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *missing], check=True)


def preload_cuda12_libs():
    # Modern CUDA 13 containers may lack libcublas.so.12 / libcudnn.so.9 on LD_LIBRARY_PATH.
    lib_dirs = []
    for base in site.getsitepackages():
        nvidia = Path(base) / 'nvidia'
        if not nvidia.is_dir():
            continue
        for sub in ('cublas', 'cudnn', 'cuda_runtime', 'cuda_nvrtc'):
            libdir = nvidia / sub / 'lib'
            if libdir.is_dir():
                lib_dirs.append(str(libdir))
    if lib_dirs:
        cur = os.environ.get('LD_LIBRARY_PATH', '')
        os.environ['LD_LIBRARY_PATH'] = ':'.join(lib_dirs + ([cur] if cur else []))
    for name in ('libcublasLt.so.12', 'libcublas.so.12', 'libcudnn.so.9', 'libcudnn_ops.so.9', 'libcudnn_cnn.so.9'):
        loaded = False
        for d in lib_dirs:
            cand = Path(d) / name
            if cand.is_file():
                try:
                    ctypes.CDLL(str(cand), mode=ctypes.RTLD_GLOBAL)
                    loaded = True
                    break
                except OSError:
                    pass
        if not loaded:
            for d in lib_dirs:
                for p in sorted(Path(d).glob(name.split('.')[0] + '*.so*')):
                    try:
                        ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
                        loaded = True
                        break
                    except OSError:
                        pass
                if loaded:
                    break


def verified(path, meta):
    return path.is_file() and path.stat().st_size == meta['size'] and sha(path) == meta['sha256']


def fetch_model_files(cfg):
    model_dir = MODELS_ROOT / (CONFIG['language'] + '-' + cfg['revision'])
    model_dir.mkdir(parents=True, exist_ok=True)
    for filename, meta in cfg['files'].items():
        dest = model_dir / filename
        if verified(dest, meta):
            continue
        part = model_dir / (filename + '.part')
        for attempt in range(5):
            if verified(part, meta):
                os.replace(part, dest)
                break
            offset = part.stat().st_size if part.exists() else 0
            if offset >= meta['size']:
                part.unlink(missing_ok=True)
                offset = 0
            headers = {{'User-Agent': 'Chep colab-worker/1'}}
            if offset:
                headers['Range'] = f'bytes={{offset}}-'
            try:
                req = urllib.request.Request(meta['url'], headers=headers)
                with urllib.request.urlopen(req, timeout=180) as resp, part.open('ab' if offset else 'wb') as out:
                    while block := resp.read(4 * 1024 * 1024):
                        out.write(block)
                        offset += len(block)
                        if offset > meta['size']:
                            raise RuntimeError('Downloaded model artifact exceeded expected size.')
                if verified(part, meta):
                    os.replace(part, dest)
                    break
                if part.stat().st_size == meta['size']:
                    part.unlink(missing_ok=True)
                    raise RuntimeError('Downloaded model artifact checksum mismatch.')
            except (OSError, urllib.error.URLError) as exc:
                if attempt == 4:
                    raise RuntimeError(f'Model download failed: {{type(exc).__name__}}') from exc
                time.sleep(min(2 ** attempt, 8))
        if not verified(dest, meta):
            raise RuntimeError('Pinned model artifact could not be verified.')
    return model_dir


def assemble_audio():
    parts_dir = JOB_DIR / 'parts'
    audio_path = JOB_DIR / 'input.audio'
    with audio_path.open('wb') as out:
        for idx in range(N_CHUNKS):
            p = parts_dir / f'part_{{idx:04d}}.bin'
            if not p.is_file():
                raise RuntimeError(f'Missing uploaded audio chunk {{idx}}.')
            with p.open('rb') as src:
                shutil.copyfileobj(src, out, length=1024 * 1024)
            p.unlink(missing_ok=True)
    shutil.rmtree(parts_dir, ignore_errors=True)
    if sha(audio_path) != SOURCE_SHA256:
        raise RuntimeError('Uploaded audio checksum mismatch on Colab.')
    return audio_path


def main():
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    (WORK_ROOT / 'local_app').mkdir(parents=True, exist_ok=True)
    for name, content in LOCAL_APP_FILES.items():
        (WORK_ROOT / 'local_app' / name).write_text(content, encoding='utf-8')
    if str(WORK_ROOT) not in sys.path:
        sys.path.insert(0, str(WORK_ROOT))

    result = {{
        'schema_version': 1,
        'job_id': JOB_ID,
        'source_sha256': SOURCE_SHA256,
        'ok': False,
    }}
    write_status({{'status': 'LOADING', 'progress': 0.16, 'done': False, 'pid': os.getpid()}})
    try:
        gpu = subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=name', '--format=csv,noheader'], text=True).strip()
        if not gpu:
            raise RuntimeError('No CUDA GPU was found in the Colab runtime.')
        ensure_packages()
        preload_cuda12_libs()

        cfg = CONFIG['model_config']
        write_status({{'status': 'LOADING', 'progress': 0.19, 'done': False, 'pid': os.getpid()}})
        model_dir = fetch_model_files(cfg)
        audio_path = assemble_audio()

        from local_app import worker

        profile = {{
            **cfg['sweet_spot'],
            'device': 'cuda',
            'device_index': 0,
            'cpu_threads': 4,
            'device_name': gpu,
            'reason': f'Google Colab {{gpu}}; validated model-specific profile.',
            'provider': 'colab',
        }}
        audio_work = JOB_DIR / 'work'
        audio_work.mkdir(exist_ok=True)
        last_emit = [0.0]

        def emit(event, **values):
            if event == 'validated':
                result['profile'] = values['profile']
            elif event == 'done':
                result['duration'] = values['duration']
                result['source_duration'] = values['source_duration']
            elif event == 'progress':
                now = time.monotonic()
                raw_p = float(values.get('progress', 0.0))
                mapped = min(0.95, 0.22 + 0.73 * raw_p)
                if now - last_emit[0] >= 1.0 or values.get('status') != 'TRANSCRIBING':
                    last_emit[0] = now
                    write_status({{
                        'status': values.get('status', 'REMOTE_RUNNING'),
                        'progress': round(mapped, 4),
                        'duration': values.get('duration'),
                        'processed': values.get('processed', 0.0),
                        'done': False,
                        'pid': os.getpid(),
                    }})

        worker.emit = emit
        worker.run({{
            'config': CONFIG,
            'profile': profile,
            'model': str(model_dir),
            'source': str(audio_path),
            'work': str(audio_work),
            'output': str(JOB_DIR / JOB_ID),
            'source_duration': SOURCE_DURATION,
            'include_segments': False,
        }})
        result['files'] = {{}}
        for ext in ('txt', 'srt'):
            p = JOB_DIR / (JOB_ID + '.' + ext)
            os.replace(str(p) + '.part', p)
            result['files'][ext] = {{'sha256': sha(p), 'size': p.stat().st_size}}
        audio_path.unlink(missing_ok=True)
        shutil.rmtree(audio_work, ignore_errors=True)
        result['ok'] = True
        write_status({{'status': 'DOWNLOADING', 'progress': 0.96, 'duration': result.get('duration'), 'processed': result.get('duration') or 0.0, 'done': True, 'ok': True}})
    except Exception as exc:
        result['error'] = str(exc)
        traceback.print_exc()
        write_status({{'status': 'FAILED', 'progress': 1.0, 'done': True, 'ok': False, 'error': str(exc)}})
    finally:
        RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
"""
    return runner
