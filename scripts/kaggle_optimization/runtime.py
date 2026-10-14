"""Embedded in each experimental notebook alongside the frozen production worker."""
import importlib.metadata
import json
import os
import subprocess
import time
import wave
from pathlib import Path
from core import discover, save, sha, validate_model, gpu_valid


def run(spec):
    out = Path('/kaggle/working')
    result = {'ok': False, 'fingerprint': spec['fingerprint'], 'inference_windows': [],
              'segments': [], 'timings': {}, 'cache': 'fresh per-notebook path; host cache uncontrolled'}
    timings = result['timings']
    started = time.perf_counter()
    try:
        result['stage'] = 'gpu'
        result['gpu'] = subprocess.check_output(
            ['nvidia-smi', '--id=0', '--query-gpu=name', '--format=csv,noheader'], text=True).strip()
        result['nvidia_smi'] = subprocess.check_output(['nvidia-smi'], text=True)
        if 'T4' not in result['gpu']:
            raise ValueError('Requested T4 missing')
        result['stage'] = 'dependencies'
        subprocess.run([os.sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        result['versions'] = {d.metadata['Name'].lower(): d.version for d in importlib.metadata.distributions()}
        result['python'] = os.sys.version
        timings['notebook_start_to_dependencies_ready'] = time.perf_counter() - started
        from huggingface_hub import snapshot_download
        from local_app import worker
        result['stage'] = 'audio_integrity'
        sources = list(Path('/kaggle/input').rglob('benchmark.audio'))
        if len(sources) != 1 or sha(sources[0]) != spec['audio_sha256']:
            raise ValueError('Audio checksum mismatch')
        cfg = spec['config']['model_config']
        begin = time.perf_counter()
        result['stage'] = 'model_source_prepare'
        if spec['source'] == 'dataset':
            model = discover('/kaggle/input', cfg['files'])
        else:
            model = Path(snapshot_download(cfg['ct2_model'], revision=cfg['revision'],
                         allow_patterns=list(cfg['files']), cache_dir='/tmp/chep-bench-models'))
            validate_model(model, cfg['files'])
        timings['model_source_prepare'] = time.perf_counter() - begin
        result['model_path'] = str(model)
        result['model_copy_bytes'] = 0
        profile = {**cfg['sweet_spot'], 'device': 'cuda', 'device_index': 0, 'cpu_threads': 4,
                   'device_name': result['gpu'], 'provider': 'kaggle'}
        work = Path('/tmp/chep-bench-audio')
        work.mkdir(exist_ok=True)
        original_normalize = worker.normalize
        def normalize(source, target):
            begin = time.perf_counter()
            duration = original_normalize(source, target)
            result['original_duration'] = duration
            if abs(duration - spec['duration']) > .1:
                raise ValueError('Declared duration differs from decoded audio')
            low, high = spec['span']
            if low != 0 or abs(high - duration) > .1:
                # Decode identical original in each job, then lossless PCM range selection.
                full = target.with_name('full.wav')
                target.rename(full)
                with wave.open(str(full), 'rb') as src, wave.open(str(target), 'wb') as dst:
                    dst.setparams(src.getparams())
                    src.setpos(round(low * 16000))
                    remaining = round(high * 16000) - round(low * 16000)
                    while remaining:
                        data = src.readframes(min(remaining, 16000 * 60))
                        if not data:
                            raise ValueError('Short decoded range')
                        dst.writeframesraw(data)
                        remaining -= len(data) // 2
                duration = (round(high * 16000) - round(low * 16000)) / 16000
            timings['decode_and_range'] = time.perf_counter() - begin
            return duration
        worker.normalize = normalize
        base_pool = worker.ThreadPoolExecutor
        class MeasuredPool(base_pool):
            def map(self, fn, *args, **kwargs):
                if fn.__name__ != 'infer':
                    return super().map(fn, *args, **kwargs)
                begin, wall = time.perf_counter(), time.time()
                batches = list(super().map(fn, *args, **kwargs))
                result['inference_windows'].append([wall, time.time()])
                timings['inference'] = timings.get('inference', 0) + time.perf_counter() - begin
                for batch in batches:
                    result['segments'].extend(batch)
                return iter(batches)
        worker.ThreadPoolExecutor = MeasuredPool
        marks = {}
        def emit(event, **values):
            now = time.perf_counter()
            result['stage'] = values.get('status', event)
            if values.get('status') == 'LOADING':
                marks['load'] = now
            if event == 'validated':
                result['profile'] = values['profile']
                timings['ctranslate2_model_load_and_warmup'] = now - marks['load']
            if values.get('status') == 'TRANSCRIBING':
                marks.setdefault('transcribe', now)
            if values.get('status') == 'FINALIZING':
                timings['output_write_and_loop_overhead'] = now - marks['transcribe'] - timings['inference']
        worker.emit = emit
        worker.run({'config': spec['config'], 'profile': profile, 'model': str(model),
                    'work': str(work), 'source': str(sources[0]), 'output': str(out / 'transcript')})
        if not gpu_valid(result['gpu'], result['profile']):
            raise ValueError('GPU profile mismatch')
        result['files'] = {}
        for ext in ('txt', 'srt'):
            target = out / ('transcript.' + ext)
            os.replace(str(target) + '.part', target)
            result['files'][ext] = {'sha256': sha(target), 'size': target.stat().st_size}
        result['ok'] = True
        result['stage'] = 'complete'
    except Exception as exc:
        # Error type only: third-party messages can contain signed URLs or credentials.
        result['error_type'] = type(exc).__name__
    finally:
        timings['notebook_runtime'] = time.perf_counter() - started
        save(out / 'result.json', result)


if __name__ == '__main__':
    run(json.loads(Path('spec.json').read_text(encoding='utf-8')))
