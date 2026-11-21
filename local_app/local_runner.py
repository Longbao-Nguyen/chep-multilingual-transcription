"""Local subprocess transcription execution and candidate-profile fallback."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from .catalog import resolve_artifact
from .components import Components
from .config import ROOT, atomic_json
from .hardware import candidates
from .models import ModelRegistry
from .processes import WorkerJob
from .runtime import RuntimeResolver
from .store import Store


def run_local_job(
    *,
    job: dict[str, Any],
    store: Store,
    lock: Any,
    components: Components,
    models: ModelRegistry,
    runtime: RuntimeResolver,
    worker_job: WorkerJob,
    selected_fn: Callable[[], dict[str, Any]],
    source_path_fn: Callable[[str], Path],
    popen_fn: Callable[..., subprocess.Popen[str]],
    is_closing_fn: Callable[[], bool],
    set_process_fn: Callable[[subprocess.Popen[str] | None], None],
    set_hardware_fn: Callable[[dict[str, Any]], None],
) -> None:
    """Execute one local job in isolated worker subprocesses with profile fallback."""
    job_id = job['id']
    work = store.root / 'work' / job_id
    output = store.root / 'outputs' / job_id
    work.mkdir(exist_ok=True)
    try:
        config = json.loads(job['config'])
        hardware = components.hardware(config['device_preference'])
        set_hardware_fn(hardware)
        profiles = list(candidates(config['model_config'], hardware, config['device_preference']))
        manager = models.get(config['language'], config['model_config'])
        config = {**config, 'model_config': resolve_artifact(config['language'], config['model_config'])}
        if manager.snapshot()['status'] != 'ready':
            raise RuntimeError('The model required by this file is not available. Download / repair it and retry.')
        attempts = []
        store.update(job_id, attempts=None)
        for index, current in enumerate(profiles):
            with lock:
                if is_closing_fn() or store.get(job_id)['status'] == 'CANCELLED':
                    return
                store.update(job_id, profile=None, progress=0, processed=0, status='LOADING', error=None)
            requested = current.json()
            if attempts:
                requested['reason'] = (
                    'GPU fallback to CPU. ' if current.device == 'cpu' else 'Trying the next configuration. '
                ) + attempts[-1]['error']
            spec = {
                'profile': requested,
                'model': str(manager.path),
                'source': str(source_path_fn(job_id)),
                'work': str(work),
                'output': str(output),
                'config': config,
            }
            atomic_json(work / 'spec.json', spec)
            done, failure, gpu_failure, validated = False, None, False, False
            with (store.root / 'logs' / f'{job_id}.log').open('a', encoding='utf-8') as log_file:
                with lock:
                    if is_closing_fn() or store.get(job_id)['status'] == 'CANCELLED':
                        return
                    process = popen_fn(
                        [sys.executable, '-m', 'local_app.worker', str(work / 'spec.json')],
                        cwd=ROOT,
                        stdout=subprocess.PIPE,
                        stderr=log_file,
                        text=True,
                        encoding='utf-8',
                        env={
                            **os.environ,
                            'HF_HUB_OFFLINE': '1',
                            **({'CUDA_VISIBLE_DEVICES': ''} if current.device == 'cpu' else components.environment()),
                        },
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                    )
                    set_process_fn(process)
                    worker_job.attach(process)
                for line in process.stdout:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    with lock:
                        if is_closing_fn() or store.get(job_id)['status'] == 'CANCELLED':
                            continue
                        if event.get('event') == 'progress':
                            store.update(job_id, **{k: v for k, v in event.items() if k != 'event'})
                        elif event.get('event') == 'validated':
                            validated = True
                            store.update(job_id, profile=json.dumps(event['profile']))
                            selected = selected_fn()
                            if (
                                config['language'],
                                config['device_preference'],
                                config['model_config']['revision'],
                            ) == (
                                selected['language'],
                                selected['device_preference'],
                                selected['model_config']['revision'],
                            ):
                                runtime.confirm(selected, event['profile'], hardware)
                        elif event.get('event') == 'done':
                            done = True
                        elif event.get('event') == 'error':
                            failure, gpu_failure = event['message'], event['gpu_failure']
                return_code = process.wait()
                process.stdout.close()
                with lock:
                    set_process_fn(None)
            with lock:
                if is_closing_fn() or store.get(job_id)['status'] == 'CANCELLED':
                    return
                if return_code == 0 and done and validated:
                    for extension in ('.txt', '.srt'):
                        os.replace(output.with_suffix(extension + '.part'), output.with_suffix(extension))
                    store.update(job_id, status='COMPLETED', progress=1, error=None)
                    source_path_fn(job_id).unlink(missing_ok=True)
                    return
            # Native CUDA crashes may terminate the process before emitting an error.
            if not failure and return_code != 0 and current.device == 'cuda':
                gpu_failure = True
            attempts.append({'profile': requested, 'error': failure or f'Worker exited with code {return_code}.'})
            store.update(job_id, attempts=json.dumps(attempts))
            if (gpu_failure or not validated) and index + 1 < len(profiles):
                continue
            raise RuntimeError(failure or f'Worker stopped unexpectedly (exit {return_code}). See the logs folder.')
        raise RuntimeError('No execution profile is available for this model.')
    except Exception as error:
        with lock:
            if not is_closing_fn() and store.get(job_id)['status'] != 'CANCELLED':
                store.update(job_id, status='FAILED', error=str(error))
    finally:
        shutil.rmtree(work, ignore_errors=True)
        for extension in ('.txt.part', '.srt.part'):
            output.with_suffix(extension).unlink(missing_ok=True)
        with lock:
            if store.get(job_id)['status'] == 'CANCELLED':
                source_path_fn(job_id).unlink(missing_ok=True)
