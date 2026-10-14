"""Isolated inference process. Only structured events go to stdout."""
from __future__ import annotations

import json
import math
import os
import sys
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import setup_dlls


def emit(event, **payload):
    print(json.dumps({"event": event, **payload}, ensure_ascii=True), flush=True)


def normalize(source: Path, target: Path):
    import av
    # Decode as a stream; neither the uploaded file nor decoded audio is held in RAM.
    with av.open(str(source)) as container, wave.open(str(target), "wb") as output:
        if not container.streams.audio:
            raise ValueError("This file has no audio stream.")
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        count = 0
        for frame in container.decode(stream):
            for converted in resampler.resample(frame):
                data = converted.to_ndarray().tobytes()
                output.writeframesraw(data)
                count += len(data) // 2
        for converted in resampler.resample(None):
            data = converted.to_ndarray().tobytes()
            output.writeframesraw(data)
            count += len(data) // 2
    if count == 0:
        raise ValueError("The audio file is empty.")
    return count / 16000


def decoded_duration(source: Path) -> float:
    """Count decoded 16 kHz samples so shard boundaries match the remote worker."""
    import av
    count = 0
    with av.open(str(source)) as container:
        if not container.streams.audio:
            raise ValueError('This file has no audio stream.')
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
        for frame in container.decode(stream):
            for converted in resampler.resample(frame):
                count += converted.to_ndarray().size
        for converted in resampler.resample(None):
            count += converted.to_ndarray().size
    if not count:
        raise ValueError('The audio file is empty.')
    return count / 16000


def select_range(audio: Path, start: float, stop: float, source_duration: float) -> float:
    """Keep a lossless sample-aligned range from the fully decoded WAV."""
    if (not all(math.isfinite(value) for value in (start, stop, source_duration)) or
            not 0 <= start < stop <= source_duration + .001):
        raise ValueError('Invalid audio shard range.')
    full = audio.with_name('full.wav')
    audio.replace(full)
    low, high = round(start * 16000), min(round(stop * 16000), round(source_duration * 16000))
    with wave.open(str(full), 'rb') as original, wave.open(str(audio), 'wb') as selected:
        selected.setparams(original.getparams())
        original.setpos(low)
        remaining = high - low
        while remaining:
            frames = original.readframes(min(remaining, 16000 * 60))
            if not frames:
                raise ValueError('Decoded audio ended before the shard range.')
            selected.writeframesraw(frames)
            remaining -= len(frames) // 2
    return (high - low) / 16000


def stamp(seconds: float, srt=False):
    millis = max(0, round(seconds * 1000))
    whole, ms = divmod(millis, 1000)
    hours, rest = divmod(whole, 3600)
    minutes, sec = divmod(rest, 60)
    return f"{hours:02}:{minutes:02}:{sec:02}" + (f",{ms:03}" if srt else "")


def run(spec):
    profile = spec["profile"]
    if profile['device'] == 'cuda':
        setup_dlls()
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    import ctranslate2
    from faster_whisper import WhisperModel
    from .hardware import available_ram_mib

    available = available_ram_mib()
    if available is not None and available < 768:
        raise RuntimeError('Not enough available RAM. Close other applications and try again.')

    if profile['compute_type'] not in ctranslate2.get_supported_compute_types(profile['device'], profile['device_index']):
        raise RuntimeError(f"Unsupported compute type {profile['compute_type']} on {profile['device']}.")
    config = spec['config']
    model_config = config['model_config']
    language = config['language']
    chunk_seconds = model_config['chunk_seconds']
    emit("progress", status="LOADING", progress=0.01)
    model = WhisperModel(spec["model"], device=profile["device"], compute_type=profile["compute_type"],
                         device_index=profile["device_index"], num_workers=profile["concurrency"],
                         cpu_threads=profile["cpu_threads"], local_files_only=True)
    profile = {**profile, 'compute_type': model.model.compute_type, 'language': language,
               'model_id': model_config['model'], 'artifact_id': model_config.get('ct2_model') or model_config['model'],
               'revision': model_config['revision'], 'backend': model_config['backend'], 'chunk_seconds': chunk_seconds}
    # Exercise all workers before declaring the profile valid. This is a feasibility
    # probe, not an optimal-profile benchmark. Runtime OOM still retries in a fresh process.
    import threading
    barrier = threading.Barrier(profile['concurrency'])
    def warmup(_):
        barrier.wait(timeout=30)
        sample = np.random.default_rng(17).normal(0, .025, chunk_seconds*16000).astype(np.float32)
        segments, _ = model.transcribe(sample, language=language, task='transcribe', **model_config['transcribe'])
        return list(segments)
    with ThreadPoolExecutor(max_workers=profile['concurrency']) as pool:
        list(pool.map(warmup, range(profile['concurrency'])))
    available = available_ram_mib()
    if available is not None and available < 384:
        raise RuntimeError('Not enough RAM safety margin after model initialization. Close other applications and retry.')
    if profile['device'] == 'cuda':
        from .hardware import detect
        gpu = next((g for g in detect()['gpus'] if g['index'] == profile['device_index']), None)
        if gpu and gpu['free_mib'] < max(384, gpu['total_mib']*.10):
            raise RuntimeError('Insufficient free VRAM safety margin after model initialization and inference.')
    emit('validated', profile=profile)
    if spec.get('mode') == 'probe':
        return
    emit("progress", status="PREPROCESSING", progress=0.03)
    work = Path(spec["work"])
    audio = work / "audio.wav"
    duration = normalize(Path(spec["source"]), audio)
    source_duration = duration
    if spec.get('source_duration') is not None and abs(duration - spec['source_duration']) > .1:
        raise ValueError('The decoded audio duration changed since shard planning.')
    if spec.get('span') is not None:
        duration = select_range(audio, *spec['span'], source_duration)
    emit("progress", status="TRANSCRIBING", progress=0.08, duration=duration)

    def infer(item):
        offset, chunk = item
        segments, _ = model.transcribe(chunk, language=language, task="transcribe", **model_config['transcribe'])
        return [{"start": min(duration, offset + s.start), "end": min(duration, offset + s.end), "text": s.text.strip()}
                for s in segments if s.text.strip()]

    output = Path(spec["output"])
    collected = [] if spec.get('include_segments') else None
    # Partial output never becomes a downloadable completed transcript.
    with (output.with_suffix(".txt.part")).open("w", encoding="utf-8") as text_file, \
         (output.with_suffix(".srt.part")).open("w", encoding="utf-8") as subtitle_file, \
         wave.open(str(audio), "rb") as wav, ThreadPoolExecutor(max_workers=profile["concurrency"]) as pool:
        index = 0
        segment_index = 0
        while True:
            batch = []
            for _ in range(profile["concurrency"]):
                frames = wav.readframes(chunk_seconds * 16000)
                if not frames:
                    break
                batch.append((index * chunk_seconds, np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0))
                index += 1
            if not batch:
                break
            for segments in pool.map(infer, batch):
                for segment in segments:
                    if collected is not None:
                        collected.append(segment)
                    segment_index += 1
                    start, end, text = segment["start"], segment["end"], segment["text"]
                    text_file.write(f"[{stamp(start)} - {stamp(end)}] {text}\n")
                    subtitle_file.write(f"{segment_index}\n{stamp(start, True)} --> {stamp(end, True)}\n{text}\n\n")
            processed = min(duration, index * chunk_seconds)
            emit("progress", status="TRANSCRIBING", progress=min(0.96, 0.08 + 0.88 * processed / duration),
                 duration=duration, processed=processed)
        for stream in (text_file, subtitle_file):
            stream.flush()
            os.fsync(stream.fileno())
    emit("progress", status="FINALIZING", progress=0.98)
    if collected is not None:
        with output.with_suffix('.segments.json.part').open('w', encoding='utf-8') as stream:
            json.dump(collected, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
    emit("done", duration=duration, source_duration=source_duration)


if __name__ == "__main__":
    try:
        run(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
    except BaseException as error:
        message = str(error)
        emit("error", message=message, gpu_failure=any(token in message.lower() for token in
             ("cuda", "cublas", "cudnn", "out of memory", "out_of_memory", "insufficient driver", "vram", "unsupported compute")))
        raise SystemExit(1)
