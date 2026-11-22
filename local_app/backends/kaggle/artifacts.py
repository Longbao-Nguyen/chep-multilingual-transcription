"""Packaging, integrity verification, and transcript formatting for Kaggle jobs."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from ...config import atomic_json
from .notebook import notebook


def sha(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_result(
    directory: Path,
    job_id: str,
    source_hash: str,
    *,
    stem: str | None = None,
    part_index: int | None = None,
    span: list[float] | tuple[float, float] | None = None,
    source_duration: float | None = None,
    model_dataset: str | None = None,
) -> dict[str, Any]:
    """Validate result.json metadata, GPU identity, and output artifact checksums."""
    manifest = directory / 'result.json'
    if not manifest.is_file() or manifest.stat().st_size > 1024**2:
        raise ValueError('Kaggle did not return a valid result manifest.')
    result = json.loads(manifest.read_text(encoding='utf-8'))
    if (
        result.get('schema_version') != 1
        or result.get('job_id') != job_id
        or result.get('source_sha256') != source_hash
    ):
        raise ValueError('Kaggle result does not match this recording.')
    if result.get('ok') is not True:
        raise ValueError('Kaggle transcription failed: ' + str(result.get('error', 'Unknown remote error.'))[:1000])
    if part_index is not None and result.get('part_index') != part_index:
        raise ValueError('Kaggle shard index mismatch.')
    if span is not None and result.get('span') != list(span):
        raise ValueError('Kaggle shard range mismatch.')
    if source_duration is not None:
        actual = result.get('source_duration')
        if not isinstance(actual, (int, float)) or abs(actual - source_duration) > .1:
            raise ValueError('Kaggle decoded a different recording duration.')
    if model_dataset is not None and result.get('model_dataset') != model_dataset:
        raise ValueError('Kaggle returned a different public model Dataset.')
    profile = result.get('profile', {})
    if profile.get('device') != 'cuda' or profile.get('provider') != 'kaggle' or 'T4' not in profile.get('device_name', ''):
        raise ValueError('Kaggle did not confirm the requested GPU profile.')
    for ext in (('txt', 'srt', 'segments.json') if span is not None else ('txt', 'srt')):
        path = directory / ((stem or job_id) + '.' + ext)
        meta = result.get('files', {}).get(ext, {})
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != meta.get('size')
            or sha(path) != meta.get('sha256')
        ):
            raise ValueError('Kaggle transcript integrity check failed.')
    return result


def validate_model_profile(result: dict[str, Any], config: dict[str, Any]) -> None:
    """Confirm the remote notebook executed the frozen language, revision, and sweet-spot profile."""
    profile = result['profile']
    model = config['model_config']
    expected = model['sweet_spot']
    if (
        profile.get('language') != config['language']
        or profile.get('revision') != model['revision']
        or profile.get('compute_type') != expected['compute_type']
        or profile.get('concurrency') != expected['concurrency']
    ):
        raise ValueError('Kaggle returned a different model, language, or GPU profile.')


def prepare_audio_dataset_dir(inputs_dir: Path, source: Path, job_id: str, dataset_ref: str) -> None:
    """Stage the private audio upload and Kaggle dataset metadata."""
    audio = inputs_dir / (job_id + '.audio')
    if not audio.exists():
        try:
            os.link(source, audio)
        except OSError:
            shutil.copyfile(source, audio)
    # A private dataset with 'other' license grants no new open license to user recordings.
    atomic_json(
        inputs_dir / 'dataset-metadata.json',
        {'id': dataset_ref, 'title': dataset_ref.split('/')[1], 'licenses': [{'name': 'other'}]},
    )


def write_single_kernel_bundle(
    kernel_dir: Path,
    config: dict[str, Any],
    job_id: str,
    remote: dict[str, Any],
    model_dataset: str,
) -> None:
    """Generate notebook and metadata files for a single-notebook Kaggle job."""
    atomic_json(kernel_dir / 'run.ipynb', notebook(config, job_id, remote['source_sha256']))
    atomic_json(
        kernel_dir / 'kernel-metadata.json',
        {
            'id': remote['kernel'],
            'title': remote['kernel'].split('/')[1],
            'code_file': 'run.ipynb',
            'language': 'python',
            'kernel_type': 'notebook',
            'is_private': True,
            'enable_gpu': True,
            'enable_internet': True,
            'dataset_sources': [remote['dataset'], model_dataset],
            'competition_sources': [],
            'kernel_sources': [],
            'model_sources': [],
        },
    )


def write_dual_kernel_bundles(
    kernel_dir: Path,
    config: dict[str, Any],
    job_id: str,
    remote: dict[str, Any],
    model_dataset: str,
) -> None:
    """Generate per-shard notebook and metadata bundles for dual-notebook Kaggle jobs."""
    for index, entry in enumerate(remote['kernels']):
        folder = kernel_dir / str(index)
        folder.mkdir(exist_ok=True)
        atomic_json(
            folder / 'run.ipynb',
            notebook(
                config,
                job_id,
                remote['source_sha256'],
                part_index=index,
                span=remote['spans'][index],
                source_duration=remote['source_duration'],
            ),
        )
        atomic_json(
            folder / 'kernel-metadata.json',
            {
                'id': entry['ref'],
                'title': entry['ref'].split('/')[1],
                'code_file': 'run.ipynb',
                'language': 'python',
                'kernel_type': 'notebook',
                'is_private': True,
                'enable_gpu': True,
                'enable_internet': True,
                'machine_shape': 'NvidiaTeslaT4',
                'dataset_sources': [remote['dataset'], model_dataset],
                'competition_sources': [],
                'kernel_sources': [],
                'model_sources': [],
            },
        )


def format_merged_transcripts(segments: list[dict[str, Any]]) -> tuple[str, str]:
    """Render merged timestamped segments into TXT and SRT strings."""
    from ...worker import stamp

    text = ''.join(f"[{stamp(s['start'])} - {stamp(s['end'])}] {s['text']}\n" for s in segments)
    srt = ''.join(
        f"{number}\n{stamp(s['start'], True)} --> {stamp(s['end'], True)}\n{s['text']}\n\n"
        for number, s in enumerate(segments, 1)
    )
    return text, srt
