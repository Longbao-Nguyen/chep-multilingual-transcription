"""Publish pinned, checksum-verified CTranslate2 artifacts as public Kaggle Datasets.

Run one language at a time. This script never publishes user audio or credentials.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from local_app.catalog import model_config
from local_app.kaggle_cli import KaggleCLI


SLUGS = {
    'en': 'english-whisper-ct2',
    'es': 'spanish-whisper-ct2',
    'de': 'german-whisper-ct2',
    'vi': 'vietnamese-whisper-ct2',
}


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verified(path: Path, meta: dict) -> bool:
    return path.is_file() and path.stat().st_size == meta['size'] and sha256(path) == meta['sha256']


def fetch(path: Path, meta: dict) -> None:
    if verified(path, meta):
        return
    partial = path.with_name(path.name + '.part')
    for attempt in range(5):
        if verified(partial, meta):
            partial.replace(path)
            return
        offset = partial.stat().st_size if partial.exists() else 0
        if offset >= meta['size']:
            partial.unlink()
            offset = 0
        headers = {'User-Agent': 'Chep model-dataset publisher/1'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        try:
            with urllib.request.urlopen(urllib.request.Request(meta['url'], headers=headers), timeout=120) as response:
                if offset and response.status != 206:
                    partial.unlink(missing_ok=True)
                    offset = 0
                if response.status == 206 and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise ValueError('Unexpected HTTP resume range')
                with partial.open('ab' if offset else 'wb') as output:
                    while block := response.read(4 * 1024 * 1024):
                        output.write(block)
                        offset += len(block)
                        if offset > meta['size']:
                            raise ValueError('Artifact exceeds pinned size')
            if verified(partial, meta):
                partial.replace(path)
                return
            if partial.stat().st_size == meta['size']:
                partial.unlink()
                raise ValueError('Artifact checksum mismatch')
        except (OSError, ValueError, urllib.error.URLError):
            if attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError('Download did not finish')


def prepare(language: str, root: Path, owner: str) -> tuple[Path, str]:
    cfg = model_config(language)
    directory = root / language
    directory.mkdir(parents=True, exist_ok=True)
    for name, meta in cfg['files'].items():
        if Path(name).name != name:
            raise ValueError('Invalid artifact name')
        print(f'{language}: verifying {name}', flush=True)
        fetch(directory / name, meta)
    source = cfg['ct2_model']
    readme = [f'# Chep {cfg["label"]} CTranslate2 model', '',
              f'Source: https://huggingface.co/{source}/tree/{cfg["revision"]}',
              f'Pinned revision: `{cfg["revision"]}`',
              f'Source model: https://huggingface.co/{cfg["model"]}',
              f'Declared upstream license: `{cfg["license"]}`. See the source model and',
              'CTranslate2 conversion repository for author attribution and license terms.',
              'These are inference artifacts only; no user recordings are included.', '',
              '| File | Bytes | SHA-256 |', '| --- | ---: | --- |']
    readme.extend(f'| `{name}` | {meta["size"]} | `{meta["sha256"]}` |'
                  for name, meta in cfg['files'].items())
    (directory / 'README.md').write_text('\n'.join(readme) + '\n', encoding='utf-8')
    ref = f'{owner}/{SLUGS[language]}'
    metadata = {'id': ref, 'title': f'Chep {cfg["label"]} CTranslate2 model',
                'subtitle': 'Pinned, verified speech-recognition inference artifacts',
                'description': (f'Files from https://huggingface.co/{source} at revision '
                                f'{cfg["revision"]}; original model https://huggingface.co/{cfg["model"]}. '
                                f'Upstream license: {cfg["license"]}. No user audio.'),
                'licenses': [{'name': 'other'}]}
    (directory / 'dataset-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    return directory, ref


def publish(directory: Path, ref: str, cli: KaggleCLI) -> None:
    output, code = cli.run(['datasets', 'status', ref], check=False)
    if code == 0:
        raise ValueError(f'Dataset already exists: {ref}; inspect it instead of creating a new version')
    print(f'publishing {ref}', flush=True)
    cli.run(['datasets', 'create', '-p', str(directory), '--public'], timeout=7200)
    for _ in range(120):
        status, code = cli.run(['datasets', 'status', ref], check=False)
        if code == 0 and status.strip().lower() == 'ready':
            print(f'ready {ref}/1', flush=True)
            return
        if code == 0 and status.strip().lower() in ('error', 'failed'):
            raise RuntimeError(f'Kaggle rejected dataset {ref}')
        time.sleep(10)
    raise TimeoutError(f'Dataset was not ready: {ref}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--language', choices=SLUGS, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cli-root', type=Path, required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    cli = KaggleCLI(args.cli_root)
    owner = cli.identity()
    if owner != 'longnj':
        raise ValueError('Publication requires the authenticated longnj account')
    directory, ref = prepare(args.language, args.root, owner)
    if args.publish:
        publish(directory, ref, cli)
    else:
        print(f'prepared {ref} at {directory}', flush=True)


if __name__ == '__main__':
    main()
