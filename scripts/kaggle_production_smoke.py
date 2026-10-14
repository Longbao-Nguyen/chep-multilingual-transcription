"""Run one isolated production-backend Kaggle smoke job and retain local evidence."""
from __future__ import annotations

import argparse
import json
import shutil
import uuid
from pathlib import Path

from local_app.catalog import selection
from local_app.config import prepare_data
from local_app.kaggle_backend import KaggleBackend
from local_app.kaggle_cli import KaggleCLI
from local_app.store import Store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--language', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cli-root', type=Path, required=True)
    args = parser.parse_args()
    store = Store(prepare_data(args.output))
    job_id = uuid.uuid4().hex
    source = store.root / 'sources' / job_id
    shutil.copyfile(args.audio, source)
    store.add(job_id, args.audio.name, source.stat().st_size, selection(args.language, 'kaggle'))
    backend = KaggleBackend(store)
    backend.cli = KaggleCLI(args.cli_root)
    print(json.dumps({'job_id': job_id, 'language': args.language}), flush=True)
    try:
        backend.run(store.get(job_id), lambda: False)
    finally:
        try:
            backend.cleanup(job_id)
        finally:
            job = store.get(job_id)
            remote = json.loads(job.get('remote') or '{}')
            print(json.dumps({'status': job['status'], 'duration': job['duration'],
                              'dual': remote.get('dual'), 'cleaned': remote.get('cleaned'),
                              'output_txt': (store.root / 'outputs' / (job_id + '.txt')).is_file(),
                              'output_srt': (store.root / 'outputs' / (job_id + '.srt')).is_file()}), flush=True)


if __name__ == '__main__':
    main()
