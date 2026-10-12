"""Verify published CT2 caches and repair via the app downloader; no inference."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--app', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.app.resolve()))
    from local_app.catalog import model_config
    from local_app.models import ModelManager
    assert all(importlib.util.find_spec(name) is None for name in ('torch', 'transformers'))
    checks = {}
    parent = ROOT / 'test-results'
    parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='ct2-download-', dir=parent) as folder:
        root = Path(folder)
        for code in ('es', 'vi'):
            cfg = model_config(code)
            target = root / 'models' / cfg['revision']
            target.mkdir(parents=True)
            for name in cfg['files']:
                if name == 'config.json':
                    continue  # Exercise real anonymous repair through the application's downloader.
                source = args.artifacts / code / name
                if name == 'model.bin':
                    os.link(source, target / name)
                else:
                    shutil.copy2(source, target / name)
            manager = ModelManager(root, language=code)
            try:
                manager.thread.join(120)
                assert manager.snapshot()['status'] == 'error'
                manager.start()
                manager.thread.join(180)
                state = manager.snapshot()
                assert state['status'] == 'ready', state
                checks[code] = {'revision': cfg['revision'], 'verified_files': list(cfg['files']),
                               'anonymous_config_repair': True, 'status': state['status']}
            finally:
                manager.close()
    report = {'no_pytorch_or_transformers': True, 'inference_performed': False, 'models': checks}
    (ROOT / 'docs/evidence/v3-packaged-model-downloads.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
