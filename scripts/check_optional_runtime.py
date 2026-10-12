"""Exercise the real optional runtime queue without initializing GPU inference."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--app',type=Path,required=True)
args=parser.parse_args()
sys.path.insert(0,str(args.app.resolve()))
from local_app.config import prepare_data
from local_app.models import ModelRegistry
from local_app.components import Components

data=prepare_data(ROOT/'test-results/v4-optional-runtime')
models=ModelRegistry(data)
components=Components(data,models)
components.start()
try:
    deadline=time.monotonic()+30
    while components.snapshot()['gpu']['status']=='checking' and time.monotonic()<deadline: time.sleep(.1)
    assert components.snapshot()['gpu']['compatible'],components.snapshot()
    components.enqueue('gpu:auto')
    deadline=time.monotonic()+1800
    last=None
    while time.monotonic()<deadline:
        task=components.snapshot()['tasks'][-1]
        marker=(task['status'],int(task['progress']*10))
        if marker!=last: print(json.dumps(task),flush=True); last=marker
        if task['status']=='error': raise RuntimeError(task['error'])
        if task['status']=='complete': break
        time.sleep(.5)
    assert task['status']=='complete'
    assert components.provider.ready()
    report={'gpu_inference_executed':False,'download_queue':'passed','gpu':components.snapshot()['gpu'],
            'runtime_dir':str(components.provider.path),'installed_bytes':sum(p.stat().st_size for p in components.provider.path.rglob('*') if p.is_file())}
    (ROOT/'docs/evidence/v4-optional-runtime.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
finally:
    components.close(); models.close()
