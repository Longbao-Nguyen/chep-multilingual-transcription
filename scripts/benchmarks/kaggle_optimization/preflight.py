"""Read account/model metadata and submit a benchmark-owned GPU eligibility probe."""
import argparse
import json
import time
import uuid
from pathlib import Path
from .core import save
from .cli import cleanup
from local_app.kaggle_cli import KaggleCLI
from local_app.kaggle_backend import kernel_status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cli-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--owner', required=True)
    args = parser.parse_args()
    cli = KaggleCLI(args.cli_root)
    if cli.identity() != args.owner:
        raise ValueError('Account mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    nonce = uuid.uuid4().hex[:16]
    ref = args.owner + '/chep-opt-' + nonce + '-00-a-0'
    save(args.output/'ownership.json', {'owner': args.owner, 'experiment': nonce,
         'resources': [{'kind': 'kernels', 'ref': ref, 'deleted': False}]})
    source = '''import json, subprocess, time
from pathlib import Path
r = {'ok': False, 'at': time.time()}
try:
    r['gpu'] = subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=name','--format=csv,noheader'],text=True).strip()
    r['ok'] = 'T4' in r['gpu']
except Exception as e:
    r['error_type'] = type(e).__name__
Path('/kaggle/working/preflight.json').write_text(json.dumps(r))
'''
    save(args.output/'probe.ipynb', {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'name': 'python3', 'language': 'python'}}, 'cells': [
        {'id': 'gpu-probe', 'cell_type': 'code', 'metadata': {}, 'execution_count': None,
         'outputs': [], 'source': source.splitlines(True)}]})
    save(args.output/'kernel-metadata.json', {'id': ref, 'title': ref.split('/')[1],
        'code_file': 'probe.ipynb', 'language': 'python', 'kernel_type': 'notebook',
        'is_private': True, 'enable_gpu': True, 'enable_internet': True,
        'machine_shape': 'NvidiaTeslaT4', 'dataset_sources': [], 'competition_sources': [],
        'kernel_sources': [], 'model_sources': []})
    events = []
    try:
        cli.run(['kernels','push','-p',str(args.output),'--accelerator','NvidiaTeslaT4','-t','600'])
        deadline = time.monotonic() + 1800
        while True:
            status = kernel_status(cli.run(['kernels','status',ref])[0])
            events.append({'status': status, 'at': time.time()})
            save(args.output/'events.json', events)
            print(status, flush=True)
            if status == 'COMPLETE':
                break
            if status is None or any(s in status for s in ('ERROR','FAILED','CANCEL')) or time.monotonic() > deadline:
                # Retrieve diagnostics before removing our failed probe. Never print raw CLI logs.
                cli.run(['kernels','output',ref,'-p',str(args.output),'--force'], check=False)
                raise ValueError('PENDING: GPU probe failed')
            time.sleep(10)
        cli.run(['kernels','output',ref,'-p',str(args.output),'--file-pattern',r'^preflight\.json$','--force'])
        print((args.output/'preflight.json').read_text(), flush=True)
    finally:
        cleanup(args.output, cli)


if __name__ == '__main__':
    main()
