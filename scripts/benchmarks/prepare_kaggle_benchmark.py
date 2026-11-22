import json
import argparse
import os
import re
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--gpu-only', action='store_true')
parser.add_argument('--owner', default=os.environ.get('CHEP_KAGGLE_OWNER'), help='Kaggle username for kernel-metadata.json (or set CHEP_KAGGLE_OWNER).')
args=parser.parse_args()
if not args.owner or not re.fullmatch(r'[a-zA-Z0-9_-]+', args.owner):
    parser.error('Provide --owner <kaggle-username> or set CHEP_KAGGLE_OWNER.')
root=Path(__file__).resolve().parents[2]
source=root/'benchmarks'/'kaggle_multilingual'
target=root/'benchmarks'/('kaggle_profiles' if args.gpu_only else 'kaggle_multilingual')
target.mkdir(parents=True,exist_ok=True)
catalog=(root/'config'/'transcription_models.json').read_text(encoding='utf-8')
script=(source/'benchmark.py').read_text(encoding='utf-8')
cell='from pathlib import Path\nimport subprocess,sys,os\n'
if args.gpu_only:
    cell+="os.environ['BENCHMARK_GPU_ONLY']='1'\n"
cell+=f"Path('/kaggle/working/catalog.json').write_text({catalog!r})\n"
cell+=f"Path('/kaggle/working/benchmark.py').write_text({script!r})\n"
cell+="subprocess.run([sys.executable, '/kaggle/working/benchmark.py'], check=True)\n"
notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
          'cells':[{'id':'multilingual-validation','cell_type':'code','metadata':{},'source':cell.splitlines(True),'outputs':[],'execution_count':None}]}
(target/'benchmark.ipynb').write_text(json.dumps(notebook),encoding='utf-8')
(target/'kernel-metadata.json').write_text(json.dumps({'id':f"{args.owner}/"+('local-transcript-gpu-profile-validation' if args.gpu_only else 'local-transcript-multilingual-validation'),
    'title':'Local Transcript GPU Profile Validation' if args.gpu_only else 'Local Transcript Multilingual Validation','code_file':'benchmark.ipynb','language':'python','kernel_type':'notebook',
    'is_private':'true','enable_gpu':'true','enable_internet':'true','machine_shape':'NvidiaTeslaT4',
    'dataset_sources':[],'competition_sources':[],'kernel_sources':[],'model_sources':[]},indent=2))
