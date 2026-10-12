"""Generate a private, CPU-only Kaggle conversion notebook from pinned evidence."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
folder = root/'benchmarks/ct2_release'
cell = 'from pathlib import Path\nimport subprocess,sys\n'
for source, target in [
    (folder/'source-catalog.json','catalog.json'),
    (folder/'convert.py','convert.py'),
    (root/'docs/evidence/kaggle-profiles/es-artifact.json','es-expected.json'),
    (root/'docs/evidence/kaggle-profiles/vi-artifact.json','vi-expected.json')]:
    cell += f"Path('/kaggle/working/{target}').write_text({source.read_text(encoding='utf-8')!r}, encoding='utf-8')\n"
cell += "subprocess.run([sys.executable,'/kaggle/working/convert.py'],check=True)\n"
notebook = {'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
            'cells':[{'id':'ct2-release','cell_type':'code','metadata':{},'source':cell.splitlines(True),'outputs':[],'execution_count':None}]}
(folder/'convert.ipynb').write_text(json.dumps(notebook),encoding='utf-8')
metadata = {'id':'longnj/chep-ct2-release','title':'Chep CT2 Release','code_file':'convert.ipynb','language':'python',
    'kernel_type':'notebook','is_private':'true','enable_gpu':'false','enable_internet':'true',
    'dataset_sources':[],'competition_sources':[],'kernel_sources':[],'model_sources':[]}
(folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
