"""Pin official binary wheel downloads. No pip runs on end-user machines."""
import json
from pathlib import Path
import urllib.request
root=Path(__file__).resolve().parents[1]
packages=[]
for name,version in [('nvidia-cublas-cu12','12.4.5.8'),('nvidia-cudnn-cu12','9.1.1.17'),('nvidia-cuda-nvrtc-cu12','12.4.99')]:
    with urllib.request.urlopen(f'https://pypi.org/pypi/{name}/{version}/json') as response:
        data=json.load(response)
    wheel=next(f for f in data['urls'] if f['filename'].endswith('win_amd64.whl'))
    packages.append({'name':wheel['filename'],'url':wheel['url'],'size':wheel['size'],'sha256':wheel['digests']['sha256']})
catalog={'schema_version':1,'providers':{
 'cuda':{'enabled':True,'vendors':['nvidia'],'backend':'ctranslate2','device':'cuda','version':'cuda12.4-cudnn9.1-v1',
         'min_driver':'551.61','min_compute':6.0,'max_compute':9.0,'files':packages,'dll_dirs':['nvidia/cublas/bin','nvidia/cudnn/bin','nvidia/cuda_nvrtc/bin']},
 'openvino':{'enabled':False,'vendors':['intel'],'backend':'openvino','reason':'This GPU is detected, but its transcription backend is not available in this release. Using CPU.'},
 'vulkan':{'enabled':False,'vendors':['amd'],'backend':'whispercpp','reason':'This GPU is detected, but its transcription backend is not available in this release. Using CPU.'}}}
(root/'config/gpu_runtimes.json').write_text(json.dumps(catalog,indent=2)+'\n')
print('GPU download bytes',sum(f['size'] for f in packages))
