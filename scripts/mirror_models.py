"""Mirror pinned CT2 files with upstream attribution and byte identity checks."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {'fr':'whisper-large-v3-french-distil-dec16-ct2', 'en':'distil-large-v3.5-ct2', 'de':'whisper-large-v3-turbo-german-ct2'}
HF = Path.home()/'anaconda3/envs/french_transcript/Scripts/hf.exe'

def main():
    path = ROOT/'config/transcription_models.json'
    catalog = json.loads(path.read_text(encoding='utf-8'))
    evidence = {}
    for code, name in TARGETS.items():
        cfg = catalog['languages'][code]
        source = cfg['ct2_model'] or cfg['model']
        revision = cfg['revision']
        target = 'nunu-willump/'+name
        if source == target:
            continue
        folder = ROOT/'.build/mirrors'/code
        folder.mkdir(parents=True, exist_ok=True)
        info = HfApi().model_info(source, revision=revision)
        assert info.card_data.get('license') == cfg['license']
        upstream = hf_hub_download(source, 'README.md', revision=revision)
        shutil.copy2(upstream, folder/'UPSTREAM_MODEL_CARD.md')
        for filename, meta in cfg['files'].items():
            prefix = 'ctranslate2/' if code == 'fr' else ''
            cached = hf_hub_download(source, prefix+filename, revision=revision)
            with open(cached, 'rb') as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == meta['sha256']
            shutil.copy2(cached, folder/filename)
        licenses = {'LICENSE-Whisper':'https://raw.githubusercontent.com/openai/whisper/main/LICENSE'}
        licenses['LICENSE'] = ('https://www.apache.org/licenses/LICENSE-2.0.txt' if code == 'de' else
                               'https://raw.githubusercontent.com/huggingface/distil-whisper/main/LICENSE' if code == 'en' else licenses['LICENSE-Whisper'])
        for filename, url in licenses.items():
            with urllib.request.urlopen(url) as response:
                (folder/filename).write_bytes(response.read())
        if code == 'de':
            p = hf_hub_download(cfg['model'], 'README.md', revision=cfg['source_revision'])
            shutil.copy2(p, folder/'ORIGINAL_MODEL_CARD.md')
            # Retain any original notices when supplied by either upstream repository.
            original = HfApi().model_info(cfg['model'], revision=cfg['source_revision'])
            for origin, rev, records in [(source,revision,info.siblings),(cfg['model'],cfg['source_revision'],original.siblings)]:
                for f in records:
                    if f.rfilename.upper() in ('NOTICE','NOTICE.TXT','LICENSE','LICENSE.TXT'):
                        p=hf_hub_download(origin,f.rfilename,revision=rev)
                        shutil.copy2(p,folder/(origin.split('/')[0]+'-'+f.rfilename))
        manifest = {'source':source,'revision':revision,'original_model':cfg['model'],'original_revision':cfg['source_revision'],
                    'files':{n:{k:m[k] for k in ('size','sha256')} for n,m in cfg['files'].items()},'weights_modified':False}
        (folder/'provenance.json').write_text(json.dumps(manifest,indent=2))
        (folder/'README.md').write_text(f'''---
license: {cfg['license']}
language: {code}
library_name: ctranslate2
pipeline_tag: automatic-speech-recognition
base_model: {cfg['model']}
---
# {name} — preservation mirror

Byte-identical CTranslate2 inference files from [{source}](https://huggingface.co/{source}) at `{revision}`.
Mirrored by nunu-willump for Chep download availability. No training, conversion or weight modification was performed.
The French CT2 subfolder is placed at the repository root. Other changes are attribution and provenance documents only.

All credit belongs to the original model authors and CT2 converters named in `UPSTREAM_MODEL_CARD.md`.
Read the retained model cards for citations, datasets, restrictions and limitations. Model license: `{cfg['license']}`.
The original Whisper copyright and license are retained in `LICENSE-Whisper`.
No additional fine-tuner copyright notice was supplied in the source repository unless included as a separate retained file.
See `provenance.json` for immutable source references, sizes and SHA-256 hashes.

```python
from faster_whisper import WhisperModel
model = WhisperModel("{target}", device="cpu", compute_type="float32")
segments, info = model.transcribe("audio.wav", language="{code}")
for segment in segments:
    print(segment.text)
```
''', encoding='utf-8')
        subprocess.run([str(HF),'upload',target,str(folder),'.','--no-private','--commit-message','Preserve pinned CT2 weights with upstream licenses and attribution'],check=True)
        published=HfApi(token=False).model_info(target,files_metadata=True)
        assert not published.private
        remote={f.rfilename:f for f in published.siblings}
        for filename,meta in cfg['files'].items():
            assert remote[filename].size == meta['size']
            url=hf_hub_url(target,filename,revision=published.sha)
            if remote[filename].lfs:
                assert remote[filename].lfs.sha256 == meta['sha256']
            else:
                with urllib.request.urlopen(url) as response:
                    assert hashlib.sha256(response.read()).hexdigest()==meta['sha256']
            meta['url']=url
        cfg.update(ct2_model=target,revision=published.sha,legacy_ct2_revisions=list(dict.fromkeys([*cfg.get('legacy_ct2_revisions',[]),revision])))
        evidence[code]={**manifest,'mirror':target,'mirror_revision':published.sha}
        path.write_text(json.dumps(catalog,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        (ROOT/'docs/evidence/v4-mirrors.json').write_text(json.dumps(evidence,indent=2)+'\n')
        print(code,target,published.sha,flush=True)

if __name__=='__main__':
    main()
