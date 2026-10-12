import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
models=json.loads((root/'config/transcription_models.json').read_text(encoding='utf-8'))['languages']
lines=[]
for code,cfg in models.items():
    size=sum(f['size'] for f in cfg['files'].values())
    lines.extend([f'#define Size_{code} {size}',f'#define Label_{code} "{cfg["label"]} Transcription Model ({size/1024**2:.0f} MB download)"'])
(root/'installer/component-sizes.iss').write_text('\n'.join(lines)+'\n')
