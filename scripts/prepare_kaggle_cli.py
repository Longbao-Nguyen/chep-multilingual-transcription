"""Resolve with pip --dry-run --ignore-installed --only-binary=:all: first."""
import concurrent.futures
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
report = json.loads((ROOT/'kaggle-dependencies-report.json').read_text(encoding='utf-8'))
def entry(item):
    info = item['download_info']
    url = info['url']
    name = url.rsplit('/', 1)[-1]
    with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60) as r:
        size = int(r.headers['Content-Length'])
    return {'name': name, 'url': url, 'size': size, 'sha256': info['archive_info']['hashes']['sha256']}
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    files = list(pool.map(entry, report['install']))
with urllib.request.urlopen('https://pypi.org/pypi/pip/25.3/json', timeout=60) as response:
    pip_meta = json.load(response)
wheel = next(f for f in pip_meta['urls'] if f['packagetype'] == 'bdist_wheel')
pip = {'name':wheel['filename'], 'url':wheel['url'], 'size':wheel['size'], 'sha256':wheel['digests']['sha256']}
(ROOT/'config/kaggle_cli.json').write_text(json.dumps({'version':'2.2.4', 'python':'3.12', 'platform':'win_amd64', 'files':files, 'pip':pip}, indent=2))
print('CLI download bytes:', sum(f['size'] for f in files))
