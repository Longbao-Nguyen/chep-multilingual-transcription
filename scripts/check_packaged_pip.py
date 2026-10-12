"""Install with actual embedded Python + pip; verify real CLI OAuth startup.

An empty profile is used. Browser opening is intercepted, approval is not performed.
"""
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
data = ROOT / 'test-results' / ('pip-cli-' + uuid.uuid4().hex[:8])
data.mkdir(parents=True)
env = {k:v for k,v in os.environ.items() if not k.startswith(('KAGGLE', 'PYTHON'))}
env.update(USERPROFILE=str(data / 'profile'), KAGGLE_CONFIG_DIR=str(data / 'profile/.kaggle'))
code = '''import sys, json
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from local_app.kaggle_cli import KaggleCLI
cli = KaggleCLI(Path(sys.argv[2]))
cli.external = None
assert not cli.installed
cli.install()
assert cli.installed
version, status = cli.run(['--version'])
assert status == 0 and '2.2.4' in version
command = cli.command(['auth', 'login'])
marker = Path(sys.argv[2]) / 'browser-host.txt'
hook = 'import webbrowser; from pathlib import Path; from urllib.parse import urlparse; webbrowser.open = lambda url, *a, **k: (Path(' + repr(str(marker)) + ').write_text(urlparse(url).hostname or "") or True); '
command[2] = hook + command[2]
(Path(sys.argv[2]) / 'command.json').write_text(json.dumps(command))
print('PIP_INSTALL_AND_CLI_IMPORT_PASS')
'''
python = ROOT / 'release/app/runtime/python.exe'
subprocess.run([str(python), '-c', code, str(ROOT), str(data)], env=env, check=True)
command = json.loads((data / 'command.json').read_text())
process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
try:
    marker = data / 'browser-host.txt'
    for _ in range(150):
        if marker.exists() or process.poll() is not None: break
        time.sleep(.2)
    assert marker.exists(), 'Packaged CLI failed to request browser authentication'
    assert marker.read_text() in ('www.kaggle.com', 'kaggle.com')
    assert process.poll() is None, 'CLI did not wait for OAuth callback'
    print('PACKAGED_OAUTH_BROWSER_AND_WAIT_PASS (approval not performed)')
finally:
    if process.poll() is None: process.kill()
    process.communicate()
