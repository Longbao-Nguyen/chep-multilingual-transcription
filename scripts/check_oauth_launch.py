"""Verify the installed CLI reaches browser OAuth under an empty profile.

Intercept browser opening, then stop before approval; never reads user credentials.
"""
import os
from pathlib import Path
import subprocess
import time
import uuid

root = Path(__file__).resolve().parents[1]
folder = root / 'test-results' / ('oauth-launch-' + uuid.uuid4().hex[:8])
folder.mkdir(parents=True)
marker = folder / 'browser-host.txt'
code = '''import sys, webbrowser
from pathlib import Path
from urllib.parse import urlparse
marker = sys.argv[1]
def opened(url, *args, **kwargs):
    Path(marker).write_text(urlparse(url).hostname or '')
    return True
webbrowser.open = opened
sys.argv = ['kaggle', 'auth', 'login', '--force']
from kaggle.cli import main
main()
'''
env = {k: v for k, v in os.environ.items() if not k.startswith(('KAGGLE', 'PYTHON'))}
env.update(USERPROFILE=str(folder), KAGGLE_CONFIG_DIR=str(folder / '.kaggle'))
python = Path(r'C:\Users\nguye\anaconda3\envs\french_transcript\pythonw.exe')
process = subprocess.Popen([str(python), '-c', code, str(marker)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
try:
    for _ in range(100):
        if marker.exists() or process.poll() is not None: break
        time.sleep(.2)
    assert marker.exists(), 'CLI did not attempt browser authentication'
    assert marker.read_text() in ('www.kaggle.com', 'kaggle.com')
    assert process.poll() is None, 'CLI did not wait for browser approval'
    print('PASS: real CLI under pythonw opens Kaggle OAuth and waits for approval with no saved credentials. Approval not performed.')
finally:
    if process.poll() is None: process.kill()
    process.communicate()
