"""Isolated helper scripts executed inside the google-colab-cli environment."""
from __future__ import annotations

PASSIVE_AUTH_HELPER = r"""
import json, os, sys, urllib.parse, urllib.request
from pathlib import Path

payload = json.loads(sys.stdin.readline())
fallback = Path(payload['fallback_token'])
result = {'authenticated': False, 'email': None}

import colab_cli.auth as auth
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

creds = None
source = None
try:
    token_json = auth._get_token_from_keyring()
    if token_json:
        creds = Credentials.from_authorized_user_info(json.loads(token_json), auth.PUBLIC_SCOPES)
        source = 'keyring'
except Exception:
    creds = None

if not creds and fallback.is_file():
    try:
        creds = Credentials.from_authorized_user_file(str(fallback), auth.PUBLIC_SCOPES)
        source = 'fallback'
    except Exception:
        creds = None

if not creds and os.path.exists(auth.LEGACY_TOKEN_PATH):
    try:
        creds = Credentials.from_authorized_user_file(auth.LEGACY_TOKEN_PATH, auth.PUBLIC_SCOPES)
        source = 'legacy'
    except Exception:
        creds = None

if creds:
    if not creds.valid and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        serialized = creds.to_json()
        try:
            auth._save_token_to_keyring(serialized)
        except Exception:
            fallback.parent.mkdir(parents=True, exist_ok=True)
            fallback.write_text(serialized, encoding='utf-8')
    if creds.valid or creds.token:
        qs = urllib.parse.urlencode({'access_token': creds.token})
        with urllib.request.urlopen(f'https://oauth2.googleapis.com/tokeninfo?{qs}', timeout=15) as resp:
            info = json.loads(resp.read().decode('utf-8'))
        result = {'authenticated': True, 'email': info.get('email') or 'Google user', 'source': source}

print('CHEP_COLAB_JSON:' + json.dumps(result), flush=True)
"""


LOGIN_HELPER = r"""
import json, sys, urllib.parse, urllib.request
from pathlib import Path
from importlib import resources
import colab_cli.auth as auth
from google_auth_oauthlib.flow import InstalledAppFlow

payload = json.loads(sys.stdin.readline())
fallback = Path(payload['fallback_token'])

config_res = resources.files('colab_cli').joinpath('oauth_config.json')
client_config = json.loads(config_res.read_text(encoding='utf-8'))
flow = InstalledAppFlow.from_client_config(client_config, auth.PUBLIC_SCOPES)
flow.redirect_uri = auth.REMOTE_REDIRECT_URI
auth_url, _ = flow.authorization_url(prompt='consent', token_usage='remote')
print('CHEP_COLAB_AUTH_URL:' + auth_url, flush=True)

code = sys.stdin.readline().strip()
if not code:
    print('CHEP_COLAB_JSON:' + json.dumps({'ok': False, 'error': 'Missing authorization code.'}), flush=True)
    raise SystemExit(1)

flow.fetch_token(code=code)
creds = flow.credentials
serialized = creds.to_json()
try:
    auth._save_token_to_keyring(serialized)
    fallback.unlink(missing_ok=True)
except Exception:
    fallback.parent.mkdir(parents=True, exist_ok=True)
    fallback.write_text(serialized, encoding='utf-8')

qs = urllib.parse.urlencode({'access_token': creds.token})
with urllib.request.urlopen(f'https://oauth2.googleapis.com/tokeninfo?{qs}', timeout=15) as resp:
    info = json.loads(resp.read().decode('utf-8'))

email = info.get('email') or 'Google user'
print('CHEP_COLAB_JSON:' + json.dumps({'ok': True, 'email': email}), flush=True)
"""


LOGOUT_HELPER = r"""
import json, sys
from pathlib import Path

payload = json.loads(sys.stdin.readline())
fallback = Path(payload['fallback_token'])
fallback.unlink(missing_ok=True)

try:
    import keyring
    import colab_cli.auth as auth
    try:
        keyring.delete_password(auth.KEYRING_SERVICE_NAME, auth.KEYRING_USERNAME)
    except Exception:
        pass
except Exception:
    pass

print('CHEP_COLAB_JSON:' + json.dumps({'ok': True}), flush=True)
"""


HELPER_AUTH_PREAMBLE = r"""
def _load_colab_session(fallback_path):
    import json, os, urllib.parse, urllib.request
    from pathlib import Path
    import colab_cli.auth as auth
    from colab_cli.client import Client, Prod
    from google.auth.transport import requests as g_requests
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    fallback = Path(fallback_path)
    creds = None
    try:
        token_json = auth._get_token_from_keyring()
        if token_json:
            creds = Credentials.from_authorized_user_info(json.loads(token_json), auth.PUBLIC_SCOPES)
    except Exception:
        creds = None
    if not creds and fallback.is_file():
        creds = Credentials.from_authorized_user_file(str(fallback), auth.PUBLIC_SCOPES)
    if not creds and os.path.exists(auth.LEGACY_TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(auth.LEGACY_TOKEN_PATH, auth.PUBLIC_SCOPES)
    if not creds:
        raise RuntimeError('Sign in to Google Colab before starting online transcription.')
    if not creds.valid and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        serialized = creds.to_json()
        try:
            auth._save_token_to_keyring(serialized)
        except Exception:
            fallback.parent.mkdir(parents=True, exist_ok=True)
            fallback.write_text(serialized, encoding='utf-8')
    qs = urllib.parse.urlencode({'access_token': creds.token})
    with urllib.request.urlopen(f'https://oauth2.googleapis.com/tokeninfo?{qs}', timeout=15) as resp:
        email = json.loads(resp.read().decode('utf-8')).get('email') or 'Google user'
    return email, Client(Prod(), g_requests.AuthorizedSession(creds))
"""



def refresh_session_helper() -> str:
    return HELPER_AUTH_PREAMBLE + r"""
import json, sys, uuid
from colab_cli.client import Accelerator, Variant
from colab_cli.state import SessionState

payload = json.loads(sys.stdin.read())
email, client = _load_colab_session(payload['fallback_token'])
target_ep = payload.get('endpoint')
allow_create = payload.get('allow_create', True)

assignments = {a.endpoint: a for a in client.list_assignments()}
if target_ep and target_ep in assignments:
    info = assignments[target_ep].runtime_proxy_info
    session = {
        'owner': email,
        'endpoint': target_ep,
        'url': info.url,
        'token': info.token,
        'expires_at': info.expires_at().isoformat(),
        'reused': True,
    }
elif target_ep and not allow_create:
    raise RuntimeError('The Google Colab runtime disconnected. Retry to run on a new Colab GPU session.')
else:
    res = client.assign(uuid.uuid4(), variant=Variant.GPU, accelerator=Accelerator.T4)
    info = res.runtime_proxy_info
    session = {
        'owner': email,
        'endpoint': res.endpoint,
        'url': info.url,
        'token': info.token,
        'expires_at': info.expires_at().isoformat(),
        'reused': False,
    }

print('CHEP_COLAB_JSON:' + json.dumps(session), flush=True)
"""


def upload_file_helper() -> str:
    return r"""
import json, sys
from colab_cli.contents import ContentsClient
from colab_cli.state import SessionState

payload = json.loads(sys.stdin.read())
s = SessionState(name='chep', token=payload['token'], url=payload['url'], endpoint=payload['endpoint'])
client = ContentsClient(s)
client.upload(payload['local_path'], payload['remote_path'])
print('CHEP_COLAB_JSON:' + json.dumps({'ok': True}), flush=True)
"""


def exec_helper() -> str:
    return r"""
import json, sys
from colab_cli.runtime import ColabRuntime

payload = json.loads(sys.stdin.read())
runtime = ColabRuntime(payload['url'], payload['token'])
try:
    outputs = runtime.execute_code(payload['code'], timeout=float(payload.get('timeout', 60)))
finally:
    runtime.stop(shutdown_kernel=False)
text = ''.join(o.get('text', '') for o in outputs if o.get('output_type') == 'stream')
errors = [o.get('evalue', '') for o in outputs if o.get('output_type') == 'error']
if errors:
    raise RuntimeError('; '.join(errors))
print('CHEP_COLAB_JSON:' + json.dumps({'ok': True, 'stdout': text}), flush=True)
"""


def poll_status_helper() -> str:
    return r"""
import json, sys, tempfile
from pathlib import Path
from colab_cli.contents import ContentsClient
from colab_cli.state import SessionState

payload = json.loads(sys.stdin.read())
s = SessionState(name='chep', token=payload['token'], url=payload['url'], endpoint=payload['endpoint'])
client = ContentsClient(s)
status_data = None
with tempfile.TemporaryDirectory() as td:
    dst = Path(td) / 'status.json'
    try:
        client.download(payload['status_path'], str(dst))
        status_data = json.loads(dst.read_text(encoding='utf-8'))
    except FileNotFoundError:
        status_data = None
print('CHEP_COLAB_JSON:' + json.dumps({'ok': True, 'status': status_data}), flush=True)
"""


def download_results_helper() -> str:
    return r"""
import json, sys
from pathlib import Path
from colab_cli.contents import ContentsClient
from colab_cli.state import SessionState

payload = json.loads(sys.stdin.read())
s = SessionState(name='chep', token=payload['token'], url=payload['url'], endpoint=payload['endpoint'])
client = ContentsClient(s)
out_dir = Path(payload['local_dir'])
out_dir.mkdir(parents=True, exist_ok=True)
for pair in payload['files']:
    client.download(pair['remote'], str(out_dir / pair['name']))
print('CHEP_COLAB_JSON:' + json.dumps({'ok': True}), flush=True)
"""


def unassign_helper() -> str:
    return HELPER_AUTH_PREAMBLE + r"""
import json, sys
from colab_cli.client import ColabRequestError
from colab_cli.runtime import ColabRuntime
from colab_cli.utils import get_status_code

payload = json.loads(sys.stdin.read())
email, client = _load_colab_session(payload['fallback_token'])
ep = payload['endpoint']
assignments = {a.endpoint: a for a in client.list_assignments()}
if ep not in assignments:
    print('CHEP_COLAB_JSON:' + json.dumps({'ok': True, 'missing': True}), flush=True)
else:
    info = assignments[ep].runtime_proxy_info
    try:
        ColabRuntime(info.url, info.token).stop(shutdown_kernel=True)
    except Exception:
        pass
    try:
        client.unassign(ep)
    except ColabRequestError as exc:
        if get_status_code(exc) != 404:
            raise
    print('CHEP_COLAB_JSON:' + json.dumps({'ok': True, 'missing': False}), flush=True)
"""

