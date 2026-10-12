"""UI states and Start behavior; authentication response is simulated, no cloud job."""
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
import httpx
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
data = ROOT / 'test-results' / ('connection-' + uuid.uuid4().hex[:8])
data.mkdir(parents=True)
env = {k: v for k, v in os.environ.items() if not k.startswith('KAGGLE')}
env.update(USERPROFILE=str(data / 'profile'), KAGGLE_CONFIG_DIR=str(data / 'profile' / '.kaggle'))
log = (data / 'server.log').open('w')
process = subprocess.Popen([str(ROOT / '.venv/Scripts/python.exe'), 'launch.py', '--no-browser', '--port', '0', '--data-dir', str(data)], cwd=ROOT, env=env, stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW)
client = None
try:
    for _ in range(200):
        if (data / 'instance.json').exists(): break
        time.sleep(.2)
    instance = json.loads((data / 'instance.json').read_text())
    url, token = instance['url'].split('#token=')
    client = httpx.Client(base_url=url, headers={'Authorization': 'Bearer ' + token})
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width': 1280, 'height': 900})
        errors = []
        page.on('pageerror', lambda e: errors.append(str(e)))
        state = {'installed': False, 'status': 'sign_in_required', 'username': None, 'message': '', 'error': None, 'download_bytes': 3516025, 'progress': 0}
        def status_route(route):
            response = route.fetch()
            body = response.json()
            body['kaggle'] = state.copy()
            route.fulfill(json=body)
        page.route('**/api/status', status_route)
        page.goto(instance['url'])
        card = page.get_by_role('region', name='Kaggle CLI Connection')
        card.get_by_text('Not ready', exact=False).wait_for()
        button = card.get_by_role('button', name='Authenticate Kaggle CLI')
        button.click()
        page.get_by_role('alert').filter(has_text='Download Kaggle CLI below').wait_for()
        state['installed'] = True
        page.wait_for_timeout(1700)
        requests = []
        def login(route):
            requests.append('login')
            state.update(status='connecting', message='Waiting for approval...')
            route.fulfill(json={'ok': True})
        page.route('**/api/kaggle/login', login)
        button.click()
        card.get_by_text('Waiting for approval...').wait_for()
        assert button.is_disabled()
        state.update(status='ready', username='test-user', message='')
        card.get_by_text('test-user', exact=True).wait_for()
        assert button.is_disabled() and requests == ['login']
        assert page.get_by_role('button', name='Check connection').count() == 0
        assert page.get_by_text('Multilingual Transcription', exact=True).count() == 1
        assert page.get_by_text('From sound to words.', exact=True).count() == 0
        assert page.get_by_text('YOUR VOICE, SIMPLY WRITTEN', exact=True).count() == 0
        for language in ['fr', 'en', 'es', 'de', 'vi']:
            page.locator('#language').select_option(language)
            page.wait_for_timeout(500)
            assert page.locator('.model-card').get_by_text('CT2:', exact=False).count() == 0
            assert page.locator('.model-card').get_by_text('Chunk:', exact=False).count() == 0
            assert page.get_by_text('Download and verify the selected model first.', exact=True).count() == 0
        assert page.locator('.kaggle-account .card-kicker').evaluate('(e) => getComputedStyle(e).fontSize') == page.locator('.sidebar-storage .card-kicker').evaluate('(e) => getComputedStyle(e).fontSize')
        page.locator('#device').select_option('kaggle')
        page.get_by_text('Audio is processed on your private Kaggle account', exact=False).wait_for()
        page.locator('input[type=file]').set_input_files({'name':'fixture.wav', 'mimeType':'audio/wav', 'buffer':b'test'})
        page.get_by_role('heading', name='fixture.wav').wait_for()
        starts = []
        page.route('**/api/queue/start', lambda route: (starts.append(True), route.fulfill(json={'ok':True})))
        dialogs = []
        page.on('dialog', lambda dialog: (dialogs.append(dialog.message), dialog.dismiss()))
        page.get_by_role('button', name='Start', exact=True).click()
        page.wait_for_timeout(500)
        assert starts == [True] and dialogs == [] and errors == []
        page.screenshot(path=str(data / 'connection.png'), full_page=True)
        client.post('/api/lifecycle/choose', json={'choice':'background'})
        browser.close()
    print('PASS: missing CLI guidance, OAuth request/wait/ready states, layout, Start without popup; OAuth approval simulated.')
finally:
    if client:
        client.post('/api/shutdown')
        client.close()
    try: process.wait(timeout=20)
    except subprocess.TimeoutExpired: process.kill()
    log.close()
