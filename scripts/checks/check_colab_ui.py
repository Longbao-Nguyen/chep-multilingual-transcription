"""Playwright UI verification for Google Colab CLI onboarding, OAuth code flow, GPU check, and Start."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--app', type=Path, default=None)
    parser.add_argument('--python', type=Path, default=None)
    args = parser.parse_args()

    data = ROOT / 'test-results' / ('colab-ui-' + uuid.uuid4().hex[:8])
    data.mkdir(parents=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith('KAGGLE')}
    env.update(USERPROFILE=str(data / 'profile'), KAGGLE_CONFIG_DIR=str(data / 'profile' / '.kaggle'))
    log = (data / 'server.log').open('w', encoding='utf-8')

    if args.app:
        app_root = args.app.resolve()
        if (app_root / 'Chep.exe').is_file():
            command = [str(app_root / 'Chep.exe'), '--no-browser', '--port', '0', '--data-dir', str(data)]
        else:
            embedded_py = app_root / 'runtime' / 'python.exe'
            python_exe = (
                str(args.python.resolve()) if args.python
                else str(embedded_py) if embedded_py.is_file()
                else str(ROOT / '.venv/Scripts/python.exe')
            )
            command = [python_exe, 'launch.py', '--no-browser', '--port', '0', '--data-dir', str(data)]
        cwd = app_root
    else:
        venv_python = args.python or (ROOT / '.venv/Scripts/python.exe')
        python_exe = str(venv_python) if Path(venv_python).is_file() else sys.executable
        command = [python_exe, 'launch.py', '--no-browser', '--port', '0', '--data-dir', str(data)]
        cwd = ROOT

    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=log,
        stderr=log,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
    )
    client = None
    try:
        for _ in range(300):
            if (data / 'instance.json').exists():
                break
            time.sleep(0.2)
        if not (data / 'instance.json').exists():
            raise RuntimeError('Server did not write instance.json in time.')
        instance = json.loads((data / 'instance.json').read_text(encoding='utf-8'))
        url, token = instance['url'].split('#token=')
        client = httpx.Client(base_url=url, headers={'Authorization': 'Bearer ' + token})

        with sync_playwright() as pw:
            browser = pw.chromium.launch(channel='msedge', headless=True)
            page = browser.new_page(viewport={'width': 1280, 'height': 900})
            errors = []
            dialogs = []
            page.on('pageerror', lambda e: errors.append(str(e)))
            page.on('dialog', lambda d: (dialogs.append(d.message), d.dismiss()))

            colab_state = {
                'installed': False,
                'status': 'idle',
                'username': None,
                'message': '',
                'error': None,
                'auth_url': None,
                'download_bytes': 45388389,
                'progress': 0.0,
            }
            gpu_state = {'status': 'unchecked', 'username': None, 'message': '', 'quota': None}

            def status_route(route):
                response = route.fetch()
                body = response.json()
                body['colab'] = colab_state.copy()
                body['colab_gpu'] = gpu_state.copy()
                route.fulfill(json=body)

            page.route('**/api/status', status_route)
            page.goto(instance['url'])

            page.locator('#device').select_option('colab')
            page.get_by_text('Audio is uploaded to a temporary Google Colab T4 session', exact=False).wait_for()

            card = page.get_by_role('region', name='Colab CLI Connection')
            card.get_by_text('Not ready', exact=False).wait_for()
            auth_btn = card.get_by_role('button', name='Authenticate Colab CLI')
            auth_btn.click()
            page.get_by_role('alert').filter(has_text='Download Colab CLI below').wait_for()

            # Simulate Colab CLI download & installation
            page.route('**/api/colab/download', lambda route: (colab_state.update(installed=True, status='sign_in_required'), route.fulfill(json={'ok': True})))
            card.get_by_role('button', name='Download Colab CLI', exact=False).click()
            page.wait_for_timeout(1500)

            # Simulate OAuth URL + awaiting_code flow
            requests = []
            def login_route(route):
                requests.append('login')
                colab_state.update(
                    status='awaiting_code',
                    auth_url='https://accounts.google.com/o/oauth2/auth?client_id=test',
                    message='Open the Google sign-in page, approve access, then paste the code below.',
                )
                route.fulfill(json={'ok': True})

            submitted_codes = []
            def code_route(route):
                body = json.loads(route.request.post_data or '{}')
                submitted_codes.append(body.get('code'))
                colab_state.update(
                    status='ready',
                    username='tester@example.com',
                    auth_url=None,
                    message='Connected as tester@example.com',
                )
                route.fulfill(json={'ok': True})

            page.route('**/api/colab/login', login_route)
            page.route('**/api/colab/code', code_route)

            auth_btn.click()
            card.get_by_role('link', name='Open Google sign-in page').wait_for()
            assert auth_btn.is_disabled()

            code_input = card.get_by_role('textbox', name='Google authorization code')
            code_input.fill('4/0TESTCODE')
            card.get_by_role('button', name='Submit code').click()
            card.get_by_text('tester@example.com', exact=True).wait_for()
            assert submitted_codes == ['4/0TESTCODE']
            assert auth_btn.is_disabled()

            # Simulate GPU check
            page.route(
                '**/api/colab/gpu/check',
                lambda route: (
                    gpu_state.update(
                        status='ready',
                        username='tester@example.com',
                        message='Colab Tesla T4 confirmed.',
                        quota={'balance': '95.25', 'rate': '1.96', 'active': 0},
                    ),
                    route.fulfill(json={'ok': True}),
                ),
            )
            card.get_by_role('button', name='Check GPU condition').click()
            card.get_by_text('Colab Tesla T4 confirmed.', exact=False).wait_for()
            assert card.get_by_text('YOUR COLAB USAGE').count() == 0
            assert page.locator('.provider-tabs').count() == 0
            assert card.get_by_role('link', name='Open Google Colab').count() == 0

            for language in ['fr', 'en', 'es', 'vi']:
                page.locator('#language').select_option(language)
                page.wait_for_timeout(300)
                assert page.get_by_text('Download and verify the selected model first.', exact=True).count() == 0

            page.locator('input[type=file]').set_input_files({'name': 'fixture.wav', 'mimeType': 'audio/wav', 'buffer': b'test'})
            page.get_by_role('heading', name='fixture.wav').wait_for()
            starts = []
            page.route('**/api/queue/start', lambda route: (starts.append(True), route.fulfill(json={'ok': True})))
            page.get_by_role('button', name='Start', exact=True).click()
            page.wait_for_timeout(500)
            assert starts == [True]
            assert dialogs == []
            assert errors == []

            page.screenshot(path=str(data / 'colab_ui.png'), full_page=True)
            client.post('/api/lifecycle/choose', json={'choice': 'background'})
            browser.close()
        print('PASS: Colab provider UI, code submission, GPU check quota, language switching, and Start verified.')
    finally:
        if client:
            try:
                client.post('/api/shutdown')
            except Exception:
                pass
            client.close()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
        log.close()
        shutil.rmtree(data, ignore_errors=True)


if __name__ == '__main__':
    main()
