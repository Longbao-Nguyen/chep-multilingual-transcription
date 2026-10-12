"""Opt-in real Kaggle/browser smoke. Uploads only a fixed, generated speech sample."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from local_app.config import prepare_data
from local_app.store import Store

DATA = ROOT / 'test-results/kaggle-e2e/data'
PORT = 18765

def wait(fn, seconds=60):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        value = fn()
        if value:
            return value
        time.sleep(.5)
    raise TimeoutError('Timed out waiting for expected state')

def main():
    import httpx
    from playwright.sync_api import sync_playwright
    prepare_data(DATA)
    Store(DATA).set_selection('en', 'kaggle')
    audio = DATA.parent / 'synthetic.wav'
    ps = "Add-Type -AssemblyName System.Speech; $v=New-Object System.Speech.Synthesis.SpeechSynthesizer; $v.SetOutputToWaveFile('" + str(audio).replace("'", "''") + "'); $v.Speak('Hello, this is a test of online transcription. The quick brown fox jumps over the lazy dog.'); $v.Dispose()"
    subprocess.run(['powershell', '-NoProfile', '-Command', ps], check=True)
    log = (DATA/'logs/smoke.log').open('w', encoding='utf-8')
    process = subprocess.Popen([sys.executable, __file__, '--server'], cwd=ROOT, stdout=log, stderr=log,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    client = httpx.Client(base_url=f'http://127.0.0.1:{PORT}', headers={'Authorization':'Bearer smoke-token'}, timeout=30)
    def status():
        try:
            return client.get('/api/status').json()
        except httpx.ConnectError:
            return None
    report = {'checks': []}
    try:
        wait(status)
        with sync_playwright() as pw:
            browser = pw.chromium.launch(channel='msedge', headless=True)
            page = browser.new_page(viewport={'width':1440, 'height':1100})
            errors = []
            page.on('pageerror', lambda e: errors.append(str(e)))
            page.goto(f'http://127.0.0.1:{PORT}/#token=smoke-token')
            page.get_by_role('heading', name='Online GPU', exact=True).wait_for()
            assert page.locator('#device option').count() == 3
            assert page.locator('.sidebar .storage-card').count() == 1
            assert page.locator('.details-column .storage-card').count() == 0
            if not status()['kaggle']['installed']:
                page.get_by_role('button', name='Download Kaggle CLI', exact=True).click()
            wait(lambda: status()['kaggle']['status'] == 'ready', 120)
            report['checks'].append('optional CLI wheel download and existing OAuth login; no pip or local model')
            (ROOT/'docs/screenshots').mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(ROOT/'docs/screenshots/kaggle-desktop.png'), full_page=True)
            page.locator('input[type=file]').set_input_files(str(audio))
            wait(lambda: any(j['status']=='QUEUED' for j in status()['queue']['jobs']))
            page.get_by_role('button', name='Start', exact=True).click()
            last = None
            def completed():
                nonlocal last
                jobs = status()['queue']['jobs']
                job = jobs[-1]
                if job['status'] != last:
                    print('KAGGLE_STAGE', job['status'], flush=True)
                    last = job['status']
                if job['status']=='FAILED':
                    raise RuntimeError(job['error'])
                return job if job['status']=='COMPLETED' and status()['queue']['state']=='PAUSED' else None
            job = wait(completed, 1200)
            txt = client.get(f"/api/jobs/{job['id']}/transcript").text
            srt = client.get(f"/api/jobs/{job['id']}/transcript?format=srt").text
            assert 'quick brown fox' in txt.lower() and '-->' in srt
            remote = json.loads(job['remote'])
            assert remote.get('cleaned'), remote
            page.get_by_role('button', name='History', exact=False).first.click()
            page.get_by_role('button', name='View', exact=True).last.click()
            page.get_by_role('dialog').wait_for()
            assert page.locator('dialog pre').inner_text() == txt
            page.get_by_role('button', name='Close transcript').click()
            with page.expect_download() as download:
                page.get_by_role('button', name='TXT', exact=True).last.click()
            assert download.value.suggested_filename == 'synthetic.txt'
            page.screenshot(path=str(ROOT/'docs/screenshots/kaggle-history.png'), full_page=True)
            page.set_viewport_size({'width':390,'height':844})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            page.screenshot(path=str(ROOT/'docs/screenshots/kaggle-mobile.png'), full_page=True)
            assert not errors, errors
            report.update(profile=json.loads(job['profile']), transcript=txt, remote_cleaned=True)
            report['checks'].extend(['browser queue upload -> private Kaggle dataset -> T4 -> pinned HF model -> TXT/SRT -> History',
                                    'remote dataset/notebook removed after verified local import',
                                    'desktop/mobile/sidebar storage/preview/download; no browser errors'])
            browser.close()
        (ROOT/'docs/evidence').mkdir(parents=True, exist_ok=True)
        (ROOT/'docs/evidence/kaggle-e2e.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report), flush=True)
    finally:
        try:
            client.post('/api/shutdown')
        except Exception:
            pass
        client.close()
        process.terminate()
        process.wait(timeout=15)
        log.close()

if __name__ == '__main__':
    if '--server' in sys.argv:
        import uvicorn
        from local_app.api import create_app
        app = create_app(DATA, 'smoke-token')
        app.state.engine.kaggle.cli.external = None
        uvicorn.run(app, host='127.0.0.1', port=PORT)
    else:
        main()
