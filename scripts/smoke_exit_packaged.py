"""Packaged runtime + browser + native close dialog, with an empty Kaggle profile.

No user audio/credentials are used; no Kaggle notebook is submitted.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import httpx
from playwright.sync_api import sync_playwright
from check_native_close import find_dialog, user

ROOT=Path(__file__).resolve().parents[1]

def wait(fn,seconds=60):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        result=fn()
        if result: return result
        time.sleep(.15)
    raise TimeoutError('Expected state was not reached')

def main():
    testroot=ROOT/'test-results'/('exit-clean-'+uuid.uuid4().hex[:8])
    app=testroot/'app'
    profile=testroot/'empty-user-profile'
    data=testroot/'data'
    profile.mkdir(parents=True)
    data.mkdir()
    shutil.copytree(ROOT/'release/app',app)
    system=os.environ.get('SystemRoot',r'C:\Windows')
    env={k:v for k,v in os.environ.items() if not k.startswith(('KAGGLE','PYTHON'))}
    env.update(USERPROFILE=str(profile),APPDATA=str(profile/'AppData/Roaming'),LOCALAPPDATA=str(profile/'AppData/Local'),
               PATH=os.pathsep.join([str(app/'runtime'),system,system+r'\System32',system+r'\System32\WindowsPowerShell\v1.0']))
    log=(testroot/'launcher.log').open('w',encoding='utf-8')
    process=None
    client=None
    checks=[]
    screenshots=ROOT/'docs/screenshots'
    screenshots.mkdir(exist_ok=True)

    def launch():
        nonlocal process,client
        process=subprocess.Popen([str(app/'runtime/pythonw.exe'),str(app/'launch.py'),'--no-browser','--data-dir',str(data),'--port','0'],
            cwd=app,env=env,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
        wait(lambda:(data/'instance.json').exists(),45)
        details=json.loads((data/'instance.json').read_text())
        url,token=details['url'].split('#token=')
        client=httpx.Client(base_url=url,headers={'Authorization':'Bearer '+token},timeout=15)
        return details['url']

    def status():
        return client.get('api/status').json()

    try:
        url=launch()
        assert status()['kaggle']['installed'] is False
        with sync_playwright() as pw:
            browser=pw.chromium.launch(channel='msedge',headless=True)
            errors=[]
            context=browser.new_context(viewport={'width':1440,'height':1050})
            def open_page():
                page=context.new_page()
                page.on('pageerror',lambda e:errors.append(str(e)))
                connected=[]
                page.on('websocket',lambda ws:ws.on('framereceived',lambda payload: connected.append(payload)))
                page.goto(url)
                page.get_by_role('button',name='Close Chep',exact=True).wait_for()
                wait(lambda:any('connected' in str(frame) for frame in connected),15)
                return page
            page=open_page()
            page.locator('#device').select_option('kaggle')
            page.get_by_role('button',name='Download Kaggle CLI',exact=True).click()
            def cli_ready():
                state=status()['kaggle']
                if state['status']=='error': raise RuntimeError(state['error'])
                return state['installed'] and state['status']=='sign_in_required'
            wait(cli_ready,120)
            assert status()['kaggle']['username'] is None
            page.get_by_role('button',name='Authenticate Kaggle CLI',exact=True).wait_for()
            wait(lambda:page.get_by_role('button',name='Authenticate Kaggle CLI',exact=True).is_enabled())
            assert page.get_by_role('link',name='Create/Sign in an account').count()==1
            page.screenshot(path=str(screenshots/'v06-fresh-kaggle.png'),full_page=True)
            checks.append('embedded Python with empty profile and restricted PATH: CLI absent -> optional download -> sign-in required, no credentials inherited')
            # Queue metadata survives Quit; no online upload or transcription is started.
            client.put('api/settings',json={'device':'cpu'})
            job=client.post('api/jobs?name=queue-persistence.wav',content=b'local-only smoke fixture').json()
            page.reload()
            page.wait_for_timeout(3800)
            assert not find_dialog(process.pid)
            other=open_page()
            page.close()
            other.wait_for_timeout(3500)
            assert not find_dialog(process.pid)
            other.close()
            hwnd=wait(lambda:find_dialog(process.pid),15)
            user.PostMessageW(hwnd,0x400+102,1002,0)
            wait(lambda:not find_dialog(process.pid))
            assert process.poll() is None and status()['queue']['jobs'][0]['id']==job['id']
            checks.append('refresh and one-of-two-tab closure do not prompt; last-tab native Run in background keeps app alive')
            page=open_page()
            page.get_by_role('button',name='Close Chep',exact=True).click()
            page.get_by_role('dialog',name='Quit Chep or keep it running?').wait_for()
            page.screenshot(path=str(screenshots/'v06-close-dialog.png'),full_page=True)
            page.get_by_role('button',name='Run in background',exact=False).click()
            page.get_by_role('heading',name='Chep is running in background.').wait_for()
            assert process.poll() is None
            page.get_by_role('button',name='Return to Chep').click()
            page.get_by_role('button',name='Close Chep',exact=True).click()
            page.get_by_role('button',name='Quit completely',exact=False).click()
            process.wait(timeout=120)
            assert process.returncode==0
            assert not (data/'instance.json').exists()
            page.get_by_role('heading',name='Chep has stopped.').wait_for(timeout=15000)
            page.close()
            client.close()
            checks.append('in-app background/return/quit choices; actual packaged process exits with code 0 and removes instance lock metadata')
            url=launch()
            assert status()['queue']['jobs'][0]['id']==job['id']
            page=open_page()
            page.close()
            wait(lambda:find_dialog(process.pid),15)
            page=open_page()
            wait(lambda:not find_dialog(process.pid),10)
            assert process.poll() is None
            page.close()
            hwnd=wait(lambda:find_dialog(process.pid),15)
            user.PostMessageW(hwnd,0x400+102,1001,0)
            process.wait(timeout=120)
            assert process.returncode==0
            assert not errors,errors
            browser.close()
            checks.append('queue survives restart; reopening dismisses stale native dialog; native Quit completely exits actual pythonw process')
        # Validate that the packaged application directory is no longer locked.
        assert app.resolve().parent==testroot.resolve() and not app.is_symlink()
        shutil.rmtree(app)
        assert not app.exists()
        checks.append('entire test application folder deleted successfully after Quit; data folder retained')
        report={'version':'0.7.0','checks':checks,'scope':'Isolated profile on development Windows, not a separate clean Windows VM.'}
        (ROOT/'docs/evidence/v06-packaged-exit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report),flush=True)
    finally:
        if process and process.poll() is None:
            if client:
                try: client.post('api/shutdown')
                except Exception: pass
            try: process.wait(timeout=120)
            except subprocess.TimeoutExpired: process.kill();process.wait(10)
        if client: client.close()
        log.close()

if __name__=='__main__': main()
