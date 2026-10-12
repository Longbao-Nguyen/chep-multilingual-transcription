"""Browser -> local API -> CPU inference -> saved transcript. Never uses CUDA."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import httpx
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from local_app.config import prepare_data
from local_app.store import Store

def wait_for(fn, timeout=300):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        result=fn()
        if result: return result
        time.sleep(.5)
    raise TimeoutError('Expected state was not reached.')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model-dir',type=Path)
    parser.add_argument('--audio',type=Path)
    parser.add_argument('--app',type=Path,default=ROOT)
    parser.add_argument('--python',type=Path,default=Path(sys.executable))
    parser.add_argument('--label',default='v2-cpu')
    args=parser.parse_args()
    args.app=args.app.resolve()
    args.python=args.python.resolve()
    if args.model_dir: args.model_dir=args.model_dir.resolve()
    if args.audio: args.audio=args.audio.resolve()
    data=prepare_data(ROOT/'test-results'/args.label/'data')
    store=Store(data)
    store.set_selection('fr','cpu')
    screenshots=ROOT/'docs'/'screenshots'
    screenshots.mkdir(exist_ok=True)
    evidence=ROOT/'docs'/'evidence'
    evidence.mkdir(exist_ok=True)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1'}
    command=[str(args.python),str(args.app/'launch.py'),'--data-dir',str(data),'--no-browser','--port','0']
    if args.model_dir: command.extend(['--model-dir',str(args.model_dir)])
    log=(data/'logs'/'smoke-server.log').open('w',encoding='utf-8')
    server=subprocess.Popen(command,cwd=args.app,env=env,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    client=None
    report={'device':'cpu','cuda_visible_devices':'','checks':[]}
    try:
        wait_for(lambda:(data/'instance.json').is_file(),60)
        instance=json.loads((data/'instance.json').read_text())
        url,token=instance['url'].split('#token=')
        client=httpx.Client(base_url=url,headers={'Authorization':'Bearer '+token},timeout=30)
        with sync_playwright() as api:
            browser=api.chromium.launch(channel='msedge',headless=True)
            page=browser.new_page(viewport={'width':1440,'height':1100})
            errors=[]
            page.on('pageerror',lambda error:errors.append(str(error)))
            page.goto(instance['url'])
            page.get_by_role('button',name='Close Chep',exact=True).wait_for()
            wait_for(lambda:page.locator('#language option').count()==5,30)
            assert page.locator('#device option').count()==3
            assert page.get_by_role('button',name='Settings',exact=True).count()==0
            assert page.locator('footer,.sidebar-note,.topbar-actions,.chunk-note,.ready-label').count()==0
            assert page.locator('html').get_attribute('lang')=='en'
            for language,model in [('en','distil-whisper/distil-large-v3.5-ct2'),('es','adriszmar/whisper-large-v3-turbo-es'),('de','primeline/whisper-large-v3-turbo-german'),('vi','vinai/PhoWhisper-medium')]:
                page.locator('#language').select_option(language)
                page.get_by_text(model,exact=True).wait_for()
                wait_for(lambda:not page.locator('#language').is_disabled(),30)
                assert page.get_by_text('Compute type',exact=True).count()==0
                assert page.get_by_role('button',name='Download / repair model').count()==1
            report['checks'].append('five language choices, CPU selector, missing models, no stale precision, removed UI regions')
            wait_for(lambda: client.get('api/status').json()['components']['gpu']['status']!='checking',30)
            compatible=client.get('api/status').json()['components']['gpu']['compatible']
            warning=page.get_by_text('CPU processing may be significantly slower than GPU. For faster transcription, we recommend using a compatible GPU.',exact=True)
            wait_for(lambda:warning.count()==int(compatible),30)
            # The selected Vietnamese model is missing, so Auto cannot initiate inference.
            page.locator('#device').select_option('auto')
            wait_for(lambda:client.get('api/status').json()['settings']['device']=='auto',30)
            if compatible:
                wait_for(lambda:page.get_by_role('button',name='Download GPU Runtime',exact=True).count()==1,30)
                page.once('dialog',lambda dialog:dialog.dismiss())
                page.get_by_role('button',name='Download GPU Runtime',exact=True).click()
                assert not client.get('api/status').json()['components']['tasks']
            else:
                assert page.get_by_role('button',name='Download GPU Runtime',exact=True).count()==0
            page.locator('#device').select_option('cpu')
            wait_for(lambda:client.get('api/status').json()['settings']['device']=='cpu',30)
            report['checks'].append('CPU recommendation follows compatibility; optional GPU runtime confirmation can be dismissed without downloading')
            page.screenshot(path=str(screenshots/'v2-desktop.png'),full_page=True)
            page.get_by_role('button',name='History',exact=False).first.click()
            assert page.locator('footer,.sidebar-note,.topbar-actions').count()==0
            page.screenshot(path=str(screenshots/'v2-history.png'),full_page=True)
            page.get_by_role('button',name='Transcribe',exact=False).first.click()
            page.set_viewport_size({'width':390,'height':844})
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
            page.screenshot(path=str(screenshots/'v2-mobile.png'),full_page=True)
            page.set_viewport_size({'width':1440,'height':1100})
            if args.model_dir and args.audio:
                page.locator('#language').select_option('fr')
                def ready():
                    payload=client.get('api/status').json()
                    if payload['runtime']['status']=='error': raise RuntimeError(payload['runtime'])
                    return payload['runtime']['status']=='ready'
                wait_for(ready,360)
                profile=client.get('api/status').json()['runtime']['profile']
                assert profile['device']=='cpu'
                page.locator('input[type=file]').set_input_files(str(args.audio))
                wait_for(lambda:any(j['status']=='QUEUED' for j in client.get('api/status').json()['queue']['jobs']),30)
                wait_for(lambda:page.get_by_role('button',name='Start',exact=True).is_enabled(),30)
                page.get_by_role('button',name='Start',exact=True).click()
                def completed():
                    jobs=client.get('api/status').json()['queue']['jobs']
                    if jobs and jobs[-1]['status']=='FAILED': raise RuntimeError(jobs[-1]['error'])
                    return jobs[-1] if jobs and jobs[-1]['status']=='COMPLETED' else None
                job=wait_for(completed,420)
                actual=json.loads(job['profile'])
                assert actual['device']=='cpu' and actual['language']=='fr'
                txt=client.get(f"api/jobs/{job['id']}/transcript").text
                srt=client.get(f"api/jobs/{job['id']}/transcript?format=srt").text
                assert len(txt)>50 and '-->' in srt
                page.get_by_role('button',name='History',exact=False).first.click()
                page.get_by_role('button',name='View',exact=True).first.click()
                page.get_by_role('dialog').wait_for()
                assert page.locator('dialog pre').inner_text()==txt
                page.get_by_role('button',name='Close transcript').click()
                with page.expect_download() as download:
                    page.get_by_role('button',name='TXT',exact=True).first.click()
                assert download.value.suggested_filename.endswith('.txt')
                report['profile']=actual
                report['transcript_characters']=len(txt)
                report['checks'].append('actual CPU probe and transcription; frozen job config; TXT/SRT; preview and browser download')
                page.screenshot(path=str(screenshots/'v2-completed.png'),full_page=True)
            assert not errors,errors
            report['checks'].append('desktop/mobile/history rendering; no browser errors; no mobile horizontal overflow')
            browser.close()
        (evidence/(args.label+'.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report),flush=True)
    finally:
        if client:
            try:client.post('api/shutdown')
            except Exception:pass
            client.close()
        try:server.wait(timeout=30)
        except subprocess.TimeoutExpired:server.terminate();server.wait(timeout=10)
        log.close()

if __name__=='__main__':
    main()
