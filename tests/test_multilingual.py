import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from local_app.api import create_app
from local_app.catalog import model_config, selection
from local_app.config import prepare_data
from local_app.hardware import candidates, detect
from local_app.models import ModelRegistry
from local_app.runtime import RuntimeResolver
from local_app.store import Store

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr('local_app.engine.detect', lambda *args: {'gpus':[], 'cpu_name':'Test CPU'})
    monkeypatch.setattr('local_app.runtime.detect', lambda *args: {'gpus':[], 'cpu_name':'Test CPU'})
    monkeypatch.setattr(RuntimeResolver, 'probe', lambda self, selected, path, profile, generation: profile)
    with TestClient(create_app(tmp_path, 'test'), headers={'Authorization':'Bearer test'}) as client:
        yield client

def test_no_download_on_startup_and_all_languages_are_available(client, monkeypatch):
    download = Mock(side_effect=AssertionError('Unexpected network request'))
    monkeypatch.setattr('urllib.request.urlopen', download)
    payload = client.get('/api/status').json()
    assert [c['language'] for c in payload['languages']] == ['fr','en','es','de','vi']
    assert payload['model']['status'] == 'missing'
    assert payload['runtime']['profile'] is None
    for language in ['en','de','es','vi','fr']:
        assert client.put('/api/settings',json={'language':language}).status_code == 200
        result=client.get('/api/status').json()
        assert result['model']['language'] == language
        assert result['runtime']['profile'] is None
    download.assert_not_called()

def test_job_selection_is_frozen_across_changes_and_retry(client):
    client.put('/api/settings',json={'language':'en','device':'cpu'})
    job=client.post('/api/jobs?name=english.wav', content=b'audio').json()
    original=json.loads(job['config'])
    assert original['language']=='en' and original['device_preference']=='cpu'
    client.put('/api/settings',json={'language':'vi','device':'auto'})
    engine=client.app.state.engine
    engine.store.update(job['id'],status='FAILED',profile='{"device":"cpu"}',attempts='[{"error":"old attempt"}]')
    assert client.post(f"/api/jobs/{job['id']}/retry").status_code==200
    assert json.loads(engine.store.get(job['id'])['config'])==original
    assert engine.store.get(job['id'])['profile'] is None
    assert engine.store.get(job['id'])['attempts'] is None
    assert client.put('/api/settings',json={'language':'xx'}).status_code==400

def test_block_selection_and_preparation_during_running_and_pausing(client):
    for state in ['RUNNING','PAUSING']:
        client.app.state.engine.state=state
        assert client.put('/api/settings',json={'language':'vi'}).status_code==409
        assert client.post('/api/model/download').status_code==409
        assert client.post('/api/runtime/recheck').status_code==409

def test_queue_reports_the_missing_model_from_the_job_not_current_dropdown(client):
    client.put('/api/settings',json={'language':'en'})
    job=client.post('/api/jobs?name=english.wav',content=b'audio').json()
    # Validate fixture policy so this check isolates model availability.
    config=json.loads(job['config']); config['model_config']['validation_status']='validated'
    with client.app.state.engine.store.connect() as db:
        db.execute('UPDATE jobs SET config=? WHERE id=?',(json.dumps(config),job['id']))
    client.put('/api/settings',json={'language':'vi'})
    result=client.post('/api/queue/start')
    assert result.status_code==409 and 'English' in result.json()['detail']

def test_migration_preserves_existing_history_and_recorded_profile(tmp_path):
    root=prepare_data(tmp_path)
    with sqlite3.connect(root/'queue.sqlite3') as db:
        db.executescript("CREATE TABLE jobs(id TEXT PRIMARY KEY,name TEXT,size INTEGER,status TEXT,progress REAL,duration REAL,processed REAL,error TEXT,created REAL,updated REAL,profile TEXT);")
        db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?)',('old','old.wav',10,'COMPLETED',1,30,30,None,1,2,'{"device":"cpu","compute_type":"int8","concurrency":1}'))
    (root/'outputs'/'old.txt').write_text('Bonjour',encoding='utf-8')
    store=Store(root)
    row=store.get('old')
    assert row['status']=='COMPLETED' and row['updated']==2
    assert json.loads(row['config'])['legacy'] is True
    assert json.loads(row['profile'])['compute_type']=='int8'
    assert (root/'outputs'/'old.txt').read_text()=='Bonjour'
    assert Store(root).get('old')==row

def test_cpu_detection_never_queries_cuda(monkeypatch):
    monkeypatch.setattr('local_app.hardware.platform.processor',lambda:'Test CPU')
    forbidden=Mock(side_effect=AssertionError('CUDA was queried'))
    monkeypatch.setattr(subprocess,'run',forbidden)
    assert detect('cpu')['gpus']==[]
    forbidden.assert_not_called()

def test_pending_model_never_claims_a_confirmed_profile(tmp_path):
    resolver=RuntimeResolver(prepare_data(tmp_path))
    manager=Mock(); manager.snapshot.return_value={'status':'ready'}
    config=selection('vi','cpu'); config['model_config']['validation_status']='pending'
    resolver.ensure(config,manager)
    assert resolver.snapshot()['status']=='blocked'
    assert resolver.snapshot()['profile'] is None
    resolver.close()

def test_resolver_tries_candidates_in_order_and_reports_actual_precision(tmp_path,monkeypatch):
    resolver=RuntimeResolver(prepare_data(tmp_path))
    gpu={'index':0,'name':'Test GPU','total_mib':4096,'free_mib':3900}
    monkeypatch.setattr('local_app.runtime.detect',lambda *args:{'gpus':[gpu]})
    seen=[]
    def probe(selected,path,profile,generation):
        seen.append((profile['compute_type'],profile['concurrency']))
        if len(seen)<4: raise RuntimeError('CUDA out of memory')
        return {**profile,'compute_type':'int8_float16'}
    monkeypatch.setattr(resolver,'probe',probe)
    manager=Mock(); manager.snapshot.return_value={'status':'ready'}
    resolver.ensure(selection('fr','auto'),manager)
    resolver.thread.join(5)
    state=resolver.snapshot()
    assert state['status']=='ready'
    assert seen==[('float16',2),('float16',1),('int8_float16',2),('int8_float16',1)]
    assert len(state['attempts'])==3 and state['profile']['concurrency']==1
    resolver.close()

def test_new_selection_cannot_receive_a_stale_probe_result(tmp_path,monkeypatch):
    import threading
    resolver=RuntimeResolver(prepare_data(tmp_path))
    monkeypatch.setattr('local_app.runtime.detect',lambda *args:{'gpus':[]})
    entered, release=threading.Event(),threading.Event()
    def probe(*args):
        entered.set(); release.wait(5)
        return {'device':'cpu','compute_type':'int8','concurrency':1}
    monkeypatch.setattr(resolver,'probe',probe)
    manager=Mock(); manager.snapshot.return_value={'status':'ready'}
    resolver.ensure(selection('fr','cpu'),manager)
    assert entered.wait(2)
    resolver.invalidate()
    release.set(); resolver.thread.join(5)
    assert resolver.snapshot()['profile'] is None
    resolver.close()

def test_registry_isolates_language_downloads(tmp_path):
    registry=ModelRegistry(prepare_data(tmp_path))
    fr,en=registry.get('fr'),registry.get('en')
    fr.update(status='downloading',progress=.4)
    assert en.snapshot()['status']=='missing'
    assert fr.path != en.path
    registry.close()

def test_fresh_process_fallback_replaces_partial_output_and_preserves_metadata(client, tmp_path, monkeypatch):
    """Run real child processes, but fake inference so no local GPU is used."""
    engine=client.app.state.engine
    gpu={'index':0,'name':'Test GPU','total_mib':4096,'free_mib':3900}
    monkeypatch.setattr('local_app.engine.detect',lambda *args:{'gpus':[gpu]})
    monkeypatch.setattr(engine.components,'hardware',lambda *args:{'gpus':[gpu]})
    manager=engine.models.get('fr'); manager.update(status='ready')
    worker=tmp_path/'fake_worker.py'
    worker.write_text('''import json,pathlib,sys
spec=json.loads(pathlib.Path(sys.argv[1]).read_text())
p=spec['profile']; out=pathlib.Path(spec['output'])
for ext in ('.txt.part','.srt.part'): out.with_suffix(ext).write_text('partial')
if p['concurrency']==2:
 print(json.dumps({'event':'error','message':'CUDA out of memory','gpu_failure':True}),flush=True)
 sys.exit(1)
actual={**p,'language':spec['config']['language'],'model_id':spec['config']['model_config']['model'],'chunk_seconds':spec['config']['model_config']['chunk_seconds'],'backend':'ctranslate2','compute_type':'float16'}
print(json.dumps({'event':'validated','profile':actual}),flush=True)
for ext in ('.txt.part','.srt.part'): out.with_suffix(ext).write_text('complete transcript')
print(json.dumps({'event':'done'}),flush=True)
''',encoding='utf-8')
    real_popen=subprocess.Popen
    processes=[]
    def popen(args,**kwargs):
        if processes: assert processes[-1].poll() is not None
        process=real_popen([sys.executable,str(worker),args[-1]],**kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr('local_app.engine.subprocess.Popen',popen)
    job=client.post('/api/jobs?name=french.wav',content=b'audio').json()
    assert client.post('/api/queue/start').status_code==200
    engine.thread.join(15)
    row=engine.store.get(job['id'])
    assert row['status']=='COMPLETED'
    assert len(processes)==2
    assert json.loads(row['profile'])['language']=='fr'
    assert json.loads(row['profile'])['concurrency']==1
    assert len(json.loads(row['attempts']))==1
    assert (engine.store.root/'outputs'/f"{job['id']}.txt").read_text()=='complete transcript'
    assert not (engine.store.root/'outputs'/f"{job['id']}.txt.part").exists()
