import json
import time
from pathlib import Path
from unittest.mock import Mock
import pytest
from local_app.components import Components
from local_app.config import prepare_data
from local_app.gpu_components import validate_member
from local_app.models import ModelRegistry

def wait_for(fn):
    end=time.monotonic()+5
    while time.monotonic()<end:
        if fn(): return
        time.sleep(.02)
    assert fn()

def test_initial_choices_consumed_once_and_downloads_can_be_skipped(tmp_path):
    root=prepare_data(tmp_path)
    (root/'initial_setup.json').write_text(json.dumps({'schema_version':1,'setup_id':'one','components':['model:fr','model:vi']}))
    models=ModelRegistry(root)
    components=Components(root,models)
    assert [t['id'] for t in components.snapshot()['tasks']]==['model:fr','model:vi']
    components.control('pause')
    components.control('cancel','model:vi')
    restarted=Components(root,models)
    assert restarted.snapshot()['paused']
    assert len(restarted.snapshot()['tasks'])==2
    assert restarted.snapshot()['tasks'][1]['status']=='cancelled'
    models.close()

def test_unsupported_gpu_download_is_rejected_without_network(tmp_path,monkeypatch):
    root=prepare_data(tmp_path); models=ModelRegistry(root)
    components=Components(root,models)
    download=Mock(side_effect=AssertionError('Unexpected network'))
    monkeypatch.setattr('urllib.request.urlopen',download)
    components._check()
    with pytest.raises(ValueError,match='No compatible GPU'): components.enqueue('gpu:auto')
    assert components.snapshot()['tasks']==[]
    assert components.hardware('auto')['gpus']==[]
    download.assert_not_called()
    models.close()

def test_failure_does_not_block_later_download_and_retry_is_explicit(tmp_path):
    root=prepare_data(tmp_path)
    bad=Mock(thread=None); bad.snapshot.return_value={'status':'error','error':'Network unavailable'}
    good=Mock(thread=None); good.snapshot.return_value={'status':'ready'}
    models=Mock(); models.get.side_effect=lambda lang: bad if lang=='fr' else good
    components=Components(root,models)
    components.enqueue('model:fr'); components.enqueue('model:en'); components.start()
    try:
        wait_for(lambda: components.snapshot()['tasks'][1]['status']=='complete')
        assert components.snapshot()['tasks'][0]['status']=='error'
        assert bad.start.call_count==1
        components.control('pause')
        components.enqueue('model:fr')
        assert components.snapshot()['tasks'][-1]['status']=='queued'
    finally: components.close()

def test_interrupted_download_is_requeued_on_restart(tmp_path):
    root=prepare_data(tmp_path)
    (root/'download_queue.json').write_text(json.dumps({'tasks':[{'id':'model:en','status':'downloading'}],'paused':False,'consumed_setups':[]}))
    assert Components(root,Mock()).snapshot()['tasks'][0]['status']=='queued'

@pytest.mark.parametrize('name',['../escape.dll','/absolute.dll','C:/evil.dll','nvidia/../../bad.dll','x\\bad.dll'])
def test_runtime_archive_rejects_unsafe_paths(name):
    with pytest.raises(ValueError): validate_member(name)

def test_runtime_completion_invalidates_cpu_fallback_and_activates_provider(tmp_path,monkeypatch):
    class Provider:
        config={'files':[{'size':5}]}
        installed=False
        def __init__(self,root): pass
        def ready(self): return self.installed
        def install(self,stop,update): Provider.installed=True; return True
        def environment(self): return {'CHEP_GPU_RUNTIME':'test-runtime'}
    monkeypatch.setitem(__import__('local_app.components',fromlist=['PROVIDERS']).PROVIDERS,'test',Provider)
    monkeypatch.setattr('local_app.components.discover',lambda:{'compatible':True,'provider':'test','reason':'Compatible', 'eligible_indices':[1], 'adapters':[]})
    changed=Mock(); components=Components(prepare_data(tmp_path),Mock(),changed)
    components._check()
    assert components.snapshot()['gpu']['status']=='missing'
    components.enqueue('gpu:auto'); components.start()
    try:
        wait_for(lambda: components.snapshot()['tasks'][0]['status']=='complete')
        assert components.snapshot()['gpu']['status']=='ready'
        assert components.environment()=={'CHEP_GPU_RUNTIME':'test-runtime'}
        assert changed.call_count>=2
    finally: components.close()
