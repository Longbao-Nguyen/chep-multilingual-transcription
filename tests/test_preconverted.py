import copy
import hashlib

import pytest

from local_app.catalog import model_config, resolve_artifact
from local_app.models import ModelManager, ModelRegistry


@pytest.mark.parametrize('language', ['es', 'vi'])
def test_catalog_uses_ready_ct2_and_preserves_legacy_job_settings(language):
    current = model_config(language)
    assert not current['requires_conversion']
    assert current['ct2_model'].startswith('nunu-willump/')
    assert 'model.bin' in current['files']
    assert not any(name.endswith(('.safetensors', '.pt')) for name in current['files'])
    legacy = copy.deepcopy(current)
    legacy.update(requires_conversion=True, ct2_model=None, revision=current['source_revision'],
                  files={'pytorch_model.bin': {}}, chunk_seconds=17,
                  transcribe={'beam_size': 2}, cpu_fallback_order=[{'compute_type': 'float32', 'concurrency': 1}])
    before = copy.deepcopy(legacy)
    effective = resolve_artifact(language, legacy)
    assert legacy == before
    for key in ('chunk_seconds', 'transcribe', 'cpu_fallback_order'):
        assert effective[key] == legacy[key]
    for key in ('revision', 'files', 'ct2_model'):
        assert effective[key] == current[key]
    legacy['source_revision'] = 'different'
    with pytest.raises(ValueError, match='compatible preconverted'):
        resolve_artifact(language, legacy)


@pytest.mark.parametrize('corrupt', [False, True])
def test_legacy_cache_is_reused_only_after_integrity_verification(tmp_path, monkeypatch, corrupt):
    data = b'converted model'
    cfg = model_config('vi')
    cfg['files'] = {'model.bin': {'size': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'url': 'https://example.com/model.bin'}}
    legacy_path = tmp_path / 'models' / cfg['source_revision']
    legacy_path.mkdir(parents=True)
    (legacy_path / 'model.bin').write_bytes(b'x' * len(data) if corrupt else data)
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **k: pytest.fail('Unexpected automatic download'))
    manager = ModelManager(tmp_path, language='vi', config=cfg)
    manager.thread.join(5)
    assert manager.path == legacy_path
    assert manager.snapshot()['status'] == ('error' if corrupt else 'ready')
    manager.close()


def test_old_and_current_jobs_share_same_manager(tmp_path):
    current = model_config('es')
    old = {**current, 'revision': current['source_revision'], 'requires_conversion': True, 'ct2_model': None}
    registry = ModelRegistry(tmp_path)
    try:
        assert registry.get('es', old) is registry.get('es', current)
    finally:
        registry.close()

@pytest.mark.parametrize('language',['fr','en','de'])
def test_mirrored_artifacts_preserve_queued_settings_and_require_identity(language):
    current=model_config(language)
    assert current['ct2_model'].startswith('nunu-willump/')
    old=copy.deepcopy(current)
    old.update(revision=current['legacy_ct2_revisions'][-1],ct2_model='upstream/original',chunk_seconds=19)
    for meta in old['files'].values(): meta['url']='https://old.example/file'
    resolved=resolve_artifact(language,old)
    assert resolved['revision']==current['revision']
    assert resolved['chunk_seconds']==19
    assert old['revision']!=current['revision']
    next(iter(old['files'].values()))['sha256']='0'*64
    assert resolve_artifact(language,old)['revision']==old['revision']
