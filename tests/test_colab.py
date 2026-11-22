import hashlib
import json
import sys
import threading
import time
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from local_app import colab_backend
from local_app.api import create_app
from local_app.catalog import selection
from local_app.cleanup_manager import CleanupManager
from local_app.colab_backend import ColabBackend, MAX_REMOTE_BYTES, validate_colab_result
from local_app.colab_cli import AUTH_URL_PREFIX, ColabCLI, ColabStopped, JSON_PREFIX, safe_error
from local_app.colab_gpu import ColabGpuCheck
from local_app.colab_script import build_worker_script
from local_app.config import ROOT, prepare_data
from local_app.store import Store


def _mark_colab_installed(cli: ColabCLI) -> None:
    cli.packages.mkdir(parents=True, exist_ok=True)
    (cli.packages / 'chep-ready.json').write_text('{"version": "0.8.0"}', encoding='utf-8')


def _colab_result_files(directory: Path, job_id: str = 'job', source_sha256: str = 'source', config: dict | None = None) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    result = {
        'schema_version': 1,
        'job_id': job_id,
        'source_sha256': source_sha256,
        'ok': True,
        'duration': 2.5,
        'profile': {'device': 'cuda', 'provider': 'colab', 'device_name': 'Tesla T4'},
        'files': {},
    }
    if config:
        result['profile'].update(
            language=config['language'],
            revision=config['model_config']['revision'],
            **config['model_config']['sweet_spot'],
        )
    for ext in ('txt', 'srt'):
        data = f'synthetic colab transcript ({ext})'.encode('utf-8')
        (directory / f'{job_id}.{ext}').write_bytes(data)
        result['files'][ext] = {'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    (directory / 'result.json').write_text(json.dumps(result), encoding='utf-8')
    return result


@pytest.fixture
def colab_app(tmp_path, monkeypatch):
    monkeypatch.setattr('local_app.engine.detect', lambda *args: {'gpus': [], 'cpu_name': 'Test CPU'})
    app = create_app(tmp_path, 'token')
    engine = app.state.engine
    _mark_colab_installed(engine.colab.cli)
    engine.colab.cli.state.update(status='ready', username='colab.user@example.com')
    with TestClient(app, headers={'Authorization': 'Bearer token'}) as client:
        yield client, engine


def test_colab_catalog_schema():
    catalog = json.loads((ROOT / 'config' / 'colab_cli.json').read_text(encoding='utf-8'))
    assert catalog['version'] == '0.8.0'
    entries = [catalog['pip'], *catalog['files']]
    assert len(entries) >= 2
    for entry in entries:
        assert entry['url'].startswith('https://')
        assert entry['name']
        assert entry['size'] > 0
        assert len(entry['sha256']) == 64
        int(entry['sha256'], 16)
    names = {entry['name'] for entry in catalog['files']}
    assert 'google_colab_cli-0.8.0-py3-none-any.whl' in names


def test_colab_settings_status_and_upload_limit(colab_app, monkeypatch):
    client, engine = colab_app
    monkeypatch.setattr(engine.runtime, 'ensure', Mock(side_effect=AssertionError('Local probe must not run')))
    assert client.put('/api/settings', json={'device': 'unknown'}).status_code == 400
    assert client.put('/api/settings', json={'device': 'colab', 'language': 'fr'}).status_code == 200

    status = client.get('/api/status').json()
    assert status['runtime']['status'] == 'online'
    assert status['colab']['installed'] is True
    assert status['colab']['download_bytes'] > 0
    assert status['colab_gpu']['status'] == 'unchecked'

    too_big = client.post(
        '/api/jobs?name=big.wav',
        content=b'x',
        headers={'Content-Length': str(3 * 1024**3)},
    )
    assert too_big.status_code == 413

    job = client.post('/api/jobs?name=sample.wav', content=b'audio').json()
    assert json.loads(job['config'])['device_preference'] == 'colab'

    engine.colab.cli.state['status'] = 'sign_in_required'
    assert client.post('/api/queue/start').status_code == 409

    engine.colab.cli.state['status'] = 'ready'
    with engine.store.connect() as db:
        db.execute('UPDATE jobs SET size=? WHERE id=?', (MAX_REMOTE_BYTES + 1, job['id']))
    assert client.post('/api/queue/start').status_code == 409

    with engine.store.connect() as db:
        db.execute('UPDATE jobs SET size=? WHERE id=?', (5, job['id']))
    dispatched = threading.Event()
    monkeypatch.setattr(engine, '_online_job', lambda j: dispatched.set())
    assert client.post('/api/queue/start').status_code == 200
    assert dispatched.wait(2)


def test_cpu_jobs_never_enter_colab_backend(colab_app, monkeypatch):
    client, engine = colab_app
    client.put('/api/settings', json={'device': 'cpu'})
    job = client.post('/api/jobs?name=local.wav', content=b'local').json()
    colab_run = Mock(side_effect=AssertionError('Colab backend must not run for CPU jobs'))
    monkeypatch.setattr(engine.colab, 'run', colab_run)
    monkeypatch.setattr('local_app.engine.run_local_job', Mock())
    engine._job(job)
    colab_run.assert_not_called()


def test_colab_cli_safe_error_and_actions(tmp_path, monkeypatch):
    secret = 'https://accounts.google.com/o/oauth2/auth?code=4/secret-token&token=abc'
    assert 'secret-token' not in safe_error(secret)
    assert 'quota' in safe_error('412 TooManyAssignmentsError allocation refused').lower()
    assert 'sign in again' in safe_error('401 invalid_grant').lower()
    assert 'sign in to google colab' in safe_error('Please Sign in to Google Colab first').lower()

    cli = ColabCLI(tmp_path)
    assert not cli.installed
    with pytest.raises(ValueError, match='Download Colab CLI first'):
        cli.action('login')
    assert cli.thread is None

    _mark_colab_installed(cli)
    assert cli.installed
    monkeypatch.setattr(cli, 'identity', Mock(side_effect=ValueError('Sign in to Google Colab before starting online transcription.')))
    cli._action('check')
    assert cli.snapshot()['status'] == 'sign_in_required'
    assert cli.snapshot()['error'] is None
    assert cli.snapshot()['username'] is None

    monkeypatch.setattr(cli, 'identity', lambda: 'ready.user@example.com')
    cli._action('check')
    assert cli.snapshot()['status'] == 'ready'
    assert cli.snapshot()['username'] == 'ready.user@example.com'


def test_colab_cli_oauth_code_flow_cancel_and_logout(colab_app, monkeypatch):
    client, engine = colab_app
    cli = engine.colab.cli
    cli.state.update(status='sign_in_required', username=None)
    monkeypatch.setattr('local_app.colab_cli.webbrowser.open', lambda url: True)

    fake_helper = (
        'import json, sys\n'
        'json.loads(sys.stdin.readline())\n'
        'print("CHEP_COLAB_AUTH_URL:https://accounts.google.com/o/oauth2/test?state=1", flush=True)\n'
        'code = sys.stdin.readline().strip()\n'
        'if code == "4/0TESTCODE":\n'
        '    print("CHEP_COLAB_JSON:" + json.dumps({"ok": True, "email": "signed.in@example.com"}), flush=True)\n'
        'else:\n'
        '    print("CHEP_COLAB_JSON:" + json.dumps({"ok": False, "error": "Invalid code."}), flush=True)\n'
        '    raise SystemExit(1)\n'
    )
    monkeypatch.setattr(cli, '_helper_command', lambda code: [sys.executable, '-c', fake_helper])

    assert client.post('/api/colab/login').status_code == 200
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and cli.snapshot()['status'] != 'awaiting_code':
        time.sleep(0.05)
    snap = cli.snapshot()
    assert snap['status'] == 'awaiting_code'
    assert snap['auth_url'] == 'https://accounts.google.com/o/oauth2/test?state=1'

    assert client.post('/api/colab/code', json={'code': '   '}).status_code == 409
    assert client.post('/api/colab/code', json={'code': '4/0TESTCODE'}).status_code == 200
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and cli.snapshot()['status'] != 'ready':
        time.sleep(0.05)
    assert cli.snapshot()['status'] == 'ready'
    assert cli.snapshot()['username'] == 'signed.in@example.com'
    assert cli.snapshot()['auth_url'] is None

    # Cancel flow
    cli.state.update(status='sign_in_required', username=None)
    assert client.post('/api/colab/login').status_code == 200
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and cli.snapshot()['status'] != 'awaiting_code':
        time.sleep(0.05)
    assert client.post('/api/colab/cancel').status_code == 200
    assert cli.thread is not None
    cli.thread.join(timeout=5)
    assert cli.snapshot()['status'] == 'sign_in_required'
    assert cli.snapshot()['error'] is None

    # Logout cleans fallback token
    cli.fallback_token_path.write_text('{"token":"dummy"}', encoding='utf-8')
    monkeypatch.setattr(cli, 'run_helper', lambda code, payload=None, **kw: {'ok': True})
    assert client.post('/api/colab/logout').status_code == 200
    assert not cli.fallback_token_path.exists()
    assert cli.snapshot()['status'] == 'sign_in_required'


def test_colab_gpu_check_states_and_pending_cleanup(tmp_path, monkeypatch):
    cli = ColabCLI(tmp_path)
    _mark_colab_installed(cli)
    gpu = ColabGpuCheck(tmp_path, cli)

    with pytest.raises(ValueError, match='Authenticate Colab CLI'):
        gpu.start()

    cli.state.update(status='ready', username='tester@example.com')
    monkeypatch.setattr(cli, 'identity', lambda: 'tester@example.com')

    # Pending cleanup case (cleaned=False) -> followed by clean check clearing pending file
    responses = iter([
        {
            'ok': True,
            'status': 'ready',
            'email': 'tester@example.com',
            'quota': {'balance': '98.50', 'rate': '1.96', 'active': 1},
            'gpu_name': 'Tesla T4',
            'endpoint': 'ep-pending',
            'cleaned': False,
            'message': '',
        },
        {
            'ok': True,
            'status': 'ready',
            'email': 'tester@example.com',
            'quota': {'balance': '98.40', 'rate': '1.96', 'active': 0},
            'gpu_name': 'Tesla T4',
            'endpoint': 'ep-clean',
            'cleaned': True,
            'message': '',
        },
        {
            'ok': False,
            'status': 'quota_exhausted',
            'email': 'tester@example.com',
            'quota': {'balance': '0.00', 'rate': '1.96', 'active': 0},
            'gpu_name': '',
            'endpoint': None,
            'cleaned': True,
            'message': 'No T4 allocation.',
        },
    ])
    seen_payloads = []

    def fake_helper(code, payload=None, **kw):
        seen_payloads.append(payload)
        return next(responses)

    monkeypatch.setattr(cli, 'run_helper', fake_helper)

    gpu._check('tester@example.com')
    assert gpu.pending_path.is_file()
    assert gpu.snapshot()['status'] == 'cleanup_pending'

    gpu._check('tester@example.com')
    assert seen_payloads[1]['pending_endpoint'] == 'ep-pending'
    assert not gpu.pending_path.exists()
    assert gpu.snapshot()['status'] == 'ready'
    assert gpu.snapshot()['quota']['balance'] == '98.40'

    gpu._check('tester@example.com')
    assert gpu.snapshot()['status'] == 'quota_exhausted'
    assert 'No T4' in gpu.snapshot()['message']


@pytest.mark.parametrize('language', ['en', 'fr', 'es', 'vi'])
def test_colab_worker_script_compiles_and_contains_pinned_model(language):
    cfg = selection(language, 'colab')
    script = build_worker_script(cfg, 'a' * 32, 'b' * 64, 3)
    compile(script, f'<colab_worker_{language}>', 'exec')
    assert cfg['model_config']['revision'] in script
    for meta in cfg['model_config']['files'].values():
        assert meta['sha256'] in script
        assert meta['url'] in script
    assert 'preload_cuda12_libs' in script
    assert "'provider': 'colab'" in script
    assert 'status.json' in script
    assert 'result.json' in script
    assert 'N_CHUNKS = 3' in script
    assert 'oauth_config' not in script


def test_validate_colab_result_checks_manifest_provider_and_checksum(tmp_path):
    _colab_result_files(tmp_path)
    assert validate_colab_result(tmp_path, 'job', 'source')['ok']

    with pytest.raises(ValueError, match='match'):
        validate_colab_result(tmp_path, 'wrong-job', 'source')

    manifest = json.loads((tmp_path / 'result.json').read_text(encoding='utf-8'))
    manifest['profile']['provider'] = 'kaggle'
    (tmp_path / 'result.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='CUDA GPU profile'):
        validate_colab_result(tmp_path, 'job', 'source')

    _colab_result_files(tmp_path)
    (tmp_path / 'job.txt').write_text('tampered', encoding='utf-8')
    with pytest.raises(ValueError, match='integrity'):
        validate_colab_result(tmp_path, 'job', 'source')


def test_colab_backend_chunked_upload_resume_validation_and_session_reuse(tmp_path, monkeypatch):
    store = Store(prepare_data(tmp_path))
    config = selection('en', 'colab')
    job_id = 'a' * 32
    job_id_2 = 'b' * 32
    audio_bytes = b'0123456789A'  # 11 bytes -> 3 chunks at 4 bytes each
    source_sha = hashlib.sha256(audio_bytes).hexdigest()

    store.add(job_id, 'part.wav', len(audio_bytes), config)
    (tmp_path / 'sources' / job_id).write_bytes(audio_bytes)
    store.add(job_id_2, 'queued2.wav', 5, config)
    (tmp_path / 'sources' / job_id_2).write_bytes(b'audio')

    monkeypatch.setattr(colab_backend, 'UPLOAD_CHUNK_BYTES', 4)
    backend = ColabBackend(store)
    _mark_colab_installed(backend.cli)
    backend.cli.identity = lambda: 'colab.user@example.com'

    uploaded_parts = []
    uploaded_script_text = []
    unassign_calls = []

    def fake_helper(code, payload=None, **kw):
        if 'client.assign' in code:
            return {
                'owner': 'colab.user@example.com',
                'endpoint': 'ep-warm-1',
                'url': 'https://proxy.example/ep-warm-1',
                'token': 'tok-1',
                'expires_at': '2099-01-01T00:00:00+00:00',
                'reused': False,
            }
        if 'ContentsClient(s)' in code and 'client.upload' in code:
            remote_path = payload['remote_path']
            data = Path(payload['local_path']).read_bytes()
            if remote_path.endswith('.py'):
                uploaded_script_text.append(data.decode('utf-8'))
            else:
                uploaded_parts.append((remote_path, data))
            return {'ok': True}
        if 'ColabRuntime' in code and 'execute_code' in code:
            return {'ok': True, 'stdout': ''}
        if 'status_path' in (payload or {}):
            return {'ok': True, 'status': {'status': 'DOWNLOADING', 'progress': 0.96, 'duration': 2.5, 'done': True, 'ok': True}}
        if 'files' in (payload or {}):
            _colab_result_files(Path(payload['local_dir']), job_id=job_id, source_sha256=source_sha, config=config)
            return {'ok': True}
        if 'unassign' in code:
            unassign_calls.append(payload['endpoint'])
            return {'ok': True, 'missing': False}
        raise AssertionError(f'Unexpected helper: {payload}')

    monkeypatch.setattr(backend.cli, 'run_helper', fake_helper)

    backend.run(store.get(job_id), lambda: False)
    assert [p[0].rsplit('/', 1)[-1] for p in uploaded_parts] == ['part_0000.bin', 'part_0001.bin', 'part_0002.bin']
    assert b''.join(p[1] for p in uploaded_parts) == audio_bytes
    assert 'N_CHUNKS = 3' in uploaded_script_text[0]
    assert store.get(job_id)['status'] == 'COMPLETED'
    assert (tmp_path / 'outputs' / f'{job_id}.txt').is_file()
    assert (tmp_path / 'outputs' / f'{job_id}.srt').is_file()
    assert not (tmp_path / 'sources' / job_id).exists()
    # Second colab job is still QUEUED, so warm session MUST be preserved without calling unassign!
    assert unassign_calls == []
    assert backend._warm is not None and backend._warm['endpoint'] == 'ep-warm-1'

    # Now cancel queued job 2 -> no more queued colab jobs -> cleanup MUST unassign warm endpoint.
    store.update(job_id_2, status='CANCELLED', remote=json.dumps({'provider': 'colab', 'owner': 'colab.user@example.com', 'endpoint': 'ep-warm-1'}))
    backend.cleanup(job_id_2)
    assert unassign_calls == ['ep-warm-1']
    assert backend._warm is None


def test_colab_backend_resume_and_disconnect_guard(tmp_path, monkeypatch):
    store = Store(prepare_data(tmp_path))
    config = selection('en', 'colab')
    job_id = 'c' * 32
    audio_bytes = b'audio'
    source_sha = hashlib.sha256(audio_bytes).hexdigest()
    store.add(job_id, 'resume.wav', len(audio_bytes), config)
    (tmp_path / 'sources' / job_id).write_bytes(audio_bytes)
    store.update(
        job_id,
        remote=json.dumps({
            'provider': 'colab',
            'owner': 'colab.user@example.com',
            'endpoint': 'ep-existing',
            'source_sha256': source_sha,
            'submitted': True,
            'stage': 'running',
            'cleaned': False,
        }),
    )

    backend = ColabBackend(store)
    _mark_colab_installed(backend.cli)
    backend.cli.identity = lambda: 'colab.user@example.com'
    allow_create_flags = []

    def fake_helper(code, payload=None, **kw):
        if 'client.assign' in code:
            allow_create_flags.append(payload.get('allow_create'))
            return {
                'owner': 'colab.user@example.com',
                'endpoint': 'ep-existing',
                'url': 'https://proxy.example/ep-existing',
                'token': 'tok-r',
                'expires_at': '2020-01-01T00:00:00+00:00' if len(allow_create_flags) == 1 else '2099-01-01T00:00:00+00:00',
                'reused': True,
            }
        if 'client.upload' in code:
            raise AssertionError('Resume must not upload chunks again')
        if 'status_path' in (payload or {}):
            return {'ok': True, 'status': {'done': True, 'ok': True, 'duration': 2.5}}
        if 'files' in (payload or {}):
            _colab_result_files(Path(payload['local_dir']), job_id=job_id, source_sha256=source_sha, config=config)
            return {'ok': True}
        if 'unassign' in code:
            return {'ok': True}
        raise AssertionError(f'Unexpected helper: {payload}')

    monkeypatch.setattr(backend.cli, 'run_helper', fake_helper)
    backend.run(store.get(job_id), lambda: False)
    assert store.get(job_id)['status'] == 'COMPLETED'
    # Resume check used default allow_create=True, expired proxy refresh during polling forced allow_create=False.
    assert allow_create_flags == [True, False]


def test_cleanup_manager_routes_colab_and_engine_close_releases_session(colab_app, monkeypatch):
    client, engine = colab_app
    kaggle_cleanup = Mock(side_effect=AssertionError('Kaggle cleanup must not run for Colab job'))
    colab_cleanup = Mock()
    monkeypatch.setattr(engine.kaggle, 'cleanup', kaggle_cleanup)
    monkeypatch.setattr(engine.colab, 'cleanup', colab_cleanup)

    job_id = 'd' * 32
    engine.store.add(job_id, 'del.wav', 5, selection('en', 'colab'))
    engine.store.update(
        job_id,
        status='FAILED',
        remote=json.dumps({'provider': 'colab', 'owner': 'colab.user@example.com', 'endpoint': 'ep-del', 'cleaned': False}),
    )
    res = client.delete(f'/api/jobs/{job_id}')
    assert res.status_code == 200
    colab_cleanup.assert_called_once_with(job_id)
    kaggle_cleanup.assert_not_called()

    # Cancel during online Colab job preserves CANCELLED status.
    engine.store.add('e' * 32, 'cancel.wav', 5, selection('en', 'colab'))
    def stopped_run(j, stopped, guard):
        engine.store.update(j['id'], status='CANCELLED')
        raise ColabStopped()
    monkeypatch.setattr(engine.colab, 'run', stopped_run)
    engine._online_job(engine.store.get('e' * 32))
    assert engine.store.get('e' * 32)['status'] == 'CANCELLED'

    # Engine.close releases warm session BEFORE setting colab.cli.stop.
    events = []
    monkeypatch.setattr(engine.colab, 'release_session', lambda: events.append(('release', engine.colab.cli.stop.is_set())))
    orig_cli_close = engine.colab.cli.close
    monkeypatch.setattr(engine.colab.cli, 'close', lambda: (events.append(('cli_close', True)), orig_cli_close()))
    engine.close()
    assert ('release', False) in events
    assert events.index(('release', False)) < events.index(('cli_close', True))
