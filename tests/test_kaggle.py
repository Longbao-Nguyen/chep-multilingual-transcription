import hashlib
import io
import json
import subprocess
import threading
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from local_app.api import create_app
from local_app.catalog import selection
from local_app.config import prepare_data
from local_app.kaggle_backend import KaggleBackend, kernel_status, silent_startup_overdue, validate_result
from local_app.kaggle_cli import FirstLogMonitor, KaggleCLI, KaggleStopped, safe_error
from local_app.kaggle_notebook import notebook
from local_app.store import Store


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr('local_app.engine.detect', lambda *args: {'gpus': [], 'cpu_name': 'Test CPU'})
    app = create_app(tmp_path, 'token')
    engine = app.state.engine
    engine.kaggle.cli.external = 'kaggle.exe'
    engine.kaggle.cli.state.update(status='ready', username='tester')
    with TestClient(app, headers={'Authorization': 'Bearer token'}) as client:
        yield client, engine


def test_online_selection_freezes_without_requiring_local_model(app, monkeypatch):
    client, engine = app
    monkeypatch.setattr(engine.runtime, 'ensure', Mock(side_effect=AssertionError('Local probe must not run')))
    assert client.put('/api/settings', json={'device': 'kaggle', 'language': 'en'}).status_code == 200
    status = client.get('/api/status').json()
    assert status['runtime']['status'] == 'online'
    job = client.post('/api/jobs?name=sample.wav', content=b'audio').json()
    assert json.loads(job['config'])['device_preference'] == 'kaggle'
    completed = threading.Event()
    def online(job):
        completed.set()
    monkeypatch.setattr(engine, '_online_job', online)
    assert client.post('/api/queue/start').status_code == 200
    assert completed.wait(2)


def test_cpu_jobs_never_enter_online_backend(app, monkeypatch):
    client, engine = app
    client.put('/api/settings', json={'device': 'cpu'})
    job = client.post('/api/jobs?name=local.wav', content=b'local').json()
    online = Mock(side_effect=AssertionError('No upload allowed'))
    monkeypatch.setattr(engine.kaggle, 'run', online)
    engine._job(job)
    online.assert_not_called()


def test_online_requires_connection_and_remote_limit(app):
    client, engine = app
    client.put('/api/settings', json={'device': 'kaggle'})
    job = client.post('/api/jobs?name=sample.wav', content=b'audio').json()
    engine.kaggle.cli.state['status'] = 'error'
    assert client.post('/api/queue/start').status_code == 409
    engine.kaggle.cli.state['status'] = 'ready'
    with engine.store.connect() as db:
        db.execute('UPDATE jobs SET size=? WHERE id=?', (3 * 1024**3, job['id']))
    assert client.post('/api/queue/start').status_code == 409


def test_restart_preserves_remote_identity(tmp_path):
    store = Store(prepare_data(tmp_path))
    store.add('job', 'test.wav', 1, selection('en', 'kaggle'))
    remote = json.dumps({'kernel': 'tester/chep-run-123'})
    store.update('job', status='REMOTE_RUNNING', remote=remote)
    store.recover()
    assert store.get('job')['status'] == 'QUEUED'
    assert store.get('job')['remote'] == remote


def result_files(tmp_path):
    result = {'schema_version': 1, 'job_id': 'job', 'source_sha256': 'source', 'ok': True,
              'profile': {'device': 'cuda', 'provider': 'kaggle', 'device_name': 'Tesla T4'}, 'files': {}}
    for ext in ('txt', 'srt'):
        data = b'synthetic transcript'
        (tmp_path/('job.'+ext)).write_bytes(data)
        result['files'][ext] = {'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    (tmp_path/'result.json').write_text(json.dumps(result))
    return result


def test_result_rejects_wrong_job_and_modified_transcript(tmp_path):
    result_files(tmp_path)
    assert validate_result(tmp_path, 'job', 'source')['ok']
    with pytest.raises(ValueError, match='match'):
        validate_result(tmp_path, 'another-job', 'source')
    (tmp_path/'job.txt').write_text('tampered')
    with pytest.raises(ValueError, match='integrity'):
        validate_result(tmp_path, 'job', 'source')


def test_kernel_metadata_code_is_private_input_and_pinned_model():
    data = notebook(selection('en', 'kaggle'), 'a'*32, 'b'*64)
    code = ''.join(data['cells'][0]['source'])
    compile(code, '<notebook>', 'exec')
    assert 'nunu-willump/distil-large-v3.5-ct2' in code
    assert 'source_sha256' in code
    assert 'T4' in code
    assert 'snapshot_download' not in code
    assert 'Pinned public model Dataset' in code
    assert 'auth' not in code.lower()


def test_first_log_monitor_detects_live_cli_output(monkeypatch):
    class Process:
        stdout = io.StringIO('\n{"event":"progress"}\n')
        def poll(self):
            return 0
        def communicate(self):
            return '', ''
    commands = []
    def popen(command, **kwargs):
        commands.append(command)
        assert kwargs['stderr'] == subprocess.DEVNULL
        return Process()
    monkeypatch.setattr('local_app.kaggle_cli.subprocess.Popen', popen)
    cli = Mock()
    cli.command.side_effect = lambda args: ['kaggle', *args]
    monitor = FirstLogMonitor(cli, 'tester/chep-run-' + 'a' * 24)
    try:
        assert monitor.seen.wait(2)
        assert monitor.first_log_at is not None
        assert commands == [['kaggle', 'kernels', 'logs', 'tester/chep-run-' + 'a' * 24, '--follow']]
    finally:
        monitor.close()


def test_first_log_timeout_requires_120_seconds_and_a_live_probe():
    monitor = Mock(seen=threading.Event(), started_at=100)
    assert not silent_startup_overdue(100, monitor, now=219.999)
    assert silent_startup_overdue(100, monitor, now=220)
    monitor.started_at = 215  # Allow a resumed live stream to replay old logs.
    assert not silent_startup_overdue(100, monitor, now=220)
    monitor.seen.set()
    assert not silent_startup_overdue(100, monitor, now=300)


def test_status_parser_does_not_treat_error_message_as_remote_error():
    assert kernel_status('x has status "KernelWorkerStatus.COMPLETE"') == 'COMPLETE'
    assert kernel_status('403 Client Error: Forbidden') is None
    assert kernel_status('Cannot access kernel') is None


def test_auth_errors_do_not_leak_output():
    secret = 'https://example.test/oauth?token=private-token'
    assert 'private-token' not in safe_error(secret)
    assert 'quota' in safe_error('429 Too Many Requests').lower()


def test_fresh_cli_is_ready_for_sign_in_not_a_broken_download(tmp_path, monkeypatch):
    cli = KaggleCLI(tmp_path)
    cli.external = 'kaggle.exe'
    monkeypatch.setattr(cli, 'identity', Mock(side_effect=RuntimeError('Sign in to Kaggle before starting online transcription.')))
    cli._action('check')
    assert cli.snapshot()['status'] == 'sign_in_required'
    assert cli.snapshot()['error'] is None
    assert cli.snapshot()['username'] is None


def test_missing_credentials_message_is_actionable():
    assert safe_error('Please run kaggle auth login to authenticate.') == 'Sign in to Kaggle before starting online transcription.'


def test_authenticate_requires_cli_download(tmp_path):
    cli = KaggleCLI(tmp_path)
    cli.external = None
    with pytest.raises(ValueError, match='Download Kaggle CLI first'):
        cli.action('login')
    assert cli.thread is None


def test_authenticate_awaits_browser_approval_before_ready(tmp_path, monkeypatch):
    cli = KaggleCLI(tmp_path)
    cli.external = 'kaggle.exe'
    def run(args, **kwargs):
        assert args == ['auth', 'login']
        assert kwargs == {'timeout': 600, 'check': False}
        assert cli.snapshot()['username'] is None
        assert 'Waiting for approval' in cli.snapshot()['message']
        return '', 0
    monkeypatch.setattr(cli, 'run', run)
    monkeypatch.setattr(cli, 'identity', lambda: 'new-user')
    cli._action('login')
    assert cli.snapshot()['status'] == 'ready'
    assert cli.snapshot()['username'] == 'new-user'


def test_authentication_failure_cannot_reuse_saved_username(tmp_path, monkeypatch):
    cli = KaggleCLI(tmp_path)
    cli.external = 'kaggle.exe'
    cli.state.update(status='ready', username='old-user')
    monkeypatch.setattr(cli, 'run', Mock(side_effect=RuntimeError('Authentication timed out.')))
    identity = Mock(return_value='old-user')
    monkeypatch.setattr(cli, 'identity', identity)
    cli._action('login')
    assert cli.snapshot()['status'] == 'error'
    assert cli.snapshot()['username'] is None
    identity.assert_not_called()


def test_login_nonzero_does_not_become_ready(tmp_path, monkeypatch):
    cli = KaggleCLI(tmp_path)
    cli.external = 'kaggle.exe'
    monkeypatch.setattr(cli, 'run', Mock(return_value=('OAuth callback failed', 1)))
    identity = Mock(return_value='old-user')
    monkeypatch.setattr(cli, 'identity', identity)
    cli._action('login')
    assert cli.snapshot()['status'] == 'error'
    identity.assert_not_called()


def test_already_authenticated_cli_is_accepted(tmp_path, monkeypatch):
    cli = KaggleCLI(tmp_path)
    cli.external = 'kaggle.exe'
    monkeypatch.setattr(cli, 'run', Mock(return_value=('You are already logged-in to Kaggle as [tester].', 1)))
    monkeypatch.setattr(cli, 'identity', lambda: 'tester')
    cli._action('login')
    assert cli.snapshot()['status'] == 'ready'
    assert cli.snapshot()['username'] == 'tester'


def test_resume_does_not_upload_or_push_again(tmp_path):
    store = Store(prepare_data(tmp_path))
    config = selection('en', 'kaggle')
    store.add('job', 'sample.wav', 5, config)
    (tmp_path/'sources/job').write_bytes(b'audio')
    backend = KaggleBackend(store)
    backend.cli.identity = lambda: 'tester'
    remote = {'owner': 'tester', 'dataset': 'tester/chep-input-'+'a'*24,
              'kernel': 'tester/chep-run-'+'a'*24, 'stage': 'submitted', 'source_sha256': 'source', 'cleaned': False}
    store.update('job', remote=json.dumps(remote))
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        if args[:2] == ['kernels', 'status']:
            return 'x has status "KernelWorkerStatus.COMPLETE"', 0
        if args[:2] == ['kernels', 'output']:
            directory = Path(args[args.index('-p')+1])
            result = result_files(directory)
            result.update(duration=1)
            result['profile'].update(language='en', revision=config['model_config']['revision'],
                                     **config['model_config']['sweet_spot'])
            (directory/'result.json').write_text(json.dumps(result))
        return '', 0
    backend.cli.run = run
    backend.run(store.get('job'), lambda: False)
    assert store.get('job')['status'] == 'COMPLETED'
    assert (tmp_path/'outputs/job.txt').is_file()
    assert not any('push' in call or 'create' in call for call in calls)


def test_cleanup_cannot_delete_unowned_resource(tmp_path):
    store = Store(prepare_data(tmp_path))
    store.add('job', 'sample.wav', 1, selection('en', 'kaggle'))
    store.update('job', remote=json.dumps({'owner': 'tester', 'kernel': 'someone/valuable',
        'dataset': 'tester/chep-input-'+'a'*24, 'kernel_attempted': True}))
    backend = KaggleBackend(store)
    backend.cli.identity = lambda: 'tester'
    backend.cli.run = Mock()
    with pytest.raises(ValueError):
        backend.cleanup('job')
    backend.cli.run.assert_not_called()


def test_cancel_does_not_mark_online_job_failed(app, monkeypatch):
    client, engine = app
    client.put('/api/settings', json={'device': 'kaggle'})
    job = client.post('/api/jobs?name=sample.wav', content=b'audio').json()
    def run(job, stopped, guard):
        engine.store.update(job['id'], status='CANCELLED')
        raise KaggleStopped()
    monkeypatch.setattr(engine.kaggle, 'run', run)
    engine._online_job(job)
    assert engine.store.get(job['id'])['status'] == 'CANCELLED'


def test_account_switch_cannot_resume_another_users_job(tmp_path):
    store = Store(prepare_data(tmp_path))
    store.add('job', 'sample.wav', 1, selection('en', 'kaggle'))
    (tmp_path/'sources/job').write_bytes(b'audio')
    store.update('job', remote=json.dumps({'owner': 'original-user'}))
    backend = KaggleBackend(store)
    backend.cli.identity = lambda: 'new-user'
    with pytest.raises(ValueError, match='original-user'):
        backend.run(store.get('job'), lambda: False)
