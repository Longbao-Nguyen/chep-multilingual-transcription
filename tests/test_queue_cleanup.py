import json
import sqlite3
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from local_app.api import create_app
from local_app.catalog import selection
from local_app.config import prepare_data
from local_app.kaggle_backend import KaggleBackend, cleanup_response
from local_app.store import Store


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr('local_app.engine.detect', lambda *args: {'gpus': [], 'cpu_name': 'Test CPU'})
    application = create_app(tmp_path, 'token')
    engine = application.state.engine
    engine.kaggle.cli.external = 'kaggle.exe'
    engine.kaggle.cli.state.update(status='ready', username='tester')
    engine.kaggle.cli.identity = lambda: 'tester'
    engine.kaggle.cli.run = Mock(return_value=('', 0))
    with TestClient(application, headers={'Authorization': 'Bearer token'}) as client:
        yield client, engine, tmp_path


def add(app, name='audio.wav', status='QUEUED', remote=False):
    client, engine, root = app
    job = client.post('/api/jobs?name=' + name, content=b'audio').json()
    engine.store.update(job['id'], status=status)
    if remote:
        record = {'owner': 'tester', 'kernel': 'tester/chep-run-' + 'a' * 24,
                  'dataset': 'tester/chep-input-' + 'a' * 24,
                  'kernel_attempted': True, 'dataset_attempted': True, 'cleaned': False}
        engine.store.update(job['id'], remote=json.dumps(record))
    return job['id']


@pytest.mark.parametrize('status', ['QUEUED', 'FAILED', 'CANCELLED', 'COMPLETED'])
def test_remove_job_states_and_refresh(app, status):
    client, engine, root = app
    job_id = add(app, status=status)
    response = client.delete('/api/jobs/' + job_id)
    assert response.status_code == 200
    assert response.json() == {'removed': True, 'cleanup_pending': False, 'errors': {}}
    assert engine.store.get(job_id) is None
    assert not (root / 'sources' / job_id).exists()
    assert job_id not in [job['id'] for job in client.get('/api/status').json()['queue']['jobs']]
    assert client.delete('/api/jobs/' + job_id).json() == {'removed': True, 'cleanup_pending': False}


@pytest.mark.parametrize('missing', ['kernels', 'datasets', 'both'])
def test_missing_kaggle_resources_are_idempotent(app, missing):
    client, engine, root = app
    job_id = add(app, status='FAILED', remote=True)
    def run(args, **kwargs):
        if args[0] == missing or missing == 'both':
            return ('404 Notebook not found' if args[0] == 'kernels' else '404 Dataset not found'), 1
        return '', 0
    engine.kaggle.cli.run = Mock(side_effect=run)
    assert client.delete('/api/jobs/' + job_id).json()['cleanup_pending'] is False
    assert engine.store.pending_cleanup() == []
    assert engine.kaggle.cli.run.call_count == 2


def test_missing_local_and_remote_resources_do_not_block_removal(app):
    client, engine, root = app
    job_id = add(app, status='FAILED', remote=True)
    (root / 'sources' / job_id).unlink()
    engine.kaggle.cli.run = Mock(return_value=('404 Notebook not found', 1))
    response = client.delete('/api/jobs/' + job_id)
    assert response.status_code == 200
    assert response.json()['cleanup_pending'] is False
    assert engine.store.get(job_id) is None


def test_all_local_resources_missing_is_safe_to_repeat(app):
    client, engine, root = app
    job_id = add(app, status='FAILED')
    (root / 'sources' / job_id).unlink()
    assert client.delete('/api/jobs/' + job_id).status_code == 200
    assert client.delete('/api/jobs/' + job_id).json()['removed'] is True


@pytest.mark.parametrize('output,expected', [
    ('401 Unauthorized', 'error'), ('403 Forbidden', 'error'),
    ('404 Notebook not found', 'missing'), ('429 Too Many Requests', 'error'),
    ('503 Server Error', 'error'), ('Credentials file not found', 'error')])
def test_cleanup_response_classification(output, expected):
    assert cleanup_response(output, 'kernels', 'tester/chep-run-' + 'a' * 24) == expected


@pytest.mark.parametrize('output,kind', [('401 Unauthorized', 'authentication'), ('403 Forbidden', 'permission'),
    ('429 Too Many Requests', 'rate_limit'), ('503 Server Error', 'server')])
def test_remote_failure_leaves_hidden_recoverable_job(app, output, kind):
    client, engine, root = app
    job_id = add(app, status='FAILED', remote=True)
    engine.kaggle.cli.run = Mock(return_value=(output, 1))
    response = client.delete('/api/jobs/' + job_id)
    assert response.status_code == 200
    assert response.json()['cleanup_pending'] is True
    assert engine.store.get(job_id) is None
    assert engine.store.get(job_id, include_removed=True)['removed'] == 1
    assert json.loads(engine.store.get(job_id, include_removed=True)['remote'])['cleanup_kind'] == kind
    assert job_id in [row['id'] for row in client.get('/api/status').json()['queue']['cleanup_pending']]
    assert client.delete('/api/jobs/' + job_id).json()['cleanup_pending'] is True


def test_timeout_preserves_cleanup_and_retry(app):
    client, engine, root = app
    job_id = add(app, status='FAILED', remote=True)
    engine.kaggle.cli.run = Mock(side_effect=RuntimeError('Kaggle request timed out.'))
    assert client.delete('/api/jobs/' + job_id).json()['cleanup_pending']
    assert json.loads(engine.store.get(job_id, include_removed=True)['remote'])['cleanup_kind'] == 'timeout'
    engine.kaggle.cli.run = Mock(return_value=('', 0))
    assert client.post('/api/jobs/' + job_id + '/cleanup').json()['cleanup_pending'] is False
    assert engine.store.pending_cleanup() == []


def test_local_permission_failure_is_tracked_without_losing_remote_cleanup(app, monkeypatch):
    client, engine, root = app
    job_id = add(app, status='FAILED', remote=True)
    real_source = engine._source
    denied = Mock()
    denied.unlink.side_effect = PermissionError('access denied')
    monkeypatch.setattr(engine, '_source', lambda _: denied)
    result = client.delete('/api/jobs/' + job_id).json()
    assert result['cleanup_pending'] is True
    assert 'source' in result['errors']
    assert json.loads(engine.store.get(job_id, include_removed=True)['remote'])['cleaned'] is True
    monkeypatch.setattr(engine, '_source', real_source)
    assert client.post('/api/jobs/' + job_id + '/cleanup').json()['cleanup_pending'] is False


def test_removing_one_job_does_not_change_others(app):
    client, engine, root = app
    first, second = add(app), add(app, 'other.wav')
    client.delete('/api/jobs/' + first)
    assert engine.store.get(second)['status'] == 'QUEUED'
    assert (root / 'sources' / second).exists()


def test_active_job_must_be_cancelled_and_worker_stopped(app):
    client, engine, root = app
    job_id = add(app, status='LOADING')
    engine.active = job_id
    assert client.delete('/api/jobs/' + job_id).status_code == 409
    assert client.post('/api/jobs/' + job_id + '/cancel').status_code == 200
    assert engine.state == 'PAUSING'
    assert client.delete('/api/jobs/' + job_id).status_code == 409
    engine.active = None
    assert client.delete('/api/jobs/' + job_id).status_code == 200


def test_restart_preserves_hidden_pending_cleanup(tmp_path):
    store = Store(prepare_data(tmp_path))
    store.add('job', 'audio.wav', 1, selection('en', 'kaggle'))
    store.update('job', status='FAILED', remote=json.dumps({'owner': 'tester', 'kernel': 'tester/chep-run-' + 'a' * 24,
        'kernel_attempted': True, 'cleaned': False}))
    store.mark_removed('job')
    reopened = Store(tmp_path)
    reopened.recover()
    assert reopened.get('job') is None
    assert reopened.pending_cleanup()[0]['id'] == 'job'
    assert reopened.claim() is None


def test_startup_recovery_retries_saved_cleanup(app):
    client, engine, root = app
    job_id = add(app, status='FAILED', remote=True)
    engine.store.mark_removed(job_id)
    engine.start_cleanup_recovery()
    engine.cleanup_thread.join(timeout=3)
    assert engine.store.pending_cleanup() == []


def test_corrupt_remote_metadata_does_not_trap_queue_entry(app):
    client, engine, root = app
    job_id = add(app, status='FAILED')
    engine.store.update(job_id, remote='{invalid json')
    response = client.delete('/api/jobs/' + job_id)
    assert response.status_code == 200
    assert response.json()['cleanup_pending'] is True
    assert engine.store.get(job_id) is None
    assert client.get('/api/status').status_code == 200


def test_existing_database_migrates_without_losing_job(tmp_path):
    with sqlite3.connect(tmp_path / 'queue.sqlite3') as db:
        db.execute("CREATE TABLE jobs (id TEXT PRIMARY KEY, name TEXT NOT NULL, size INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'QUEUED', progress REAL NOT NULL DEFAULT 0, duration REAL, processed REAL NOT NULL DEFAULT 0, error TEXT, created REAL NOT NULL, updated REAL NOT NULL, profile TEXT, config TEXT, attempts TEXT, remote TEXT)")
        db.execute("INSERT INTO jobs (id,name,size,created,updated) VALUES ('legacy','old.wav',1,0,0)")
    store = Store(prepare_data(tmp_path))
    assert store.get('legacy')['removed'] == 0
    assert store.get('legacy')['cleanup'] is None
    assert store.get('legacy')['name'] == 'old.wav'


def test_partial_cleanup_retries_only_remaining_resource(app):
    client, engine, root = app
    job_id = add(app, status='FAILED', remote=True)
    calls = []
    def first(args, **kwargs):
        calls.append(args[0])
        return ('503 Server Error', 1) if args[0] == 'datasets' else ('', 0)
    engine.kaggle.cli.run = Mock(side_effect=first)
    assert client.delete('/api/jobs/' + job_id).json()['cleanup_pending']
    remote = json.loads(engine.store.get(job_id, include_removed=True)['remote'])
    assert remote['kernel_cleaned'] is True
    assert remote.get('dataset_cleaned') is not True
    calls.clear()
    engine.kaggle.cli.run = Mock(side_effect=lambda args, **kw: (calls.append(args[0]) or '', 0))
    assert client.post('/api/jobs/' + job_id + '/cleanup').json()['cleanup_pending'] is False
    assert calls == ['datasets']


def test_cleanup_after_credentials_restored(app):
    client, engine, root = app
    job_id = add(app, status='FAILED', remote=True)
    engine.kaggle.cli.identity = lambda: 'other-user'
    assert client.delete('/api/jobs/' + job_id).json()['cleanup_pending']
    assert engine.store.get(job_id) is None
    engine.kaggle.cli.identity = lambda: 'tester'
    assert client.post('/api/jobs/' + job_id + '/cleanup').json()['cleanup_pending'] is False
    assert engine.store.pending_cleanup() == []
