import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from local_app.api import create_app
from local_app.hardware import candidates
from local_app.catalog import model_config
from local_app.processes import WorkerJob
from local_app.store import Store


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("local_app.engine.detect", lambda *args: {"gpus": [], "cpu_name": "Test CPU"})
    monkeypatch.setattr("local_app.runtime.detect", lambda *args: {"gpus": [], "cpu_name": "Test CPU"})
    monkeypatch.setattr("local_app.runtime.RuntimeResolver.probe", lambda self, selected, path, profile, generation: profile)
    app = create_app(tmp_path, "test-token")
    with TestClient(app, headers={"Authorization": "Bearer test-token"}) as client:
        yield client


def test_local_api_auth_and_cross_origin(client):
    assert client.get("/api/status", headers={"Authorization": ""}).status_code == 401
    assert client.get("/api/status", headers={"Origin": "https://hostile.example"}).status_code == 403
    assert client.get("/api/status", headers={"Host": "hostile.example"}).status_code == 403
    assert client.post("/api/queue/start").status_code == 409


def test_upload_persists_cancel_only_selected_file(client):
    first = client.post("/api/jobs?name=first.wav", content=b"audio").json()
    second = client.post("/api/jobs?name=second.wav", content=b"audio2").json()
    assert client.post(f"/api/jobs/{first['id']}/cancel").status_code == 200
    payload = client.get("/api/status").json()
    assert [row["status"] for row in payload["queue"]["jobs"]] == ["CANCELLED", "QUEUED"]
    root = Path(payload["data_dir"])
    assert not (root / "sources" / first["id"]).exists()
    assert (root / "sources" / second["id"]).read_bytes() == b"audio2"
    assert len(Store(root).list()) == 2
    assert client.get(f"/api/jobs/{second['id']}/transcript").status_code == 409


def test_empty_upload_and_path_names(client):
    assert client.post("/api/jobs?name=empty.wav", content=b"").status_code == 400
    result = client.post("/api/jobs?name=..%5C..%5Caudio.wav", content=b"media").json()
    assert result["name"] == "audio.wav"
    assert len(result["id"]) == 32


def test_completed_download_delete_and_interruption(client):
    job = client.post("/api/jobs?name=le%C3%A7on.wav", content=b"media").json()
    root = Path(client.get("/api/status").json()["data_dir"])
    store = Store(root)
    store.update(job["id"], status="TRANSCRIBING")
    store.recover()
    assert store.get(job["id"])["status"] == "INTERRUPTED"
    assert client.post(f"/api/jobs/{job['id']}/retry").status_code == 200
    (root / "outputs" / f"{job['id']}.txt").write_text("Bonjour à tous", encoding="utf-8")
    store.update(job["id"], status="COMPLETED", progress=1)
    response = client.get(f"/api/jobs/{job['id']}/transcript")
    assert response.text == "Bonjour à tous"
    assert "attachment" in response.headers["content-disposition"]
    assert client.delete(f"/api/jobs/{job['id']}").status_code == 200
    assert not list((root / "outputs").iterdir())


def test_profiles_follow_model_policy_and_cpu_only_excludes_cuda():
    gpu = {"index": 0, 'name': 'Test GPU', "total_mib": 4096, "free_mib": 3900}
    config = model_config('fr')
    profiles = list(candidates(config, {'gpus':[gpu]}, 'auto'))
    assert [(p.device, p.compute_type, p.concurrency) for p in profiles][:5] == [
        ('cuda','float16',2), ('cuda','float16',1), ('cuda','int8_float16',2), ('cuda','int8_float16',1), ('cuda','int8',1)]
    assert all(p.device == 'cpu' for p in candidates(config, {'gpus':[gpu]}, 'cpu'))


@pytest.mark.skipif(os.name != "nt", reason="Windows process lifetime")
def test_worker_terminates_when_job_handle_closes():
    job = WorkerJob()
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        job.attach(process)
        assert process.poll() is None
        start = time.monotonic()
        job.close()
        process.wait(timeout=10)
        assert time.monotonic() - start < 10
    finally:
        if process.poll() is None:
            process.kill()


def test_cancel_active_pauses_and_preserves_next(client):
    from unittest.mock import Mock
    first = client.post("/api/jobs?name=a.wav", content=b"a").json()
    second = client.post("/api/jobs?name=b.wav", content=b"b").json()
    engine = client.app.state.engine
    claimed = engine.store.claim()
    assert claimed["id"] == first["id"]
    engine.active = first["id"]
    engine.state = "RUNNING"
    engine.process = Mock()
    engine.process.poll.return_value = None
    assert client.post(f"/api/jobs/{first['id']}/cancel").status_code == 200
    engine.process.terminate.assert_called_once()
    assert engine.state == "PAUSING"
    assert engine.store.get(second["id"])["status"] == "QUEUED"
    assert client.post('/api/queue/start').status_code == 409
    engine.process = None


def test_settings_cannot_change_while_queue_runs(client):
    assert client.put('/api/settings', json={"device": "cpu"}).status_code == 200
    assert client.put('/api/settings', json={"device": "fake"}).status_code == 400
    client.app.state.engine.state = "RUNNING"
    assert client.put('/api/settings', json={"device": "auto"}).status_code == 409
