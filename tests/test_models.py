import hashlib
import io

from local_app.models import ModelManager
from local_app.catalog import model_config


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}


def tiny_model(monkeypatch, root):
    data = b"a small verified model"
    config = model_config('fr')
    config['files'] = {'model.bin': {'size': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'url': 'https://example.com/model.bin'}}
    return ModelManager(root, config=config), data


def test_model_resumes_partial_download_and_verifies_hash(tmp_path, monkeypatch):
    manager, data = tiny_model(monkeypatch, tmp_path)
    (manager.path / "model.bin.part").write_bytes(data[:5])
    requests = []

    def download(request, **kwargs):
        requests.append(request)
        return Response(data[5:], 206, {"Content-Range": f"bytes 5-{len(data)-1}/{len(data)}"})

    monkeypatch.setattr("urllib.request.urlopen", download)
    manager.start()
    manager.thread.join(5)
    assert manager.snapshot()["status"] == "ready"
    assert requests[0].get_header("Range") == "bytes=5-"
    assert (manager.path / "model.bin").read_bytes() == data
    assert not (manager.path / "model.bin.part").exists()


def test_download_restarts_if_server_does_not_support_range(tmp_path, monkeypatch):
    manager, data = tiny_model(monkeypatch, tmp_path)
    (manager.path / "model.bin.part").write_bytes(data[:5])
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response(data))
    manager.start()
    manager.thread.join(5)
    assert manager.snapshot()["status"] == "ready"
    assert (manager.path / "model.bin").read_bytes() == data


def test_corrupt_model_never_becomes_ready(tmp_path, monkeypatch):
    manager, data = tiny_model(monkeypatch, tmp_path)
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response(b"x" * len(data)))
    monkeypatch.setattr(manager.stop, "wait", lambda timeout: None)
    manager.start()
    manager.thread.join(5)
    assert manager.snapshot()["status"] == "error"
    assert "SHA-256" in manager.snapshot()["error"]
    assert not (manager.path / "model.bin").exists()
