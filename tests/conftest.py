import pytest

@pytest.fixture(autouse=True)
def isolate_optional_hardware(monkeypatch):
    monkeypatch.setattr('local_app.components.discover',lambda:{'compatible':False,'provider':None,
        'reason':'No compatible GPU was detected. Using CPU.','eligible_indices':[],'adapters':[]})
