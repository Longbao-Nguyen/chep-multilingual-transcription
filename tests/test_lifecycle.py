import threading
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from local_app.api import create_app
from local_app.lifecycle import Lifecycle


def test_last_tab_only_and_quit_once():
    stopped = threading.Event()
    prompt = Mock(return_value='quit')
    lifecycle = Lifecycle(stopped.set, prompt, delay=.01)
    a, b = lifecycle.connect(), lifecycle.connect()
    lifecycle.disconnect(a)
    assert not stopped.wait(.04)
    prompt.assert_not_called()
    lifecycle.disconnect(b)
    assert stopped.wait(1)
    lifecycle.choose('quit')
    prompt.assert_called_once()
    lifecycle.close()


def test_refresh_cancels_pending_prompt():
    prompt = Mock(return_value='quit')
    lifecycle = Lifecycle(Mock(), prompt, delay=.1)
    lifecycle.disconnect(lifecycle.connect())
    lifecycle.connect()
    assert not threading.Event().wait(.15)
    prompt.assert_not_called()
    lifecycle.close()


def test_background_choice_keeps_process_and_reopening_rearms_prompt():
    asked = threading.Event()
    def prompt(cancelled):
        asked.set()
        return 'background'
    shutdown = Mock()
    lifecycle = Lifecycle(shutdown, prompt, delay=.01)
    client = lifecycle.connect()
    lifecycle.choose('background')
    lifecycle.disconnect(client)
    assert not asked.wait(.05)
    new = lifecycle.connect()
    lifecycle.disconnect(new)
    assert asked.wait(1)
    shutdown.assert_not_called()
    lifecycle.close()


def test_reconnect_while_native_prompt_is_open_cancels_stale_quit():
    opened, released, done = threading.Event(), threading.Event(), threading.Event()
    def prompt(cancelled):
        opened.set()
        released.wait(1)
        assert cancelled.is_set()
        done.set()
        return 'quit'
    shutdown = Mock()
    lifecycle = Lifecycle(shutdown, prompt, delay=.01)
    lifecycle.disconnect(lifecycle.connect())
    assert opened.wait(1)
    lifecycle.connect()
    released.set()
    assert done.wait(1)
    shutdown.assert_not_called()
    lifecycle.close()


def test_no_connected_browser_does_not_schedule_shutdown():
    shutdown = Mock()
    lifecycle = Lifecycle(shutdown, delay=.01)
    assert lifecycle.timer is None
    with pytest.raises(ValueError):
        lifecycle.choose('invalid')
    shutdown.assert_not_called()
    lifecycle.close()


def test_websocket_requires_same_origin_and_token(tmp_path):
    app = create_app(tmp_path, 'token')
    with TestClient(app) as client:
        for origin in ('http://hostile.example', None):
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect('/api/ui-session', headers={'origin': origin} if origin else {}) as socket:
                    socket.send_json({'token': 'token'})
        with client.websocket_connect('/api/ui-session', headers={'origin': 'http://testserver'}) as socket:
            socket.send_json({'token': 'wrong'})
            with pytest.raises(WebSocketDisconnect):
                socket.receive_json()
        with client.websocket_connect('/api/ui-session', headers={'origin': 'http://testserver'}) as socket:
            socket.send_json({'token': 'token'})
            assert socket.receive_json() == {'connected': True}
            assert len(app.state.lifecycle.clients) == 1
            assert client.post('/api/lifecycle/choose', json={'choice': 'background'}).status_code == 401
            assert client.post('/api/lifecycle/choose', headers={'Authorization': 'Bearer token'}, json={'choice': 'background'}).status_code == 200
        assert not app.state.lifecycle.clients
