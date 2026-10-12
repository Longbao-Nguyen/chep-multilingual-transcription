"""Entry point for both the Windows shortcut and source development."""
from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

from local_app.config import atomic_json, default_data_dir, prepare_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--model-dir", type=Path, help="Development only: use an existing pinned model")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-auto-download", action="store_true", help="Compatibility option; downloads are always manual")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument('--stop', action='store_true', help='Stop the running application for this data folder')
    args = parser.parse_args()
    root = prepare_data(args.data_dir)
    if args.stop:
        import urllib.request
        instance_path = root / 'instance.json'
        if instance_path.is_file():
            instance = json.loads(instance_path.read_text(encoding='utf-8'))
            url, token = instance['url'].split('#token=', 1)
            request = urllib.request.Request(url + 'api/shutdown', method='POST', headers={'Authorization':'Bearer '+token})
            with urllib.request.urlopen(request, timeout=60):
                pass
            # Wait until the server releases DLLs and the application lock.
            until = time.monotonic() + 120
            while instance_path.exists() and time.monotonic() < until:
                time.sleep(.2)
        return
    lock_file = (root / "app.lock").open("a+b")
    lock_file.seek(0)
    if os.name == "nt":
        import msvcrt
        try:
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            # Another instance is initializing, or already serves this data directory.
            for _ in range(30):
                try:
                    details = json.loads((root / "instance.json").read_text())
                    if not args.no_browser:
                        webbrowser.open(details["url"])
                    return
                except (OSError, ValueError):
                    time.sleep(0.2)
            raise RuntimeError("The application is starting. Please try again in a few seconds.")
    log_stream = (root / "logs" / "app.log").open("a", encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = log_stream
    if sys.stderr is None:
        sys.stderr = log_stream
    logging.basicConfig(stream=log_stream, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import uvicorn
    from local_app.api import create_app
    token = secrets.token_urlsafe(32)
    app = create_app(root, token, args.model_dir)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", args.port))
    except OSError:
        sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}/#token={token}"
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, access_log=False, log_config=None))
    app.state.shutdown = lambda: setattr(server, "should_exit", True)

    def open_when_ready():
        while not server.started and not server.should_exit:
            time.sleep(0.1)
        if server.started:
            atomic_json(root / "instance.json", {"url": url, "pid": os.getpid()})
            if not args.no_browser:
                webbrowser.open(url)

    threading.Thread(target=open_when_ready, daemon=True).start()
    try:
        server.run(sockets=[sock])
    finally:
        app.state.lifecycle.close()
        app.state.engine.close()
        app.state.models.close()
        (root / "instance.json").unlink(missing_ok=True)
        sock.close()
        lock_file.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if os.name == "nt" and "--no-browser" not in sys.argv:
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, str(error), "Chep", 0x10)
        raise
