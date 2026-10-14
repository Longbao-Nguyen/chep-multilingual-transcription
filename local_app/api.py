from __future__ import annotations

import asyncio
import os
import secrets
import shutil
import uuid
import threading
from contextlib import asynccontextmanager
from pathlib import Path, PureWindowsPath

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import MAX_UPLOAD, ROOT, prepare_data
from .engine import Engine
from .models import ModelRegistry
from .catalog import LANGUAGES, public_catalog
from .store import Store
from .lifecycle import Lifecycle


class Settings(BaseModel):
    device: str | None = None
    language: str | None = None

class DownloadAction(BaseModel):
    action: str
    component: str | None = None

class ExitChoice(BaseModel):
    choice: str


def create_app(data_dir: Path, token: str, model_dir: Path | None = None):
    root = prepare_data(data_dir)
    store = Store(root)
    models = ModelRegistry(root, model_dir)
    engine = Engine(store, models)
    cleanup_lock = threading.Lock()
    resources_closed = False

    def close_resources():
        nonlocal resources_closed
        with cleanup_lock:
            if not resources_closed:
                engine.close()
                models.close()
                resources_closed = True

    def shutdown_now():
        try:
            close_resources()
        finally:
            if app.state.shutdown:
                app.state.shutdown()

    lifecycle = Lifecycle(shutdown_now)

    @asynccontextmanager
    async def lifespan(app):
        engine.components.start()
        engine.start_cleanup_recovery()
        yield
        lifecycle.close()
        await asyncio.to_thread(close_resources)

    app = FastAPI(title="Chep", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.engine = engine
    app.state.models = models
    app.state.shutdown = None
    app.state.lifecycle = lifecycle

    @app.websocket('/api/ui-session')
    async def ui_session(socket: WebSocket):
        host = socket.url.hostname
        origin = socket.headers.get('origin')
        expected = 'http://' + socket.headers.get('host', '')
        if host not in ('127.0.0.1', 'localhost', 'testserver') or origin != expected:
            await socket.close(code=1008)
            return
        await socket.accept()
        client_id = None
        try:
            auth = await asyncio.wait_for(socket.receive_json(), timeout=5)
            supplied = auth.get('token', '') if isinstance(auth, dict) else ''
            if not isinstance(supplied, str) or not secrets.compare_digest(supplied, token):
                await socket.close(code=1008)
                return
            client_id = lifecycle.connect()
            await socket.send_json({'connected': True})
            while True:
                await socket.receive_text()
        except (WebSocketDisconnect, asyncio.TimeoutError, ValueError):
            pass
        finally:
            if client_id:
                lifecycle.disconnect(client_id)

    @app.post('/api/lifecycle/choose')
    def exit_choice(body: ExitChoice):
        lifecycle.choose(body.choice)
        return {'ok': True, 'choice': body.choice}

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        host = request.url.hostname
        if host not in ("127.0.0.1", "localhost", "testserver"):
            return JSONResponse({"detail": "Invalid host"}, status_code=403)
        if request.url.path.startswith("/api/"):
            origin = request.headers.get("origin")
            if origin and origin != f"{request.url.scheme}://{request.headers.get('host')}":
                return JSONResponse({"detail": "Invalid origin"}, status_code=403)
            auth = request.headers.get("authorization", "")
            if not secrets.compare_digest(auth, "Bearer " + token):
                return JSONResponse({"detail": "Open the application from its shortcut to reconnect."}, status_code=401)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; object-src 'none'; base-uri 'none'"
        return response

    @app.exception_handler(ValueError)
    async def value_error(_, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(KeyError)
    async def key_error(_, exc):
        return JSONResponse({"detail": "File not found."}, status_code=404)

    @app.get("/api/status")
    def status():
        model, runtime = engine.configuration()
        return {"queue": engine.snapshot(), "model": model, "runtime": runtime, "languages": public_catalog(), "hardware": runtime.get('hardware', engine.hardware),
                "settings": store.settings(), "data_dir": str(root), "free_bytes": shutil.disk_usage(root).free,
                "version": "0.7.0", "components": engine.components.snapshot(), "kaggle": engine.kaggle.cli.snapshot(),
                "kaggle_gpu": engine.kaggle_gpu.snapshot()}

    @app.post('/api/kaggle/gpu/check')
    def check_kaggle_gpu():
        engine.kaggle_gpu.start()
        return {'ok': True}

    @app.post('/api/kaggle/{action}')
    def kaggle_action(action: str):
        with engine.lock:
            if engine.state != 'PAUSED':
                raise ValueError('Pause the queue before changing the Kaggle connection.')
            engine.kaggle.cli.action(action)
        return {'ok': True}

    @app.post('/api/jobs/{job_id}/cleanup')
    def cleanup_kaggle(job_id: str):
        with engine.lock:
            removed = store.get(job_id, include_removed=True)
            if removed and removed['removed']:
                return engine.retry_cleanup(job_id)
            job = store.get(job_id)
            if not job:
                return {'removed': True, 'cleanup_pending': False}
            if engine.active == job_id or job['status'] not in ('COMPLETED', 'CANCELLED', 'FAILED'):
                raise ValueError('Wait for the job to stop before removing its Kaggle files.')
            engine.kaggle.cleanup(job_id)
        return {'ok': True}

    @app.post("/api/model/download")
    def download_model():
        with engine.lock:
            if engine.state != 'PAUSED':
                raise ValueError('Pause the queue before preparing a model.')
            engine.components.enqueue('model:'+store.settings()['language'])
        return {"ok": True}

    @app.post('/api/components/download')
    def download_component(body: DownloadAction):
        if body.action != 'download' or not body.component:
            raise ValueError('Choose a component to download.')
        with engine.lock:
            if body.component=='gpu:auto' and engine.state!='PAUSED':
                raise ValueError('Pause the transcription queue before changing the GPU runtime.')
            engine.components.enqueue(body.component)
        return {'ok': True}

    @app.post('/api/components/control')
    def control_downloads(body: DownloadAction):
        engine.components.control(body.action,body.component)
        return {'ok':True}

    @app.post('/api/components/recheck')
    def check_components():
        engine.components.refresh()
        return {'ok':True}

    @app.post('/api/runtime/recheck')
    def recheck():
        with engine.lock:
            if engine.state != 'PAUSED':
                raise ValueError('Pause the queue before checking the configuration.')
            engine.runtime.invalidate()
        return {'ok': True}

    @app.post("/api/jobs")
    async def upload(request: Request, name: str):
        with engine.lock:
            selected = engine.selected()
        name = PureWindowsPath(name).name.strip()[:240]
        if not name or any(ord(char) < 32 for char in name):
            raise HTTPException(400, "Invalid filename.")
        content_length = request.headers.get("content-length")
        upload_limit = 2 * 1024**3 if selected['device_preference'] == 'kaggle' else MAX_UPLOAD
        limit_message = 'Online GPU supports files up to 2 GB.' if selected['device_preference'] == 'kaggle' else 'Each file must be smaller than 20 GB.'
        if content_length and int(content_length) > upload_limit:
            raise HTTPException(413, limit_message)
        job_id = uuid.uuid4().hex
        part = root / "sources" / (job_id + ".part")
        destination = root / "sources" / job_id
        size = 0
        try:
            with part.open("wb") as stream:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > upload_limit:
                        raise HTTPException(413, limit_message)
                    if shutil.disk_usage(root).free < len(chunk) + 128 * 1024**2:
                        raise HTTPException(507, "The data drive is almost full.")
                    await asyncio.to_thread(stream.write, chunk)
            if not size:
                raise HTTPException(400, "The file is empty.")
            os.replace(part, destination)
            store.add(job_id, name, size, selected)
        except BaseException:
            part.unlink(missing_ok=True)
            destination.unlink(missing_ok=True)
            raise
        return store.get(job_id)

    @app.post("/api/queue/start")
    def start():
        engine.start()
        return {"ok": True}

    @app.post("/api/queue/pause")
    def pause():
        engine.pause()
        return {"ok": True}

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        engine.cancel(job_id)
        return {"ok": True}

    @app.post("/api/jobs/{job_id}/retry")
    def retry(job_id: str):
        engine.retry(job_id)
        return {"ok": True}

    @app.delete("/api/jobs/{job_id}")
    def delete(job_id: str):
        return engine.delete(job_id)

    @app.get("/api/jobs/{job_id}/transcript")
    def transcript(job_id: str, format: str = "txt"):
        job = store.get(job_id)
        if not job:
            raise KeyError(job_id)
        if job["status"] != "COMPLETED" or format not in ("txt", "srt"):
            raise HTTPException(409, "The transcript is not ready.")
        path = root / "outputs" / f"{job_id}.{format}"
        if not path.is_file():
            raise HTTPException(404, "The transcript was not found on disk.")
        return FileResponse(path, media_type="text/plain; charset=utf-8", filename=f"{Path(job['name']).stem}.{format}")

    @app.put("/api/settings")
    def settings(body: Settings):
        if body.device is not None and body.device not in ("auto", "cpu", "kaggle"):
            raise HTTPException(400, "Choose Local GPU, Online GPU or CPU only.")
        if body.language is not None and body.language not in LANGUAGES:
            raise HTTPException(400, 'Unsupported language.')
        with engine.lock:
            if engine.state != "PAUSED":
                raise ValueError("Pause the queue before changing the language or device.")
            previous = store.settings()
            store.set_selection(body.language or previous['language'], body.device or previous['device'])
            engine.runtime.invalidate()
            if body.device == 'auto':
                engine.components.refresh()
        return store.settings()

    @app.post("/api/storage/open")
    def open_storage():
        if os.name == "nt":
            os.startfile(root)
        return {"ok": True}

    @app.post("/api/shutdown")
    async def shutdown():
        lifecycle.choose('quit')
        return {"ok": True}

    static_dir = ROOT / "dist"
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
    return app
