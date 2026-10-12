from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "Chep"
MAX_UPLOAD = 20 * 1024**3


def default_data_dir() -> Path:
    return Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "FrenchTranscriptLocal" / "data"


def prepare_data(path: Path) -> Path:
    path = path.expanduser().resolve()
    for name in ("sources", "outputs", "work", "models", "logs"):
        (path / name).mkdir(parents=True, exist_ok=True)
    return path


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


_dll_handles = []


def setup_dlls() -> None:
    """Use only this application's NVIDIA runtime; never alter machine PATH."""
    if os.name != "nt":
        return
    runtime = os.environ.get('CHEP_GPU_RUNTIME')
    if not runtime:
        return
    candidates = [Path(runtime)]
    paths = []
    for base in candidates:
        for suffix in ("nvidia/cublas/bin", "nvidia/cudnn/bin", "nvidia/cuda_runtime/bin", "nvidia/cuda_nvrtc/bin"):
            location = base / suffix
            if location.is_dir():
                paths.append(str(location))
                _dll_handles.append(os.add_dll_directory(str(location)))
    if paths:
        os.environ["PATH"] = os.pathsep.join(paths + [os.environ.get("PATH", "")])
