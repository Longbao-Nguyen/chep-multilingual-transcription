"""Shared pure-Python helper code embedded into Kaggle evaluation notebooks."""
from __future__ import annotations

REMOTE_COMMON_PY = r'''
import concurrent.futures
import hashlib
import importlib.metadata
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time
import traceback
import unicodedata

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

WORK = pathlib.Path("/kaggle/working")
TMP = pathlib.Path("/tmp/chep-eval")
TMP.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[{ts}] {msg}", flush=True)


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def ensure_kaggle_t4() -> dict:
    if not pathlib.Path("/kaggle/input").exists():
        raise SystemExit("Refusing to run outside Kaggle (/kaggle/input missing).")
    gpu_name = subprocess.check_output(
        ["nvidia-smi", "--id=0", "--query-gpu=name", "--format=csv,noheader"],
        text=True,
    ).strip()
    if "T4" not in gpu_name:
        raise RuntimeError(f"Expected Nvidia Tesla T4 GPU, got: {gpu_name}")
    full_smi = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"],
        text=True,
    ).strip()
    log(f"Verified GPU: {full_smi}")
    return {"gpu_0": gpu_name, "nvidia_smi_header": full_smi}


def preload_cuda12_libs() -> list[str]:
    """Ensure CTranslate2 can find CUDA 12 cuBLAS/cuDNN on CUDA 13 Kaggle images."""
    import ctypes
    import site
    import sysconfig

    roots = {sysconfig.get_paths().get("purelib", ""), *site.getsitepackages()}
    nv_dirs = [pathlib.Path(r) / "nvidia" for r in roots if r and (pathlib.Path(r) / "nvidia").is_dir()]

    def _find(pattern: str) -> list[pathlib.Path]:
        hits = []
        for nd in nv_dirs:
            hits.extend(sorted(nd.rglob(pattern)))
        return hits

    if not _find("libcublas.so.12*"):
        log("libcublas.so.12 not found in site-packages; installing nvidia-cublas-cu12 nvidia-cudnn-cu12...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nvidia-cublas-cu12", "nvidia-cudnn-cu12"], check=True)
        nv_dirs = [pathlib.Path(r) / "nvidia" for r in roots if r and (pathlib.Path(r) / "nvidia").is_dir()]

    lib_dirs = sorted({str(p.parent) for nd in nv_dirs for p in nd.rglob("*.so*")})
    if lib_dirs:
        cur_ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(lib_dirs + ([cur_ld] if cur_ld else []))

    loaded = []
    for pattern in ("libcublasLt.so.12*", "libcublas.so.12*", "libcudnn.so.9*"):
        for cand in _find(pattern):
            try:
                ctypes.CDLL(str(cand), mode=ctypes.RTLD_GLOBAL)
                loaded.append(str(cand))
                break
            except OSError:
                continue
    return loaded


def discover_pinned_ct2_model(files_meta: dict) -> pathlib.Path:
    matches = []
    for candidate_bin in pathlib.Path("/kaggle/input").rglob("model.bin"):
        directory = candidate_bin.parent
        ok = True
        for name, meta in files_meta.items():
            target = directory / name
            if not target.is_file() or target.stat().st_size != meta["size"]:
                ok = False
                break
            if sha256_file(target) != meta["sha256"]:
                ok = False
                break
        if ok:
            matches.append(directory)
    if len(matches) != 1:
        raise RuntimeError(f"Expected 1 verified model directory under /kaggle/input, found {len(matches)}: {matches}")
    return matches[0]


def decode_audio_bytes_16k(raw_bytes: bytes) -> "np.ndarray":
    import av
    import numpy as np
    import soundfile as sf

    try:
        data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1).astype(np.float32)
        if sr == 16000:
            return np.ascontiguousarray(data, dtype=np.float32)
    except Exception:
        pass

    # Fallback / resampling through PyAV AudioResampler (identical to local_app/worker.py)
    chunks = []
    with av.open(io.BytesIO(raw_bytes)) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)
        for frame in container.decode(stream):
            for converted in resampler.resample(frame):
                arr = converted.to_ndarray()
                chunks.append(arr.reshape(-1))
        for converted in resampler.resample(None):
            arr = converted.to_ndarray()
            chunks.append(arr.reshape(-1))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)


def installed_versions(pkgs: list[str]) -> dict[str, str]:
    out = {}
    for p in pkgs:
        try:
            out[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            out[p] = "missing"
    return out


class VramMonitor:
    def __init__(self, interval_s: float = 0.15):
        self.interval_s = interval_s
        self.peak_vram_mib = 0
        self.peak_ram_mib = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        import psutil
        proc = psutil.Process()
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--id=0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    text=True,
                ).strip()
                used = int(out.splitlines()[0])
                if used > self.peak_vram_mib:
                    self.peak_vram_mib = used
            except Exception:
                pass
            try:
                rss = proc.memory_info().rss / (1024.0 * 1024.0)
                if rss > self.peak_ram_mib:
                    self.peak_ram_mib = rss
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_args):
        self._stop.set()
        self._thread.join(timeout=3.0)


def score_texts_jiwer(ref_raw: str, hyp_raw: str, basic_norm, en_norm=None) -> dict:
    import jiwer

    r_nfc = unicodedata.normalize("NFC", ref_raw).replace("\r\n", " ").replace("\n", " ")
    h_nfc = unicodedata.normalize("NFC", hyp_raw).replace("\r\n", " ").replace("\n", " ")
    ref_norm = " ".join(basic_norm(r_nfc).split())
    hyp_norm = " ".join(basic_norm(h_nfc).split())

    if ref_norm:
        w = jiwer.process_words(ref_norm, hyp_norm)
        c = jiwer.process_characters(ref_norm, hyp_norm)
        w_edits = int(w.substitutions + w.deletions + w.insertions)
        c_edits = int(c.substitutions + c.deletions + c.insertions)
        res = {
            "ref_norm": ref_norm,
            "hyp_norm": hyp_norm,
            "word_subs": int(w.substitutions),
            "word_dels": int(w.deletions),
            "word_ins": int(w.insertions),
            "word_hits": int(w.hits),
            "word_edits": w_edits,
            "ref_words": int(w.hits + w.substitutions + w.deletions),
            "hyp_words": len(hyp_norm.split()),
            "char_edits": c_edits,
            "ref_chars": int(c.hits + c.substitutions + c.deletions),
        }
    else:
        res = {
            "ref_norm": "",
            "hyp_norm": hyp_norm,
            "word_subs": 0,
            "word_dels": 0,
            "word_ins": len(hyp_norm.split()),
            "word_hits": 0,
            "word_edits": len(hyp_norm.split()),
            "ref_words": 0,
            "hyp_words": len(hyp_norm.split()),
            "char_edits": len(hyp_norm),
            "ref_chars": 0,
        }

    if en_norm is not None:
        r_en = " ".join(en_norm(r_nfc).split())
        h_en = " ".join(en_norm(h_nfc).split())
        if r_en:
            we = jiwer.process_words(r_en, h_en)
            ce = jiwer.process_characters(r_en, h_en)
            res["en_word_edits"] = int(we.substitutions + we.deletions + we.insertions)
            res["en_ref_words"] = int(we.hits + we.substitutions + we.deletions)
            res["en_char_edits"] = int(ce.substitutions + ce.deletions + ce.insertions)
            res["en_ref_chars"] = int(ce.hits + ce.substitutions + ce.deletions)
        else:
            res["en_word_edits"] = len(h_en.split())
            res["en_ref_words"] = 0
            res["en_char_edits"] = len(h_en)
            res["en_ref_chars"] = 0
    return res


def aggregate_scores(items: list[dict], prefix: str = "") -> dict:
    subs = sum(it.get(f"{prefix}word_subs", 0) for it in items)
    dels = sum(it.get(f"{prefix}word_dels", 0) for it in items)
    ins = sum(it.get(f"{prefix}word_ins", 0) for it in items)
    hits = sum(it.get(f"{prefix}word_hits", 0) for it in items)
    w_edits = sum(it[f"{prefix}word_edits"] for it in items)
    w_ref = sum(it[f"{prefix}ref_words"] for it in items)
    c_edits = sum(it[f"{prefix}char_edits"] for it in items)
    c_ref = sum(it[f"{prefix}ref_chars"] for it in items)
    return {
        "word_subs": subs,
        "word_dels": dels,
        "word_ins": ins,
        "word_hits": hits,
        "word_edits": w_edits,
        "ref_words": w_ref,
        "wer_pct": (w_edits / w_ref * 100.0) if w_ref else 0.0,
        "del_rate_pct": (dels / w_ref * 100.0) if w_ref else 0.0,
        "sub_rate_pct": (subs / w_ref * 100.0) if w_ref else 0.0,
        "ins_rate_pct": (ins / w_ref * 100.0) if w_ref else 0.0,
        "char_edits": c_edits,
        "ref_chars": c_ref,
        "cer_pct": (c_edits / c_ref * 100.0) if c_ref else 0.0,
    }
'''
