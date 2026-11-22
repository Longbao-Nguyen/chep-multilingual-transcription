"""Local Kaggle CLI orchestrator with strict ownership ledger & silent-startup recovery."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OWNER = os.environ.get("CHEP_KAGGLE_OWNER", "longnj")
KAGGLE_EXE = os.environ.get("KAGGLE_CLI") or shutil.which("kaggle") or "kaggle"
RESULTS_ROOT = Path("test-results/eval-2026-11")
LEDGER_PATH = RESULTS_ROOT / "ownership.json"
SLUG_RE = re.compile(rf"^{re.escape(OWNER)}/chep-eval-[a-z0-9-]+$")
STATUS_RE = re.compile(r'has status "(?:KernelWorkerStatus\.)?([A-Z_]+)"')
FLAGS = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


class LiveFirstLogWatcher:
    """Attach to Kaggle's --follow SSE log stream to detect real live startup."""

    def __init__(self, ref: str) -> None:
        self.ref = ref
        self.seen = threading.Event()
        self.stop = threading.Event()
        self.started_at = time.time()
        self.first_line: str | None = None
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        env = _utf8_env()
        while not self.stop.is_set() and not self.seen.is_set():
            proc: subprocess.Popen[str] | None = None
            try:
                proc = subprocess.Popen(
                    [KAGGLE_EXE, "kernels", "logs", self.ref, "--follow"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                    creationflags=FLAGS,
                )
                with self._lock:
                    self._proc = proc
                assert proc.stdout is not None
                for line in proc.stdout:
                    if self.stop.is_set():
                        break
                    s = line.strip()
                    if not s or s.startswith("Warning: Looks like you're using an outdated `kaggle` version"):
                        continue
                    self.first_line = s
                    self.seen.set()
                    break
            except OSError:
                pass
            finally:
                if proc is not None:
                    if proc.poll() is None:
                        try:
                            proc.kill()
                        except OSError:
                            pass
                    try:
                        proc.communicate(timeout=2)
                    except Exception:
                        pass
                    with self._lock:
                        if self._proc is proc:
                            self._proc = None
            if not self.seen.is_set():
                self.stop.wait(2.0)

    def close(self) -> None:
        self.stop.set()
        with self._lock:
            proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass
        self._thread.join(timeout=3.0)


@dataclass
class KernelSpec:
    wave: str
    code: str
    dataset_sources: list[str]
    out_dir: Path


def _utf8_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run_kaggle(
    args: list[str],
    *,
    timeout: int = 180,
    check: bool = True,
    retries: int = 3,
) -> tuple[str, int]:
    last_out = ""
    last_code = 1
    for attempt in range(retries):
        proc = subprocess.run(
            [KAGGLE_EXE, *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=_utf8_env(),
        )
        last_out, last_code = proc.stdout, proc.returncode
        if last_code == 0 or attempt == retries - 1:
            break
        lower = last_out.lower()
        if any(tok in lower for tok in ("429", "timeout", "temporarily", "connection", "502", "503", "504")):
            time.sleep(5 * (attempt + 1))
            continue
        break
    if check and last_code != 0:
        raise RuntimeError(f"kaggle {' '.join(args)} failed (code={last_code}):\n{last_out}")
    return last_out, last_code


def _load_ledger() -> dict[str, Any]:
    if LEDGER_PATH.is_file():
        return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    return {"owner": OWNER, "kernels": []}


def _save_ledger(data: dict[str, Any]) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = LEDGER_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, LEDGER_PATH)


def record_owned_kernel(ref: str, wave: str) -> None:
    if not SLUG_RE.fullmatch(ref):
        raise ValueError(f"Refusing to record non-campaign kernel ref: {ref}")
    ledger = _load_ledger()
    if not any(item["ref"] == ref for item in ledger["kernels"]):
        ledger["kernels"].append(
            {
                "ref": ref,
                "wave": wave,
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "deleted": False,
                "status": "pending",
            }
        )
    _save_ledger(ledger)


def update_owned_kernel(ref: str, **updates: Any) -> None:
    ledger = _load_ledger()
    for item in ledger["kernels"]:
        if item["ref"] == ref:
            item.update(updates)
            break
    _save_ledger(ledger)


def make_slug(wave: str) -> str:
    clean_wave = re.sub(r"[^a-z0-9-]+", "-", wave.lower()).strip("-")
    return f"chep-eval-{clean_wave}-{uuid.uuid4().hex[:8]}"


def write_notebook_bundle(bundle_dir: Path, slug: str, script_source: str, dataset_sources: list[str]) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bootstrap = (
        "import os, pathlib, subprocess, sys\n"
        "os.environ['PYTHONUNBUFFERED'] = '1'\n"
        "driver = pathlib.Path('/tmp/chep_eval_driver.py')\n"
        f"driver.write_text({script_source!r}, encoding='utf-8')\n"
        "proc = subprocess.Popen(\n"
        "    [sys.executable, '-u', str(driver)],\n"
        "    stdout=subprocess.PIPE,\n"
        "    stderr=subprocess.STDOUT,\n"
        "    text=True,\n"
        "    encoding='utf-8',\n"
        "    errors='replace',\n"
        "    bufsize=1,\n"
        ")\n"
        "assert proc.stdout is not None\n"
        "for line in proc.stdout:\n"
        "    print(line, end='', flush=True)\n"
        "rc = proc.wait()\n"
        "if rc != 0:\n"
        "    raise subprocess.CalledProcessError(rc, proc.args)\n"
    )
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
        "cells": [
            {
                "id": "chep-eval-cell",
                "cell_type": "code",
                "metadata": {},
                "source": bootstrap.splitlines(True),
                "outputs": [],
                "execution_count": None,
            }
        ],
    }
    (bundle_dir / "notebook.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = {
        "id": f"{OWNER}/{slug}",
        "title": slug,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": dataset_sources,
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    (bundle_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def parse_kernel_status(output: str) -> str | None:
    m = STATUS_RE.search(output)
    return m.group(1) if m else None


def _has_real_log_output(log_text: str) -> bool:
    for line in log_text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("Warning: Looks like you're using an outdated `kaggle` version"):
            continue
        return True
    return False


def delete_owned_kernel(ref: str) -> bool:
    if not SLUG_RE.fullmatch(ref):
        raise ValueError(f"Refusing to delete non-campaign kernel ref: {ref}")
    out, code = run_kaggle(["kernels", "delete", ref, "-y"], timeout=90, check=False)
    lower = out.lower()
    if code == 0 or "not found" in lower or "404" in lower:
        update_owned_kernel(ref, deleted=True, deleted_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        return True
    print(f"[cleanup warning] Failed deleting {ref}: {out.strip()}", flush=True)
    return False


def cleanup_ledger(only_refs: list[str] | None = None) -> dict[str, Any]:
    ledger = _load_ledger()
    results = {}
    for item in ledger["kernels"]:
        ref = item["ref"]
        if item.get("deleted"):
            continue
        if only_refs is not None and ref not in only_refs:
            continue
        ok = delete_owned_kernel(ref)
        results[ref] = ok
    return results


def submit_and_wait_wave(
    specs: list[KernelSpec],
    *,
    poll_interval_s: int = 20,
    max_wait_s: int = 12600,
    queued_timeout_s: int = 420,
    max_queued_retries: int = 2,
    silent_timeout_s: int = 180,
    max_silent_retries: int = 2,
    sequential: bool = False,
) -> dict[str, dict[str, Any]]:
    """Submit Kaggle T4 kernels, monitor, download outputs, and clean up.

    When sequential=True, runs one KernelSpec at a time to avoid dual-queue deadlocks
    on single-GPU waves. Use sequential=False when real 2-GPU concurrency is required.
    """
    if not specs:
        return {}
    if sequential and len(specs) > 1:
        merged: dict[str, dict[str, Any]] = {}
        for sp in specs:
            merged.update(
                submit_and_wait_wave(
                    [sp],
                    poll_interval_s=poll_interval_s,
                    max_wait_s=max_wait_s,
                    queued_timeout_s=queued_timeout_s,
                    max_queued_retries=max_queued_retries,
                    silent_timeout_s=silent_timeout_s,
                    max_silent_retries=max_silent_retries,
                    sequential=False,
                )
            )
        return merged

    states: dict[str, dict[str, Any]] = {}
    bundles_root = RESULTS_ROOT / "bundles"
    bundles_root.mkdir(parents=True, exist_ok=True)

    def _submit_one(spec: KernelSpec, tag: str, silent_retries: int = 0, queued_retries: int = 0) -> dict[str, Any]:
        slug = make_slug(spec.wave if tag == "" else f"{spec.wave}-{tag}")
        ref = f"{OWNER}/{slug}"
        bdir = bundles_root / slug
        write_notebook_bundle(bdir, slug, spec.code, spec.dataset_sources)
        record_owned_kernel(ref, spec.wave)
        t_push = time.time()
        out, _ = run_kaggle(
            ["kernels", "push", "-p", str(bdir), "--accelerator", "NvidiaTeslaT4", "-t", "14400"],
            timeout=180,
        )
        print(f"[submitted] {ref} -> {out.strip().splitlines()[-1] if out.strip() else 'ok'}", flush=True)
        update_owned_kernel(ref, status="SUBMITTED")
        return {
            "spec": spec,
            "slug": slug,
            "ref": ref,
            "bundle_dir": bdir,
            "submitted_at": t_push,
            "running_since": None,
            "first_log_seen": False,
            "log_watcher": None,
            "silent_retries": silent_retries,
            "queued_retries": queued_retries,
            "last_queue_log": t_push,
            "status": "SUBMITTED",
            "complete": False,
            "downloaded": False,
        }

    for idx, sp in enumerate(specs):
        if idx > 0:
            time.sleep(8)
        st = _submit_one(sp, "")
        states[sp.wave] = st

    submitted_refs: list[str] = [st["ref"] for st in states.values()]
    start_wall = time.time()
    try:
        while True:
            pending = [st for st in states.values() if not st["complete"]]
            if not pending:
                break
            if time.time() - start_wall > max_wait_s:
                raise TimeoutError(f"Wave timed out after {max_wait_s}s; pending={[s['ref'] for s in pending]}")

            for st in list(pending):
                ref = st["ref"]
                out, _ = run_kaggle(["kernels", "status", ref], timeout=90, check=False)
                kstatus = parse_kernel_status(out) or "UNKNOWN"
                st["status"] = kstatus
                update_owned_kernel(ref, status=kstatus)
                now = time.time()

                if kstatus == "COMPLETE":
                    if st.get("log_watcher") is not None:
                        st["log_watcher"].close()
                        st["log_watcher"] = None
                    st["complete"] = True
                    st["completed_at"] = now
                    print(f"[complete] {ref} (wall={now - st['submitted_at']:.1f}s)", flush=True)
                    dest = st["spec"].out_dir
                    if dest.exists():
                        shutil.rmtree(dest)
                    dest.mkdir(parents=True, exist_ok=True)
                    dl_out, _ = run_kaggle(["kernels", "output", ref, "-p", str(dest), "--force"], timeout=600)
                    st["downloaded"] = True
                    st["download_log"] = dl_out.strip()
                    print(f"[downloaded] {ref} -> {dest}", flush=True)
                    delete_owned_kernel(ref)
                elif any(tok in kstatus for tok in ("ERROR", "CANCEL", "FAILED")):
                    if st.get("log_watcher") is not None:
                        st["log_watcher"].close()
                        st["log_watcher"] = None
                    log_out, _ = run_kaggle(["kernels", "logs", ref], timeout=90, check=False)
                    dest = st["spec"].out_dir
                    dest.mkdir(parents=True, exist_ok=True)
                    (dest / f"{st['slug']}.error.log").write_text(log_out, encoding="utf-8")
                    run_kaggle(["kernels", "output", ref, "-p", str(dest), "--force"], timeout=120, check=False)
                    raise RuntimeError(f"Kernel {ref} entered terminal failure status {kstatus}. Saved log under {dest}.")
                elif kstatus == "RUNNING":
                    if st["running_since"] is None:
                        st["running_since"] = now
                        print(f"[running] {ref} (queue wait={now - st['submitted_at']:.1f}s)", flush=True)
                    if not st["first_log_seen"]:
                        watcher = st.get("log_watcher")
                        if watcher is None:
                            watcher = LiveFirstLogWatcher(ref)
                            st["log_watcher"] = watcher
                        if watcher.seen.is_set():
                            st["first_log_seen"] = True
                            first_line = watcher.first_line or ""
                            watcher.close()
                            st["log_watcher"] = None
                            print(f"[live log] {ref}: {first_line[:120]}", flush=True)
                        elif now - st["running_since"] >= silent_timeout_s and now - watcher.started_at >= 15.0:
                            watcher.close()
                            st["log_watcher"] = None
                            if st["silent_retries"] >= max_silent_retries:
                                raise RuntimeError(
                                    f"Kernel {ref} remained RUNNING with zero live SSE logs for >{silent_timeout_s}s after {max_silent_retries} retries."
                                )
                            print(f"[silent restart] {ref} had 0 live SSE logs for {now - st['running_since']:.0f}s; deleting & retrying", flush=True)
                            delete_owned_kernel(ref)
                            new_st = _submit_one(
                                st["spec"],
                                f"r{st['silent_retries'] + 1}",
                                silent_retries=st["silent_retries"] + 1,
                                queued_retries=st["queued_retries"],
                            )
                            states[st["spec"].wave] = new_st
                            submitted_refs.append(new_st["ref"])
                    elif now - st["last_queue_log"] >= 180:
                        st["last_queue_log"] = now
                        print(f"[poll] {ref} status=RUNNING elapsed={now - st['submitted_at']:.0f}s", flush=True)
                else:
                    q_elapsed = now - st["submitted_at"]
                    if q_elapsed >= queued_timeout_s:
                        if st["queued_retries"] >= max_queued_retries:
                            raise RuntimeError(
                                f"Kernel {ref} stayed {kstatus} for >{queued_timeout_s}s after {max_queued_retries} retries."
                            )
                        print(f"[queue restart] {ref} stuck in {kstatus} for {q_elapsed:.0f}s; deleting & retrying", flush=True)
                        delete_owned_kernel(ref)
                        new_st = _submit_one(
                            st["spec"],
                            f"q{st['queued_retries'] + 1}",
                            silent_retries=st["silent_retries"],
                            queued_retries=st["queued_retries"] + 1,
                        )
                        states[st["spec"].wave] = new_st
                        submitted_refs.append(new_st["ref"])
                    elif now - st["last_queue_log"] >= 120:
                        st["last_queue_log"] = now
                        print(f"[poll] {ref} status={kstatus} elapsed={q_elapsed:.0f}s", flush=True)

            if any(not st["complete"] for st in states.values()):
                time.sleep(poll_interval_s)
    finally:
        for st in states.values():
            if st.get("log_watcher") is not None:
                st["log_watcher"].close()
        cleanup_ledger(only_refs=submitted_refs)

    return {
        wave: {
            "ref": st["ref"],
            "submitted_at": st["submitted_at"],
            "completed_at": st.get("completed_at"),
            "out_dir": str(st["spec"].out_dir),
            "silent_retries": st["silent_retries"],
        }
        for wave, st in states.items()
    }
