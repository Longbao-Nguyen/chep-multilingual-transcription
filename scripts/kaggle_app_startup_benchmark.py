"""Measure first-log startup using Chep's real dual-notebook Kaggle setup.

Each trial creates the same private audio Dataset and two notebooks as the app.
The app-generated notebook is stopped after its normal audio/model/GPU checks,
before inference. All owned remote resources are then removed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import site
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from local_app.catalog import selection
from local_app.kaggle_merge import shard_ranges
from local_app.kaggle_notebook import notebook
from local_app.worker import decoded_duration


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def attach_only_notebook(config, job_id, source_hash, index, span, duration):
    """Keep Chep's generated notebook and stop after its verified model mount."""
    result = notebook(config, job_id, source_hash,
                      part_index=index, span=span, source_duration=duration)
    source = "".join(result["cells"][0]["source"])
    lines = source.splitlines(True)
    indexes = [i for i, line in enumerate(lines)
               if line.startswith("(root / 'run.py').write_text(")]
    if len(indexes) != 1:
        raise ValueError("Chep runner write statement changed")
    line_index = indexes[0]
    call = ast.parse(lines[line_index]).body[0].value
    runner = ast.literal_eval(call.args[0])
    anchor = "    model = matching[0]\n"
    if runner.count(anchor) != 1:
        raise ValueError("Chep mount verification changed")
    extra = (
        "    attach = {'event': 'attach_confirmed', 'job_id': JOB_ID,\n"
        "              'part_index': PART_INDEX, 'span': SPAN,\n"
        "              'source_sha256': SOURCE_SHA256,\n"
        "              'model_dataset': MODEL_DATASET, 'gpu': gpu}\n"
        "    (out / 'attach-confirmed.json').write_text(json.dumps(attach), encoding='utf-8')\n"
        "    print(json.dumps(attach), flush=True)\n"
        "    raise SystemExit(0)\n"
    )
    runner = runner.replace(anchor, anchor + extra)
    lines[line_index] = f"(root / 'run.py').write_text({runner!r}, encoding='utf-8')\n"
    result["cells"][0]["source"] = lines
    return result


def status(api, ref):
    return str(api.kernels_status(ref).status).split(".")[-1].upper()


def start_live_monitor(api, row, ui_observe=False):
    """Observe RUNNING and the first streamed log from before kernels_push."""
    stop = threading.Event()
    running = threading.Event()
    first_log = threading.Event()

    def watch_status():
        while not stop.is_set() and not running.is_set():
            try:
                current = status(api, row["ref"])
                stamp = now()
                tick = time.monotonic_ns()
                row["last_polled_status"] = current
                if current == "RUNNING":
                    row["running_observed_utc"] = stamp
                    row["running_observed_monotonic_ns"] = tick
                    running.set()
                    if ui_observe:
                        print(json.dumps({"event": "running_observed", "ref": row["ref"],
                                          "utc": stamp}), flush=True)
                    return
                if current == "QUEUED":
                    row["last_queued_observed_utc"] = stamp
                    row["last_queued_observed_monotonic_ns"] = tick
                if current in ("COMPLETE", "ERROR", "CANCELLED"):
                    return
            except Exception as exc:
                row["status_monitor_last_error"] = f"{type(exc).__name__}: {exc}"
            stop.wait(2)

    def watch_logs():
        while not stop.is_set() and not first_log.is_set():
            try:
                for event in api.kernels_logs_stream(row["ref"]):
                    data = str(event.get("data", ""))
                    if not data.strip():
                        continue
                    row["first_live_log_received_utc"] = now()
                    row["first_live_log_received_monotonic_ns"] = time.monotonic_ns()
                    row["first_live_log_event_time"] = event.get("time")
                    row["first_live_log_data"] = data[:300]
                    first_log.set()
                    if ui_observe:
                        print(json.dumps({"event": "first_live_log", "ref": row["ref"],
                                          "utc": row["first_live_log_received_utc"],
                                          "seconds_since_running_observed": round(
                                              (row["first_live_log_received_monotonic_ns"] -
                                               row["running_observed_monotonic_ns"]) / 1e9, 3)
                                          if running.is_set() else None}), flush=True)
                    return
            except Exception as exc:
                row["log_monitor_last_error"] = f"{type(exc).__name__}: {exc}"
            stop.wait(1)

    for target in (watch_status, watch_logs):
        threading.Thread(target=target, daemon=True).start()
    return stop, running, first_log


def missing(exc):
    return "404" in str(exc)


def collect(api, folder, row, trial, model_dataset):
    events = list(api.kernels_logs_stream(row["ref"]))
    logs = [{"time": event.get("time"), "stream": event.get("stream_name"),
             "data": str(event.get("data", ""))[:1500]} for event in events]
    save(folder / "logs.json", logs)
    first = next((event for event in logs
                  if event["data"].strip() and isinstance(event["time"], (int, float))), None)
    row["first_log_seconds"] = float(first["time"]) if first else None
    marker = next((event for event in logs
                   if '"event": "attach_confirmed"' in event["data"]), None)
    row["attach_log_seconds"] = float(marker["time"]) if marker else None
    output = folder / "output"
    output.mkdir(exist_ok=True)
    api.kernels_output(row["ref"], path=str(output),
                       file_pattern=r"^attach-confirmed\.json$", force=True, quiet=True)
    manifest = output / "attach-confirmed.json"
    if not manifest.is_file():
        raise ValueError("Missing attach confirmation output")
    check = json.loads(manifest.read_text(encoding="utf-8"))
    expected = {
        "event": "attach_confirmed",
        "job_id": trial["job_id"],
        "part_index": row["part_index"],
        "span": list(row["span"]),
        "source_sha256": trial["source_sha256"],
        "model_dataset": model_dataset,
    }
    if any(check.get(key) != value for key, value in expected.items()):
        raise ValueError("Attach confirmation does not match the requested job")
    if "T4" not in check.get("gpu", "") or row["first_log_seconds"] is None or marker is None:
        raise ValueError("T4 or timestamped attach log was not verified")
    row["gpu"] = check["gpu"]
    row["attach_verified"] = True


def run_trial(api, output, filename, trial_number, attempt_number, source,
              file_info, config, result, ui_observe=False):
    owner = result["account"]
    nonce = uuid.uuid4().hex[:24]
    job_id = uuid.uuid4().hex
    trial_key = f"{filename}-trial-{trial_number:02d}-attempt-{attempt_number:02d}"
    folder = output / trial_key
    folder.mkdir(parents=True)
    dataset = f"{owner}/chep-input-{nonce}"
    kernel_base = f"{owner}/chep-run-{nonce}"
    spans = shard_ranges(file_info["duration_seconds"])
    trial = {
        "file": filename, "trial": trial_number, "attempt": attempt_number, "job_id": job_id,
        "source_sha256": file_info["sha256"], "duration_seconds": file_info["duration_seconds"],
        "overlap_seconds": 5.0, "dataset": dataset, "dataset_attempted": False,
        "dataset_deleted": False, "status": "PREPARED", "started_utc": now(),
        "notebooks": [
            {"part_index": index, "span": list(spans[index]),
             "ref": kernel_base + f"-{index}", "push_attempted": False,
             "first_log_seconds": None, "attach_log_seconds": None,
             "attach_verified": False, "deleted": False, "error": None}
            for index in range(2)
        ],
    }
    result["trials"].append(trial)
    save(output / "results.json", result)
    monitors = {}
    try:
        inputs = folder / "input"
        inputs.mkdir()
        audio_link = inputs / f"{job_id}.audio"
        try:
            os.link(source, audio_link)
        except OSError:
            shutil.copyfile(source, audio_link)
        save(inputs / "dataset-metadata.json", {
            "id": dataset, "title": dataset.split("/")[1],
            "licenses": [{"name": "other"}],
        })
        trial["dataset_attempted"] = True
        trial["status"] = "UPLOADING"
        save(output / "results.json", result)
        api.dataset_create_new(str(inputs), public=False, quiet=True)
        deadline = time.monotonic() + 900
        while True:
            try:
                state = api.dataset_status(dataset).lower()
            except Exception as exc:
                # Newly created private Datasets can briefly return 403 before they are ready.
                if not any(code in str(exc) for code in ("403", "404")):
                    raise
                state = "pending"
            if state == "ready":
                break
            if state in ("error", "failed"):
                raise RuntimeError("Kaggle rejected the input Dataset")
            if time.monotonic() > deadline:
                raise TimeoutError("Input Dataset did not become ready")
            time.sleep(5)
        audio_link.unlink()
        trial["status"] = "DATASET_READY"
        save(output / "results.json", result)
        for row in trial["notebooks"]:
            index = row["part_index"]
            kernel_folder = folder / str(index)
            kernel_folder.mkdir()
            save(kernel_folder / "run.ipynb", attach_only_notebook(
                config, job_id, file_info["sha256"], index, spans[index],
                file_info["duration_seconds"]))
            save(kernel_folder / "kernel-metadata.json", {
                "id": row["ref"], "title": row["ref"].split("/")[1],
                "code_file": "run.ipynb", "language": "python", "kernel_type": "notebook",
                "is_private": True, "enable_gpu": True, "enable_internet": True,
                "machine_shape": "NvidiaTeslaT4",
                "dataset_sources": [dataset, result["model_dataset"]],
                "competition_sources": [], "kernel_sources": [], "model_sources": [],
            })
        if ui_observe:
            print(json.dumps({"event": "pair_ready", "file": filename,
                              "trial": trial_number, "refs": [row["ref"]
                              for row in trial["notebooks"]]}), flush=True)
            if input("Press Enter to submit this pair after the user is ready: ").strip():
                raise RuntimeError("Unexpected input at pre-submit gate")
        for row in trial["notebooks"]:
            index = row["part_index"]
            kernel_folder = folder / str(index)
            row["push_attempted"] = True
            row["submitted_utc"] = now()
            row["push_begin_monotonic_ns"] = time.monotonic_ns()
            monitors[row["ref"]] = start_live_monitor(api, row, ui_observe)
            save(output / "results.json", result)
            api.kernels_push(str(kernel_folder), timeout="14400", acc="NvidiaTeslaT4")
        trial["status"] = "RUNNING"
        save(output / "results.json", result)
        pending = {row["ref"] for row in trial["notebooks"]}
        deadline = time.monotonic() + 1800
        while pending:
            for row in trial["notebooks"]:
                if row["ref"] not in pending:
                    continue
                try:
                    current = status(api, row["ref"])
                    row["status_check_errors"] = 0
                except Exception as exc:
                    row["status_check_errors"] = row.get("status_check_errors", 0) + 1
                    row["status_check_last_error"] = f"{type(exc).__name__}: {exc}"
                    if row["status_check_errors"] >= 20:
                        raise RuntimeError(
                            f"Repeated status failures for {row['ref']}") from exc
                    continue
                row["remote_status"] = current
                if current == "COMPLETE":
                    monitor = monitors[row["ref"]]
                    monitor[1].wait(6)
                    monitor[2].wait(15)
                    if not monitor[1].is_set() or not monitor[2].is_set():
                        raise RuntimeError(f"Live RUNNING or first log was not observed for {row['ref']}")
                    first_tick = row["first_live_log_received_monotonic_ns"]
                    running_tick = row["running_observed_monotonic_ns"]
                    row["running_to_first_live_log_observed_seconds"] = round(
                        (first_tick - running_tick) / 1e9, 3)
                    lower_tick = row.get("last_queued_observed_monotonic_ns",
                                         row["push_begin_monotonic_ns"])
                    row["running_to_first_live_log_upper_bound_seconds"] = round(
                        (first_tick - lower_tick) / 1e9, 3)
                    collect(api, folder / str(row["part_index"]), row, trial,
                            result["model_dataset"])
                    if not ui_observe:
                        api.kernels_delete(row["ref"], no_confirm=True)
                        row["deleted"] = True
                    pending.remove(row["ref"])
                    save(output / "results.json", result)
                elif any(item in current for item in ("ERROR", "FAIL", "CANCEL")):
                    raise RuntimeError(f"{row['ref']} ended with {current}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"Trial {trial_key} did not finish within 30 minutes")
            if pending:
                time.sleep(8)
        trial["status"] = "VERIFIED"
        save(output / "results.json", result)
        if ui_observe:
            print(json.dumps({"event": "pair_verified", "file": filename,
                              "trial": trial_number, "results": [
                                  {"ref": row["ref"],
                                   "lower_seconds": row["running_to_first_live_log_observed_seconds"],
                                   "upper_seconds": row["running_to_first_live_log_upper_bound_seconds"]}
                                  for row in trial["notebooks"]]}), flush=True)
            if input("Press Enter to delete this pair after UI comparison: ").strip():
                raise RuntimeError("Unexpected input at cleanup gate")
    except Exception as exc:
        trial["status"] = "ERROR"
        trial["error"] = f"{type(exc).__name__}: {exc}"
        save(output / "results.json", result)
        raise
    finally:
        for monitor in monitors.values():
            monitor[0].set()
        (folder / "input" / f"{job_id}.audio").unlink(missing_ok=True)
        for row in trial["notebooks"]:
            if row["push_attempted"] and not row["deleted"]:
                try:
                    api.kernels_delete(row["ref"], no_confirm=True)
                    row["deleted"] = True
                except Exception as exc:
                    if missing(exc):
                        row["deleted"] = True
                    else:
                        row["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                save(output / "results.json", result)
        if trial["dataset_attempted"] and not trial["dataset_deleted"]:
            for _ in range(24):
                try:
                    api.dataset_delete(owner, dataset.split("/")[1], no_confirm=True)
                    trial["dataset_deleted"] = True
                    break
                except Exception as exc:
                    if missing(exc):
                        trial["dataset_deleted"] = True
                        break
                    if "403" in str(exc):
                        time.sleep(5)
                        continue
                    trial["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                    break
            save(output / "results.json", result)
        trial["ended_utc"] = now()
        save(output / "results.json", result)
    print(f"{filename} trial {trial_number} verified and cleaned", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rdv", type=Path, required=True)
    parser.add_argument("--b2", type=Path, required=True)
    parser.add_argument("--max-trials", type=int, default=10)
    parser.add_argument("--ui-observe", action="store_true",
                        help="Pause before each pair and after verification for live UI comparison")
    args = parser.parse_args()
    if not 1 <= args.max_trials <= 10:
        raise ValueError("max-trials must be between 1 and 10")
    site.addsitedir(str(args.cli_root))
    sys.path.insert(0, str(args.cli_root))
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    owner = api.get_config_value(api.CONFIG_NAME_USER)
    if owner != "longnj":
        raise ValueError(f"Unexpected Kaggle account: {owner}")
    config = selection("fr", "kaggle")
    model_dataset = config["model_config"]["kaggle_dataset"]
    sources = {"Rdv.m4a": args.rdv.resolve(), "b2.m4a": args.b2.resolve()}
    file_info = {
        name: {"path": str(source), "size_bytes": source.stat().st_size,
               "sha256": sha(source), "duration_seconds": decoded_duration(source)}
        for name, source in sources.items()
    }
    for name, info in file_info.items():
        shard_ranges(info["duration_seconds"])
    report = args.output / "results.json"
    if report.exists():
        result = json.loads(report.read_text(encoding="utf-8"))
        if result["account"] != owner or result["files"] != file_info or result["model_dataset"] != model_dataset:
            raise ValueError("Existing benchmark input metadata differs")
    else:
        if args.output.exists():
            raise FileExistsError("Output folder exists without a results ledger")
        args.output.mkdir(parents=True)
        result = {"account": owner, "started_utc": now(), "files": file_info,
                  "model_dataset": model_dataset, "trials": []}
        save(report, result)
    planned = [(name, number) for number in range(1, 6) for name in ("Rdv.m4a", "b2.m4a")]
    for name, number in planned[:args.max_trials]:
        previous = [trial for trial in result["trials"]
                    if trial["file"] == name and trial["trial"] == number]
        verified = [trial for trial in previous if trial["status"] == "VERIFIED"]
        if verified:
            if len(verified) == 1 and verified[0]["dataset_deleted"] and all(
                    row["deleted"] and row["attach_verified"] for row in verified[0]["notebooks"]):
                continue
            raise ValueError(f"Trial {name} {number} has incomplete verification or cleanup")
        if any(not trial["dataset_deleted"] or any(
                row["push_attempted"] and not row["deleted"] for row in trial["notebooks"])
               for trial in previous):
            raise ValueError(f"Trial {name} {number} needs manual recovery before resume")
        run_trial(api, args.output, name, number, len(previous) + 1,
                  sources[name], file_info[name], config, result, args.ui_observe)
    valid = [trial for trial in result["trials"] if trial["status"] == "VERIFIED"]
    if len(valid) == 10:
        result["completed_utc"] = now()
    save(report, result)
    print(json.dumps({"valid_trials": len(valid),
                      "valid_notebooks": sum(len(trial["notebooks"]) for trial in valid)}), flush=True)


if __name__ == "__main__":
    main()
