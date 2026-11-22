"""Generate Kaggle T4 notebooks for Campaign 2: Long-form / Real-world evaluation (EN, ES, VI)."""
from __future__ import annotations

from .kaggle_runner import KernelSpec, RESULTS_ROOT
from .metrics import load_catalog
from .remote_common import REMOTE_COMMON_PY

_LONGFORM_BODY = r'''
ARM_WORKER_PY = r"""
import concurrent.futures
import json
import pathlib
import sys
import time
import numpy as np

sys.path.insert(0, "/tmp/chep-eval")
from remote_common import VramMonitor, aggregate_scores, log, preload_cuda12_libs, score_texts_jiwer

spec = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
out_path = pathlib.Path(sys.argv[2])

preload_cuda12_libs()
from faster_whisper import WhisperModel
from whisper_normalizer.basic import BasicTextNormalizer
try:
    from whisper_normalizer.english import EnglishTextNormalizer
except Exception:
    EnglishTextNormalizer = None

basic_norm = BasicTextNormalizer()
en_norm = EnglishTextNormalizer() if (spec["language"] == "en" and EnglishTextNormalizer is not None) else None
items = json.loads(pathlib.Path(spec["manifest_path"]).read_text(encoding="utf-8"))

result = {
    "language": spec["language"],
    "dataset": spec["dataset"],
    "profile": spec["profile"],
    "chunk_seconds": spec["chunk_seconds"],
    "is_production": spec["is_production"],
    "n_recordings": len(items),
    "total_audio_s": sum(it["duration_s"] for it in items),
    "status": "error",
}

with VramMonitor() as mon:
    try:
        t_load0 = time.perf_counter()
        model = WhisperModel(
            spec["model_dir"],
            device="cuda",
            compute_type=spec["profile"]["compute_type"],
            num_workers=spec["profile"]["concurrency"],
            cpu_threads=4,
            device_index=0,
            local_files_only=True,
        )
        actual_compute_type = model.model.compute_type
        load_s = time.perf_counter() - t_load0

        # Warmup matching local_app/worker.py (fixed seeded noise of chunk_seconds)
        chunk_samples = int(spec["chunk_seconds"] * 16000)
        rng = np.random.default_rng(17)
        warm = rng.normal(0.0, 0.025, chunk_samples).astype(np.float32)
        def _warm(_idx):
            segs, _ = model.transcribe(warm, language=spec["language"], task="transcribe", **spec["transcribe"])
            return list(segs)
        with concurrent.futures.ThreadPoolExecutor(max_workers=spec["profile"]["concurrency"]) as pool:
            list(pool.map(_warm, range(spec["profile"]["concurrency"])))

        def _infer(chunk):
            segs, _ = model.transcribe(chunk, language=spec["language"], task="transcribe", **spec["transcribe"])
            return " ".join(s.text.strip() for s in segs if s.text.strip())

        workers = spec["profile"]["concurrency"]
        per_rec = []
        total_inf_s = 0.0
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for r_idx, item in enumerate(items, start=1):
                arr = np.load(item["npy_path"])
                chunks = [arr[i : i + chunk_samples] for i in range(0, len(arr), chunk_samples)]
                t0 = time.perf_counter()
                texts = []
                for b0 in range(0, len(chunks), workers):
                    batch = chunks[b0 : b0 + workers]
                    texts.extend(t for t in pool.map(_infer, batch) if t)
                rec_inf_s = time.perf_counter() - t0
                total_inf_s += rec_inf_s
                hyp = " ".join(texts)
                sc = score_texts_jiwer(item["reference"], hyp, basic_norm, en_norm=en_norm)
                rec_entry = {
                    "id": item["id"],
                    "duration_s": item["duration_s"],
                    "n_chunks": len(chunks),
                    "inference_s": rec_inf_s,
                    "rtf": rec_inf_s / item["duration_s"] if item["duration_s"] else 0.0,
                    "word_subs": sc["word_subs"],
                    "word_dels": sc["word_dels"],
                    "word_ins": sc["word_ins"],
                    "word_hits": sc["word_hits"],
                    "word_edits": sc["word_edits"],
                    "ref_words": sc["ref_words"],
                    "char_edits": sc["char_edits"],
                    "ref_chars": sc["ref_chars"],
                }
                if en_norm is not None:
                    rec_entry.update({
                        "en_word_edits": sc["en_word_edits"],
                        "en_ref_words": sc["en_ref_words"],
                        "en_char_edits": sc["en_char_edits"],
                        "en_ref_chars": sc["en_ref_chars"],
                    })
                per_rec.append(rec_entry)
                if r_idx % 5 == 0 or r_idx == len(items):
                    log(f"  [{spec['language']} {actual_compute_type}x{workers}] {r_idx}/{len(items)} recordings ({total_inf_s:.1f}s)")

        agg = aggregate_scores(per_rec)
        total_audio_s = sum(it["duration_s"] for it in items)
        result.update({
            "status": "ok",
            "actual_compute_type": actual_compute_type,
            "load_s": load_s,
            "inference_s": total_inf_s,
            "rtf": total_inf_s / total_audio_s if total_audio_s else 0.0,
            "aggregate": agg,
            "per_recording": per_rec,
        })
        if en_norm is not None:
            tot_ew = sum(x["en_word_edits"] for x in per_rec)
            tot_rw = sum(x["en_ref_words"] for x in per_rec)
            tot_ec = sum(x["en_char_edits"] for x in per_rec)
            tot_rc = sum(x["en_ref_chars"] for x in per_rec)
            result["aggregate_english_normalizer"] = {
                "wer_pct": (tot_ew / tot_rw * 100.0) if tot_rw else 0.0,
                "cer_pct": (tot_ec / tot_rc * 100.0) if tot_rc else 0.0,
                "ref_words": tot_rw,
                "ref_chars": tot_rc,
            }
    except Exception as exc:
        result["error"] = str(exc)
        import traceback
        result["traceback"] = traceback.format_exc()

result["peak_vram_mib"] = mon.peak_vram_mib
result["peak_ram_mib"] = mon.peak_ram_mib
out_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
"""


def prepare_tedlium_en(work_dir):
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq
    import numpy as np

    ds = "distil-whisper/tedlium-long-form"
    files = sorted(f for f in HfApi().list_repo_files(ds, repo_type="dataset") if f.startswith("data/test-") and f.endswith(".parquet"))
    dl_dir = TMP / "dl_en"
    dl_dir.mkdir(parents=True, exist_ok=True)
    parquet_hashes = {}
    items = []
    idx = 0
    for rf in files:
        lp = pathlib.Path(hf_hub_download(ds, rf, repo_type="dataset", cache_dir=str(dl_dir)))
        parquet_hashes[rf] = sha256_file(lp)
        table = pq.read_table(lp)
        for row in table.to_pylist():
            raw = row["audio"]["bytes"]
            arr = decode_audio_bytes_16k(raw)
            npy_path = work_dir / f"en_{idx:03d}.npy"
            np.save(npy_path, arr)
            dur = len(arr) / 16000.0
            items.append({
                "id": row.get("speaker_id") or f"talk_{idx}",
                "npy_path": str(npy_path),
                "duration_s": dur,
                "reference": row["text"],
            })
            idx += 1
    shutil.rmtree(dl_dir, ignore_errors=True)
    return {
        "dataset": ds,
        "split": "test",
        "grouping": "native continuous TED talks (unsegmented)",
        "parquet_sha256": parquet_hashes,
        "n_recordings": len(items),
        "total_audio_s": sum(x["duration_s"] for x in items),
        "durations_s": [round(x["duration_s"], 2) for x in items],
    }, items


def prepare_mtedx_grouped(lang, work_dir):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    import numpy as np

    ds = "deepdml/mtedx"
    rf = f"{lang}/test-00000-of-00001.parquet"
    dl_dir = TMP / f"dl_{lang}"
    dl_dir.mkdir(parents=True, exist_ok=True)
    lp = pathlib.Path(hf_hub_download(ds, rf, repo_type="dataset", cache_dir=str(dl_dir)))
    pq_sha = sha256_file(lp)
    table = pq.read_table(lp)

    talks = {}
    total_segs = 0
    for row in table.to_pylist():
        total_segs += 1
        talks.setdefault(row["talk_id"], []).append(row)

    items = []
    for t_idx, (talk_id, segs) in enumerate(sorted(talks.items())):
        segs.sort(key=lambda r: (int(r["segment_id"]), float(r["start"])))
        parts = []
        texts = []
        for s_i, r in enumerate(segs):
            if s_i > 0:
                gap = float(r["start"]) - float(segs[s_i - 1]["end"])
                sil_s = min(max(gap, 0.2), 1.5)
                parts.append(np.zeros(int(round(sil_s * 16000)), dtype=np.float32))
            parts.append(decode_audio_bytes_16k(r["audio"]["bytes"]))
            texts.append(r["transcript"].strip())
        arr = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        npy_path = work_dir / f"{lang}_{t_idx:03d}.npy"
        np.save(npy_path, arr)
        items.append({
            "id": talk_id,
            "npy_path": str(npy_path),
            "duration_s": len(arr) / 16000.0,
            "n_segments": len(segs),
            "reference": " ".join(t for t in texts if t),
        })
    shutil.rmtree(dl_dir, ignore_errors=True)
    durs = sorted(x["duration_s"] for x in items)
    return {
        "dataset": f"{ds} ({lang}/test)",
        "split": "test",
        "grouping": "talk_id chronological reconstruction with clipped natural gaps [0.2s, 1.5s]",
        "parquet_sha256": {rf: pq_sha},
        "total_segments": total_segs,
        "n_recordings": len(items),
        "total_audio_s": sum(durs),
        "duration_min_med_max_s": [round(durs[0], 2), round(durs[len(durs)//2], 2), round(durs[-1], 2)],
    }, items


def _trailing_int(text: str) -> int:
    import re
    m = re.search(r"(\d+)$", text.strip())
    return int(m.group(1)) if m else 0


def prepare_vietmed_grouped(work_dir):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    import numpy as np

    ds = "leduckhai/VietMed"
    rf = "data/test-00000-of-00001.parquet"
    dl_dir = TMP / "dl_vi"
    dl_dir.mkdir(parents=True, exist_ok=True)
    lp = pathlib.Path(hf_hub_download(ds, rf, repo_type="dataset", cache_dir=str(dl_dir)))
    pq_sha = sha256_file(lp)
    table = pq.read_table(lp)

    sessions = {}
    total_segs = 0
    for row in table.to_pylist():
        total_segs += 1
        key = row.get("speaker_name") or row.get("audio_name") or "unknown"
        sessions.setdefault(key, []).append(row)

    sil = np.zeros(int(0.5 * 16000), dtype=np.float32)
    items = []
    for s_idx, (sess_id, segs) in enumerate(sorted(sessions.items())):
        segs.sort(key=lambda r: (_trailing_int(r.get("seq_name", "")), r.get("utterance_id", "")))
        parts = []
        texts = []
        for k, r in enumerate(segs):
            if k > 0:
                parts.append(sil)
            parts.append(decode_audio_bytes_16k(r["audio"]["bytes"]))
            texts.append(r["text"].strip())
        arr = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        npy_path = work_dir / f"vi_{s_idx:03d}.npy"
        np.save(npy_path, arr)
        items.append({
            "id": sess_id,
            "npy_path": str(npy_path),
            "duration_s": len(arr) / 16000.0,
            "n_segments": len(segs),
            "reference": " ".join(t for t in texts if t),
        })
    shutil.rmtree(dl_dir, ignore_errors=True)
    durs = sorted(x["duration_s"] for x in items)
    return {
        "dataset": ds,
        "split": "test",
        "grouping": "speaker_name chronological session stitching (0.5s silence separators)",
        "parquet_sha256": {rf: pq_sha},
        "total_segments": total_segs,
        "n_recordings": len(items),
        "total_audio_s": sum(durs),
        "duration_min_med_max_s": [round(durs[0], 2), round(durs[len(durs)//2], 2), round(durs[-1], 2)],
    }, items


def main():
    import pathlib
    import shutil
    import subprocess
    import sys

    (TMP / "remote_common.py").write_text(REMOTE_COMMON_SPEC, encoding="utf-8")
    (TMP / "arm_worker.py").write_text(ARM_WORKER_PY, encoding="utf-8")

    log(f"[chep-eval] Starting long-form/domain wave={WAVE_NAME} langs={LANGS}")
    gpu_info = ensure_kaggle_t4()

    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "faster-whisper==1.2.1", "ctranslate2==4.8.2",
        "jiwer==3.1.0", "whisper-normalizer==0.1.12",
        "soundfile", "pyarrow", "psutil", "huggingface_hub",
    ], check=True)

    report = {
        "campaign": "longform_multilingual",
        "wave": WAVE_NAME,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": gpu_info,
        "versions": installed_versions(["faster-whisper", "ctranslate2", "jiwer", "whisper-normalizer", "soundfile", "av", "huggingface_hub"]),
        "languages": {},
    }

    for code in LANGS:
        cfg = CATALOG[code]
        model_dir = discover_pinned_ct2_model(cfg["files"])
        work_dir = TMP / f"audio_{code}"
        shutil.rmtree(work_dir, ignore_errors=True)
        work_dir.mkdir(parents=True, exist_ok=True)

        log(f"Preparing {code} corpus...")
        if code == "en":
            meta, items = prepare_tedlium_en(work_dir)
        elif code == "es":
            meta, items = prepare_mtedx_grouped(code, work_dir)
        elif code == "vi":
            meta, items = prepare_vietmed_grouped(work_dir)
        else:
            raise ValueError(f"Unsupported lang in nb_longform: {code}")

        manifest_path = TMP / f"manifest_{code}_long.json"
        manifest_path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
        log(f"Prepared {code}: {meta['n_recordings']} recordings, {meta['total_audio_s']:.1f}s")

        lang_out = {
            **meta,
            "model": cfg["model"],
            "ct2_model": cfg["ct2_model"],
            "kaggle_dataset": cfg["kaggle_dataset"],
            "revision": cfg["revision"],
            "chunk_seconds": cfg["chunk_seconds"],
            "transcribe": cfg["transcribe"],
            "profiles": [],
        }

        sweet = cfg["sweet_spot"]
        other_compute = "int8_float16" if sweet["compute_type"] == "float16" else "float16"
        candidate_profiles = [
            (sweet["compute_type"], sweet["concurrency"], True),
            (other_compute, sweet["concurrency"], False),
        ]

        for compute_type, concurrency, is_prod in candidate_profiles:
            spec_file = TMP / f"spec_{code}_{compute_type}.json"
            out_file = TMP / f"out_{code}_{compute_type}.json"
            out_file.unlink(missing_ok=True)
            spec_file.write_text(json.dumps({
                "language": code,
                "dataset": meta["dataset"],
                "model_dir": str(model_dir),
                "manifest_path": str(manifest_path),
                "chunk_seconds": cfg["chunk_seconds"],
                "transcribe": cfg["transcribe"],
                "profile": {"device": "cuda", "compute_type": compute_type, "concurrency": concurrency},
                "is_production": is_prod,
            }), encoding="utf-8")

            proc = subprocess.run([sys.executable, str(TMP / "arm_worker.py"), str(spec_file), str(out_file)], capture_output=False)
            arm_res = json.loads(out_file.read_text(encoding="utf-8")) if out_file.is_file() else {
                "status": "error", "profile": {"compute_type": compute_type, "concurrency": concurrency}, "returncode": proc.returncode
            }
            lang_out["profiles"].append(arm_res)
            log(f"Finished {code} {compute_type}x{concurrency}: status={arm_res.get('status')} wer={arm_res.get('aggregate', {}).get('wer_pct')}")

        report["languages"][code] = lang_out
        (WORK / "result.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        shutil.rmtree(work_dir, ignore_errors=True)

    log("[chep-eval] Completed long-form/domain wave successfully.")


if __name__ == "__main__":
    main()
'''


def build_longform_script(wave_name: str, langs: list[str]) -> str:
    catalog = load_catalog()["languages"]
    subset = {k: catalog[k] for k in langs}
    header = (
        f"WAVE_NAME = {wave_name!r}\n"
        f"LANGS = {langs!r}\n"
        f"CATALOG = {subset!r}\n"
        f"REMOTE_COMMON_SPEC = {REMOTE_COMMON_PY!r}\n"
    )
    return REMOTE_COMMON_PY + "\n" + header + _LONGFORM_BODY


def build_longform_wave_specs() -> list[KernelSpec]:
    catalog = load_catalog()["languages"]
    return [
        KernelSpec(
            wave="long-envi",
            code=build_longform_script("long-envi", ["en", "vi"]),
            dataset_sources=[catalog["en"]["kaggle_dataset"], catalog["vi"]["kaggle_dataset"]],
            out_dir=RESULTS_ROOT / "long-envi",
        ),
        KernelSpec(
            wave="long-es",
            code=build_longform_script("long-es", ["es"]),
            dataset_sources=[catalog["es"]["kaggle_dataset"]],
            out_dir=RESULTS_ROOT / "long-es",
        ),
    ]
