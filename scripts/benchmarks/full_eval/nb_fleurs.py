"""Generate Kaggle T4 notebooks for Campaign 1: Full FLEURS test evaluation (all 4 languages)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .kaggle_runner import KernelSpec, RESULTS_ROOT
from .metrics import load_catalog
from .remote_common import REMOTE_COMMON_PY

LOCALES = {
    "fr": "fr_fr",
    "en": "en_us",
    "es": "es_419",
    "vi": "vi_vn",
}

_FLEURS_BODY = r'''
ARM_WORKER_PY = r"""
import concurrent.futures
import json
import pathlib
import sys
import time
import wave
import numpy as np

sys.path.insert(0, "/tmp/chep-eval")
from remote_common import VramMonitor, aggregate_scores, decode_audio_bytes_16k, log, preload_cuda12_libs, score_texts_jiwer

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

rows = json.loads(pathlib.Path(spec["manifest_path"]).read_text(encoding="utf-8"))

# Preload audio into RAM as float32 16 kHz once so inference_s measures pure GPU inference, not tar/disk I/O
arrays = []
total_audio_s = 0.0
for row in rows:
    arr = decode_audio_bytes_16k(pathlib.Path(row["wav_path"]).read_bytes())
    arrays.append(arr)
    total_audio_s += len(arr) / 16000.0

result = {
    "language": spec["language"],
    "locale": spec["locale"],
    "profile": spec["profile"],
    "is_production": spec["is_production"],
    "n_utterances": len(rows),
    "total_audio_s": total_audio_s,
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

        # Warmup all workers cleanly (same seed style as local_app/worker.py)
        rng = np.random.default_rng(17)
        warm = rng.normal(0.0, 0.02, 5 * 16000).astype(np.float32)
        def _warm(_idx):
            segs, _ = model.transcribe(warm, language=spec["language"], task="transcribe", **spec["transcribe"])
            return list(segs)
        with concurrent.futures.ThreadPoolExecutor(max_workers=spec["profile"]["concurrency"]) as pool:
            list(pool.map(_warm, range(spec["profile"]["concurrency"])))

        def _infer(idx_and_arr):
            idx, arr = idx_and_arr
            if len(arr) == 0:
                return idx, ""
            segs, _ = model.transcribe(arr, language=spec["language"], task="transcribe", **spec["transcribe"])
            return idx, " ".join(s.text.strip() for s in segs if s.text.strip())

        t0 = time.perf_counter()
        hyps = [""] * len(arrays)
        workers = spec["profile"]["concurrency"]
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for done_count, (idx, hyp) in enumerate(pool.map(_infer, enumerate(arrays)), start=1):
                hyps[idx] = hyp
                if done_count % 200 == 0 or done_count == len(arrays):
                    log(f"  [{spec['language']} {actual_compute_type}x{workers}] {done_count}/{len(arrays)} utterances")
        inference_s = time.perf_counter() - t0

        per_item = []
        per_item_col3 = []
        for row, hyp in zip(rows, hyps):
            sc = score_texts_jiwer(row["raw_transcription"], hyp, basic_norm, en_norm=en_norm)
            per_item.append({
                "id": row["id"],
                "file": row["file_name"],
                "dur_s": row["dur_s"],
                "word_subs": sc["word_subs"],
                "word_dels": sc["word_dels"],
                "word_ins": sc["word_ins"],
                "word_hits": sc["word_hits"],
                "word_edits": sc["word_edits"],
                "ref_words": sc["ref_words"],
                "char_edits": sc["char_edits"],
                "ref_chars": sc["ref_chars"],
                **({
                    "en_word_edits": sc["en_word_edits"],
                    "en_ref_words": sc["en_ref_words"],
                    "en_char_edits": sc["en_char_edits"],
                    "en_ref_chars": sc["en_ref_chars"],
                } if en_norm is not None else {}),
            })
            if row.get("norm_col3"):
                sc3 = score_texts_jiwer(row["norm_col3"], hyp, basic_norm)
                per_item_col3.append(sc3)

        agg = aggregate_scores(per_item)
        result.update({
            "status": "ok",
            "actual_compute_type": actual_compute_type,
            "load_s": load_s,
            "inference_s": inference_s,
            "rtf": inference_s / total_audio_s if total_audio_s else 0.0,
            "aggregate": agg,
            "per_utterance": per_item,
        })
        if en_norm is not None:
            tot_ew = sum(x["en_word_edits"] for x in per_item)
            tot_rw = sum(x["en_ref_words"] for x in per_item)
            tot_ec = sum(x["en_char_edits"] for x in per_item)
            tot_rc = sum(x["en_ref_chars"] for x in per_item)
            result["aggregate_english_normalizer"] = {
                "wer_pct": (tot_ew / tot_rw * 100.0) if tot_rw else 0.0,
                "cer_pct": (tot_ec / tot_rc * 100.0) if tot_rc else 0.0,
                "ref_words": tot_rw,
                "ref_chars": tot_rc,
            }
        if per_item_col3:
            agg3 = aggregate_scores(per_item_col3)
            result["aggregate_fleurs_col3"] = {
                "wer_pct": agg3["wer_pct"],
                "cer_pct": agg3["cer_pct"],
                "ref_words": agg3["ref_words"],
            }
    except Exception as exc:
        result["error"] = str(exc)
        import traceback
        result["traceback"] = traceback.format_exc()

result["peak_vram_mib"] = mon.peak_vram_mib
result["peak_ram_mib"] = mon.peak_ram_mib
out_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
"""

def main():
    import csv
    import pathlib
    import shutil
    import subprocess
    import sys
    import tarfile
    import wave

    (TMP / "remote_common.py").write_text(REMOTE_COMMON_SPEC, encoding="utf-8")
    (TMP / "arm_worker.py").write_text(ARM_WORKER_PY, encoding="utf-8")

    log(f"[chep-eval] Starting FLEURS full-test benchmark wave={WAVE_NAME} langs={LANGS}")
    gpu_info = ensure_kaggle_t4()

    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "faster-whisper==1.2.1", "ctranslate2==4.8.2",
        "jiwer==3.1.0", "whisper-normalizer==0.1.12",
        "soundfile", "pyarrow", "psutil", "huggingface_hub",
    ], check=True)

    from huggingface_hub import hf_hub_download

    report = {
        "campaign": "fleurs_full_test",
        "wave": WAVE_NAME,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": gpu_info,
        "versions": installed_versions(["faster-whisper", "ctranslate2", "jiwer", "whisper-normalizer", "soundfile", "av", "huggingface_hub"]),
        "languages": {},
    }

    for code in LANGS:
        cfg = CATALOG[code]
        locale = LOCALES[code]
        log(f"Preparing {code} ({locale}) model + FLEURS test...")
        model_dir = discover_pinned_ct2_model(cfg["files"])

        fleurs_cache = TMP / "fleurs_dl" / locale
        fleurs_cache.mkdir(parents=True, exist_ok=True)
        tsv_path = pathlib.Path(hf_hub_download("google/fleurs", f"data/{locale}/test.tsv", repo_type="dataset", cache_dir=str(fleurs_cache)))
        tar_path = pathlib.Path(hf_hub_download("google/fleurs", f"data/{locale}/audio/test.tar.gz", repo_type="dataset", cache_dir=str(fleurs_cache)))

        extract_dir = TMP / "fleurs_wavs" / locale
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(extract_dir)
        wav_map = {p.name: p for p in extract_dir.rglob("*.wav")}

        import soundfile as sf

        rows = []
        with tsv_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for r in reader:
                if not r:
                    continue
                utt_id, fname, raw_txt, norm_txt = r[0], r[1], r[2], r[3]
                wp = wav_map.get(fname)
                if wp is None:
                    raise RuntimeError(f"Missing wav for {locale} row {utt_id}: {fname}")
                try:
                    dur_s = float(sf.info(str(wp)).duration)
                except Exception:
                    dur_s = len(decode_audio_bytes_16k(wp.read_bytes())) / 16000.0
                rows.append({
                    "id": utt_id,
                    "file_name": fname,
                    "wav_path": str(wp),
                    "raw_transcription": raw_txt,
                    "norm_col3": norm_txt,
                    "dur_s": dur_s,
                })

        manifest_path = TMP / f"manifest_{code}.json"
        manifest_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        log(f"Prepared {code} ({locale}): {len(rows)} utterances, {sum(x['dur_s'] for x in rows):.1f}s audio")

        lang_out = {
            "locale": locale,
            "model": cfg["model"],
            "ct2_model": cfg["ct2_model"],
            "kaggle_dataset": cfg["kaggle_dataset"],
            "revision": cfg["revision"],
            "tsv_sha256": sha256_file(tsv_path),
            "tar_sha256": sha256_file(tar_path),
            "n_utterances": len(rows),
            "total_audio_s": sum(x["dur_s"] for x in rows),
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
                "locale": locale,
                "model_dir": str(model_dir),
                "manifest_path": str(manifest_path),
                "transcribe": cfg["transcribe"],
                "profile": {"device": "cuda", "compute_type": compute_type, "concurrency": concurrency},
                "is_production": is_prod,
            }), encoding="utf-8")

            proc = subprocess.run([sys.executable, str(TMP / "arm_worker.py"), str(spec_file), str(out_file)], capture_output=False)
            if out_file.is_file():
                arm_res = json.loads(out_file.read_text(encoding="utf-8"))
            else:
                arm_res = {"status": "error", "profile": {"compute_type": compute_type, "concurrency": concurrency}, "returncode": proc.returncode}
            lang_out["profiles"].append(arm_res)
            log(f"Finished {code} {compute_type}x{concurrency}: status={arm_res.get('status')} wer={arm_res.get('aggregate', {}).get('wer_pct')}")

        report["languages"][code] = lang_out
        (WORK / "result.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        # Free extracted audio before next language
        shutil.rmtree(extract_dir, ignore_errors=True)
        shutil.rmtree(fleurs_cache, ignore_errors=True)

    log("[chep-eval] Completed FLEURS wave successfully.")

if __name__ == "__main__":
    main()
'''


def build_fleurs_script(wave_name: str, langs: list[str]) -> str:
    catalog = load_catalog()["languages"]
    subset = {k: catalog[k] for k in langs}
    header = (
        f"WAVE_NAME = {wave_name!r}\n"
        f"LANGS = {langs!r}\n"
        f"LOCALES = {LOCALES!r}\n"
        f"CATALOG = {subset!r}\n"
        f"REMOTE_COMMON_SPEC = {REMOTE_COMMON_PY!r}\n"
    )
    return REMOTE_COMMON_PY + "\n" + header + _FLEURS_BODY


def build_fleurs_wave_specs() -> list[KernelSpec]:
    catalog = load_catalog()["languages"]
    return [
        KernelSpec(
            wave="fleurs-a",
            code=build_fleurs_script("fleurs-a", ["fr", "en"]),
            dataset_sources=[catalog["fr"]["kaggle_dataset"], catalog["en"]["kaggle_dataset"]],
            out_dir=RESULTS_ROOT / "fleurs-a",
        ),
        KernelSpec(
            wave="fleurs-b",
            code=build_fleurs_script("fleurs-b", ["es", "vi"]),
            dataset_sources=[
                catalog["es"]["kaggle_dataset"],
                catalog["vi"]["kaggle_dataset"],
            ],
            out_dir=RESULTS_ROOT / "fleurs-b",
        ),
    ]
