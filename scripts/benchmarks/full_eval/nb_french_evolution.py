"""Generate Kaggle T4 notebooks for Campaign 3: Table 2 French optimization evolution (all 5 steps)."""
from __future__ import annotations

from .kaggle_runner import KernelSpec, RESULTS_ROOT
from .metrics import load_catalog
from .remote_common import REMOTE_COMMON_PY

_FR_PREP_HELPER = r'''
def prepare_french_longform(work_dir):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    import numpy as np

    ds = "eustlb/french-long-form-test"
    rf = "data/test-00000-of-00001.parquet"
    dl_dir = TMP / "dl_fr_long"
    dl_dir.mkdir(parents=True, exist_ok=True)
    lp = pathlib.Path(hf_hub_download(ds, rf, repo_type="dataset", cache_dir=str(dl_dir)))
    pq_sha = sha256_file(lp)
    table = pq.read_table(lp)

    items = []
    for idx, row in enumerate(table.to_pylist()):
        raw = row["audio"]["bytes"]
        arr = decode_audio_bytes_16k(raw)
        npy_path = work_dir / f"fr_{idx:02d}_{row['uid']}.npy"
        np.save(npy_path, arr)
        items.append({
            "idx": idx,
            "uid": row["uid"],
            "audio_sha256": hashlib.sha256(raw).hexdigest(),
            "npy_path": str(npy_path),
            "duration_s": len(arr) / 16000.0,
            "reference": row["sentence"],
        })
    import shutil
    shutil.rmtree(dl_dir, ignore_errors=True)
    return {
        "dataset": ds,
        "split": "test",
        "parquet_sha256": pq_sha,
        "n_recordings": len(items),
        "total_audio_s": sum(x["duration_s"] for x in items),
        "durations_s": [round(x["duration_s"], 3) for x in items],
    }, items
'''

_FR_PT_BODY = r'''
PT_WORKER_PY = r"""
import json
import pathlib
import sys
import time
import numpy as np
import torch

sys.path.insert(0, "/tmp/chep-eval")
from remote_common import VramMonitor, aggregate_scores, log, score_texts_jiwer

spec = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
out_path = pathlib.Path(sys.argv[2])
items = json.loads(pathlib.Path(spec["manifest_path"]).read_text(encoding="utf-8"))

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from huggingface_hub import HfApi
from whisper_normalizer.basic import BasicTextNormalizer

basic_norm = BasicTextNormalizer()
result = {
    "step": spec["step"],
    "label": spec["label"],
    "backend": "pytorch",
    "model_id": spec["model_id"],
    "revision": spec.get("revision"),
    "chunk_seconds": spec["chunk_seconds"],
    "batch_size": spec["batch_size"],
    "status": "error",
}

with VramMonitor() as mon:
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available for PyTorch arm")
        resolved_sha = HfApi().model_info(spec["model_id"], revision=spec.get("revision")).sha
        result["resolved_revision"] = resolved_sha

        t_load0 = time.perf_counter()
        kw = {"revision": spec["revision"]} if spec.get("revision") else {}
        processor = AutoProcessor.from_pretrained(spec["model_id"], cache_dir=spec["hf_cache"], **kw)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            spec["model_id"],
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            cache_dir=spec["hf_cache"],
            **kw,
        ).to("cuda:0")
        if hasattr(model, "generation_config") and getattr(model.generation_config, "forced_decoder_ids", None) is not None:
            model.generation_config.forced_decoder_ids = None
        model.eval()
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            feature_extractor=processor.feature_extractor,
            tokenizer=processor.tokenizer,
            torch_dtype=torch.float16,
            device="cuda:0",
            chunk_length_s=30,
            generate_kwargs={"language": "french", "task": "transcribe"},
        )
        torch.cuda.synchronize()
        load_s = time.perf_counter() - t_load0

        # Check legacy return_timestamps=False vs transformers >=4.50 requirement on >30s audio
        return_timestamps_arg = False
        warm = np.random.default_rng(17).normal(0.0, 0.02, 40 * 16000).astype(np.float32)
        try:
            _ = pipe({"array": warm, "sampling_rate": 16000})
        except ValueError as ve:
            if "return_timestamps" in str(ve):
                return_timestamps_arg = True
                _ = pipe({"array": warm, "sampling_rate": 16000}, return_timestamps=True)
            else:
                raise
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        seg_samples = int(spec["chunk_seconds"] * 16000)
        batch_size = int(spec["batch_size"])
        per_rec = []
        total_inf_s = 0.0

        for r_idx, item in enumerate(items, start=1):
            arr = np.load(item["npy_path"])
            segments = [arr[i : i + seg_samples] for i in range(0, len(arr), seg_samples)]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            texts = []
            for b0 in range(0, len(segments), batch_size):
                batch_inputs = [{"array": s, "sampling_rate": 16000} for s in segments[b0 : b0 + batch_size]]
                if return_timestamps_arg:
                    outputs = pipe(batch_inputs, batch_size=batch_size, return_timestamps=True)
                else:
                    outputs = pipe(batch_inputs, batch_size=batch_size)
                if isinstance(outputs, dict):
                    outputs = [outputs]
                for o in outputs:
                    txt = (o.get("text") or "").strip()
                    if txt:
                        texts.append(txt)
            torch.cuda.synchronize()
            rec_inf_s = time.perf_counter() - t0
            total_inf_s += rec_inf_s

            hyp = " ".join(texts)
            sc = score_texts_jiwer(item["reference"], hyp, basic_norm)
            per_rec.append({
                "uid": item["uid"],
                "duration_s": item["duration_s"],
                "n_segments": len(segments),
                "inference_s": rec_inf_s,
                "rtf": rec_inf_s / item["duration_s"],
                "word_subs": sc["word_subs"],
                "word_dels": sc["word_dels"],
                "word_ins": sc["word_ins"],
                "word_hits": sc["word_hits"],
                "word_edits": sc["word_edits"],
                "ref_words": sc["ref_words"],
                "char_edits": sc["char_edits"],
                "ref_chars": sc["ref_chars"],
            })
            log(f"  [PyTorch {spec['step']}] {r_idx}/{len(items)} uid={item['uid']} inf={rec_inf_s:.1f}s wer={sc['word_edits']/sc['ref_words']*100:.2f}%")

        agg = aggregate_scores(per_rec)
        total_audio_s = sum(it["duration_s"] for it in items)
        result.update({
            "status": "ok",
            "return_timestamps_used": return_timestamps_arg,
            "load_s": load_s,
            "inference_s": total_inf_s,
            "rtf": total_inf_s / total_audio_s,
            "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
            "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024.0 * 1024.0),
            "aggregate": agg,
            "per_recording": per_rec,
        })
    except Exception as exc:
        result["error"] = str(exc)
        import traceback
        result["traceback"] = traceback.format_exc()

result["peak_vram_mib"] = mon.peak_vram_mib
result["peak_ram_mib"] = mon.peak_ram_mib
out_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
"""


def main():
    import pathlib
    import shutil
    import subprocess
    import sys

    (TMP / "remote_common.py").write_text(REMOTE_COMMON_SPEC, encoding="utf-8")
    (TMP / "pt_worker.py").write_text(PT_WORKER_PY, encoding="utf-8")

    log("[chep-eval] Starting Wave 3A PyTorch baselines (Table 2 Steps 1 & 2)")
    gpu_info = ensure_kaggle_t4()

    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "transformers==4.57.6", "accelerate",
        "jiwer==3.1.0", "whisper-normalizer==0.1.12",
        "soundfile", "pyarrow", "psutil", "huggingface_hub", "av",
    ], check=True)

    work_dir = TMP / "fr_audio"
    work_dir.mkdir(parents=True, exist_ok=True)
    meta, items = prepare_french_longform(work_dir)
    manifest_path = TMP / "fr_manifest.json"
    manifest_path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    log(f"Prepared French long-form: {meta['n_recordings']} talks, {meta['total_audio_s']:.1f}s")

    arms_spec = [
        {
            "step": "step1_pytorch_full_120s_b4",
            "label": "Full French model, baseline (PyTorch, 120s, batch=4)",
            "model_id": "bofenghuang/whisper-large-v3-french",
            "revision": "90a7cd1fea9cd69f80f5a0fc49738418dba96083",
            "chunk_seconds": 120,
            "batch_size": 4,
        },
        {
            "step": "step2_pytorch_dec16_120s_b4",
            "label": "Distil-dec16, baseline (PyTorch, 120s, batch=4)",
            "model_id": "bofenghuang/whisper-large-v3-french-distil-dec16",
            "revision": "16bd02185bdaa6b00fb4b0deb46e47ac1b754b8e",
            "chunk_seconds": 120,
            "batch_size": 4,
        },
    ]

    report = {
        "campaign": "french_evolution_pytorch",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": gpu_info,
        "corpus": meta,
        "versions": installed_versions(["torch", "transformers", "accelerate", "jiwer", "whisper-normalizer", "soundfile", "av"]),
        "arms": {},
    }

    for arm in arms_spec:
        hf_cache = TMP / "hf_cache" / arm["step"]
        hf_cache.mkdir(parents=True, exist_ok=True)
        spec_file = TMP / f"spec_{arm['step']}.json"
        out_file = TMP / f"out_{arm['step']}.json"
        out_file.unlink(missing_ok=True)
        spec_file.write_text(json.dumps({**arm, "manifest_path": str(manifest_path), "hf_cache": str(hf_cache)}), encoding="utf-8")
        proc = subprocess.run([sys.executable, str(TMP / "pt_worker.py"), str(spec_file), str(out_file)], capture_output=False)
        res = json.loads(out_file.read_text(encoding="utf-8")) if out_file.is_file() else {"status": "error", "returncode": proc.returncode}
        report["arms"][arm["step"]] = res
        (WORK / "result.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        log(f"Finished {arm['step']}: status={res.get('status')} inf_s={res.get('inference_s')} vram={res.get('peak_vram_mib')} wer={res.get('aggregate', {}).get('wer_pct')}")
        shutil.rmtree(hf_cache, ignore_errors=True)

    log("[chep-eval] Completed Wave 3A PyTorch baselines.")


if __name__ == "__main__":
    main()
'''


_FR_CT2_BODY = r'''
CT2_WORKER_PY = r"""
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
items = json.loads(pathlib.Path(spec["manifest_path"]).read_text(encoding="utf-8"))

preload_cuda12_libs()
from faster_whisper import WhisperModel
from whisper_normalizer.basic import BasicTextNormalizer

basic_norm = BasicTextNormalizer()
result = {
    "step": spec["step"],
    "label": spec["label"],
    "backend": "ctranslate2",
    "compute_type": spec["compute_type"],
    "chunk_seconds": spec["chunk_seconds"],
    "workers": spec["workers"],
    "status": "error",
}

with VramMonitor() as mon:
    try:
        t_load0 = time.perf_counter()
        model = WhisperModel(
            spec["model_dir"],
            device="cuda",
            compute_type=spec["compute_type"],
            num_workers=spec["workers"],
            cpu_threads=4,
            device_index=0,
            local_files_only=True,
        )
        actual_compute_type = model.model.compute_type
        load_s = time.perf_counter() - t_load0

        chunk_samples = int(spec["chunk_seconds"] * 16000)
        workers = int(spec["workers"])
        warm = np.random.default_rng(17).normal(0.0, 0.025, chunk_samples).astype(np.float32)
        def _warm(_idx):
            segs, _ = model.transcribe(warm, language="fr", task="transcribe", **spec["transcribe"])
            return list(segs)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_warm, range(workers)))

        def _infer(chunk):
            segs, _ = model.transcribe(chunk, language="fr", task="transcribe", **spec["transcribe"])
            return " ".join(s.text.strip() for s in segs if s.text.strip())

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
                sc = score_texts_jiwer(item["reference"], hyp, basic_norm)
                entry = {
                    "uid": item["uid"],
                    "duration_s": item["duration_s"],
                    "n_chunks": len(chunks),
                    "inference_s": rec_inf_s,
                    "rtf": rec_inf_s / item["duration_s"],
                    "word_subs": sc["word_subs"],
                    "word_dels": sc["word_dels"],
                    "word_ins": sc["word_ins"],
                    "word_hits": sc["word_hits"],
                    "word_edits": sc["word_edits"],
                    "ref_words": sc["ref_words"],
                    "char_edits": sc["char_edits"],
                    "ref_chars": sc["ref_chars"],
                }
                if spec.get("save_texts"):
                    entry["raw_hyp"] = hyp
                    entry["norm_ref"] = sc["ref_norm"]
                    entry["norm_hyp"] = sc["hyp_norm"]
                per_rec.append(entry)
                log(f"  [CT2 {spec['step']}] {r_idx}/{len(items)} uid={item['uid']} inf={rec_inf_s:.1f}s wer={sc['word_edits']/sc['ref_words']*100:.2f}% del={sc['word_dels']/sc['ref_words']*100:.2f}%")

        agg = aggregate_scores(per_rec)
        total_audio_s = sum(it["duration_s"] for it in items)
        result.update({
            "status": "ok",
            "actual_compute_type": actual_compute_type,
            "load_s": load_s,
            "inference_s": total_inf_s,
            "rtf": total_inf_s / total_audio_s,
            "aggregate": agg,
            "per_recording": per_rec,
        })
    except Exception as exc:
        result["error"] = str(exc)
        import traceback
        result["traceback"] = traceback.format_exc()

result["peak_vram_mib"] = mon.peak_vram_mib
result["peak_ram_mib"] = mon.peak_ram_mib
out_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
"""


def main():
    import pathlib
    import subprocess
    import sys

    (TMP / "remote_common.py").write_text(REMOTE_COMMON_SPEC, encoding="utf-8")
    (TMP / "ct2_worker.py").write_text(CT2_WORKER_PY, encoding="utf-8")

    log("[chep-eval] Starting Wave 3A CT2 single-GPU steps (Table 2 Steps 3, 4 + sensitivity + INT8)")
    gpu_info = ensure_kaggle_t4()

    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "faster-whisper==1.2.1", "ctranslate2==4.8.2",
        "jiwer==3.1.0", "whisper-normalizer==0.1.12",
        "soundfile", "pyarrow", "psutil", "huggingface_hub",
    ], check=True)

    model_dir = discover_pinned_ct2_model(FR_CFG["files"])
    work_dir = TMP / "fr_audio"
    work_dir.mkdir(parents=True, exist_ok=True)
    meta, items = prepare_french_longform(work_dir)
    manifest_path = TMP / "fr_manifest.json"
    manifest_path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    log(f"Prepared French long-form: {meta['n_recordings']} talks, {meta['total_audio_s']:.1f}s")

    arms_spec = [
        {
            "step": "step3_ct2_120s_w1",
            "label": "Distil-dec16, long chunks (CT2 float16, 120s, 1 worker)",
            "compute_type": "float16",
            "chunk_seconds": 120,
            "workers": 1,
            "save_texts": False,
        },
        {
            "step": "step4a_ct2_40s_w1",
            "label": "Distil-dec16, 40s, 1 worker (sensitivity)",
            "compute_type": "float16",
            "chunk_seconds": 40,
            "workers": 1,
            "save_texts": False,
        },
        {
            "step": "step4_ct2_40s_w2",
            "label": "Distil-dec16, shorter concurrent chunks (CT2 float16, 40s, 2 workers)",
            "compute_type": "float16",
            "chunk_seconds": 40,
            "workers": 2,
            "save_texts": True,
        },
        {
            "step": "step4c_ct2_40s_w4",
            "label": "Distil-dec16, 40s, 4 workers (sensitivity)",
            "compute_type": "float16",
            "chunk_seconds": 40,
            "workers": 4,
            "save_texts": False,
        },
        {
            "step": "step4d_ct2_int8fp16_40s_w2",
            "label": "Distil-dec16, 40s, 2 workers, int8_float16 (quantization counterfactual)",
            "compute_type": "int8_float16",
            "chunk_seconds": 40,
            "workers": 2,
            "save_texts": False,
        },
    ]

    report = {
        "campaign": "french_evolution_ct2",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": gpu_info,
        "corpus": meta,
        "model": FR_CFG["model"],
        "ct2_model": FR_CFG["ct2_model"],
        "kaggle_dataset": FR_CFG["kaggle_dataset"],
        "revision": FR_CFG["revision"],
        "transcribe": FR_CFG["transcribe"],
        "versions": installed_versions(["faster-whisper", "ctranslate2", "jiwer", "whisper-normalizer", "soundfile", "av"]),
        "arms": {},
    }

    for arm in arms_spec:
        spec_file = TMP / f"spec_{arm['step']}.json"
        out_file = TMP / f"out_{arm['step']}.json"
        out_file.unlink(missing_ok=True)
        spec_file.write_text(json.dumps({
            **arm,
            "model_dir": str(model_dir),
            "manifest_path": str(manifest_path),
            "transcribe": FR_CFG["transcribe"],
        }), encoding="utf-8")
        proc = subprocess.run([sys.executable, str(TMP / "ct2_worker.py"), str(spec_file), str(out_file)], capture_output=False)
        res = json.loads(out_file.read_text(encoding="utf-8")) if out_file.is_file() else {"status": "error", "returncode": proc.returncode}
        report["arms"][arm["step"]] = res
        (WORK / "result.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        log(f"Finished {arm['step']}: status={res.get('status')} inf_s={res.get('inference_s')} vram={res.get('peak_vram_mib')} wer={res.get('aggregate', {}).get('wer_pct')}")

    log("[chep-eval] Completed Wave 3A CT2 single-GPU steps.")


if __name__ == "__main__":
    main()
'''


_FR_DUAL_BODY = r'''
def main():
    import concurrent.futures
    import pathlib
    import subprocess
    import sys
    import numpy as np

    (TMP / "remote_common.py").write_text(REMOTE_COMMON_SPEC, encoding="utf-8")
    log(f"[chep-eval] Starting Wave 3B Dual-GPU Shard {SHARD_INDEX} (Table 2 Step 5)")
    gpu_info = ensure_kaggle_t4()

    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "faster-whisper==1.2.1", "ctranslate2==4.8.2",
        "soundfile", "pyarrow", "psutil", "huggingface_hub",
    ], check=True)

    loaded_libs = preload_cuda12_libs()
    log(f"Preloaded CUDA12 libs: {loaded_libs}")
    from faster_whisper import WhisperModel

    model_dir = discover_pinned_ct2_model(FR_CFG["files"])
    work_dir = TMP / "fr_audio"
    work_dir.mkdir(parents=True, exist_ok=True)
    meta, items = prepare_french_longform(work_dir)
    log(f"Shard {SHARD_INDEX}: prepared {meta['n_recordings']} talks ({meta['total_audio_s']:.1f}s full)")

    report = {
        "campaign": "french_evolution_dual_shard",
        "shard_index": SHARD_INDEX,
        "overlap_seconds": 5.0,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": gpu_info,
        "corpus": meta,
        "model": FR_CFG["model"],
        "kaggle_dataset": FR_CFG["kaggle_dataset"],
        "revision": FR_CFG["revision"],
        "versions": installed_versions(["faster-whisper", "ctranslate2", "soundfile", "av"]),
        "status": "error",
    }

    with VramMonitor() as mon:
        try:
            t_load0 = time.perf_counter()
            model = WhisperModel(
                str(model_dir),
                device="cuda",
                compute_type="float16",
                num_workers=2,
                cpu_threads=4,
                device_index=0,
                local_files_only=True,
            )
            load_s = time.perf_counter() - t_load0

            chunk_seconds = int(FR_CFG["chunk_seconds"])
            chunk_samples = chunk_seconds * 16000
            warm = np.random.default_rng(17).normal(0.0, 0.025, chunk_samples).astype(np.float32)
            def _warm(_idx):
                segs, _ = model.transcribe(warm, language="fr", task="transcribe", **FR_CFG["transcribe"])
                return list(segs)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(_warm, range(2)))

            def _infer(offset_and_chunk):
                offset_s, chunk = offset_and_chunk
                segs, _ = model.transcribe(chunk, language="fr", task="transcribe", **FR_CFG["transcribe"])
                return [
                    {"start": offset_s + float(s.start), "end": offset_s + float(s.end), "text": s.text.strip()}
                    for s in segs
                    if s.text.strip()
                ]

            per_rec = []
            utc_inf_start = time.time()
            total_inf_s = 0.0
            total_shard_audio_s = 0.0

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                for r_idx, item in enumerate(items, start=1):
                    full_arr = np.load(item["npy_path"])
                    n_total = len(full_arr)
                    dur_s = n_total / 16000.0
                    mid_s = dur_s / 2.0
                    if SHARD_INDEX == 0:
                        span_start = 0.0
                        span_end = mid_s + 2.5
                    else:
                        span_start = mid_s - 2.5
                        span_end = dur_s
                    lo = round(span_start * 16000)
                    hi = min(round(span_end * 16000), n_total)
                    shard_arr = full_arr[lo:hi]
                    # Use exact local slice length as upper bound (same convention as local_app/worker.py:select_range + kaggle_merge.py)
                    actual_span = (lo / 16000.0, hi / 16000.0)
                    shard_dur_s = (hi - lo) / 16000.0
                    total_shard_audio_s += shard_dur_s

                    chunks = [
                        (i / 16000.0, shard_arr[i : i + chunk_samples])
                        for i in range(0, len(shard_arr), chunk_samples)
                    ]
                    w_start = time.time()
                    t0 = time.perf_counter()
                    talk_segments = []
                    for b0 in range(0, len(chunks), 2):
                        for seg_list in pool.map(_infer, chunks[b0 : b0 + 2]):
                            for s in seg_list:
                                s["start"] = min(shard_dur_s, max(0.0, s["start"]))
                                s["end"] = min(shard_dur_s, max(s["start"], s["end"]))
                                talk_segments.append(s)
                    rec_inf_s = time.perf_counter() - t0
                    w_end = time.time()
                    total_inf_s += rec_inf_s

                    per_rec.append({
                        "uid": item["uid"],
                        "source_duration_s": dur_s,
                        "span": list(actual_span),
                        "shard_duration_s": shard_dur_s,
                        "inference_s": rec_inf_s,
                        "utc_window": [w_start, w_end],
                        "reference": item["reference"],
                        "segments": talk_segments,
                    })
                    log(f"  [Dual shard {SHARD_INDEX}] {r_idx}/{len(items)} uid={item['uid']} span=[{actual_span[0]:.2f},{actual_span[1]:.2f}] inf={rec_inf_s:.1f}s segs={len(talk_segments)}")

            utc_inf_end = time.time()
            report.update({
                "status": "ok",
                "actual_compute_type": model.model.compute_type,
                "load_s": load_s,
                "inference_s": total_inf_s,
                "shard_audio_s": total_shard_audio_s,
                "utc_inference_window": [utc_inf_start, utc_inf_end],
                "per_recording": per_rec,
            })
        except Exception as exc:
            report["error"] = str(exc)
            import traceback
            report["traceback"] = traceback.format_exc()

    report["peak_vram_mib"] = mon.peak_vram_mib
    report["peak_ram_mib"] = mon.peak_ram_mib
    (WORK / "result.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    log(f"[chep-eval] Finished Dual-GPU Shard {SHARD_INDEX}: inf={report.get('inference_s')} vram={report['peak_vram_mib']}")


if __name__ == "__main__":
    main()
'''


def build_fr_pt_spec() -> KernelSpec:
    script = (
        REMOTE_COMMON_PY
        + f"\nREMOTE_COMMON_SPEC = {REMOTE_COMMON_PY!r}\n"
        + _FR_PREP_HELPER
        + _FR_PT_BODY
    )
    return KernelSpec(
        wave="fr-pt",
        code=script,
        dataset_sources=[],
        out_dir=RESULTS_ROOT / "fr-pt",
    )


def build_fr_ct2_spec() -> KernelSpec:
    fr_cfg = load_catalog()["languages"]["fr"]
    script = (
        REMOTE_COMMON_PY
        + f"\nFR_CFG = {fr_cfg!r}\nREMOTE_COMMON_SPEC = {REMOTE_COMMON_PY!r}\n"
        + _FR_PREP_HELPER
        + _FR_CT2_BODY
    )
    return KernelSpec(
        wave="fr-ct2",
        code=script,
        dataset_sources=[fr_cfg["kaggle_dataset"]],
        out_dir=RESULTS_ROOT / "fr-ct2",
    )


def build_fr_dual_specs() -> list[KernelSpec]:
    fr_cfg = load_catalog()["languages"]["fr"]
    specs = []
    for shard_idx in (0, 1):
        script = (
            REMOTE_COMMON_PY
            + f"\nSHARD_INDEX = {shard_idx}\nFR_CFG = {fr_cfg!r}\nREMOTE_COMMON_SPEC = {REMOTE_COMMON_PY!r}\n"
            + _FR_PREP_HELPER
            + _FR_DUAL_BODY
        )
        specs.append(
            KernelSpec(
                wave=f"fr-dual-{shard_idx}",
                code=script,
                dataset_sources=[fr_cfg["kaggle_dataset"]],
                out_dir=RESULTS_ROOT / f"fr-dual-{shard_idx}",
            )
        )
    return specs
