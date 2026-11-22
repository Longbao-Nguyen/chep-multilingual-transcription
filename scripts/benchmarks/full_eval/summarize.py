"""Aggregate Kaggle T4 evaluation results, merge Step 5 shards, verify scorer parity, and export summaries."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from local_app.kaggle_merge import merge_segments  # noqa: E402
from scripts.benchmarks.full_eval.kaggle_runner import RESULTS_ROOT  # noqa: E402
from scripts.benchmarks.full_eval.metrics import bootstrap_ci_pct, load_catalog, score_texts  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing evaluation output: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_fleurs(results_root: Path) -> dict[str, Any]:
    fa = _load_json(results_root / "fleurs-a" / "result.json")
    fb = _load_json(results_root / "fleurs-b" / "result.json")
    merged_langs = {**fa.get("languages", {}), **fb.get("languages", {})}
    out: dict[str, Any] = {
        "versions": fa.get("versions") or fb.get("versions"),
        "gpu": fa.get("gpu") or fb.get("gpu"),
        "languages": {},
    }
    for code in ("fr", "en", "es", "vi"):
        if code not in merged_langs:
            continue
        ldata = merged_langs[code]
        profs = []
        for p in ldata.get("profiles", []):
            if p.get("status") != "ok":
                profs.append(p)
                continue
            per_utt = p.get("per_utterance", [])
            wer_ci = bootstrap_ci_pct(per_utt, "word_edits", "ref_words")
            cer_ci = bootstrap_ci_pct(per_utt, "char_edits", "ref_chars")
            entry = {
                "status": "ok",
                "profile": p["profile"],
                "actual_compute_type": p.get("actual_compute_type"),
                "is_production": p.get("is_production"),
                "n_utterances": p.get("n_utterances"),
                "total_audio_s": round(p.get("total_audio_s", 0.0), 2),
                "load_s": round(p.get("load_s", 0.0), 2),
                "inference_s": round(p.get("inference_s", 0.0), 2),
                "rtf": p.get("rtf"),
                "peak_vram_mib": p.get("peak_vram_mib"),
                "peak_ram_mib": round(p.get("peak_ram_mib", 0.0), 1),
                "aggregate": p.get("aggregate"),
                "wer_ci95_pct": [round(wer_ci["ci95_low_pct"], 2), round(wer_ci["ci95_high_pct"], 2)],
                "cer_ci95_pct": [round(cer_ci["ci95_low_pct"], 2), round(cer_ci["ci95_high_pct"], 2)],
            }
            if "aggregate_english_normalizer" in p:
                entry["aggregate_english_normalizer"] = p["aggregate_english_normalizer"]
            if "aggregate_fleurs_col3" in p:
                entry["aggregate_fleurs_col3"] = p["aggregate_fleurs_col3"]
            profs.append(entry)
        out["languages"][code] = {
            "locale": ldata.get("locale"),
            "model": ldata.get("model"),
            "ct2_model": ldata.get("ct2_model"),
            "kaggle_dataset": ldata.get("kaggle_dataset"),
            "revision": ldata.get("revision"),
            "tsv_sha256": ldata.get("tsv_sha256"),
            "tar_sha256": ldata.get("tar_sha256"),
            "n_utterances": ldata.get("n_utterances"),
            "total_audio_s": round(ldata.get("total_audio_s", 0.0), 2),
            "profiles": profs,
        }
    return out


def summarize_longform(results_root: Path) -> dict[str, Any]:
    envi = _load_json(results_root / "long-envi" / "result.json")
    es = _load_json(results_root / "long-es" / "result.json")
    fr_ct2 = _load_json(results_root / "fr-ct2" / "result.json")

    merged = {**envi.get("languages", {}), **es.get("languages", {})}
    out: dict[str, Any] = {
        "versions": envi.get("versions") or es.get("versions"),
        "gpu": envi.get("gpu") or es.get("gpu"),
        "languages": {},
    }

    # Add French long-form from fr-ct2 (Step 4 FP16 & Step 4d INT8_FP16)
    fr_arms = fr_ct2.get("arms", {})
    fr_corpus = fr_ct2.get("corpus", {})
    fr_profs = []
    for arm_key, is_prod in (("step4_ct2_40s_w2", True), ("step4d_ct2_int8fp16_40s_w2", False)):
        arm = fr_arms.get(arm_key)
        if not arm:
            continue
        fr_profs.append({
            "status": "ok",
            "profile": {"device": "cuda", "compute_type": arm.get("compute_type"), "concurrency": arm.get("workers")},
            "actual_compute_type": arm.get("actual_compute_type"),
            "is_production": is_prod,
            "n_recordings": fr_corpus.get("n_recordings"),
            "total_audio_s": round(fr_corpus.get("total_audio_s", 0.0), 2),
            "load_s": round(arm.get("load_s", 0.0), 2),
            "inference_s": round(arm.get("inference_s", 0.0), 2),
            "rtf": arm.get("rtf"),
            "peak_vram_mib": arm.get("peak_vram_mib"),
            "aggregate": arm.get("aggregate"),
        })
    out["languages"]["fr"] = {
        "dataset": fr_corpus.get("dataset"),
        "split": fr_corpus.get("split"),
        "grouping": "native continuous TEDx talks (unsegmented)",
        "parquet_sha256": fr_corpus.get("parquet_sha256"),
        "n_recordings": fr_corpus.get("n_recordings"),
        "total_audio_s": round(fr_corpus.get("total_audio_s", 0.0), 2),
        "model": fr_ct2.get("model"),
        "kaggle_dataset": fr_ct2.get("kaggle_dataset"),
        "chunk_seconds": 40,
        "profiles": fr_profs,
    }

    for code in ("en", "es", "vi"):
        if code not in merged:
            continue
        ldata = merged[code]
        profs = []
        for p in ldata.get("profiles", []):
            if p.get("status") != "ok":
                profs.append(p)
                continue
            entry = {
                "status": "ok",
                "profile": p["profile"],
                "actual_compute_type": p.get("actual_compute_type"),
                "is_production": p.get("is_production"),
                "n_recordings": p.get("n_recordings"),
                "total_audio_s": round(p.get("total_audio_s", 0.0), 2),
                "load_s": round(p.get("load_s", 0.0), 2),
                "inference_s": round(p.get("inference_s", 0.0), 2),
                "rtf": p.get("rtf"),
                "peak_vram_mib": p.get("peak_vram_mib"),
                "aggregate": p.get("aggregate"),
            }
            if "aggregate_english_normalizer" in p:
                entry["aggregate_english_normalizer"] = p["aggregate_english_normalizer"]
            profs.append(entry)
        out["languages"][code] = {
            "dataset": ldata.get("dataset"),
            "split": ldata.get("split"),
            "grouping": ldata.get("grouping"),
            "parquet_sha256": ldata.get("parquet_sha256"),
            "total_segments": ldata.get("total_segments"),
            "n_recordings": ldata.get("n_recordings"),
            "total_audio_s": round(ldata.get("total_audio_s", 0.0), 2),
            "duration_min_med_max_s": ldata.get("duration_min_med_max_s"),
            "durations_s": ldata.get("durations_s"),
            "model": ldata.get("model"),
            "kaggle_dataset": ldata.get("kaggle_dataset"),
            "chunk_seconds": ldata.get("chunk_seconds"),
            "profiles": profs,
        }
    return out


def summarize_french_evolution(results_root: Path) -> dict[str, Any]:
    fr_pt = _load_json(results_root / "fr-pt" / "result.json")
    fr_ct2 = _load_json(results_root / "fr-ct2" / "result.json")
    d0 = _load_json(results_root / "fr-dual-0" / "result.json")
    d1 = _load_json(results_root / "fr-dual-1" / "result.json")

    # Cross-check pure-Python metrics.score_texts against Kaggle jiwer on Step 4!
    step4 = fr_ct2["arms"]["step4_ct2_40s_w2"]
    max_norm_mismatch = 0
    max_word_diff = 0
    max_char_diff = 0
    for r0_item, rec in zip(d0["per_recording"], step4["per_recording"]):
        orig_ref = r0_item.get("reference", rec.get("norm_ref", ""))
        local_sc = score_texts(orig_ref, rec.get("raw_hyp", ""))
        if local_sc["ref_norm"] != rec.get("norm_ref") or local_sc["hyp_norm"] != rec.get("norm_hyp"):
            max_norm_mismatch += 1
        max_word_diff = max(max_word_diff, abs(local_sc["word_edits"] - rec["word_edits"]))
        max_char_diff = max(max_char_diff, abs(local_sc["char_edits"] - rec["char_edits"]))

    # Step 5 merge & scoring via production local_app/kaggle_merge.py
    assert d0.get("status") == "ok" and d1.get("status") == "ok"
    w0 = d0["utc_inference_window"]
    w1 = d1["utc_inference_window"]
    utc_overlap_s = max(0.0, min(w0[1], w1[1]) - max(w0[0], w1[0]))
    utc_span_s = max(w0[1], w1[1]) - min(w0[0], w1[0])

    merged_per_rec = []
    seam_audits = []
    recs0 = d0["per_recording"]
    recs1 = d1["per_recording"]
    assert len(recs0) == len(recs1)
    for r0, r1 in zip(recs0, recs1):
        assert r0["uid"] == r1["uid"]
        merged_segs, audit = merge_segments([r0["segments"], r1["segments"]], [tuple(r0["span"]), tuple(r1["span"])])
        hyp = " ".join(s["text"] for s in merged_segs if s["text"].strip())
        sc = score_texts(r0["reference"], hyp)
        merged_per_rec.append({
            "uid": r0["uid"],
            "duration_s": r0["source_duration_s"],
            "word_subs": sc["word_subs"],
            "word_dels": sc["word_dels"],
            "word_ins": sc["word_ins"],
            "word_hits": sc["word_hits"],
            "word_edits": sc["word_edits"],
            "ref_words": sc["ref_words"],
            "char_edits": sc["char_edits"],
            "ref_chars": sc["ref_chars"],
            "seam_audit": audit,
        })
        if audit:
            seam_audits.append({"uid": r0["uid"], "midpoint_s": r0["source_duration_s"] / 2.0, "events": audit})

    tot_subs = sum(x["word_subs"] for x in merged_per_rec)
    tot_dels = sum(x["word_dels"] for x in merged_per_rec)
    tot_ins = sum(x["word_ins"] for x in merged_per_rec)
    tot_hits = sum(x["word_hits"] for x in merged_per_rec)
    tot_we = sum(x["word_edits"] for x in merged_per_rec)
    tot_rw = sum(x["ref_words"] for x in merged_per_rec)
    tot_ce = sum(x["char_edits"] for x in merged_per_rec)
    tot_rc = sum(x["ref_chars"] for x in merged_per_rec)
    total_audio_s = fr_ct2["corpus"]["total_audio_s"]
    inf_max_s = max(d0["inference_s"], d1["inference_s"])

    step5 = {
        "step": "step5_ct2_40s_w2_dual_2gpu",
        "label": "Distil-dec16, shorter concurrent chunks, split on 2 GPU (Notebooks)",
        "status": "ok",
        "backend": "ctranslate2",
        "compute_type": "float16",
        "chunk_seconds": 40,
        "workers_per_gpu": 2,
        "shard0_inference_s": d0["inference_s"],
        "shard1_inference_s": d1["inference_s"],
        "inference_max_s": inf_max_s,
        "inference_utc_span_s": utc_span_s,
        "utc_overlap_s": utc_overlap_s,
        "rtf_max": inf_max_s / total_audio_s,
        "rtf_utc_span": utc_span_s / total_audio_s,
        "peak_vram_mib_shard0": d0["peak_vram_mib"],
        "peak_vram_mib_shard1": d1["peak_vram_mib"],
        "peak_vram_mib_max": max(d0["peak_vram_mib"], d1["peak_vram_mib"]),
        "scorer_parity_check": {
            "norm_mismatches": max_norm_mismatch,
            "max_word_edit_diff_on_step4": max_word_diff,
            "max_char_edit_diff_on_step4": max_char_diff,
        },
        "seam_audit_events": seam_audits,
        "aggregate": {
            "word_subs": tot_subs,
            "word_dels": tot_dels,
            "word_ins": tot_ins,
            "word_hits": tot_hits,
            "word_edits": tot_we,
            "ref_words": tot_rw,
            "wer_pct": tot_we / tot_rw * 100.0,
            "del_rate_pct": tot_dels / tot_rw * 100.0,
            "sub_rate_pct": tot_subs / tot_rw * 100.0,
            "ins_rate_pct": tot_ins / tot_rw * 100.0,
            "char_edits": tot_ce,
            "ref_chars": tot_rc,
            "cer_pct": tot_ce / tot_rc * 100.0,
        },
    }

    def _compact_arm(arm: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in arm.items() if k != "per_recording"}

    steps: dict[str, Any] = {}
    for k, v in fr_pt.get("arms", {}).items():
        steps[k] = _compact_arm(v)
    for k, v in fr_ct2.get("arms", {}).items():
        steps[k] = _compact_arm(v)
    steps["step5_ct2_40s_w2_dual_2gpu"] = step5

    return {
        "corpus": fr_ct2.get("corpus"),
        "pytorch_versions": fr_pt.get("versions"),
        "ct2_versions": fr_ct2.get("versions"),
        "steps": steps,
    }


def write_full_summary(results_root: Path = RESULTS_ROOT) -> dict[str, Any]:
    summary: dict[str, Any] = {"catalog_snapshot": {k: {"model": v["model"], "sweet_spot": v["sweet_spot"], "chunk_seconds": v["chunk_seconds"]} for k, v in load_catalog()["languages"].items()}}
    if (results_root / "fleurs-a" / "result.json").is_file() and (results_root / "fleurs-b" / "result.json").is_file():
        summary["fleurs_full_test"] = summarize_fleurs(results_root)
    if (results_root / "long-envi" / "result.json").is_file() and (results_root / "long-es" / "result.json").is_file() and (results_root / "fr-ct2" / "result.json").is_file():
        summary["longform_multilingual"] = summarize_longform(results_root)
    if all((results_root / p / "result.json").is_file() for p in ("fr-pt", "fr-ct2", "fr-dual-0", "fr-dual-1")):
        summary["french_evolution"] = summarize_french_evolution(results_root)

    for dest in (results_root / "summary.json", REPO_ROOT / "docs" / "evidence" / "full-eval-2026-11" / "summary.json"):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    s = write_full_summary()
    print(json.dumps(s, indent=2, ensure_ascii=False))
