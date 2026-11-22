"""Pure-Python text normalization, exact edit distance, and bootstrap confidence intervals.

Designed to reproduce Whisper's BasicTextNormalizer(remove_diacritics=False) and
micro-averaged WER/CER identically to jiwer + whisper-normalizer, and compute
stratified/paired bootstrap 95% confidence intervals locally from saved Kaggle
per-item edit counts.
"""
from __future__ import annotations

import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Any

CATALOG_PATH = Path(__file__).resolve().parents[3] / "config" / "transcription_models.json"

_BRACKETED_RE = re.compile(r"[<\[][^>\]]*[>\]]")
_PAREN_RE = re.compile(r"\(([^)]+?)\)")
_WS_RE = re.compile(r"\s+")


def load_catalog() -> dict[str, Any]:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def basic_text_normalize(text: str) -> str:
    """Equivalent to whisper_normalizer.basic.BasicTextNormalizer(remove_diacritics=False)."""
    s = unicodedata.normalize("NFC", text).replace("\r\n", " ").replace("\n", " ")
    s = s.lower()
    s = _BRACKETED_RE.sub("", s)
    s = _PAREN_RE.sub("", s)
    s = re.sub(r"\s+'", "'", s)
    out_chars: list[str] = []
    for ch in s:
        cat = unicodedata.category(ch)
        if cat and cat[0] in ("M",):
            out_chars.append(ch)
        elif cat and cat[0] in ("S", "P"):
            out_chars.append(" ")
        else:
            out_chars.append(ch)
    s = "".join(out_chars)
    return _WS_RE.sub(" ", s).strip()


def align_counts(ref_seq: list[str] | str, hyp_seq: list[str] | str) -> dict[str, int]:
    """Compute exact unit-cost Levenshtein S, D, I, H counts between sequences.

    Uses standard Myers / Wagner-Fischer row DP in O(min(len)) space when S/D/I
    breakdown is needed, or fast Myers bit-parallel edit distance for long char strings.
    """
    n, m = len(ref_seq), len(hyp_seq)
    if n == 0:
        return {"edits": m, "subs": 0, "dels": 0, "ins": m, "hits": 0, "ref_len": 0}
    if m == 0:
        return {"edits": n, "subs": 0, "dels": n, "ins": 0, "hits": 0, "ref_len": n}

    # For very long character strings (>5000), use Myers bit-parallel distance when only total edits needed
    if isinstance(ref_seq, str) and n > 5000:
        from scripts.benchmarks.kaggle_optimization.core import edit_distance

        edits = edit_distance(ref_seq, hyp_seq)
        return {"edits": edits, "subs": -1, "dels": -1, "ins": -1, "hits": -1, "ref_len": n}

    # Rolling DP storing (cost, subs, dels, ins) packed cleanly as ints for speed
    # Upper bound per step < 2^20; single 64-bit packed int: cost<<40 | subs<<20 | dels -> ins = edits - subs - dels
    SHIFT_COST = 42
    SHIFT_SUBS = 21
    MASK21 = (1 << 21) - 1

    prev = [
        (j << SHIFT_COST) | j  # cost=j, subs=0, dels=0 (packed ins implicitly = j)
        for j in range(m + 1)
    ]
    for i in range(1, n + 1):
        rc = ref_seq[i - 1]
        # i deletions, 0 subs, 0 ins -> cost=i, subs=0, dels=i
        curr = [(i << SHIFT_COST) | i]
        for j in range(1, m + 1):
            hc = hyp_seq[j - 1]
            # Deletion: from prev[j]
            del_val = prev[j] + (1 << SHIFT_COST) + 1
            # Insertion: from curr[j-1]
            ins_val = curr[j - 1] + (1 << SHIFT_COST)
            # Match / Substitution: from prev[j-1]
            if rc == hc:
                sub_val = prev[j - 1]
            else:
                sub_val = prev[j - 1] + (1 << SHIFT_COST) + (1 << SHIFT_SUBS)
            best = sub_val if sub_val <= del_val else del_val
            if ins_val < best:
                best = ins_val
            curr.append(best)
        prev = curr

    final = prev[m]
    edits = final >> SHIFT_COST
    subs = (final >> SHIFT_SUBS) & MASK21
    dels = final & MASK21
    ins = edits - subs - dels
    hits = n - subs - dels
    return {
        "edits": edits,
        "subs": subs,
        "dels": dels,
        "ins": ins,
        "hits": hits,
        "ref_len": n,
    }


def score_texts(ref_raw: str, hyp_raw: str) -> dict[str, Any]:
    ref_norm = basic_text_normalize(ref_raw)
    hyp_norm = basic_text_normalize(hyp_raw)
    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()
    w = align_counts(ref_words, hyp_words)
    c = align_counts(ref_norm, hyp_norm)
    return {
        "ref_norm": ref_norm,
        "hyp_norm": hyp_norm,
        "word_edits": w["edits"],
        "word_subs": w["subs"],
        "word_dels": w["dels"],
        "word_ins": w["ins"],
        "word_hits": w["hits"],
        "ref_words": w["ref_len"],
        "hyp_words": len(hyp_words),
        "char_edits": c["edits"],
        "ref_chars": c["ref_len"],
        "wer_pct": (w["edits"] / w["ref_len"] * 100.0) if w["ref_len"] else 0.0,
        "cer_pct": (c["edits"] / c["ref_len"] * 100.0) if c["ref_len"] else 0.0,
    }


def bootstrap_ci_pct(
    items: list[dict[str, Any]],
    edits_key: str,
    ref_key: str,
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """Compute micro-averaged error rate (%) and 95% percentile bootstrap CI over items."""
    n = len(items)
    if n == 0:
        return {"mean_pct": 0.0, "ci95_low_pct": 0.0, "ci95_high_pct": 0.0}
    pairs = [(int(it[edits_key]), int(it[ref_key])) for it in items]
    tot_e = sum(e for e, _ in pairs)
    tot_r = sum(r for _, r in pairs)
    point = (tot_e / tot_r * 100.0) if tot_r else 0.0
    if n == 1:
        return {"mean_pct": point, "ci95_low_pct": point, "ci95_high_pct": point}

    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(n_boot):
        se = 0
        sr = 0
        for _i in range(n):
            e, r = pairs[rng.randrange(n)]
            se += e
            sr += r
        samples.append((se / sr * 100.0) if sr else 0.0)
    samples.sort()
    lo_idx = int(0.025 * n_boot)
    hi_idx = min(n_boot - 1, int(0.975 * n_boot))
    return {
        "mean_pct": point,
        "ci95_low_pct": samples[lo_idx],
        "ci95_high_pct": samples[hi_idx],
    }
