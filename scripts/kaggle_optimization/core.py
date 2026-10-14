"""Portable validation and merge helpers (standard library only)."""
import hashlib
import difflib
import json
import math
import statistics
import unicodedata
from pathlib import Path


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def validate_model(path, files):
    for name, meta in files.items():
        target = Path(path) / name
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Unsafe artifact name')
        if not target.is_file() or target.stat().st_size != meta['size'] or sha(target) != meta['sha256']:
            raise ValueError('Artifact mismatch: ' + name)


def discover(root, files):
    candidates = [p.parent for p in Path(root).rglob('model.bin')
                  if all((p.parent / name).is_file() for name in files)]
    valid = []
    for path in candidates:
        try:
            validate_model(path, files)
            valid.append(path)
        except ValueError:
            pass
    if len(valid) != 1:
        raise ValueError('Expected exactly one matching model mount')
    return valid[0]


def ranges(duration, overlap):
    if not math.isfinite(duration) or duration <= 0 or overlap not in (0, 5, 10) or overlap >= duration:
        raise ValueError('Invalid duration/overlap')
    return [(0, duration / 2 + overlap / 2), (duration / 2 - overlap / 2, duration)]


def chunk_aligned_ranges(duration, chunk_seconds):
    """Partition existing production chunks without adding a new chunk boundary."""
    if not math.isfinite(duration) or not math.isfinite(chunk_seconds) or chunk_seconds <= 0 or duration <= chunk_seconds:
        raise ValueError('Need at least two chunks')
    split = max(chunk_seconds, round(duration / (2 * chunk_seconds)) * chunk_seconds)
    if split >= duration:
        split -= chunk_seconds
    return [(0, split), (split, duration)]


def normalize(text):
    # Accent-preserving Unicode normalization, shared by both quality arms.
    return ' '.join(''.join(c if c.isalnum() else ' ' for c in
                            unicodedata.normalize('NFC', text).casefold()).split())


def merge(shards, spans):
    """Conservative whole-segment dedup; ambiguous boundary content is retained for audit."""
    if len(shards) != len(spans):
        raise ValueError('Missing shard')
    output, removed = [], []
    for index, (segments, (offset, stop)) in enumerate(zip(shards, spans)):
        for segment in segments:
            start, end = segment['start'], segment['end']
            if not all(math.isfinite(t) for t in (start, end)) or not 0 <= start <= end <= stop - offset + .001:
                raise ValueError('Invalid local timestamps')
            item = {**segment, 'start': start + offset, 'end': end + offset, 'shard': index}
            duplicate = next((old for old in output if old['shard'] != index
                              and normalize(old['text']) == normalize(item['text'])
                              and normalize(item['text'])
                              and max(old['start'], item['start']) < min(old['end'], item['end'])
                              and abs(old['start'] - item['start']) <= 2
                              and abs(old['end'] - item['end']) <= 2), None)
            if duplicate:
                removed.append(item)
            else:
                output.append(item)
    output.sort(key=lambda s: (s['start'], s['end'], s['shard']))
    return [{k: v for k, v in s.items() if k != 'shard'} for s in output], removed


def edit_distance(a, b):
    """Exact unit-cost Levenshtein, bit parallel to keep long-form CER practical."""
    if len(a) > len(b):
        a, b = b, a
    if not a:
        return len(b)
    masks = {}
    for index, token in enumerate(a):
        masks[token] = masks.get(token, 0) | (1 << index)
    limit = (1 << len(a)) - 1
    high = 1 << (len(a) - 1)
    positive, negative, score = limit, 0, len(a)
    for token in b:
        equal = masks.get(token, 0)
        diagonal = (((equal & positive) + positive) ^ positive) | equal | negative
        horizontal_positive = negative | ~(diagonal | positive)
        horizontal_negative = positive & diagonal
        score += bool(horizontal_positive & high) - bool(horizontal_negative & high)
        horizontal_positive = (horizontal_positive << 1) | 1
        horizontal_negative <<= 1
        positive = (horizontal_negative | ~(diagonal | horizontal_positive)) & limit
        negative = (diagonal & horizontal_positive) & limit
    return score


def errors(reference, hypothesis):
    def distance(a, b):
        return edit_distance(a, b) / len(a) * 100 if a else None
    ref, hyp = normalize(reference), normalize(hypothesis)
    return {'wer_percent': distance(ref.split(), hyp.split()), 'cer_percent': distance(ref, hyp)}


def compare_segments(baseline, candidate):
    """Timing differences on aligned equal-text segments, never a ground-truth claim."""
    matcher = difflib.SequenceMatcher(a=[normalize(s['text']) for s in baseline],
                                     b=[normalize(s['text']) for s in candidate], autojunk=False)
    starts, ends = [], []
    for a, b, count in matcher.get_matching_blocks():
        for offset in range(count):
            starts.append(abs(baseline[a+offset]['start'] - candidate[b+offset]['start']))
            ends.append(abs(baseline[a+offset]['end'] - candidate[b+offset]['end']))
    return {'segments_exact': baseline == candidate, 'matched_segments': len(starts),
            'unmatched_baseline_segments': len(baseline) - len(starts),
            'unmatched_candidate_segments': len(candidate) - len(starts),
            'start_difference_seconds_median': statistics.median(starts) if starts else None,
            'start_difference_seconds_max': max(starts) if starts else None,
            'end_difference_seconds_median': statistics.median(ends) if ends else None,
            'end_difference_seconds_max': max(ends) if ends else None,
            'interpretation': 'Output disagreement; baseline is not ground truth.'}


def boundary_issues(segments, midpoint, window=30):
    """Flag overlapping suffix/prefix repeats for audit; never silently rewrite text."""
    nearby = [s for s in segments if s['end'] >= midpoint-window and s['start'] <= midpoint+window]
    issues = []
    for left, right in zip(nearby, nearby[1:]):
        if right['start'] >= left['end']:
            continue
        a, b = normalize(left['text']).split(), normalize(right['text']).split()
        for count in range(min(len(a), len(b), 64), 2, -1):
            if a[-count:] == b[:count]:
                issues.append({'kind': 'overlapping_repeated_phrase', 'words': count,
                               'phrase': ' '.join(b[:count]), 'left_start': left['start'],
                               'right_start': right['start'], 'overlap_seconds': left['end']-right['start']})
                break
    return issues


def gpu_valid(name, profile):
    return ('T4' in name and profile.get('device') == 'cuda'
            and profile.get('compute_type') == 'float16' and profile.get('concurrency') == 2)


def invalid_reasons(jobs, expected, dual=False):
    reasons = []
    if len(jobs) != (2 if dual else 1):
        return ['missing jobs']
    for job in jobs:
        if not job.get('ok') or not gpu_valid(job.get('gpu', ''), job.get('profile', {})):
            reasons.append('failed or unverified GPU')
        if job.get('fingerprint') != expected:
            reasons.append('configuration/audio mismatch')
        if not job.get('files') or not job.get('versions') or not job.get('inference_windows'):
            reasons.append('incomplete artifacts/metrics')
    if dual and not any(max(a[0], b[0]) < min(a[1], b[1])
                        for a in jobs[0].get('inference_windows', [])
                        for b in jobs[1].get('inference_windows', [])):
        reasons.append('no observed inference overlap')
    if len({json.dumps(j.get('versions'), sort_keys=True) for j in jobs}) != 1:
        reasons.append('dependency mismatch')
    return reasons
