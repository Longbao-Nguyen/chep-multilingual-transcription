"""Conservative merge for two overlapping Kaggle transcription shards."""
from __future__ import annotations

import math
import re
import unicodedata


DUAL_THRESHOLD_SECONDS = 21 * 60
OVERLAP_SECONDS = 5.0
WORD = re.compile(r'\w+', re.UNICODE)


def shard_ranges(duration: float) -> list[tuple[float, float]]:
    if not math.isfinite(duration) or duration < DUAL_THRESHOLD_SECONDS:
        raise ValueError('Dual mode requires audio of at least 21 minutes.')
    midpoint = duration / 2
    return [(0.0, midpoint + OVERLAP_SECONDS / 2),
            (midpoint - OVERLAP_SECONDS / 2, duration)]


def _words(text: str):
    return [(unicodedata.normalize('NFC', match.group()).casefold(), match.end())
            for match in WORD.finditer(text)]


def _validated(shard: list[dict], offset: float, stop: float, index: int) -> list[dict]:
    if not isinstance(shard, list):
        raise ValueError('Invalid Kaggle shard segments.')
    result = []
    for segment in shard:
        if not isinstance(segment, dict) or not {'start', 'end', 'text'} <= segment.keys():
            raise ValueError('Invalid Kaggle shard segment.')
        start, end, value = segment['start'], segment['end'], segment['text']
        if (not isinstance(value, str) or not value.strip() or
                not all(type(t) in (int, float) and math.isfinite(t) for t in (start, end)) or
                not 0 <= start <= end <= stop - offset + .001):
            raise ValueError('Invalid Kaggle shard segment.')
        result.append({'start': start + offset, 'end': end + offset,
                       'text': value.strip(), 'shard': index})
    return result


def merge_segments(shards: list[list[dict]], spans: list[tuple[float, float]]):
    """Remove only an exact >=4-word right prefix in overlapping seam segments.

    The lexical score is 1.0 for an exact normalized match; it is a conservative
    heuristic, not a calibrated probability. Ambiguous or unmatched text is kept.
    """
    if len(shards) != 2 or len(spans) != 2:
        raise ValueError('Both Kaggle shards are required.')
    left = _validated(shards[0], *spans[0], 0)
    right = _validated(shards[1], *spans[1], 1)
    overlap_start, overlap_end = spans[1][0], spans[0][1]
    audit = []
    merged_right = []
    for item in right:
        if item['start'] >= overlap_end or item['end'] <= overlap_start:
            merged_right.append(item)
            continue
        right_words = _words(item['text'])
        candidates = []
        for old in left:
            if (old['start'] > item['start'] or
                    min(old['end'], item['end'], overlap_end) -
                    max(old['start'], item['start'], overlap_start) < 1.0):
                continue
            left_words = _words(old['text'])
            for count in range(min(len(left_words), len(right_words), 64), 3, -1):
                if [w for w, _ in left_words[-count:]] == [w for w, _ in right_words[:count]]:
                    candidates.append((count, old, right_words[count - 1][1]))
                    break
        candidates.sort(key=lambda match: match[0], reverse=True)
        if not candidates or (len(candidates) > 1 and candidates[0][0] == candidates[1][0]):
            merged_right.append(item)
            continue
        count, old, end_position = candidates[0]
        lexical_confidence = 1.0  # Every compared normalized word matched exactly.
        if lexical_confidence <= .90:
            merged_right.append(item)
            continue
        remaining = item['text'][end_position:].lstrip(' \t,.;:!?…-–—')
        audit.append({'left_start': old['start'], 'right_start': item['start'],
                      'removed_words': count, 'lexical_confidence': lexical_confidence,
                      'decision': 'remove_exact_overlapping_right_prefix'})
        if remaining:
            merged_right.append({**item, 'text': remaining})
    merged = sorted(left + merged_right, key=lambda item: (item['start'], item['end'], item['shard']))
    return [{key: value for key, value in item.items() if key != 'shard'} for item in merged], audit
