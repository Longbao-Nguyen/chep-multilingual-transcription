"""Error classification and normalization for Kaggle resource cleanup."""
from __future__ import annotations

import re


class CleanupError(ValueError):
    """Error raised when remote Kaggle resource cleanup fails."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def cleanup_response(output: str, kind: str, ref: str) -> str:
    """Classify a failed deletion without accepting unrelated failures as 404s."""
    lower = output.lower()
    if re.search(r'\b(401|403|429|5\d\d)\b', lower):
        return 'error'
    if re.search(r'\b404\b', lower):
        return 'missing'
    label = 'notebook' if kind == 'kernels' else 'dataset'
    if 'not found' in lower and (label in lower or kind[:-1] in lower or ref.lower() in lower):
        return 'missing'
    return 'error'


def cleanup_failure(output: str) -> CleanupError:
    """Map Kaggle CLI deletion errors to structured, user-facing CleanupError values."""
    if re.search(r'\b401\b|unauthenticated', output, re.I):
        return CleanupError(
            'authentication',
            'Kaggle authentication expired. Reauthenticate the CLI; Chep will retry cleanup automatically.',
        )
    if re.search(r'\b403\b|forbidden|permission denied', output, re.I):
        return CleanupError(
            'permission',
            'Kaggle denied deletion. Check the signed-in account; Chep will retry cleanup automatically.',
        )
    if re.search(r'\b429\b|too many requests|rate limit', output, re.I):
        return CleanupError(
            'rate_limit',
            'Kaggle is rate limiting requests. Chep will retry cleanup automatically.',
        )
    if re.search(r'\b5\d\d\b', output):
        return CleanupError('server', 'Kaggle server error. Chep will retry cleanup automatically.')
    return CleanupError(
        'unexpected',
        'Kaggle could not remove this resource. Check your account; Chep will retry cleanup automatically.',
    )
