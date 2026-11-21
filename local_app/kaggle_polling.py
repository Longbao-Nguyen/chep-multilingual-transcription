"""Polling, status parsing, and silent-startup timeout helpers for Kaggle sessions."""
from __future__ import annotations

import re
import time
from typing import Any, Callable

from .kaggle_cli import KaggleStopped

DEFAULT_FIRST_LOG_TIMEOUT_SECONDS = 120
DEFAULT_FIRST_LOG_RECONNECT_GRACE_SECONDS = 10
MAX_FIRST_LOG_RETRIES = 2


def kernel_status(output: str) -> str | None:
    """Parse Kaggle CLI kernel status text without treating error text as worker state."""
    match = re.search(r'has status "(?:KernelWorkerStatus\.)?([A-Z_]+)"', output)
    return match.group(1) if match else None


def silent_startup_overdue(
    running_since: float,
    monitor: Any,
    now: float | None = None,
    *,
    timeout_seconds: float = DEFAULT_FIRST_LOG_TIMEOUT_SECONDS,
    grace_seconds: float = DEFAULT_FIRST_LOG_RECONNECT_GRACE_SECONDS,
) -> bool:
    """Return True when a RUNNING notebook has emitted no live log within the timeout window."""
    current = time.time() if now is None else now
    return (
        not monitor.seen.is_set()
        and current - running_since >= timeout_seconds
        and current - monitor.started_at >= grace_seconds
    )


def wait_cancellable(seconds: float, stopped: Callable[[], bool]) -> None:
    """Sleep in short slices so cancellation stops waiting promptly."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if stopped():
            raise KaggleStopped()
        time.sleep(.3)


def wait_for_dataset_ready(
    dataset_ref: str,
    run: Callable[..., tuple[str, int]],
    wait: Callable[[float], None],
    attempts: int = 120,
) -> str:
    """Poll Kaggle until a newly uploaded private Dataset becomes ready, fails, or times out."""
    for _ in range(attempts):
        output, code = run(['datasets', 'status', dataset_ref], check=False)
        state = output.strip().lower()
        if not code and state == 'ready':
            return 'ready'
        if not code and state in ('error', 'failed'):
            return 'failed'
        wait(5)
    return 'timeout'
