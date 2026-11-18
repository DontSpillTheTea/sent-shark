"""
Thread-safe sliding-window rate limiter keyed by client_ip.

In production across multiple ECS Fargate / Cloud Run tasks, this interface
can be swapped to an atomic Redis Lua sorted-set / token-bucket backend
with zero changes to callers.
"""

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    remaining: int
    retry_after: float


class SlidingWindowRateLimiter:
    def __init__(self, max_requests: int = 5, window_seconds: float = 1.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        self._windows: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str, now: float | None = None) -> RateLimitDecision:
        current = time.monotonic() if now is None else now
        cutoff = current - self.window_seconds

        with self._lock:
            dq = self._windows[key]
            while dq and dq[0] <= cutoff:
                dq.popleft()

            if len(dq) >= self.max_requests:
                oldest = dq[0]
                retry_after = max(0.01, (oldest + self.window_seconds) - current)
                return RateLimitDecision(allowed=False, remaining=0, retry_after=retry_after)

            dq.append(current)
            remaining = max(0, self.max_requests - len(dq))
            return RateLimitDecision(allowed=True, remaining=remaining, retry_after=0.0)

    def reset(self, key: str | None = None) -> None:
        with self._lock:
            if key is None:
                self._windows.clear()
            else:
                self._windows.pop(key, None)


def get_client_ip(request) -> str:
    """Extract client IP, preferring X-Forwarded-For left-most entry."""
    forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR", "")
    if forwarded_for:
        first_ip = forwarded_for.split(",")[0].strip()
        if first_ip:
            return first_ip
    return request.META.get("REMOTE_ADDR", "unknown") or "unknown"


# Singleton used by judge API endpoints; configurable at runtime via reset/reconfigure
rate_limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=1.0)
