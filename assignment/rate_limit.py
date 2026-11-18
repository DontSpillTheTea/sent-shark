"""
Thread-safe sliding-window rate limiter keyed by client_ip.

In production across multiple ECS Fargate / Cloud Run tasks, this interface
can be swapped to an atomic Redis Lua sorted-set / token-bucket backend
with zero changes to callers.
"""

import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    limit: int
    remaining: int
    retry_after: float


class SlidingWindowRateLimiter:
    """Thread-safe sliding-window rate limiter with bounded LRU key tracking."""

    def __init__(
        self,
        max_requests: int = 5,
        window_seconds: float = 1.0,
        max_tracked_keys: int = 4096,
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.max_tracked_keys = max_tracked_keys
        self._lock = threading.Lock()
        self._windows: OrderedDict[str, deque[float]] = OrderedDict()

    def check(self, key: str, now: float | None = None) -> RateLimitDecision:
        current = time.monotonic() if now is None else now
        cutoff = current - self.window_seconds

        with self._lock:
            if key in self._windows:
                dq = self._windows[key]
                self._windows.move_to_end(key)
            elif self.max_tracked_keys > 0:
                while len(self._windows) >= self.max_tracked_keys:
                    self._windows.popitem(last=False)
                dq = deque()
                self._windows[key] = dq
            else:
                self._windows.clear()
                dq = deque()

            while dq and dq[0] <= cutoff:
                dq.popleft()

            if len(dq) >= self.max_requests:
                if dq:
                    oldest = dq[0]
                    retry_after = max(0.01, (oldest + self.window_seconds) - current)
                else:
                    retry_after = max(0.01, self.window_seconds)
                return RateLimitDecision(
                    allowed=False,
                    limit=self.max_requests,
                    remaining=0,
                    retry_after=retry_after,
                )

            dq.append(current)
            remaining = max(0, self.max_requests - len(dq))
            return RateLimitDecision(
                allowed=True,
                limit=self.max_requests,
                remaining=remaining,
                retry_after=0.0,
            )

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
rate_limiter = SlidingWindowRateLimiter(
    max_requests=5,
    window_seconds=1.0,
    max_tracked_keys=4096,
)
