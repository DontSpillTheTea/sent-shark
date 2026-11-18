"""
Thread-safe sliding-window rate limiter keyed by client_ip.

In production across multiple ECS Fargate / Cloud Run tasks, this interface
can be swapped to an atomic Redis Lua sorted-set / token-bucket backend
with zero changes to callers.
"""

import ipaddress
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Sequence
from dataclasses import dataclass

from django.conf import settings


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


def _normalize_ip(raw: str) -> str | None:
    """Normalize and validate an IP literal (stripping quotes, brackets, and ports)."""
    candidate = raw.strip().strip("\"'")
    if not candidate:
        return None
    if candidate.startswith("["):
        host, sep, rest = candidate[1:].partition("]")
        if not sep or (rest and not (rest.startswith(":") and rest[1:].isdigit())):
            return None
        candidate = host
    elif candidate.count(":") == 1:
        host, port = candidate.split(":", 1)
        if not port.isdigit():
            return None
        candidate = host
    try:
        return str(ipaddress.ip_address(candidate))
    except ValueError:
        return None


def _is_trusted_proxy(ip_str: str, trusted_entries: Sequence[str]) -> bool:
    """Return True if ip_str matches any configured trusted proxy IP or CIDR."""
    if not ip_str or not trusted_entries:
        return False
    try:
        ip_obj = ipaddress.ip_address(ip_str)
    except ValueError:
        ip_obj = None

    for raw_entry in trusted_entries:
        entry = str(raw_entry).strip()
        if not entry:
            continue
        if ip_obj is not None:
            try:
                if "/" in entry:
                    if ip_obj in ipaddress.ip_network(entry, strict=False):
                        return True
                elif ip_obj == ipaddress.ip_address(entry):
                    return True
                continue
            except ValueError:
                pass
        if ip_str == entry:
            return True
    return False


def get_client_ip(request) -> str:
    """
    Extract client IP safely from REMOTE_ADDR, honoring X-Forwarded-For right-to-left
    only when TRUST_PROXY_HEADERS is enabled and REMOTE_ADDR is a trusted proxy.
    """
    raw_remote = (request.META.get("REMOTE_ADDR") or "").strip()
    remote_ip = _normalize_ip(raw_remote) or raw_remote
    fallback = remote_ip or "unknown"

    if not getattr(settings, "TRUST_PROXY_HEADERS", False):
        return fallback

    raw_trusted = getattr(settings, "TRUSTED_PROXY_IPS", []) or []
    trusted_proxies = (
        [item.strip() for item in raw_trusted.split(",") if item.strip()]
        if isinstance(raw_trusted, str)
        else raw_trusted
    )
    if not raw_remote or not _is_trusted_proxy(fallback, trusted_proxies):
        return fallback

    forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR", "")
    if not forwarded_for:
        return fallback

    for token in reversed(forwarded_for.split(",")):
        if not token.strip():
            continue
        parsed_ip = _normalize_ip(token)
        if parsed_ip is None:
            return fallback
        if _is_trusted_proxy(parsed_ip, trusted_proxies):
            continue
        return parsed_ip

    return fallback


# Singleton used by judge API endpoints; configurable at runtime via reset/reconfigure
rate_limiter = SlidingWindowRateLimiter(
    max_requests=5,
    window_seconds=1.0,
    max_tracked_keys=4096,
)
