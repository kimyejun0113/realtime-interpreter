"""Tracks Groq free-tier limits per model from response headers so we can back off before hitting 429."""
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import httpx


def _parse_duration(value: str) -> float:
    """'1h10m33.6s' / '577ms' / '3' -> seconds."""
    if not value:
        return 0.0
    if re.fullmatch(r"[\d.]+", value):
        return float(value)
    total = 0.0
    for num, unit in re.findall(r"([\d.]+)(ms|h|m|s)", value):
        total += float(num) * {"h": 3600, "m": 60, "s": 1, "ms": 0.001}[unit]
    return total


@dataclass
class ModelUsage:
    rpm_limit: int  # requests/minute; Groq doesn't send it in headers, so it comes from config
    day_limit: int | None = None
    day_remaining: int | None = None
    tokens_remaining: int | None = None  # per minute
    tokens_reset_at: float = 0.0
    cooldown_until: float = 0.0
    recent: deque = field(default_factory=deque)  # timestamps of requests in the last minute


class QuotaTracker:
    def __init__(self, rpm_limits: dict[str, int]):
        self._rpm_limits = rpm_limits
        self._models: dict[str, ModelUsage] = {}
        self._lock = threading.Lock()

    def _get(self, model: str) -> ModelUsage:
        if model not in self._models:
            self._models[model] = ModelUsage(rpm_limit=self._rpm_limits.get(model, 30))
        return self._models[model]

    def can_use(self, model: str, min_tokens: int = 0, keep_day_ratio: float = 0.0, rpm_headroom: int = 0) -> bool:
        """True if a request now is unlikely to be rate-limited.

        keep_day_ratio: refuse once the daily remaining share drops below this (reserve for finals).
        rpm_headroom: leave this many requests/minute unused (reserve for finals).
        """
        now = time.time()
        with self._lock:
            u = self._get(model)
            if now < u.cooldown_until:
                return False
            while u.recent and now - u.recent[0] > 60:
                u.recent.popleft()
            if len(u.recent) >= u.rpm_limit - rpm_headroom:
                return False
            if u.tokens_remaining is not None and now < u.tokens_reset_at and u.tokens_remaining < min_tokens:
                return False
            if u.day_limit and u.day_remaining is not None and u.day_remaining < u.day_limit * keep_day_ratio:
                return False
            return True

    def record_request(self, model: str) -> None:
        with self._lock:
            self._get(model).recent.append(time.time())

    def update(self, model: str, resp: httpx.Response) -> None:
        h = resp.headers
        now = time.time()
        with self._lock:
            u = self._get(model)
            if "x-ratelimit-limit-requests" in h:
                u.day_limit = int(h["x-ratelimit-limit-requests"])
            if "x-ratelimit-remaining-requests" in h:
                u.day_remaining = int(h["x-ratelimit-remaining-requests"])
            if "x-ratelimit-remaining-tokens" in h:
                u.tokens_remaining = int(h["x-ratelimit-remaining-tokens"])
                u.tokens_reset_at = now + _parse_duration(h.get("x-ratelimit-reset-tokens", "60s"))
            if resp.status_code == 429:
                u.cooldown_until = now + max(2.0, _parse_duration(h.get("retry-after", "10")))

    def cooldown_left(self, model: str) -> float:
        with self._lock:
            return max(0.0, self._get(model).cooldown_until - time.time())

    def summary(self, labels: dict[str, str]) -> str:
        parts = []
        with self._lock:
            for model, label in labels.items():
                u = self._models.get(model)
                if u is None or u.day_remaining is None:
                    continue
                left = max(0.0, u.cooldown_until - time.time())
                state = f"대기 {left:.0f}s" if left > 0 else f"{u.day_remaining}/{u.day_limit}"
                parts.append(f"{label} {state}")
        return " · ".join(parts)
