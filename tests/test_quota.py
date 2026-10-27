import time

import httpx
import pytest

from quota import QuotaTracker, _parse_duration


@pytest.mark.parametrize("value,seconds", [
    ("1h10m33.6s", 4233.6), ("577ms", 0.577), ("3", 3.0), ("2m", 120.0), ("", 0.0), ("1m26.4s", 86.4),
])
def test_parse_duration(value, seconds):
    assert _parse_duration(value) == pytest.approx(seconds)


def response(status=200, **headers):
    return httpx.Response(status, headers=headers)


def test_rpm_limit_blocks_until_window_passes():
    q = QuotaTracker({"m": 2})
    q.record_request("m")
    assert q.can_use("m")
    q.record_request("m")
    assert not q.can_use("m")
    q._get("m").recent[0] -= 61  # first request now older than a minute
    assert q.can_use("m")


def test_rpm_headroom_reserves_requests():
    q = QuotaTracker({"m": 3})
    q.record_request("m")
    assert q.can_use("m") and not q.can_use("m", rpm_headroom=2)


def test_429_sets_cooldown_from_retry_after():
    q = QuotaTracker({})
    q.update("m", response(429, **{"retry-after": "5"}))
    assert not q.can_use("m") and 4 < q.cooldown_left("m") <= 5


def test_daily_reserve_ratio():
    q = QuotaTracker({})
    q.update("m", response(**{"x-ratelimit-limit-requests": "1000", "x-ratelimit-remaining-requests": "250"}))
    assert q.can_use("m") and not q.can_use("m", keep_day_ratio=0.3)


def test_low_minute_tokens_block_until_reset():
    q = QuotaTracker({})
    q.update("m", response(**{"x-ratelimit-remaining-tokens": "500", "x-ratelimit-reset-tokens": "30s"}))
    assert not q.can_use("m", min_tokens=1500) and q.can_use("m", min_tokens=100)
    q._get("m").tokens_reset_at = time.time() - 1  # window rolled over
    assert q.can_use("m", min_tokens=1500)


def test_summary_shows_remaining_and_waits():
    q = QuotaTracker({})
    q.update("a", response(**{"x-ratelimit-limit-requests": "1000", "x-ratelimit-remaining-requests": "990"}))
    q.update("b", response(429, **{"retry-after": "30", "x-ratelimit-limit-requests": "1000",
                                    "x-ratelimit-remaining-requests": "0"}))
    text = q.summary({"a": "번역", "b": "예비", "c": "없음"})
    assert "번역 990/1000" in text and "예비 대기" in text and "없음" not in text
