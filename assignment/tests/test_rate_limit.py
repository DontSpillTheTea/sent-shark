import time

import pytest
from rest_framework import status
from rest_framework.test import APIClient

from assignment.models import RequestAuditLog
from assignment.rate_limit import rate_limiter


@pytest.mark.django_db
class TestRateLimit:
    def setup_method(self):
        self.client = APIClient()
        rate_limiter.reset()

    def teardown_method(self):
        rate_limiter.reset()

    def test_five_requests_per_second_allowed_then_429(self):
        payload = {"sentence1": "私は猫が大好きだ。", "sentence2": "猫が好きです。"}
        client_ip = "203.0.113.10"

        for _ in range(5):
            resp = self.client.post("/judge", payload, format="json", REMOTE_ADDR=client_ip)
            assert resp.status_code == status.HTTP_200_OK

        # 6th request within the same 1-second window must return 429
        blocked = self.client.post("/judge", payload, format="json", REMOTE_ADDR=client_ip)
        assert blocked.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        assert "Retry-After" in blocked.headers

        # Independent client IP must still succeed
        other_ip_resp = self.client.post(
            "/judge", payload, format="json", REMOTE_ADDR="203.0.113.11"
        )
        assert other_ip_resp.status_code == status.HTTP_200_OK

        # Audit log recorded the rate-limited attempt
        assert (
            RequestAuditLog.objects.filter(
                client_ip=client_ip, rate_limited=True, status_code=429
            ).count()
            == 1
        )

        # After sliding window expires (>1s), client_ip can send again
        time.sleep(1.05)
        recovered = self.client.post("/judge", payload, format="json", REMOTE_ADDR=client_ip)
        assert recovered.status_code == status.HTTP_200_OK

    def test_x_forwarded_for_respected_per_client_ip(self):
        payload = {"sentence1": "私は猫が大好きだ。", "sentence2": "猫が好きです。"}
        for _ in range(5):
            resp = self.client.post(
                "/judge",
                payload,
                format="json",
                HTTP_X_FORWARDED_FOR="198.51.100.7, 10.0.0.1",
            )
            assert resp.status_code == status.HTTP_200_OK

        blocked = self.client.post(
            "/judge",
            payload,
            format="json",
            HTTP_X_FORWARDED_FOR="198.51.100.7, 10.0.0.1",
        )
        assert blocked.status_code == status.HTTP_429_TOO_MANY_REQUESTS

    def test_rate_limit_and_request_id_headers_on_all_judge_endpoints(self):
        judge_payload = {"sentence1": "私は猫が大好きだ。", "sentence2": "猫が好きです。"}
        bulk_payload = {"pairs": [judge_payload]}
        eval_payload = {
            "pairs": [{**judge_payload, "expected_label": "ENTAIL"}],
        }

        def assert_std_headers(resp, expected_remaining: str, expected_req_id: str | None = None):
            assert resp.headers["X-RateLimit-Limit"] == "5"
            assert resp.headers["X-RateLimit-Remaining"] == expected_remaining
            assert resp.headers.get("X-Request-Id")
            if expected_req_id is not None:
                assert resp.headers["X-Request-Id"] == expected_req_id

        # 1. POST /judge -> 200, 400, 429
        r_ok = self.client.post(
            "/judge",
            judge_payload,
            format="json",
            REMOTE_ADDR="192.0.2.101",
            HTTP_X_REQUEST_ID="req-judge-1",
        )
        assert r_ok.status_code == status.HTTP_200_OK
        assert_std_headers(r_ok, "4", "req-judge-1")

        r_bad = self.client.post(
            "/judge",
            {"sentence1": "", "sentence2": "猫が好きです。"},
            format="json",
            REMOTE_ADDR="192.0.2.101",
        )
        assert r_bad.status_code == status.HTTP_400_BAD_REQUEST
        assert_std_headers(r_bad, "3")

        for _ in range(3):
            self.client.post("/judge", judge_payload, format="json", REMOTE_ADDR="192.0.2.101")
        r_429 = self.client.post("/judge", judge_payload, format="json", REMOTE_ADDR="192.0.2.101")
        assert r_429.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        assert_std_headers(r_429, "0")
        assert int(r_429.headers["Retry-After"]) >= 1

        # 2. POST /judge/bulk -> 200, 400, 429
        r_bulk_ok = self.client.post(
            "/judge/bulk", bulk_payload, format="json", REMOTE_ADDR="192.0.2.102"
        )
        assert r_bulk_ok.status_code == status.HTTP_200_OK
        assert_std_headers(r_bulk_ok, "4")

        r_bulk_bad = self.client.post(
            "/judge/bulk", {"pairs": []}, format="json", REMOTE_ADDR="192.0.2.102"
        )
        assert r_bulk_bad.status_code == status.HTTP_400_BAD_REQUEST
        assert_std_headers(r_bulk_bad, "3")

        for _ in range(3):
            self.client.post("/judge/bulk", bulk_payload, format="json", REMOTE_ADDR="192.0.2.102")
        r_bulk_429 = self.client.post(
            "/judge/bulk", bulk_payload, format="json", REMOTE_ADDR="192.0.2.102"
        )
        assert r_bulk_429.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        assert_std_headers(r_bulk_429, "0")
        assert int(r_bulk_429.headers["Retry-After"]) >= 1

        # 3. GET /judge/stats -> 200, 429
        for expected_rem in ("4", "3", "2", "1", "0"):
            r_stats = self.client.get("/judge/stats", REMOTE_ADDR="192.0.2.103")
            assert r_stats.status_code == status.HTTP_200_OK
            assert_std_headers(r_stats, expected_rem)
        r_stats_429 = self.client.get("/judge/stats", REMOTE_ADDR="192.0.2.103")
        assert r_stats_429.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        assert_std_headers(r_stats_429, "0")
        assert int(r_stats_429.headers["Retry-After"]) >= 1

        # 4. POST /judge/evaluate -> 200, 400, 429
        r_eval_ok = self.client.post(
            "/judge/evaluate", eval_payload, format="json", REMOTE_ADDR="192.0.2.104"
        )
        assert r_eval_ok.status_code == status.HTTP_200_OK
        assert_std_headers(r_eval_ok, "4")

        r_eval_bad = self.client.post(
            "/judge/evaluate", {"pairs": []}, format="json", REMOTE_ADDR="192.0.2.104"
        )
        assert r_eval_bad.status_code == status.HTTP_400_BAD_REQUEST
        assert_std_headers(r_eval_bad, "3")

        for _ in range(3):
            self.client.post(
                "/judge/evaluate", eval_payload, format="json", REMOTE_ADDR="192.0.2.104"
            )
        r_eval_429 = self.client.post(
            "/judge/evaluate", eval_payload, format="json", REMOTE_ADDR="192.0.2.104"
        )
        assert r_eval_429.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        assert_std_headers(r_eval_429, "0")
        assert int(r_eval_429.headers["Retry-After"]) >= 1


def test_rate_limiter_evicts_oldest_ip_keys_when_capacity_exceeded():
    from assignment.rate_limit import SlidingWindowRateLimiter

    limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=1.0, max_tracked_keys=10)
    for i in range(25):
        limiter.check(f"192.0.2.{i}")
    assert len(limiter._windows) <= 10


def test_rate_limiter_lru_eviction_preserves_recently_active_keys_and_decision_fields():
    from assignment.rate_limit import SlidingWindowRateLimiter

    limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=1.0, max_tracked_keys=3)
    d1 = limiter.check("ip-1", now=10.0)
    assert d1.allowed is True
    assert d1.limit == 5
    assert d1.remaining == 4
    assert d1.retry_after == 0.0

    limiter.check("ip-2", now=10.0)
    limiter.check("ip-3", now=10.0)

    # Touch ip-1 so ip-2 becomes least recently used
    d1_again = limiter.check("ip-1", now=10.1)
    assert d1_again.remaining == 3

    # Adding ip-4 must evict ip-2 while retaining ip-3, ip-1, ip-4
    limiter.check("ip-4", now=10.2)
    assert len(limiter._windows) == 3
    assert "ip-2" not in limiter._windows
    assert set(limiter._windows.keys()) == {"ip-3", "ip-1", "ip-4"}

    # Exhaust ip-4 and check denied decision fields
    for idx in range(4):
        limiter.check("ip-4", now=10.2 + idx * 0.01)
    denied = limiter.check("ip-4", now=10.3)
    assert denied.allowed is False
    assert denied.limit == 5
    assert denied.remaining == 0
    assert denied.retry_after > 0.0
