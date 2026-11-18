import time
import uuid

import pytest
from rest_framework import status
from rest_framework.test import APIClient

from assignment.models import RequestAuditLog
from assignment.rate_limit import get_client_ip, rate_limiter


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

    def test_spoofed_x_forwarded_for_ignored_when_trust_proxy_disabled(self, settings):
        assert getattr(settings, "TRUST_PROXY_HEADERS", False) is False
        payload = {"sentence1": "私は猫が大好きだ。", "sentence2": "猫が好きです。"}
        client_ip = "198.51.100.50"

        for idx in range(5):
            resp = self.client.post(
                "/judge",
                payload,
                format="json",
                REMOTE_ADDR=client_ip,
                HTTP_X_FORWARDED_FOR=f"203.0.113.{idx + 1}",
            )
            assert resp.status_code == status.HTTP_200_OK

        blocked = self.client.post(
            "/judge",
            payload,
            format="json",
            REMOTE_ADDR=client_ip,
            HTTP_X_FORWARDED_FOR="203.0.113.99",
        )
        assert blocked.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        assert (
            RequestAuditLog.objects.filter(
                client_ip=client_ip, rate_limited=True, status_code=429
            ).count()
            == 1
        )
        assert (
            RequestAuditLog.objects.filter(
                client_ip=client_ip, rate_limited=False, status_code=200
            ).count()
            == 5
        )

    def test_trusted_proxy_hops_extract_rightmost_untrusted_client_ip(self, settings):
        settings.TRUST_PROXY_HEADERS = True
        settings.TRUSTED_PROXY_IPS = ["10.0.0.1", "10.0.0.2"]
        payload = {"sentence1": "私は猫が大好きだ。", "sentence2": "猫が好きです。"}

        # Trusted proxy with spoofed left-most token + real client + trusted hop
        for _ in range(5):
            resp = self.client.post(
                "/judge",
                payload,
                format="json",
                REMOTE_ADDR="10.0.0.1",
                HTTP_X_FORWARDED_FOR="spoofed, 198.51.100.7, 10.0.0.1",
            )
            assert resp.status_code == status.HTTP_200_OK

        blocked = self.client.post(
            "/judge",
            payload,
            format="json",
            REMOTE_ADDR="10.0.0.1",
            HTTP_X_FORWARDED_FOR="spoofed, 198.51.100.7, 10.0.0.1",
        )
        assert blocked.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        assert (
            RequestAuditLog.objects.filter(
                client_ip="198.51.100.7", rate_limited=True, status_code=429
            ).count()
            == 1
        )

        # Multi-proxy chain picks rightmost untrusted IP (198.51.100.8), not left-most (1.2.3.4)
        for _ in range(5):
            resp = self.client.post(
                "/judge",
                payload,
                format="json",
                REMOTE_ADDR="10.0.0.1",
                HTTP_X_FORWARDED_FOR="1.2.3.4, 198.51.100.8, 10.0.0.2, 10.0.0.1",
            )
            assert resp.status_code == status.HTTP_200_OK

        blocked_chain = self.client.post(
            "/judge",
            payload,
            format="json",
            REMOTE_ADDR="10.0.0.1",
            HTTP_X_FORWARDED_FOR="1.2.3.4, 198.51.100.8, 10.0.0.2, 10.0.0.1",
        )
        assert blocked_chain.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        assert (
            RequestAuditLog.objects.filter(client_ip="198.51.100.8", status_code=429).count() == 1
        )
        assert RequestAuditLog.objects.filter(client_ip="1.2.3.4").count() == 0

        # Malformed rightmost untrusted token falls back to REMOTE_ADDR (not left-most)
        class _DummyReq:
            def __init__(self, remote_addr: str, xff: str):
                self.META = {"REMOTE_ADDR": remote_addr, "HTTP_X_FORWARDED_FOR": xff}

        assert get_client_ip(_DummyReq("10.0.0.1", "1.2.3.4, malformed, 10.0.0.1")) == "10.0.0.1"
        assert get_client_ip(_DummyReq("10.0.0.1", "10.0.0.2, 10.0.0.1")) == "10.0.0.1"

        # Untrusted direct REMOTE_ADDR sending X-Forwarded-For must ignore XFF and track REMOTE_ADDR
        for _ in range(5):
            untrusted_resp = self.client.post(
                "/judge",
                payload,
                format="json",
                REMOTE_ADDR="203.0.113.9",
                HTTP_X_FORWARDED_FOR="198.51.100.7, 10.0.0.1",
            )
            assert untrusted_resp.status_code == status.HTTP_200_OK

        untrusted_blocked = self.client.post(
            "/judge",
            payload,
            format="json",
            REMOTE_ADDR="203.0.113.9",
            HTTP_X_FORWARDED_FOR="198.51.100.7, 10.0.0.1",
        )
        assert untrusted_blocked.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        assert (
            RequestAuditLog.objects.filter(
                client_ip="203.0.113.9", rate_limited=True, status_code=429
            ).count()
            == 1
        )

    def test_malformed_json_and_method_not_allowed_are_audited_without_sentence_content(
        self, monkeypatch
    ):
        from assignment import views as views_mod

        r_parse = self.client.post(
            "/judge",
            data=b"{bad json SECRET_SENTENCE_LEAK",
            content_type="application/json",
            REMOTE_ADDR="192.0.2.210",
        )
        assert r_parse.status_code == status.HTTP_400_BAD_REQUEST

        r_method = self.client.get("/judge", REMOTE_ADDR="192.0.2.211")
        assert r_method.status_code == status.HTTP_405_METHOD_NOT_ALLOWED

        r_not_acc = self.client.post(
            "/judge",
            {"sentence1": "SENSITIVE_ALPHA", "sentence2": "SENSITIVE_BETA"},
            format="json",
            REMOTE_ADDR="192.0.2.212",
            HTTP_ACCEPT="application/xml",
        )
        assert r_not_acc.status_code == status.HTTP_406_NOT_ACCEPTABLE

        def failing_judge_single(_s1: str, _s2: str):
            raise RuntimeError("boom SENSITIVE_EXC")

        monkeypatch.setattr(views_mod, "judge_single", failing_judge_single)
        client_500 = APIClient(raise_request_exception=False)
        r_500 = client_500.post(
            "/judge",
            {"sentence1": "SENSITIVE_500_A", "sentence2": "SENSITIVE_500_B"},
            format="json",
            REMOTE_ADDR="192.0.2.213",
        )
        assert r_500.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

        assert (
            RequestAuditLog.objects.filter(
                client_ip="192.0.2.210", method="POST", path="/judge", status_code=400
            ).count()
            == 1
        )
        assert (
            RequestAuditLog.objects.filter(
                client_ip="192.0.2.211", method="GET", path="/judge", status_code=405
            ).count()
            == 1
        )
        assert (
            RequestAuditLog.objects.filter(
                client_ip="192.0.2.212", method="POST", path="/judge", status_code=406
            ).count()
            == 1
        )
        assert (
            RequestAuditLog.objects.filter(
                client_ip="192.0.2.213", method="POST", path="/judge", status_code=500
            ).count()
            == 1
        )

        # Ensure RequestAuditLog schema and rows never store sentence content
        audit_fields = {f.name for f in RequestAuditLog._meta.get_fields()}
        assert not (audit_fields & {"sentence1", "sentence2", "sentence", "text", "payload"})
        serialized_rows = " ".join(
            f"{row.client_ip} {row.method} {row.path} {row.status_code}"
            for row in RequestAuditLog.objects.all()
        )
        for sensitive in (
            "SECRET_SENTENCE_LEAK",
            "SENSITIVE_ALPHA",
            "SENSITIVE_BETA",
            "SENSITIVE_500_A",
            "SENSITIVE_500_B",
        ):
            assert sensitive not in serialized_rows

    def test_rate_limit_and_request_id_headers_on_all_judge_endpoints(self):
        judge_payload = {"sentence1": "私は猫が大好きだ。", "sentence2": "猫が好きです。"}
        bulk_payload = {"pairs": [judge_payload]}

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

    @pytest.mark.parametrize("bad_id", ["bad\tid", "日本語id", "bad\r\nid"])
    def test_invalid_request_id_replaced_with_uuid4_including_400(self, bad_id: str):
        r_ok = self.client.post(
            "/judge",
            {"sentence1": "私は猫が大好きだ。", "sentence2": "猫が好きです。"},
            format="json",
            REMOTE_ADDR="192.0.2.150",
            HTTP_X_REQUEST_ID=bad_id,
        )
        assert r_ok.status_code == status.HTTP_200_OK
        req_id_ok = r_ok.headers["X-Request-Id"]
        assert req_id_ok != bad_id
        parsed_ok = uuid.UUID(req_id_ok)
        assert parsed_ok.version == 4
        assert str(parsed_ok) == req_id_ok

        r_bad = self.client.post(
            "/judge",
            {"sentence1": "", "sentence2": "猫が好きです。"},
            format="json",
            REMOTE_ADDR="192.0.2.150",
            HTTP_X_REQUEST_ID=bad_id,
        )
        assert r_bad.status_code == status.HTTP_400_BAD_REQUEST
        req_id_bad = r_bad.headers["X-Request-Id"]
        assert req_id_bad != bad_id
        parsed_bad = uuid.UUID(req_id_bad)
        assert parsed_bad.version == 4
        assert str(parsed_bad) == req_id_bad

    def test_request_id_present_on_parse_and_initial_errors(self):
        r_parse = self.client.post(
            "/judge",
            data=b"{bad json",
            content_type="application/json",
            REMOTE_ADDR="192.0.2.160",
            HTTP_X_REQUEST_ID="valid-req-400",
        )
        assert r_parse.status_code == status.HTTP_400_BAD_REQUEST
        assert r_parse.headers["X-Request-Id"] == "valid-req-400"

        r_not_acc = self.client.post(
            "/judge",
            {"sentence1": "私は猫が大好きだ。", "sentence2": "猫が好きです。"},
            format="json",
            REMOTE_ADDR="192.0.2.160",
            HTTP_X_REQUEST_ID="valid-req-406",
            HTTP_ACCEPT="application/xml",
        )
        assert r_not_acc.status_code == status.HTTP_406_NOT_ACCEPTABLE
        assert r_not_acc.headers["X-Request-Id"] == "valid-req-406"


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
