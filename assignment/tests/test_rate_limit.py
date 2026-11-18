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
