from concurrent.futures import ThreadPoolExecutor

import pytest
from django.db import connections
from rest_framework import status
from rest_framework.test import APIClient

from assignment.models import JudgmentRecord, RequestAuditLog
from assignment.rate_limit import rate_limiter


@pytest.mark.django_db(transaction=True)
class TestParallelJudgmentsAndDeduplication:
    """
    Verifies NFR #2 (no duplicate DB rows under 30 parallel judgments & saves)
    and NFR #3 (all requests logged to RequestAuditLog).
    """

    def setup_method(self):
        rate_limiter.reset()

    def teardown_method(self):
        rate_limiter.reset()

    def test_thirty_parallel_judgments_produce_zero_duplicates(self):
        pairs = [
            ("私は猫が大好きだ。", "猫が好きです。"),
            ("A cat is a carnivore.", "A cat is a vegetarian."),
            ("俺が守る！", "私が守ります。"),
        ]

        def worker(idx: int):
            try:
                client = APIClient()
                s1, s2 = pairs[idx % len(pairs)]
                # Distinct client_ip per worker simulates 30 concurrent callers
                # without triggering per-IP 5 req/s rate limiting.
                resp = client.post(
                    "/judge",
                    {"sentence1": s1, "sentence2": s2},
                    format="json",
                    REMOTE_ADDR=f"10.0.1.{idx + 1}",
                )
                return resp.status_code, resp.data
            finally:
                connections.close_all()

        with ThreadPoolExecutor(max_workers=30) as pool:
            results = list(pool.map(worker, range(30)))

        assert len(results) == 30
        for code, data in results:
            assert code == status.HTTP_200_OK
            assert data["label"] in ("ENTAIL", "NO_ENTAIL")
            assert 0.0 <= data["score"] <= 1.0

        # 30 parallel submissions across 3 unique pairs must persist exactly 3 rows
        assert JudgmentRecord.objects.count() == 3
        assert RequestAuditLog.objects.filter(status_code=200).count() == 30

    def test_thirty_parallel_unique_pairs_all_persist_without_locks(self):
        def worker(idx: int):
            try:
                client = APIClient()
                resp = client.post(
                    "/judge",
                    {
                        "sentence1": f"彼は昨日{idx}番の電車に乗った。",
                        "sentence2": f"昨日、彼は{idx}番の電車に乗車した。",
                    },
                    format="json",
                    REMOTE_ADDR=f"10.0.2.{idx + 1}",
                )
                return resp.status_code, resp.data
            finally:
                connections.close_all()

        with ThreadPoolExecutor(max_workers=30) as pool:
            results = list(pool.map(worker, range(30)))

        assert len(results) == 30
        assert all(code == status.HTTP_200_OK for code, _ in results)
        assert JudgmentRecord.objects.count() == 30
        assert RequestAuditLog.objects.filter(status_code=200).count() == 30
