import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from django.db import close_old_connections, connections
from rest_framework import status
from rest_framework.test import APIClient

from assignment.models import JudgmentRecord, RequestAuditLog
from assignment.rate_limit import rate_limiter


def _stub_score_pairs_batch(pairs, tier1_results=None, scoring_mode="mean"):
    from assignment.services.nli_engine import NLIScoreDetail

    return [
        NLIScoreDetail(
            score=0.92,
            label="ENTAIL",
            tier_used="tier2_nli",
            forward_entail=0.93,
            reverse_entail=0.91,
            max_contradiction=0.02,
            forward_contradict=0.02,
            reverse_contradict=0.01,
        )
        for _ in pairs
    ]


@pytest.mark.django_db(transaction=True)
class TestParallelJudgmentsAndDeduplication:
    """
    Verifies NFR #2 (no duplicate DB rows under 30 parallel judgments & saves),
    NFR #3 (all requests logged to RequestAuditLog), and F4 (5 req/s per-IP rate limit
    under concurrent single-IP bursts vs 30 distinct client IPs).
    """

    @pytest.fixture(autouse=True)
    def _mock_nli_engine(self, monkeypatch):
        from assignment.services import nli_engine as nli_mod

        monkeypatch.setattr(nli_mod.nli_engine, "score_pairs_batch", _stub_score_pairs_batch)

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
            close_old_connections()
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
            close_old_connections()
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

    def test_singleflight_coalesces_30_concurrent_cold_cache_requests_into_one_nli_call(
        self, monkeypatch
    ):
        from assignment.services import nli_engine as nli_mod

        call_count = 0
        count_lock = threading.Lock()
        original_fn = nli_mod.nli_engine.score_pairs_batch

        def wrapped_score(pairs, t1_results, scoring_mode="mean"):
            nonlocal call_count
            with count_lock:
                call_count += len(pairs)
            time.sleep(0.05)  # hold in-flight window so all 30 threads overlap
            return original_fn(pairs, t1_results, scoring_mode=scoring_mode)

        monkeypatch.setattr(nli_mod.nli_engine, "score_pairs_batch", wrapped_score)

        def worker(idx: int):
            close_old_connections()
            try:
                client = APIClient()
                resp = client.post(
                    "/judge",
                    {
                        "sentence1": "彼は昨日東京に行った。",
                        "sentence2": "昨日、彼は東京へ足を運んだ。",
                    },
                    format="json",
                    REMOTE_ADDR=f"10.0.3.{idx + 1}",
                )
                return resp.status_code, resp.data
            finally:
                connections.close_all()

        with ThreadPoolExecutor(max_workers=30) as pool:
            results = list(pool.map(worker, range(30)))

        assert all(code == status.HTTP_200_OK for code, _ in results)
        assert JudgmentRecord.objects.count() == 1
        assert call_count == 1

    def test_singleflight_overlapping_bulk_batches_no_deadlock_and_coalesced(self, monkeypatch):
        from assignment.services import nli_engine as nli_mod

        pair_a = ("彼は昨日東京に行った。", "昨日、彼は東京へ足を運んだ。")
        pair_b = ("雨が降っているので、傘を持っていく。", "雨天のため、傘を持参する。")
        pair_c = ("The movie was extremely boring.", "The film was very dull.")

        batches = [
            [
                {"sentence1": pair_a[0], "sentence2": pair_a[1]},
                {"sentence1": pair_b[0], "sentence2": pair_b[1]},
            ],
            [
                {"sentence1": pair_b[0], "sentence2": pair_b[1]},
                {"sentence1": pair_a[0], "sentence2": pair_a[1]},
            ],
            [
                {"sentence1": pair_a[0], "sentence2": pair_a[1]},
                {"sentence1": pair_c[0], "sentence2": pair_c[1]},
            ],
            [
                {"sentence1": pair_c[0], "sentence2": pair_c[1]},
                {"sentence1": pair_b[0], "sentence2": pair_b[1]},
            ],
        ]

        call_count = 0
        count_lock = threading.Lock()
        original_fn = nli_mod.nli_engine.score_pairs_batch

        def wrapped_score(pairs, t1_results, scoring_mode="mean"):
            nonlocal call_count
            with count_lock:
                call_count += len(pairs)
            time.sleep(0.05)
            return original_fn(pairs, t1_results, scoring_mode=scoring_mode)

        monkeypatch.setattr(nli_mod.nli_engine, "score_pairs_batch", wrapped_score)

        def worker(idx: int):
            close_old_connections()
            try:
                client = APIClient()
                resp = client.post(
                    "/judge/bulk",
                    {"pairs": batches[idx % len(batches)]},
                    format="json",
                    REMOTE_ADDR=f"10.0.4.{idx + 1}",
                )
                return resp.status_code, resp.data
            finally:
                connections.close_all()

        with ThreadPoolExecutor(max_workers=12) as pool:
            futures = [pool.submit(worker, i) for i in range(12)]
            results = [fut.result(timeout=15) for fut in futures]

        assert all(code == status.HTTP_200_OK and len(data) == 2 for code, data in results)
        assert JudgmentRecord.objects.count() == 3
        assert call_count == 3

    def test_singleflight_propagates_error_without_hanging_followers(self, monkeypatch):
        from assignment.services import judge_service
        from assignment.services import nli_engine as nli_mod

        original_fn = nli_mod.nli_engine.score_pairs_batch

        def failing_score(pairs, t1_results, scoring_mode="mean"):
            time.sleep(0.05)
            raise RuntimeError("simulated NLI failure")

        monkeypatch.setattr(nli_mod.nli_engine, "score_pairs_batch", failing_score)

        def worker(_idx: int):
            close_old_connections()
            try:
                return judge_service.judge_single(
                    "彼は昨日東京に行った。", "昨日、彼は東京へ足を運んだ。"
                )
            finally:
                connections.close_all()

        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(worker, i) for i in range(5)]
            errors = []
            for fut in futures:
                with pytest.raises(RuntimeError, match="simulated NLI failure"):
                    fut.result(timeout=5)
                errors.append(True)

        assert len(errors) == 5
        assert len(judge_service._INFLIGHT) == 0

        # Subsequent request after recovery must succeed cleanly
        monkeypatch.setattr(nli_mod.nli_engine, "score_pairs_batch", original_fn)
        res = judge_service.judge_single("彼は昨日東京に行った。", "昨日、彼は東京へ足を運んだ。")
        assert res.label == "ENTAIL"

    def test_hot_cache_lru_eviction_and_symmetric_hit(self, monkeypatch):
        from assignment.services import judge_service

        judge_service.reset_cache()
        monkeypatch.setattr(judge_service, "_HOT_CACHE_MAX_ENTRIES", 3)

        pairs = [(f"テスト文H{i}です。", f"テスト文H{i}だ。") for i in range(4)]
        hashes = [judge_service.compute_pair_hash(s1, s2)[0] for s1, s2 in pairs]

        for s1, s2 in pairs[:3]:
            judge_service.judge_single(s1, s2)
        assert len(judge_service._HOT_CACHE) == 3

        # Touch pair 0 in reversed order so pair 1 becomes least recently used
        hit = judge_service.judge_single(pairs[0][1], pairs[0][0])
        assert hit.cached is True

        # Insert pair 3 -> pair 1 must be evicted from _HOT_CACHE while pair 0 remains
        judge_service.judge_single(*pairs[3])
        assert len(judge_service._HOT_CACHE) == 3
        assert hashes[1] not in judge_service._HOT_CACHE
        assert set(judge_service._HOT_CACHE.keys()) == {hashes[0], hashes[2], hashes[3]}
        assert JudgmentRecord.objects.count() == 4

    def test_singleflight_acquire_many_deduplicates_duplicate_keys(self):
        from assignment.services.judge_service import SingleflightGroup

        group = SingleflightGroup()
        hits, owned, awaited = group.acquire_many(["k1", "k1", "k2", "k1"])
        assert hits == {}
        assert list(owned.keys()) == ["k1", "k2"]
        assert awaited == {}
        group.complete_many(owned, {}, error=RuntimeError("cleanup"))

    def test_persist_records_safely_concurrent_duplicate_conflicts_with_barrier(self):
        from assignment.services.judge_service import _persist_records_safely, compute_pair_hash

        shared_pairs = [
            ("共有ペアA1です。", "共有ペアA2です。"),
            ("共有ペアB1です。", "共有ペアB2です。"),
            ("共有ペアC1です。", "共有ペアC2です。"),
        ]
        shared_hashes = [compute_pair_hash(s1, s2)[0] for s1, s2 in shared_pairs]
        barrier = threading.Barrier(30)

        def worker(idx: int):
            close_old_connections()
            try:
                shared_records = [
                    JudgmentRecord(
                        pair_hash=pair_hash,
                        sentence1=s1,
                        sentence2=s2,
                        label=JudgmentRecord.LABEL_ENTAIL,
                        score=0.92,
                        tier_used=JudgmentRecord.TIER_NLI,
                    )
                    for pair_hash, (s1, s2) in zip(shared_hashes, shared_pairs, strict=True)
                ]
                if idx % 2 == 1:
                    shared_records.reverse()

                u_s1, u_s2 = f"競合テスト固有{idx}A。", f"競合テスト固有{idx}B。"
                u_hash, _, _ = compute_pair_hash(u_s1, u_s2)
                unique_rec = JudgmentRecord(
                    pair_hash=u_hash,
                    sentence1=u_s1,
                    sentence2=u_s2,
                    label=JudgmentRecord.LABEL_ENTAIL,
                    score=0.91,
                    tier_used=JudgmentRecord.TIER_NLI,
                )
                batch = [*shared_records, unique_rec]
                expected_hashes = {rec.pair_hash for rec in batch}

                barrier.wait(timeout=10)
                persisted = _persist_records_safely(batch)
                return expected_hashes, set(persisted.keys())
            finally:
                connections.close_all()

        with ThreadPoolExecutor(max_workers=30) as pool:
            futures = [pool.submit(worker, i) for i in range(30)]
            results = [fut.result(timeout=30) for fut in futures]

        assert len(results) == 30
        for expected_hashes, got_hashes in results:
            assert got_hashes == expected_hashes
        assert JudgmentRecord.objects.count() == 33

    def test_service_layer_thirty_thread_judge_single_and_bulk_concurrency(self):
        from assignment.services.judge_service import judge_bulk, judge_single

        shared_pairs = [
            ("サービス共有1Aです。", "サービス共有1Bです。"),
            ("サービス共有2Aです。", "サービス共有2Bです。"),
            ("サービス共有3Aです。", "サービス共有3Bです。"),
        ]
        barrier = threading.Barrier(30)

        def worker(idx: int):
            close_old_connections()
            try:
                shared = shared_pairs[idx % len(shared_pairs)]
                barrier.wait(timeout=10)
                if idx < 15:
                    return [judge_single(*shared)]
                unique = (f"サービス固有{idx}A。", f"サービス固有{idx}B。")
                return judge_bulk([shared, unique])
            finally:
                connections.close_all()

        with ThreadPoolExecutor(max_workers=30) as pool:
            futures = [pool.submit(worker, i) for i in range(30)]
            results = [fut.result(timeout=30) for fut in futures]

        assert len(results) == 30
        for idx, outs in enumerate(results):
            expected_len = 1 if idx < 15 else 2
            assert len(outs) == expected_len
            for out in outs:
                assert out.label in ("ENTAIL", "NO_ENTAIL")
                assert 0.0 <= out.score <= 1.0

        assert JudgmentRecord.objects.count() == 18
        assert RequestAuditLog.objects.count() == 0

    def test_http_thirty_concurrent_distinct_ips_with_barrier_all_200(self):
        pairs = [
            ("私は猫が大好きだ。", "猫が好きです。"),
            ("A cat is a carnivore.", "A cat is a vegetarian."),
            ("俺が守る！", "私が守ります。"),
        ]
        barrier = threading.Barrier(30)

        def worker(idx: int):
            close_old_connections()
            try:
                client = APIClient()
                s1, s2 = pairs[idx % len(pairs)]
                barrier.wait(timeout=10)
                resp = client.post(
                    "/judge",
                    {"sentence1": s1, "sentence2": s2},
                    format="json",
                    REMOTE_ADDR=f"198.51.100.{idx + 1}",
                )
                return resp.status_code, resp.data
            finally:
                connections.close_all()

        with ThreadPoolExecutor(max_workers=30) as pool:
            futures = [pool.submit(worker, i) for i in range(30)]
            results = [fut.result(timeout=30) for fut in futures]

        assert len(results) == 30
        for code, data in results:
            assert code == status.HTTP_200_OK
            assert data["label"] in ("ENTAIL", "NO_ENTAIL")
            assert 0.0 <= data["score"] <= 1.0

        assert JudgmentRecord.objects.count() == 3
        assert RequestAuditLog.objects.count() == 30
        assert RequestAuditLog.objects.filter(status_code=200, rate_limited=False).count() == 30
        logged_ips = set(RequestAuditLog.objects.values_list("client_ip", flat=True))
        assert logged_ips == {f"198.51.100.{i + 1}" for i in range(30)}

    def test_http_thirty_concurrent_single_ip_rate_limit_burst_5_ok_25_429(self):
        barrier = threading.Barrier(30)
        burst_ip = "203.0.113.50"

        def worker(_idx: int):
            close_old_connections()
            try:
                client = APIClient()
                barrier.wait(timeout=10)
                resp = client.post(
                    "/judge",
                    {"sentence1": "私は猫が大好きだ。", "sentence2": "猫が好きです。"},
                    format="json",
                    REMOTE_ADDR=burst_ip,
                )
                return resp.status_code, resp.data
            finally:
                connections.close_all()

        with ThreadPoolExecutor(max_workers=30) as pool:
            futures = [pool.submit(worker, i) for i in range(30)]
            results = [fut.result(timeout=30) for fut in futures]

        codes = [code for code, _ in results]
        assert len(codes) == 30
        assert codes.count(status.HTTP_200_OK) == 5
        assert codes.count(status.HTTP_429_TOO_MANY_REQUESTS) == 25

        assert RequestAuditLog.objects.filter(client_ip=burst_ip).count() == 30
        assert (
            RequestAuditLog.objects.filter(
                client_ip=burst_ip, status_code=200, rate_limited=False
            ).count()
            == 5
        )
        assert (
            RequestAuditLog.objects.filter(
                client_ip=burst_ip, status_code=429, rate_limited=True
            ).count()
            == 25
        )
