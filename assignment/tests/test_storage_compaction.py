import pytest

from assignment.models import JudgmentRecord, RequestAuditLog
from assignment.services import judge_service
from assignment.services.judge_service import judge_bulk, judge_single, prune_storage_if_needed
from assignment.views import _record_audit


@pytest.mark.django_db
def test_judgment_record_has_no_redundant_normalized_columns():
    field_names = {f.name for f in JudgmentRecord._meta.get_fields()}
    assert "normalized_s1" not in field_names
    assert "normalized_s2" not in field_names
    assert "hit_count" in field_names


@pytest.mark.django_db
def test_bounded_storage_prunes_oldest_records(settings):
    for i in range(8):
        judge_single(f"テスト文A{i}です。", f"テスト文A{i}だ。")
    assert JudgmentRecord.objects.count() == 8

    settings.MAX_STORED_JUDGMENTS = 5
    prune_storage_if_needed()
    assert JudgmentRecord.objects.count() == 5
    remaining = set(JudgmentRecord.objects.values_list("sentence1", flat=True))
    assert remaining == {f"テスト文A{i}です。" for i in range(3, 8)}


@pytest.mark.django_db
def test_bounded_audit_logs_prunes_oldest_records(settings):
    for i in range(7):
        RequestAuditLog.objects.create(
            client_ip=f"10.0.0.{i}",
            method="POST",
            path="/judge",
            status_code=200,
        )
    assert RequestAuditLog.objects.count() == 7

    settings.MAX_AUDIT_LOGS = 4
    prune_storage_if_needed()
    assert RequestAuditLog.objects.count() == 4
    remaining_ips = set(RequestAuditLog.objects.values_list("client_ip", flat=True))
    assert remaining_ips == {f"10.0.0.{i}" for i in range(3, 7)}


@pytest.mark.django_db
def test_prune_storage_scoped_flags(settings):
    for i in range(6):
        JudgmentRecord.objects.create(
            pair_hash=f"hash-{i}",
            sentence1=f"文{i}です。",
            sentence2=f"文{i}だ。",
            label=JudgmentRecord.LABEL_ENTAIL,
            score=0.95,
            tier_used=JudgmentRecord.TIER_RULE,
        )
        RequestAuditLog.objects.create(
            client_ip=f"10.0.1.{i}",
            method="GET",
            path="/judge/stats",
            status_code=200,
        )

    settings.MAX_STORED_JUDGMENTS = 3
    settings.MAX_AUDIT_LOGS = 3

    prune_storage_if_needed(judgments=True, audit_logs=False)
    assert JudgmentRecord.objects.count() == 3
    assert RequestAuditLog.objects.count() == 6

    prune_storage_if_needed(judgments=False, audit_logs=True)
    assert JudgmentRecord.objects.count() == 3
    assert RequestAuditLog.objects.count() == 3


@pytest.mark.django_db
def test_prune_storage_zero_or_negative_limit_deletes_all(settings):
    for i in range(3):
        JudgmentRecord.objects.create(
            pair_hash=f"zero-hash-{i}",
            sentence1=f"文Z{i}です。",
            sentence2=f"文Z{i}だ。",
            label=JudgmentRecord.LABEL_ENTAIL,
            score=0.95,
            tier_used=JudgmentRecord.TIER_RULE,
        )
        RequestAuditLog.objects.create(
            client_ip=f"10.0.2.{i}",
            method="POST",
            path="/judge",
            status_code=200,
        )

    settings.MAX_STORED_JUDGMENTS = 0
    settings.MAX_AUDIT_LOGS = -1
    prune_storage_if_needed()

    assert JudgmentRecord.objects.count() == 0
    assert RequestAuditLog.objects.count() == 0


@pytest.mark.django_db
def test_persist_and_audit_prune_only_own_table(settings, rf):
    for i in range(4):
        JudgmentRecord.objects.create(
            pair_hash=f"scope-hash-{i}",
            sentence1=f"文S{i}です。",
            sentence2=f"文S{i}だ。",
            label=JudgmentRecord.LABEL_ENTAIL,
            score=0.95,
            tier_used=JudgmentRecord.TIER_RULE,
        )
        RequestAuditLog.objects.create(
            client_ip=f"10.0.3.{i}",
            method="POST",
            path="/judge",
            status_code=200,
        )

    settings.MAX_STORED_JUDGMENTS = 2
    settings.MAX_AUDIT_LOGS = 2

    # Persisting a judgment should prune JudgmentRecord only, not RequestAuditLog.
    judge_single("テスト新文Sです。", "テスト新文Sだ。")
    assert JudgmentRecord.objects.count() == 2
    assert RequestAuditLog.objects.count() == 4

    # Re-exceed JudgmentRecord limit, then record an audit log.
    for i in range(2):
        JudgmentRecord.objects.create(
            pair_hash=f"extra-hash-{i}",
            sentence1=f"追加文{i}です。",
            sentence2=f"追加文{i}だ。",
            label=JudgmentRecord.LABEL_ENTAIL,
            score=0.95,
            tier_used=JudgmentRecord.TIER_RULE,
        )
    assert JudgmentRecord.objects.count() == 4

    _record_audit(rf.get("/judge/stats"), client_ip="127.0.0.1", status_code=200, started_at=0.0)
    assert RequestAuditLog.objects.count() == 2
    assert JudgmentRecord.objects.count() == 4


def test_storage_limits_default_settings(settings):
    assert settings.MAX_STORED_JUDGMENTS == 10000
    assert settings.MAX_AUDIT_LOGS == 50000


@pytest.mark.django_db
def test_zero_max_stored_judgments_leaves_db_and_hot_cache_empty(settings):
    settings.MAX_STORED_JUDGMENTS = 0

    first = judge_single("テストゼロ文です。", "テストゼロ文だ。")
    assert first.cached is False
    assert JudgmentRecord.objects.count() == 0
    assert len(judge_service._HOT_CACHE) == 0

    second = judge_single("テストゼロ文です。", "テストゼロ文だ。")
    assert second.cached is False
    assert JudgmentRecord.objects.count() == 0
    assert len(judge_service._HOT_CACHE) == 0


@pytest.mark.django_db
def test_bulk_exceeding_max_stored_judgments_evicts_pruned_from_hot_cache(settings):
    settings.MAX_STORED_JUDGMENTS = 1
    pairs = [(f"テスト一括{i}です。", f"テスト一括{i}だ。") for i in range(3)]

    outputs = judge_bulk(pairs)
    assert len(outputs) == 3
    assert all(out.cached is False for out in outputs)
    assert JudgmentRecord.objects.count() == 1

    surviving_hashes = set(JudgmentRecord.objects.values_list("pair_hash", flat=True))
    assert set(judge_service._HOT_CACHE.keys()) == surviving_hashes

    # The oldest pruned pair must not hit _HOT_CACHE on a subsequent request.
    recheck = judge_single(*pairs[0])
    assert recheck.cached is False


@pytest.mark.django_db
def test_prune_table_handles_more_than_500_deleted_rows(settings):
    records = [
        JudgmentRecord(
            pair_hash=f"batch-hash-{i}",
            sentence1=f"バッチ文{i}です。",
            sentence2=f"バッチ文{i}だ。",
            label=JudgmentRecord.LABEL_ENTAIL,
            score=0.95,
            tier_used=JudgmentRecord.TIER_RULE,
        )
        for i in range(550)
    ]
    JudgmentRecord.objects.bulk_create(records)
    judge_service._cache_put_many(
        {
            rec.pair_hash: judge_service.JudgmentOutput(
                label=rec.label,
                score=rec.score,
                tier_used=rec.tier_used,
                cached=True,
            )
            for rec in records
        },
        max_entries=600,
    )
    RequestAuditLog.objects.bulk_create(
        [
            RequestAuditLog(
                client_ip=f"10.1.{i // 250}.{i % 250}",
                method="POST",
                path="/judge",
                status_code=200,
            )
            for i in range(520)
        ]
    )

    settings.MAX_STORED_JUDGMENTS = 10
    settings.MAX_AUDIT_LOGS = 5
    prune_storage_if_needed()

    assert JudgmentRecord.objects.count() == 10
    assert RequestAuditLog.objects.count() == 5
    surviving_hashes = set(JudgmentRecord.objects.values_list("pair_hash", flat=True))
    assert set(judge_service._HOT_CACHE.keys()) == surviving_hashes
