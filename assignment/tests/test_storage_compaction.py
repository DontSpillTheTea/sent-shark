import pytest

from assignment.models import JudgmentRecord, RequestAuditLog
from assignment.services.judge_service import judge_single, prune_storage_if_needed


@pytest.mark.django_db
def test_judgment_record_has_no_redundant_normalized_columns():
    field_names = {f.name for f in JudgmentRecord._meta.get_fields()}
    assert "normalized_s1" not in field_names
    assert "normalized_s2" not in field_names
    assert "hit_count" in field_names


@pytest.mark.django_db
def test_bounded_storage_prunes_oldest_records(settings):
    settings.MAX_STORED_JUDGMENTS = 5
    for i in range(8):
        judge_single(f"テスト文A{i}です。", f"テスト文A{i}だ。")
    prune_storage_if_needed()
    assert JudgmentRecord.objects.count() == 5
    remaining = set(JudgmentRecord.objects.values_list("sentence1", flat=True))
    assert remaining == {f"テスト文A{i}です。" for i in range(3, 8)}


@pytest.mark.django_db
def test_bounded_audit_logs_prunes_oldest_records(settings):
    settings.MAX_AUDIT_LOGS = 4
    for i in range(7):
        RequestAuditLog.objects.create(
            client_ip=f"10.0.0.{i}",
            method="POST",
            path="/judge",
            status_code=200,
        )
    prune_storage_if_needed()
    assert RequestAuditLog.objects.count() == 4
    remaining_ips = set(RequestAuditLog.objects.values_list("client_ip", flat=True))
    assert remaining_ips == {f"10.0.0.{i}" for i in range(3, 7)}


def test_storage_limits_default_settings(settings):
    assert settings.MAX_STORED_JUDGMENTS == 10000
    assert settings.MAX_AUDIT_LOGS == 50000
