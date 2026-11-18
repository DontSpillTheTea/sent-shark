import pytest

from assignment.models import JudgmentRecord, RequestAuditLog
from assignment.services import judge_service
from assignment.services.judge_service import judge_bulk, judge_single


@pytest.mark.django_db
def test_judgment_record_has_no_redundant_or_removed_columns():
    field_names = {f.name for f in JudgmentRecord._meta.get_fields()}
    assert "normalized_s1" not in field_names
    assert "normalized_s2" not in field_names
    assert "expected_label" not in field_names
    assert "hit_count" in field_names


@pytest.mark.django_db
def test_pair_hash_is_order_independent_and_version_scoped(settings):
    from assignment.services.judge_service import compute_pair_hash

    h_ab, _, _ = compute_pair_hash("猫が好きです。", "私は猫が大好きだ。")
    h_ba, _, _ = compute_pair_hash("私は猫が大好きだ。", "猫が好きです。")
    assert h_ab == h_ba

    settings.NLI_MODEL_NAME = "other/nli-model-v2"
    h_new_model, _, _ = compute_pair_hash("猫が好きです。", "私は猫が大好きだ。")
    assert h_new_model != h_ab


@pytest.mark.django_db
def test_pair_hash_policy_version_and_normalization_invariants(settings):
    from assignment.services.judge_service import SCORING_POLICY_VERSION, compute_pair_hash

    h1, n1, n2 = compute_pair_hash("　猫が好きです！　", "私は猫が大好きだ。")
    h2, _, _ = compute_pair_hash("私は猫が大好きだ", "猫が好きです")
    assert (n1, n2) == ("猫が好きです", "私は猫が大好きだ")
    assert h1 == h2

    h_other_policy, _, _ = compute_pair_hash(
        "猫が好きです。",
        "私は猫が大好きだ。",
        policy_version="exact_plus_nli:v2:min",
    )
    assert h_other_policy != h1
    assert SCORING_POLICY_VERSION == "exact_plus_nli:v1:mean"


@pytest.mark.django_db
def test_symmetric_calls_share_single_db_row_and_identical_score(client):
    s1 = "雨が降っているので、傘を持っていく。"
    s2 = "雨天のため、傘を持参する。"
    r1 = client.post(
        "/judge",
        data={"sentence1": s1, "sentence2": s2},
        content_type="application/json",
        REMOTE_ADDR="198.51.100.201",
    )
    r2 = client.post(
        "/judge",
        data={"sentence1": s2, "sentence2": s1},
        content_type="application/json",
        REMOTE_ADDR="198.51.100.202",
    )
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json() == r2.json()
    assert JudgmentRecord.objects.count() == 1


@pytest.mark.django_db
def test_writes_do_not_prune_judgments_or_audit_logs(client, settings):
    assert not hasattr(settings, "MAX_STORED_JUDGMENTS")
    assert not hasattr(settings, "MAX_AUDIT_LOGS")
    assert not hasattr(judge_service, "prune_storage_if_needed")
    assert not hasattr(judge_service, "_prune_table")

    pairs = [(f"テスト文P{i}です。", f"テスト文P{i}です。") for i in range(12)]
    outputs = judge_bulk(pairs)
    assert len(outputs) == 12
    assert JudgmentRecord.objects.count() == 12

    for i in range(8):
        resp = client.post(
            "/judge",
            data={"sentence1": f"テスト監査文{i}", "sentence2": f"テスト監査文{i}"},
            content_type="application/json",
            REMOTE_ADDR=f"198.51.100.{i + 1}",
        )
        assert resp.status_code == 200

    assert JudgmentRecord.objects.count() == 20
    assert RequestAuditLog.objects.count() == 8


@pytest.mark.django_db
def test_symmetric_pair_hits_hot_cache_without_recomputing(monkeypatch):
    from assignment.services import nli_engine as nli_mod

    s1 = "雨が降っているので、傘を持っていく。"
    s2 = "雨天のため、傘を持参する。"
    first = judge_single(s1, s2)
    assert first.cached is False

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("NLI inference should not run on symmetric cache hit")

    monkeypatch.setattr(nli_mod.nli_engine, "score_pairs_batch", fail_if_called)
    second = judge_single(s2, s1)
    assert second.cached is True
    assert second.label == first.label
    assert second.score == first.score
    assert second.tier_used == first.tier_used
