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

    h_v1_policy, _, _ = compute_pair_hash(
        "猫が好きです。",
        "私は猫が大好きだ。",
        policy_version="exact_plus_nli:v1:mean",
    )
    assert h_v1_policy != h1
    assert SCORING_POLICY_VERSION == "exact_plus_nli:v2:min"


@pytest.mark.django_db
def test_policy_version_change_invalidates_hot_and_db_cache(monkeypatch):
    from assignment.services import nli_engine as nli_mod
    from assignment.services.judge_service import JudgmentOutput, _cache_put_many, compute_pair_hash

    s1, s2 = "ポリシー変更テストAです。", "ポリシー変更テストBです。"
    v1_hash, _, _ = compute_pair_hash(s1, s2, policy_version="exact_plus_nli:v1:mean")
    JudgmentRecord.objects.create(
        pair_hash=v1_hash,
        sentence1=s1,
        sentence2=s2,
        label=JudgmentRecord.LABEL_ENTAIL,
        score=0.75,
        tier_used=JudgmentRecord.TIER_NLI,
    )
    _cache_put_many(
        {
            v1_hash: JudgmentOutput(
                label="ENTAIL",
                score=0.75,
                tier_used="tier2_nli",
                cached=True,
            )
        }
    )

    calls = {"count": 0}

    def v2_nli(pairs, tier1_results=None):
        calls["count"] += 1
        return [
            nli_mod.NLIScoreDetail(
                score=0.2,
                label="NO_ENTAIL",
                tier_used="tier2_nli",
                forward_entail=0.95,
                reverse_entail=0.20,
                max_contradiction=0.02,
                forward_contradict=0.01,
                reverse_contradict=0.02,
            )
            for _ in pairs
        ]

    monkeypatch.setattr(nli_mod.nli_engine, "score_pairs_batch", v2_nli)
    out = judge_single(s1, s2)
    assert calls["count"] == 1
    assert out.cached is False
    assert out.label == "NO_ENTAIL"
    assert out.score == 0.2
    assert JudgmentRecord.objects.count() == 2


@pytest.mark.django_db
def test_symmetric_calls_share_single_db_row_and_identical_score(client, monkeypatch):
    from assignment.services import nli_engine as nli_mod

    monkeypatch.setattr(
        nli_mod.nli_engine,
        "score_pairs_batch",
        lambda pairs, tier1_results=None: [
            nli_mod.NLIScoreDetail(
                score=0.91,
                label="ENTAIL",
                tier_used="tier2_nli",
                forward_entail=0.92,
                reverse_entail=0.9,
                max_contradiction=0.02,
                forward_contradict=0.02,
                reverse_contradict=0.01,
            )
            for _ in pairs
        ],
    )
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

    monkeypatch.setattr(
        nli_mod.nli_engine,
        "score_pairs_batch",
        lambda pairs, tier1_results=None: [
            nli_mod.NLIScoreDetail(
                score=0.91,
                label="ENTAIL",
                tier_used="tier2_nli",
                forward_entail=0.92,
                reverse_entail=0.9,
                max_contradiction=0.02,
                forward_contradict=0.02,
                reverse_contradict=0.01,
            )
            for _ in pairs
        ],
    )
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


@pytest.mark.django_db
def test_pair_hash_includes_threshold_revision_enable_flag_and_label_indices(settings, monkeypatch):
    from assignment.services import nli_engine as nli_mod
    from assignment.services.judge_service import compute_pair_hash

    s1, s2 = "猫が好きです。", "私は猫が大好きだ。"
    monkeypatch.setattr(nli_mod.nli_engine, "model_revision", "rev-abc-1")
    settings.NLI_MODEL_NAME = nli_mod.nli_engine.model_name
    settings.NLI_ENABLE_MODEL = True
    settings.NLI_ENTAIL_INDEX = None
    settings.NLI_CONTRADICT_INDEX = None
    settings.ENTAIL_THRESHOLD = 0.50

    h_base, _, _ = compute_pair_hash(s1, s2)

    # Equivalent float representation must produce identical hash
    settings.ENTAIL_THRESHOLD = 0.5
    assert compute_pair_hash(s1, s2)[0] == h_base

    # Changing ENTAIL_THRESHOLD must invalidate pair_hash
    settings.ENTAIL_THRESHOLD = 0.60
    h_thresh, _, _ = compute_pair_hash(s1, s2)
    assert h_thresh != h_base

    # Changing model_revision for active model must invalidate pair_hash
    settings.ENTAIL_THRESHOLD = 0.50
    monkeypatch.setattr(nli_mod.nli_engine, "model_revision", "rev-abc-2")
    h_rev, _, _ = compute_pair_hash(s1, s2)
    assert h_rev != h_base

    # Different model_name must NOT inherit singleton nli_engine's model_revision
    monkeypatch.setattr(nli_mod, "_resolve_hf_model_revision", lambda _name: None)
    monkeypatch.setattr(nli_mod.nli_engine, "model_revision", "rev-singleton-a")
    h_other_a, _, _ = compute_pair_hash(s1, s2, model_name="custom/model-x")
    monkeypatch.setattr(nli_mod.nli_engine, "model_revision", "rev-singleton-b")
    h_other_b, _, _ = compute_pair_hash(s1, s2, model_name="custom/model-x")
    assert h_other_a == h_other_b

    # Changing NLI_ENABLE_MODEL or explicit label overrides must invalidate pair_hash
    monkeypatch.setattr(nli_mod.nli_engine, "model_revision", "rev-abc-1")
    settings.NLI_ENABLE_MODEL = False
    h_disabled, _, _ = compute_pair_hash(s1, s2)
    assert h_disabled != h_base

    settings.NLI_ENABLE_MODEL = True
    settings.NLI_ENTAIL_INDEX = 2
    settings.NLI_CONTRADICT_INDEX = 0
    h_indices, _, _ = compute_pair_hash(s1, s2)
    assert h_indices != h_base


@pytest.mark.django_db
def test_threshold_change_invalidates_hot_and_db_cache_for_api_predictions(
    client, settings, monkeypatch
):
    from assignment.services import nli_engine as nli_mod

    def threshold_aware_nli(pairs, tier1_results=None):
        threshold = float(getattr(settings, "ENTAIL_THRESHOLD", 0.50))
        score = 0.53
        label = "ENTAIL" if score >= threshold else "NO_ENTAIL"
        return [
            nli_mod.NLIScoreDetail(
                score=score,
                label=label,
                tier_used="tier2_nli",
                forward_entail=0.54,
                reverse_entail=0.52,
                max_contradiction=0.0,
                forward_contradict=0.0,
                reverse_contradict=0.0,
            )
            for _ in pairs
        ]

    monkeypatch.setattr(nli_mod.nli_engine, "score_pairs_batch", threshold_aware_nli)
    payload = {"sentence1": "境界付近のテスト文Aです。", "sentence2": "境界付近のテスト文Bです。"}

    settings.ENTAIL_THRESHOLD = 0.50
    r1 = client.post(
        "/judge", data=payload, content_type="application/json", REMOTE_ADDR="198.51.100.10"
    )
    assert r1.status_code == 200
    assert r1.json() == {"label": "ENTAIL", "score": 0.53}

    # Raising ENTAIL_THRESHOLD from 0.50 to 0.60 must NOT return stale cached ENTAIL
    settings.ENTAIL_THRESHOLD = 0.60
    r2 = client.post(
        "/judge", data=payload, content_type="application/json", REMOTE_ADDR="198.51.100.11"
    )
    assert r2.status_code == 200
    assert r2.json() == {"label": "NO_ENTAIL", "score": 0.53}
    assert JudgmentRecord.objects.count() == 2


@pytest.mark.django_db
def test_disabling_model_does_not_poison_cache_when_model_is_enabled(client, settings, monkeypatch):
    from assignment.services import nli_engine as nli_mod

    payload = {"sentence1": "猫が好きです。", "sentence2": "私は猫が大好きだ。"}

    settings.NLI_ENABLE_MODEL = False
    r_off = client.post(
        "/judge", data=payload, content_type="application/json", REMOTE_ADDR="198.51.100.20"
    )
    assert r_off.status_code == 200
    assert r_off.json() == {"label": "NO_ENTAIL", "score": 0.0}

    monkeypatch.setattr(
        nli_mod.nli_engine,
        "score_pairs_batch",
        lambda pairs, tier1_results=None: [
            nli_mod.NLIScoreDetail(
                score=0.9536,
                label="ENTAIL",
                tier_used="tier2_nli",
                forward_entail=0.96,
                reverse_entail=0.95,
                max_contradiction=0.003,
                forward_contradict=0.003,
                reverse_contradict=0.002,
            )
            for _ in pairs
        ],
    )
    settings.NLI_ENABLE_MODEL = True
    r_on = client.post(
        "/judge", data=payload, content_type="application/json", REMOTE_ADDR="198.51.100.21"
    )
    assert r_on.status_code == 200
    assert r_on.json() == {"label": "ENTAIL", "score": 0.9536}
    assert JudgmentRecord.objects.count() == 2


@pytest.mark.django_db
def test_cold_model_load_rekeys_cache_to_resolved_model_revision(monkeypatch):
    from assignment.services import nli_engine as nli_mod

    monkeypatch.setattr(nli_mod, "_resolve_hf_model_revision", lambda _name: None)
    monkeypatch.setattr(nli_mod.nli_engine, "model_revision", None)

    call_counter = {"count": 0}

    def loading_nli(pairs, tier1_results=None):
        call_counter["count"] += 1
        nli_mod.nli_engine.model_revision = "rev-downloaded-on-first-call"
        return [
            nli_mod.NLIScoreDetail(
                score=0.88,
                label="ENTAIL",
                tier_used="tier2_nli",
                forward_entail=0.90,
                reverse_entail=0.86,
                max_contradiction=0.01,
                forward_contradict=0.01,
                reverse_contradict=0.01,
            )
            for _ in pairs
        ]

    monkeypatch.setattr(nli_mod.nli_engine, "score_pairs_batch", loading_nli)
    first = judge_single("初回ロード文Aです。", "初回ロード文Bです。")
    assert first.cached is False
    assert call_counter["count"] == 1

    # Second call sees model_revision="rev-downloaded-on-first-call" and must hit hot/DB cache
    second = judge_single("初回ロード文Bです。", "初回ロード文Aです。")
    assert second.cached is True
    assert second.label == "ENTAIL"
    assert second.score == 0.88
    assert call_counter["count"] == 1
    assert JudgmentRecord.objects.count() == 1


def test_default_mdeberta_revision_pinning_and_cache_isolation(settings):
    from assignment.services import nli_engine as nli_mod
    from assignment.services.judge_service import compute_pair_hash

    assert settings.DEFAULT_NLI_MODEL_NAME == "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    assert settings.DEFAULT_NLI_MODEL_REVISION == "8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c"
    assert (
        nli_mod._configured_revision_for_model("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
        == "8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c"
    )

    h_mdeberta, _, _ = compute_pair_hash(
        "母は夕食を作る。",
        "母は晩ご飯を作る。",
        model_name="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        model_revision="8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c",
    )
    h_minilm, _, _ = compute_pair_hash(
        "母は夕食を作る。",
        "母は晩ご飯を作る。",
        model_name="MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli",
        model_revision="0a71e92a985b6e1ad1828cf67ce9c459639c1dca",
    )
    assert h_mdeberta != h_minilm
