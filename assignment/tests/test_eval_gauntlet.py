import json
from io import StringIO
from pathlib import Path

import pytest
from django.conf import settings
from django.core.management import call_command
from django.core.management.base import CommandError

from assignment.models import JudgmentRecord, RequestAuditLog
from assignment.services.normalizer import (
    contains_japanese,
    is_same_language_pair,
    normalize_sentence,
)

GAUNTLET_PATH = Path(__file__).resolve().parent / "fixtures" / "eval_gauntlet.json"
HOLDOUT_PATH = Path(__file__).resolve().parent / "fixtures" / "eval_holdout.json"
HOLDOUT_PROVENANCE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "eval_holdout_provenance.json"
)
REQUIRED_KEYS = {
    "id",
    "difficulty",
    "language",
    "phenomenon",
    "sentence1",
    "sentence2",
    "expected_label",
}
HOLDOUT_REQUIRED_KEYS = {
    "id",
    "source",
    "source_id",
    "source_label",
    "stratum",
    "language",
    "sentence1",
    "sentence2",
    "expected_label",
}
HOLDOUT_EXPECTED_STRATA = {
    "bidir_entail": (20, "ja", "ENTAIL"),
    "oneway_entail": (7, "ja", "NO_ENTAIL"),
    "contradict": (7, "ja", "NO_ENTAIL"),
    "neutral": (6, "ja", "NO_ENTAIL"),
    "paraphrase": (10, "en", "ENTAIL"),
    "adversarial_swap": (10, "en", "NO_ENTAIL"),
}
DIFFICULTIES = ("easy", "medium", "hard", "edge")


def test_gauntlet_fixture_has_60_stratified_pairs():
    assert GAUNTLET_PATH.exists()
    items = json.loads(GAUNTLET_PATH.read_text(encoding="utf-8"))
    assert len(items) == 60

    ids = [item["id"] for item in items]
    assert ids == [f"G{i:02d}" for i in range(1, 61)]

    by_diff = {d: 0 for d in DIFFICULTIES}
    by_label = {"ENTAIL": 0, "NO_ENTAIL": 0}
    by_lang = {"ja": 0, "en": 0}
    diff_labels = {d: {"ENTAIL": 0, "NO_ENTAIL": 0} for d in DIFFICULTIES}
    diff_langs = {d: {"ja": 0, "en": 0} for d in DIFFICULTIES}
    ordered_pairs: set[tuple[str, str]] = set()
    unordered_pairs: set[frozenset[str]] = set()

    for item in items:
        assert REQUIRED_KEYS.issubset(item.keys())
        diff = item["difficulty"]
        lang = item["language"]
        label = item["expected_label"]
        s1 = item["sentence1"]
        s2 = item["sentence2"]

        assert diff in DIFFICULTIES
        assert lang in ("ja", "en")
        assert label in ("ENTAIL", "NO_ENTAIL")
        assert isinstance(item["phenomenon"], str) and item["phenomenon"].strip()
        assert isinstance(s1, str) and s1.strip()
        assert isinstance(s2, str) and s2.strip()
        assert is_same_language_pair(s1, s2)
        if lang == "ja":
            assert contains_japanese(s1) and contains_japanese(s2)
        else:
            assert not contains_japanese(s1) and not contains_japanese(s2)

        n1 = normalize_sentence(s1)
        n2 = normalize_sentence(s2)
        assert (n1, n2) not in ordered_pairs, f"Duplicate pair found: {item['id']}"
        assert frozenset((n1, n2)) not in unordered_pairs, (
            f"Symmetric duplicate pair found: {item['id']}"
        )
        ordered_pairs.add((n1, n2))
        unordered_pairs.add(frozenset((n1, n2)))

        by_diff[diff] += 1
        by_label[label] += 1
        by_lang[lang] += 1
        diff_labels[diff][label] += 1
        diff_langs[diff][lang] += 1

    assert by_diff == {"easy": 15, "medium": 15, "hard": 15, "edge": 15}
    assert by_label == {"ENTAIL": 30, "NO_ENTAIL": 30}
    assert by_lang["ja"] >= 20 and by_lang["en"] >= 20
    for diff in DIFFICULTIES:
        assert diff_labels[diff]["ENTAIL"] in (7, 8)
        assert diff_labels[diff]["NO_ENTAIL"] in (7, 8)
        assert diff_langs[diff]["ja"] > 0 and diff_langs[diff]["en"] > 0


@pytest.mark.django_db
def test_run_eval_management_command_produces_provenance_report():
    assert JudgmentRecord.objects.count() == 0
    assert RequestAuditLog.objects.count() == 0

    out = StringIO()
    call_command("run_eval", "--json", stdout=out)
    report = json.loads(out.getvalue())

    # Evaluation must not mutate persistent DB tables
    assert JudgmentRecord.objects.count() == 0
    assert RequestAuditLog.objects.count() == 0

    dataset = report["dataset"]
    assert dataset["total_pairs"] == 60
    assert dataset["by_difficulty"] == {"easy": 15, "medium": 15, "hard": 15, "edge": 15}
    assert dataset["by_label"] == {"ENTAIL": 30, "NO_ENTAIL": 30}
    assert sum(dataset["by_language"].values()) == 60
    assert set(dataset["by_language"].keys()) == {"ja", "en"}

    model = report["model"]
    assert model["name"] == settings.NLI_MODEL_NAME
    assert model["threshold"] == settings.ENTAIL_THRESHOLD

    modes = report["modes"]
    assert set(modes.keys()) == {"tier1", "tier2", "hybrid"}

    for mode_name in ("tier1", "tier2", "hybrid"):
        metrics = modes[mode_name]
        for key in (
            "accuracy",
            "precision",
            "recall",
            "f1",
            "specificity",
            "brier_score",
            "confusion_matrix",
            "by_difficulty",
            "latency",
            "misclassified_ids",
        ):
            assert key in metrics, f"Missing {key} in mode {mode_name}"

        assert 0.0 <= metrics["brier_score"] <= 1.0
        cm = metrics["confusion_matrix"]
        assert set(cm.keys()) == {"tp", "fp", "tn", "fn"}
        assert cm["tp"] + cm["fp"] + cm["tn"] + cm["fn"] == 60
        assert len(metrics["misclassified_ids"]) == cm["fp"] + cm["fn"]

        for diff in DIFFICULTIES:
            diff_stats = metrics["by_difficulty"][diff]
            assert diff_stats["total"] == 15
            assert 0 <= diff_stats["correct"] <= 15
            assert diff_stats["accuracy"] == round(diff_stats["correct"] / 15, 4)

        lat = metrics["latency"]
        for lat_key in ("total_ms", "mean_ms", "p50_ms", "p95_ms", "p99_ms"):
            assert lat[lat_key] >= 0.0
        assert lat["p50_ms"] <= lat["p95_ms"] <= lat["p99_ms"]

    hybrid = modes["hybrid"]
    assert hybrid["accuracy"] >= 0.90
    assert hybrid["f1"] >= 0.90
    assert hybrid["accuracy"] >= modes["tier1"]["accuracy"]
    assert hybrid["accuracy"] >= modes["tier2"]["accuracy"]


@pytest.mark.django_db
def test_run_eval_default_markdown_output_does_not_mutate_db():
    assert JudgmentRecord.objects.count() == 0
    out = StringIO()
    call_command("run_eval", stdout=out)
    markdown = out.getvalue()

    assert JudgmentRecord.objects.count() == 0
    assert RequestAuditLog.objects.count() == 0
    assert "60" in markdown
    for term in ("tier1", "tier2", "hybrid", "Accuracy", "F1", "Brier", "p50", "p95", "p99"):
        assert term in markdown


def test_run_eval_raises_command_error_for_missing_fixture(tmp_path):
    missing_path = tmp_path / "does_not_exist.json"
    with pytest.raises(CommandError, match="not found"):
        call_command("run_eval", "--fixture", str(missing_path))


def test_run_tier2_mode_disables_lexical_surface_rewrites(monkeypatch):
    from assignment.management.commands.run_eval import _run_tier2_mode
    from assignment.services import nli_engine as nli_mod
    from assignment.services.normalizer import Tier1Result, evaluate_tier1

    calls: list[dict[str, object]] = []
    original_fn = nli_mod.nli_engine.score_pairs_batch

    def recording_score(
        pairs,
        tier1_results,
        escalation_judge=None,
        apply_lexical_surface=True,
        scoring_mode="mean",
    ):
        calls.append(
            {
                "count": len(pairs),
                "apply_lexical_surface": apply_lexical_surface,
                "scoring_mode": scoring_mode,
            }
        )
        return original_fn(
            pairs,
            tier1_results,
            escalation_judge=escalation_judge,
            apply_lexical_surface=apply_lexical_surface,
            scoring_mode=scoring_mode,
        )

    monkeypatch.setattr(nli_mod.nli_engine, "score_pairs_batch", recording_score)

    items = [
        {
            "id": "T2_ISO",
            "difficulty": "medium",
            "language": "en",
            "phenomenon": "temporal_shift",
            "sentence1": "He gave up smoking last year.",
            "sentence2": "He quit smoking a year ago.",
            "expected_label": "ENTAIL",
        }
    ]
    preds, latencies, _ = _run_tier2_mode(items)
    assert len(preds) == 1
    assert len(latencies) == 1
    # Raw Tier 2 must pass apply_lexical_surface=False on both batch and per-pair calls
    assert len(calls) == 2
    assert all(call["apply_lexical_surface"] is False for call in calls)
    # Without lexical surface rewrites, raw MiniLM misfires to NO_ENTAIL on G21
    assert preds[0][0] == "NO_ENTAIL"

    # Verify single-pass un-rewritten surface input when apply_lexical_surface=False
    s1, s2 = "He gave up smoking last year.", "He quit smoking a year ago."
    neutral = Tier1Result(
        decisive=False,
        label="NO_ENTAIL",
        score=0.0,
        reason="tier2_only",
        canonical_s1=normalize_sentence(s1),
        canonical_s2=normalize_sentence(s2),
        fallback_score=0.0,
    )
    nli_mod.nli_engine.warm_up()
    captured_sequences: list[tuple[list[str], list[str]]] = []
    orig_tokenizer = nli_mod.nli_engine._tokenizer

    def recording_tokenizer(premises, hypotheses, **kwargs):
        captured_sequences.append((list(premises), list(hypotheses)))
        return orig_tokenizer(premises, hypotheses, **kwargs)

    monkeypatch.setattr(nli_mod.nli_engine, "_tokenizer", recording_tokenizer)
    raw_detail = original_fn([(s1, s2)], [neutral], apply_lexical_surface=False)[0]
    assert raw_detail.label == "NO_ENTAIL"
    assert len(captured_sequences) == 1
    assert captured_sequences[0] == (
        ["He gave up smoking last year", "He quit smoking a year ago"],
        ["He quit smoking a year ago", "He gave up smoking last year"],
    )

    # With Tier 1 canonicals + apply_lexical_surface=True, hybrid evaluates both and gets ENTAIL
    t1 = evaluate_tier1(s1, s2)
    hybrid_detail = original_fn([(s1, s2)], [t1], apply_lexical_surface=True)[0]
    assert hybrid_detail.label == "ENTAIL"


def test_holdout_fixture_and_provenance_are_valid_and_disjoint_from_dev():
    assert HOLDOUT_PATH.exists(), f"Missing holdout fixture: {HOLDOUT_PATH}"
    assert HOLDOUT_PROVENANCE_PATH.exists(), (
        f"Missing holdout provenance: {HOLDOUT_PROVENANCE_PATH}"
    )

    holdout_items = json.loads(HOLDOUT_PATH.read_text(encoding="utf-8"))
    assert len(holdout_items) == 60
    assert [item["id"] for item in holdout_items] == [f"H{i:02d}" for i in range(1, 61)]

    dev_items = json.loads(GAUNTLET_PATH.read_text(encoding="utf-8"))
    dev_ordered: set[tuple[str, str]] = {
        (normalize_sentence(item["sentence1"]), normalize_sentence(item["sentence2"]))
        for item in dev_items
    }
    dev_unordered: set[frozenset[str]] = {
        frozenset((normalize_sentence(item["sentence1"]), normalize_sentence(item["sentence2"])))
        for item in dev_items
    }

    holdout_ordered: set[tuple[str, str]] = set()
    holdout_unordered: set[frozenset[str]] = set()
    by_lang = {"ja": 0, "en": 0}
    by_label = {"ENTAIL": 0, "NO_ENTAIL": 0}
    by_source = {"jglue_jnli": 0, "paws_x": 0}
    by_stratum = {stratum: 0 for stratum in HOLDOUT_EXPECTED_STRATA}

    for item in holdout_items:
        assert HOLDOUT_REQUIRED_KEYS.issubset(item.keys())
        stratum = item["stratum"]
        lang = item["language"]
        label = item["expected_label"]
        source = item["source"]
        s1 = item["sentence1"]
        s2 = item["sentence2"]

        assert stratum in HOLDOUT_EXPECTED_STRATA
        expected_count, expected_lang, expected_label = HOLDOUT_EXPECTED_STRATA[stratum]
        assert expected_count > 0
        assert lang == expected_lang
        assert label == expected_label
        assert source in ("jglue_jnli", "paws_x")
        assert isinstance(item["source_id"], str) and item["source_id"].strip()
        assert isinstance(item["source_label"], str) and item["source_label"].strip()
        assert isinstance(s1, str) and s1.strip()
        assert isinstance(s2, str) and s2.strip()
        assert s1 != s2
        assert is_same_language_pair(s1, s2)
        if lang == "ja":
            assert contains_japanese(s1) and contains_japanese(s2)
        else:
            assert not contains_japanese(s1) and not contains_japanese(s2)

        n1 = normalize_sentence(s1)
        n2 = normalize_sentence(s2)
        assert (n1, n2) not in holdout_ordered, f"Duplicate holdout pair: {item['id']}"
        assert frozenset((n1, n2)) not in holdout_unordered, (
            f"Symmetric duplicate holdout pair: {item['id']}"
        )
        assert (n1, n2) not in dev_ordered, f"Holdout overlaps dev ordered: {item['id']}"
        assert frozenset((n1, n2)) not in dev_unordered, (
            f"Holdout overlaps dev unordered: {item['id']}"
        )
        holdout_ordered.add((n1, n2))
        holdout_unordered.add(frozenset((n1, n2)))

        by_lang[lang] += 1
        by_label[label] += 1
        by_source[source] += 1
        by_stratum[stratum] += 1

    assert by_lang == {"ja": 40, "en": 20}
    assert by_label == {"ENTAIL": 30, "NO_ENTAIL": 30}
    assert by_source == {"jglue_jnli": 40, "paws_x": 20}
    for stratum, (expected_count, _, _) in HOLDOUT_EXPECTED_STRATA.items():
        assert by_stratum[stratum] == expected_count

    provenance = json.loads(HOLDOUT_PROVENANCE_PATH.read_text(encoding="utf-8"))
    assert provenance["sampling_seed"] == 42
    assert provenance["total_pairs"] == 60
    assert provenance["by_language"] == {"ja": 40, "en": 20}
    assert provenance["by_label"] == {"ENTAIL": 30, "NO_ENTAIL": 30}
    assert provenance["strata_counts"] == {
        stratum: count for stratum, (count, _, _) in HOLDOUT_EXPECTED_STRATA.items()
    }
    label_rules = provenance.get("label_mapping_rules")
    assert isinstance(label_rules, dict) and label_rules

    sources = provenance["sources"]
    assert "jglue_jnli" in sources and "paws_x" in sources

    jglue = sources["jglue_jnli"]
    assert "JGLUE" in jglue["dataset_name"]
    assert "JNLI" in jglue["dataset_name"]
    assert jglue["license"] == "CC BY-SA 4.0"
    assert jglue["commit_sha"] == "6f071c09316baae89c3d083a90985b4b1cb9968c"
    assert "yahoojapan/JGLUE" in jglue["source_url"]
    assert isinstance(jglue["citation"], str) and jglue["citation"].strip()

    pawsx = sources["paws_x"]
    assert "PAWS-X" in pawsx["dataset_name"]
    assert "attribution" in pawsx["license"].lower()
    assert pawsx["commit_sha"] == "4cd8187c404b"
    assert "paws-x" in pawsx["source_url"]
    assert isinstance(pawsx["citation"], str) and pawsx["citation"].strip()


@pytest.mark.django_db
def test_run_eval_supports_scoring_modes_and_provenance_metadata():
    assert JudgmentRecord.objects.count() == 0
    assert RequestAuditLog.objects.count() == 0

    out_dev_min = StringIO()
    call_command("run_eval", "--dataset", "dev", "--scoring", "min", "--json", stdout=out_dev_min)
    dev_min_report = json.loads(out_dev_min.getvalue())

    assert JudgmentRecord.objects.count() == 0
    assert RequestAuditLog.objects.count() == 0

    dev_prov = dev_min_report["provenance"]
    assert dev_prov["dataset_name"] == "dev"
    assert dev_prov["scoring_mode"] == "min"
    assert dev_prov["model_name"] == settings.NLI_MODEL_NAME
    assert isinstance(dev_prov["model_revision"], str) and dev_prov["model_revision"]
    assert isinstance(dev_prov["git_sha"], str) and dev_prov["git_sha"]
    assert isinstance(dev_prov["command"], str) and "--scoring min" in dev_prov["command"]
    assert dev_prov["fixture_path"].endswith("eval_gauntlet.json")

    hardware = dev_prov["hardware"]
    for hw_key in ("platform", "machine", "cpu_count", "torch_version", "cuda_available", "device"):
        assert hw_key in hardware

    inf_cfg = dev_prov["inference_config"]
    assert inf_cfg["max_length"] == 256
    assert inf_cfg["batch_size"] > 0
    assert inf_cfg["entail_threshold"] == settings.ENTAIL_THRESHOLD
    assert "entail_margin" in inf_cfg
    assert "contradict_cap" in inf_cfg
    assert inf_cfg["scoring_mode"] == "min"
    assert inf_cfg["apply_lexical_surface_tier2"] is False
    assert inf_cfg["apply_lexical_surface_hybrid"] is True

    assert "by_source" in dev_min_report["dataset"]
    assert "by_stratum" in dev_min_report["dataset"]
    for mode_name in ("tier1", "tier2", "hybrid"):
        mode_metrics = dev_min_report["modes"][mode_name]
        for key in (
            "by_language",
            "by_source",
            "by_stratum",
            "confusion_matrix",
            "misclassified_ids",
        ):
            assert key in mode_metrics

    out_holdout = StringIO()
    call_command("run_eval", "--dataset", "holdout", "--json", stdout=out_holdout)
    holdout_report = json.loads(out_holdout.getvalue())

    assert JudgmentRecord.objects.count() == 0
    assert RequestAuditLog.objects.count() == 0

    hold_prov = holdout_report["provenance"]
    assert hold_prov["dataset_name"] == "holdout"
    assert hold_prov["scoring_mode"] == "mean"
    assert hold_prov["model_name"] == settings.NLI_MODEL_NAME
    assert isinstance(hold_prov["model_revision"], str) and hold_prov["model_revision"]
    assert isinstance(hold_prov["git_sha"], str) and hold_prov["git_sha"]
    assert hold_prov["fixture_path"].endswith("eval_holdout.json")
    assert hold_prov["inference_config"]["scoring_mode"] == "mean"
    assert "--dataset holdout" in hold_prov["command"]

    hold_ds = holdout_report["dataset"]
    assert hold_ds["total_pairs"] == 60
    assert hold_ds["by_language"] == {"ja": 40, "en": 20}
    assert hold_ds["by_label"] == {"ENTAIL": 30, "NO_ENTAIL": 30}
    assert hold_ds["by_source"] == {"jglue_jnli": 40, "paws_x": 20}
    assert hold_ds["by_stratum"] == {
        stratum: count for stratum, (count, _, _) in HOLDOUT_EXPECTED_STRATA.items()
    }

    for mode_name in ("tier1", "tier2", "hybrid"):
        metrics = holdout_report["modes"][mode_name]
        cm = metrics["confusion_matrix"]
        assert cm["tp"] + cm["fp"] + cm["tn"] + cm["fn"] == 60
        assert len(metrics["misclassified_ids"]) == cm["fp"] + cm["fn"]
        assert set(metrics["by_language"].keys()) == {"ja", "en"}
        assert sum(v["total"] for v in metrics["by_language"].values()) == 60
        assert set(metrics["by_source"].keys()) == {"jglue_jnli", "paws_x"}
        assert sum(v["total"] for v in metrics["by_source"].values()) == 60
        assert set(metrics["by_stratum"].keys()) == set(HOLDOUT_EXPECTED_STRATA.keys())
        assert sum(v["total"] for v in metrics["by_stratum"].values()) == 60

    out_holdout_md = StringIO()
    call_command("run_eval", "--dataset", "holdout", stdout=out_holdout_md)
    md = out_holdout_md.getvalue()
    assert JudgmentRecord.objects.count() == 0
    assert RequestAuditLog.objects.count() == 0
    assert "holdout" in md
    assert "jglue_jnli" in md
    assert "paws_x" in md


def test_wilson_confidence_interval_and_diagnostics_schema(tmp_path):
    from assignment.management.commands.run_eval import (
        DEFAULT_FINAL_HOLDOUT_PATH,
        _classify_error_category,
        _load_and_validate_fixture,
        _resolve_fixture,
        _wilson_ci_95,
    )

    lo, hi = _wilson_ci_95(correct=36, total=60)
    assert 0.46 <= lo <= 0.48
    assert 0.71 <= hi <= 0.73
    assert _wilson_ci_95(0, 0) == [0.0, 0.0]

    lo_full, hi_full = _wilson_ci_95(correct=60, total=60)
    assert 0.0 <= lo_full <= hi_full <= 1.0
    assert hi_full == 1.0

    lo_zero, hi_zero = _wilson_ci_95(correct=0, total=60)
    assert 0.0 <= lo_zero <= hi_zero <= 1.0
    assert lo_zero == 0.0

    assert (
        _classify_error_category({"stratum": "adversarial_swap"}) == "word_order_or_particle_swap"
    )
    assert _classify_error_category({"stratum": "oneway_entail"}) == "oneway_specificity"
    assert _classify_error_category({"stratum": "contradict"}) == "negation_or_antonym"
    assert (
        _classify_error_category({"source": "jglue_jnli", "stratum": "bidir_entail"})
        == "caption_codescription"
    )
    assert _classify_error_category({"phenomenon": "lexical_paraphrase"}) == "lexical_synonym_gap"
    assert _classify_error_category({"phenomenon": "temporal_shift"}) == "other"

    ds_name, ds_path = _resolve_fixture("final_holdout", None)
    assert ds_name == "final_holdout"
    assert ds_path == DEFAULT_FINAL_HOLDOUT_PATH

    bad_fixture = tmp_path / "bad_source_id.json"
    bad_fixture.write_text(
        json.dumps(
            [
                {
                    "id": "B1",
                    "language": "en",
                    "source_id": True,
                    "sentence1": "A cat sleeps.",
                    "sentence2": "A cat naps.",
                    "expected_label": "ENTAIL",
                }
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(CommandError, match="source_id"):
        _load_and_validate_fixture(bad_fixture)


@pytest.mark.django_db
def test_run_eval_supports_ablation_and_misclassification_diagnostics(tmp_path, monkeypatch):
    from assignment.management.commands import run_eval as run_eval_cmd
    from assignment.services import nli_engine as nli_mod

    fixture = tmp_path / "mini_eval.json"
    fixture.write_text(
        json.dumps(
            [
                {
                    "id": "D1",
                    "language": "en",
                    "stratum": "paraphrase",
                    "source": "mini",
                    "source_id": "1",
                    "sentence1": "A cat sleeps.",
                    "sentence2": "A cat sleeps.",
                    "expected_label": "ENTAIL",
                },
                {
                    "id": "D2",
                    "language": "en",
                    "stratum": "adversarial_swap",
                    "source": "mini",
                    "source_id": 2,
                    "sentence1": "The dog chased the cat.",
                    "sentence2": "The cat chased the dog.",
                    "expected_label": "NO_ENTAIL",
                },
                {
                    "id": "D3_EXACT_MISMATCH",
                    "language": "en",
                    "stratum": "paraphrase",
                    "source": "mini",
                    "source_id": "3",
                    "sentence1": "A dog barks.",
                    "sentence2": "A dog barks.",
                    "expected_label": "NO_ENTAIL",
                },
                {
                    "id": "D4_RULE_MISMATCH",
                    "language": "en",
                    "stratum": "contradict",
                    "source": "mini",
                    "source_id": "4",
                    "sentence1": "The soup is still hot.",
                    "sentence2": "The soup is completely cold.",
                    "expected_label": "ENTAIL",
                },
            ]
        ),
        encoding="utf-8",
    )

    # 1. Test full_hybrid with --diagnostics (captures tier1_exact and tier1_rule short-circuits)
    out_full = StringIO()
    call_command(
        "run_eval",
        "--fixture",
        str(fixture),
        "--ablation",
        "full_hybrid",
        "--diagnostics",
        "--json",
        stdout=out_full,
    )
    rep_full = json.loads(out_full.getvalue())
    assert rep_full["provenance"]["ablation"] == "full_hybrid"
    hy_full = rep_full["modes"]["hybrid"]
    assert "accuracy_ci95" in hy_full
    assert len(hy_full["accuracy_ci95"]) == 2
    assert 0.0 <= hy_full["accuracy_ci95"][0] <= hy_full["accuracy_ci95"][1] <= 1.0
    for breakdown_key in ("by_language", "by_source", "by_stratum"):
        for bucket in hy_full[breakdown_key].values():
            assert "accuracy_ci95" in bucket
            assert len(bucket["accuracy_ci95"]) == 2

    full_by_id = {m["id"]: m for m in hy_full["misclassifications"]}
    assert "D3_EXACT_MISMATCH" in full_by_id
    assert full_by_id["D3_EXACT_MISMATCH"]["origin_tier"] == "tier1_exact"
    assert full_by_id["D3_EXACT_MISMATCH"]["tier1_reason"] == "exact_normalized_match"
    assert full_by_id["D3_EXACT_MISMATCH"]["directional_nli"] is None
    assert "D4_RULE_MISMATCH" in full_by_id
    assert full_by_id["D4_RULE_MISMATCH"]["origin_tier"] == "tier1_rule"
    assert full_by_id["D4_RULE_MISMATCH"]["tier1_reason"] == "contradiction_veto"
    assert full_by_id["D4_RULE_MISMATCH"]["directional_nli"] is None
    assert full_by_id["D4_RULE_MISMATCH"]["error_category"] == "negation_or_antonym"

    # 2. Test exact_plus_nli with --diagnostics (bypasses Tier 1 rules -> D2 fails via tier2_nli)
    out_exact = StringIO()
    call_command(
        "run_eval",
        "--fixture",
        str(fixture),
        "--ablation",
        "exact_plus_nli",
        "--diagnostics",
        "--json",
        stdout=out_exact,
    )
    rep_exact = json.loads(out_exact.getvalue())
    assert rep_exact["provenance"]["ablation"] == "exact_plus_nli"
    hy_exact = rep_exact["modes"]["hybrid"]
    assert "accuracy_ci95" in hy_exact and len(hy_exact["accuracy_ci95"]) == 2
    exact_by_id = {m["id"]: m for m in hy_exact["misclassifications"]}
    assert "D3_EXACT_MISMATCH" in exact_by_id
    assert exact_by_id["D3_EXACT_MISMATCH"]["origin_tier"] == "tier1_exact"
    assert exact_by_id["D3_EXACT_MISMATCH"]["directional_nli"] is None
    assert "D2" in exact_by_id
    d2_err = exact_by_id["D2"]
    for key in (
        "id",
        "source",
        "source_id",
        "language",
        "stratum",
        "sentence1",
        "sentence2",
        "expected_label",
        "predicted_label",
        "score",
        "origin_tier",
        "tier1_reason",
        "directional_nli",
        "error_category",
    ):
        assert key in d2_err
    assert d2_err["source_id"] == 2
    assert d2_err["origin_tier"] == "tier2_nli"
    assert d2_err["tier1_reason"] == "non_identical"
    assert d2_err["error_category"] == "word_order_or_particle_swap"
    dnli = d2_err["directional_nli"]
    assert isinstance(dnli, dict)
    assert set(dnli.keys()) == {"p_entail_fwd", "p_entail_rev", "p_contra_fwd", "p_contra_rev"}
    for val in dnli.values():
        assert isinstance(val, float) and 0.0 <= val <= 1.0

    # 3. Test nli_only with --diagnostics (even exact matches route to tier2_nli)
    out_nli = StringIO()
    call_command(
        "run_eval",
        "--fixture",
        str(fixture),
        "--ablation",
        "nli_only",
        "--diagnostics",
        "--json",
        stdout=out_nli,
    )
    rep_nli = json.loads(out_nli.getvalue())
    assert rep_nli["provenance"]["ablation"] == "nli_only"
    hy_nli = rep_nli["modes"]["hybrid"]
    nli_by_id = {m["id"]: m for m in hy_nli["misclassifications"]}
    assert "D3_EXACT_MISMATCH" in nli_by_id
    assert nli_by_id["D3_EXACT_MISMATCH"]["origin_tier"] == "tier2_nli"
    assert nli_by_id["D3_EXACT_MISMATCH"]["tier1_reason"] == "tier2_only"
    assert isinstance(nli_by_id["D3_EXACT_MISMATCH"]["directional_nli"], dict)
    assert "D2" in nli_by_id
    assert nli_by_id["D2"]["origin_tier"] == "tier2_nli"

    # 4. Test --dataset dev_diagnostic works cleanly with --diagnostics
    out_dd = StringIO()
    call_command(
        "run_eval",
        "--dataset",
        "dev_diagnostic",
        "--ablation",
        "exact_plus_nli",
        "--diagnostics",
        "--json",
        stdout=out_dd,
    )
    rep_dd = json.loads(out_dd.getvalue())
    assert rep_dd["provenance"]["dataset_name"] == "dev_diagnostic"
    assert rep_dd["provenance"]["fixture_path"].endswith("eval_holdout.json")
    assert rep_dd["provenance"]["ablation"] == "exact_plus_nli"
    assert rep_dd["dataset"]["total_pairs"] == 60
    for mode_name in ("tier1", "tier2", "hybrid"):
        mode = rep_dd["modes"][mode_name]
        cm = mode["confusion_matrix"]
        assert len(mode["misclassifications"]) == cm["fp"] + cm["fn"]

    # 5. Test --model-name scopes local engine; warm_up failure raises CommandError
    singleton_before = nli_mod.nli_engine.model_name

    class FakeScopedEngine:
        def __init__(self, model_name=None):
            self.model_name = model_name
            self.warmed = False

        def warm_up(self):
            self.warmed = True
            return True

        def get_model_revision(self):
            return "fake-rev-123"

        def score_pairs_batch(
            self,
            pairs,
            tier1_results,
            escalation_judge=None,
            apply_lexical_surface=True,
            scoring_mode="mean",
        ):
            return [
                nli_mod.NLIScoreDetail(
                    score=0.95,
                    label="ENTAIL",
                    tier_used="tier2_nli",
                    forward_entail=0.96,
                    reverse_entail=0.94,
                    max_contradiction=0.01,
                    forward_contradict=0.01,
                    reverse_contradict=0.005,
                )
                for _ in pairs
            ]

    monkeypatch.setattr(run_eval_cmd, "HuggingFaceNLIEngine", FakeScopedEngine)
    out_scoped = StringIO()
    call_command(
        "run_eval",
        "--fixture",
        str(fixture),
        "--model-name",
        "fake/scoped-nli",
        "--ablation",
        "nli_only",
        "--diagnostics",
        "--json",
        stdout=out_scoped,
    )
    rep_scoped = json.loads(out_scoped.getvalue())
    assert rep_scoped["provenance"]["model_name"] == "fake/scoped-nli"
    assert rep_scoped["provenance"]["model_revision"] == "fake-rev-123"
    assert rep_scoped["model"]["name"] == "fake/scoped-nli"
    assert nli_mod.nli_engine.model_name == singleton_before

    class FailingEngine(FakeScopedEngine):
        def warm_up(self):
            return False

    monkeypatch.setattr(run_eval_cmd, "HuggingFaceNLIEngine", FailingEngine)
    with pytest.raises(CommandError, match="warm up"):
        call_command("run_eval", "--fixture", str(fixture), "--model-name", "fake/broken-nli")
