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
REQUIRED_KEYS = {
    "id",
    "difficulty",
    "language",
    "phenomenon",
    "sentence1",
    "sentence2",
    "expected_label",
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

    def recording_score(pairs, tier1_results, escalation_judge=None, apply_lexical_surface=True):
        calls.append(
            {
                "count": len(pairs),
                "apply_lexical_surface": apply_lexical_surface,
            }
        )
        return original_fn(
            pairs,
            tier1_results,
            escalation_judge=escalation_judge,
            apply_lexical_surface=apply_lexical_surface,
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
