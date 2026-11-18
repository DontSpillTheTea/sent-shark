"""
Management command to benchmark Tier 1, Tier 2, and Hybrid NLI modes across the
60-pair stratified evaluation gauntlet without mutating persistent DB tables.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from assignment.services.judge_service import evaluate_uncached_pairs
from assignment.services.nli_engine import NoOpEscalationJudge, nli_engine
from assignment.services.normalizer import Tier1Result, evaluate_tier1, normalize_sentence

DEFAULT_GAUNTLET_PATH = (
    Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures" / "eval_gauntlet.json"
)
DIFFICULTIES = ("easy", "medium", "hard", "edge")
LANGUAGES = ("ja", "en")
LABELS = ("ENTAIL", "NO_ENTAIL")

_NOOP_ESCALATION_JUDGE = NoOpEscalationJudge()


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Compute linear-interpolated percentile for a pre-sorted non-empty list."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    low = int(k)
    high = min(low + 1, len(sorted_vals) - 1)
    return sorted_vals[low] + (k - low) * (sorted_vals[high] - sorted_vals[low])


def _compute_latency_stats(latencies_ms: list[float], total_ms: float) -> dict[str, float]:
    """Summarize total batch latency and single-pair latency distribution."""
    ordered = sorted(latencies_ms)
    mean_ms = (sum(ordered) / len(ordered)) if ordered else 0.0
    return {
        "total_ms": round(total_ms, 4),
        "mean_ms": round(mean_ms, 4),
        "p50_ms": round(_percentile(ordered, 50.0), 4),
        "p95_ms": round(_percentile(ordered, 95.0), 4),
        "p99_ms": round(_percentile(ordered, 99.0), 4),
    }


def _compute_mode_metrics(
    items: list[dict[str, Any]],
    predictions: list[tuple[str, float]],
    latencies_ms: list[float],
    total_ms: float,
) -> dict[str, Any]:
    """Compute classification, calibration, difficulty breakdown, and latency metrics."""
    tp = fp = tn = fn = 0
    brier_sum = 0.0
    misclassified_ids: list[str] = []
    diff_counts: dict[str, dict[str, int]] = {d: {"total": 0, "correct": 0} for d in DIFFICULTIES}

    for item, (pred_label, prob) in zip(items, predictions):
        expected = item["expected_label"]
        diff = item["difficulty"]
        y_true = 1.0 if expected == "ENTAIL" else 0.0
        brier_sum += (prob - y_true) ** 2

        diff_counts[diff]["total"] += 1
        if pred_label == expected:
            diff_counts[diff]["correct"] += 1
            if expected == "ENTAIL":
                tp += 1
            else:
                tn += 1
        else:
            misclassified_ids.append(item["id"])
            if pred_label == "ENTAIL":
                fp += 1
            else:
                fn += 1

    total = len(items)
    accuracy = round((tp + tn) / total, 4) if total else 0.0
    precision = round(tp / (tp + fp), 4) if (tp + fp) else 0.0
    recall = round(tp / (tp + fn), 4) if (tp + fn) else 0.0
    f1 = round((2 * tp) / (2 * tp + fp + fn), 4) if (2 * tp + fp + fn) else 0.0
    specificity = round(tn / (tn + fp), 4) if (tn + fp) else 0.0
    brier_score = round(brier_sum / total, 6) if total else 0.0

    by_difficulty = {
        diff: {
            "total": counts["total"],
            "correct": counts["correct"],
            "accuracy": (round(counts["correct"] / counts["total"], 4) if counts["total"] else 0.0),
        }
        for diff, counts in diff_counts.items()
    }

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "brier_score": brier_score,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "by_difficulty": by_difficulty,
        "latency": _compute_latency_stats(latencies_ms, total_ms),
        "misclassified_ids": misclassified_ids,
    }


def _run_tier1_mode(
    items: list[dict[str, Any]], threshold: float
) -> tuple[list[tuple[str, float]], list[float], float]:
    """Evaluate all pairs using Tier 1 deterministic rules and fallback score only."""
    predictions: list[tuple[str, float]] = []
    latencies_ms: list[float] = []

    batch_start = time.perf_counter()
    for item in items:
        t0 = time.perf_counter()
        t1 = evaluate_tier1(item["sentence1"], item["sentence2"])
        prob = float(t1.score if t1.decisive else t1.fallback_score)
        label = t1.label if t1.decisive else ("ENTAIL" if prob >= threshold else "NO_ENTAIL")
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        predictions.append((label, prob))
    total_ms = (time.perf_counter() - batch_start) * 1000.0
    return predictions, latencies_ms, total_ms


def _run_tier2_mode(
    items: list[dict[str, Any]],
) -> tuple[list[tuple[str, float]], list[float], float]:
    """Evaluate all pairs using raw Tier 2 NLI cross-encoder without Tier 1 rules."""
    pairs = [(item["sentence1"], item["sentence2"]) for item in items]
    neutral_t1 = [
        Tier1Result(
            decisive=False,
            label="NO_ENTAIL",
            score=0.0,
            reason="tier2_only",
            canonical_s1=normalize_sentence(s1),
            canonical_s2=normalize_sentence(s2),
            fallback_score=0.0,
        )
        for s1, s2 in pairs
    ]

    batch_start = time.perf_counter()
    batch_details = nli_engine.score_pairs_batch(
        pairs,
        neutral_t1,
        escalation_judge=_NOOP_ESCALATION_JUDGE,
    )
    total_ms = (time.perf_counter() - batch_start) * 1000.0

    latencies_ms: list[float] = []
    for pair, t1 in zip(pairs, neutral_t1):
        t0 = time.perf_counter()
        nli_engine.score_pairs_batch(
            [pair],
            [t1],
            escalation_judge=_NOOP_ESCALATION_JUDGE,
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    predictions = [(detail.label, float(detail.score)) for detail in batch_details]
    return predictions, latencies_ms, total_ms


def _run_hybrid_mode(
    items: list[dict[str, Any]],
) -> tuple[list[tuple[str, float]], list[float], float]:
    """Evaluate all pairs using judge_service's Tier 1 fast-gate + Tier 2 batch pipeline."""
    pairs = [(item["sentence1"], item["sentence2"]) for item in items]

    batch_start = time.perf_counter()
    outputs = evaluate_uncached_pairs(pairs, escalation_judge=_NOOP_ESCALATION_JUDGE)
    total_ms = (time.perf_counter() - batch_start) * 1000.0
    predictions = [(out.label, float(out.score)) for out in outputs]

    latencies_ms: list[float] = []
    for pair in pairs:
        t0 = time.perf_counter()
        evaluate_uncached_pairs([pair], escalation_judge=_NOOP_ESCALATION_JUDGE)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    return predictions, latencies_ms, total_ms


def _build_dataset_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize pair counts across difficulty, language, and ground-truth labels."""
    by_difficulty = {d: 0 for d in DIFFICULTIES}
    by_language = {lang: 0 for lang in LANGUAGES}
    by_label = {lbl: 0 for lbl in LABELS}

    for item in items:
        diff = item["difficulty"]
        lang = item["language"]
        lbl = item["expected_label"]
        by_difficulty[diff] = by_difficulty.get(diff, 0) + 1
        by_language[lang] = by_language.get(lang, 0) + 1
        by_label[lbl] = by_label.get(lbl, 0) + 1

    return {
        "total_pairs": len(items),
        "by_difficulty": by_difficulty,
        "by_language": by_language,
        "by_label": by_label,
    }


def _format_markdown_report(report: dict[str, Any]) -> str:
    """Render the evaluation provenance report as clean Markdown tables."""
    dataset = report["dataset"]
    model = report["model"]
    modes = report["modes"]

    diff_summary = ", ".join(f"{k}={v}" for k, v in dataset["by_difficulty"].items())
    lang_summary = ", ".join(f"{k}={v}" for k, v in dataset["by_language"].items())
    label_summary = ", ".join(f"{k}={v}" for k, v in dataset["by_label"].items())

    lines = [
        "# Evaluation Gauntlet Provenance Report",
        "",
        (
            f"- **Dataset:** `{dataset['total_pairs']}` pairs "
            f"(Difficulty: {diff_summary}; Language: {lang_summary}; Labels: {label_summary})"
        ),
        f"- **Model:** `{model['name']}` (threshold=`{model['threshold']}`)",
        "",
        "## Mode Comparison",
        "",
        (
            "| Mode | Accuracy | Precision | Recall | F1 | Specificity | Brier Score | "
            "Batch Total (ms) | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | CM (TP/FP/TN/FN) |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for mode_name in ("tier1", "tier2", "hybrid"):
        m = modes[mode_name]
        cm = m["confusion_matrix"]
        lat = m["latency"]
        cm_str = f"{cm['tp']}/{cm['fp']}/{cm['tn']}/{cm['fn']}"
        lines.append(
            f"| `{mode_name}` | {m['accuracy']:.4f} | {m['precision']:.4f} | "
            f"{m['recall']:.4f} | {m['f1']:.4f} | {m['specificity']:.4f} | "
            f"{m['brier_score']:.6f} | {lat['total_ms']:.2f} | {lat['mean_ms']:.2f} | "
            f"{lat['p50_ms']:.2f} | {lat['p95_ms']:.2f} | {lat['p99_ms']:.2f} | {cm_str} |"
        )

    lines.extend(
        [
            "",
            "## Per-Difficulty Accuracy",
            "",
            "| Difficulty | tier1 | tier2 | hybrid |",
            "|---|---:|---:|---:|",
        ]
    )
    for diff in DIFFICULTIES:
        t1_d = modes["tier1"]["by_difficulty"][diff]
        t2_d = modes["tier2"]["by_difficulty"][diff]
        hy_d = modes["hybrid"]["by_difficulty"][diff]
        lines.append(
            f"| `{diff}` | {t1_d['accuracy']:.4f} ({t1_d['correct']}/{t1_d['total']}) | "
            f"{t2_d['accuracy']:.4f} ({t2_d['correct']}/{t2_d['total']}) | "
            f"{hy_d['accuracy']:.4f} ({hy_d['correct']}/{hy_d['total']}) |"
        )

    lines.extend(["", "## Misclassified IDs", ""])
    for mode_name in ("tier1", "tier2", "hybrid"):
        failed = modes[mode_name]["misclassified_ids"]
        failed_str = ", ".join(failed) if failed else "None"
        lines.append(f"- **`{mode_name}` ({len(failed)}):** {failed_str}")

    return "\n".join(lines)


class Command(BaseCommand):
    """Benchmark Tier 1, Tier 2, and Hybrid entailment engines on the 60-pair gauntlet."""

    help = "Run the 60-pair stratified evaluation gauntlet across tier1, tier2, and hybrid modes."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--json",
            action="store_true",
            help="Emit structured JSON provenance report instead of Markdown.",
        )
        parser.add_argument(
            "--fixture",
            default=str(DEFAULT_GAUNTLET_PATH),
            help="Optional path to a gauntlet JSON fixture.",
        )

    def handle(self, *args, **options) -> None:
        fixture_path = Path(options["fixture"])
        if not fixture_path.exists():
            raise CommandError(f"Evaluation gauntlet fixture not found: {fixture_path}")

        items: list[dict[str, Any]] = json.loads(fixture_path.read_text(encoding="utf-8"))
        threshold = float(getattr(settings, "ENTAIL_THRESHOLD", 0.50))

        # Pre-warm NLI cross-encoder before timing so weight loading does not skew latencies
        nli_engine.warm_up()

        t1_preds, t1_lats, t1_total = _run_tier1_mode(items, threshold)
        t2_preds, t2_lats, t2_total = _run_tier2_mode(items)
        hy_preds, hy_lats, hy_total = _run_hybrid_mode(items)

        report = {
            "dataset": _build_dataset_summary(items),
            "model": {
                "name": getattr(
                    settings,
                    "NLI_MODEL_NAME",
                    "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli",
                ),
                "threshold": threshold,
            },
            "modes": {
                "tier1": _compute_mode_metrics(items, t1_preds, t1_lats, t1_total),
                "tier2": _compute_mode_metrics(items, t2_preds, t2_lats, t2_total),
                "hybrid": _compute_mode_metrics(items, hy_preds, hy_lats, hy_total),
            },
        }

        if options["json"]:
            self.stdout.write(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            self.stdout.write(_format_markdown_report(report))
