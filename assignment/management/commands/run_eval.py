"""
Management command to benchmark Tier 1, Tier 2, and Hybrid NLI modes across the
60-pair dev gauntlet or independent held-out benchmark without mutating DB tables.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from assignment.services.judge_service import evaluate_uncached_pairs
from assignment.services.nli_engine import (
    NLI_BATCH_SIZE,
    NLI_CONTRADICT_CAP,
    NLI_DIRECTIONAL_HIGH,
    NLI_DIRECTIONAL_LOW,
    NLI_LEXICAL_CONTRADICT_CAP,
    NLI_MAX_LENGTH,
    NoOpEscalationJudge,
    _resolve_hf_model_revision,
    nli_engine,
)
from assignment.services.normalizer import Tier1Result, evaluate_tier1, normalize_sentence
from assignment.types import EntailmentLabel, ScoringMode

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures"
DEFAULT_GAUNTLET_PATH = FIXTURES_DIR / "eval_gauntlet.json"
DEFAULT_HOLDOUT_PATH = FIXTURES_DIR / "eval_holdout.json"
DIFFICULTIES = ("easy", "medium", "hard", "edge")
LANGUAGES = ("ja", "en")
LABELS: tuple[EntailmentLabel, ...] = ("ENTAIL", "NO_ENTAIL")
DATASETS = ("dev", "holdout")
SCORING_MODES: tuple[ScoringMode, ...] = ("mean", "min")

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


def _item_source(item: dict[str, Any]) -> str:
    """Resolve the source identifier for an evaluation fixture item."""
    return str(item.get("source") or "curated_dev")


def _item_stratum(item: dict[str, Any]) -> str:
    """Resolve the evaluation stratum (explicit stratum, difficulty, or phenomenon)."""
    return str(item.get("stratum") or item.get("difficulty") or item.get("phenomenon") or "default")


def _increment_bucket(
    buckets: dict[str, dict[str, int]],
    key: str,
    is_correct: bool,
) -> None:
    """Increment total and correct counts for a breakdown bucket."""
    bucket = buckets.setdefault(key, {"total": 0, "correct": 0})
    bucket["total"] += 1
    if is_correct:
        bucket["correct"] += 1


def _summarize_breakdown(counts: dict[str, dict[str, int]]) -> dict[str, dict[str, Any]]:
    """Convert raw total/correct bucket counts into accuracy summaries."""
    return {
        key: {
            "total": bucket["total"],
            "correct": bucket["correct"],
            "accuracy": (round(bucket["correct"] / bucket["total"], 4) if bucket["total"] else 0.0),
        }
        for key, bucket in counts.items()
    }


def _compute_mode_metrics(
    items: list[dict[str, Any]],
    predictions: list[tuple[EntailmentLabel, float]],
    latencies_ms: list[float],
    total_ms: float,
) -> dict[str, Any]:
    """Compute classification, calibration, stratified breakdowns, and latency metrics."""
    tp = fp = tn = fn = 0
    brier_sum = 0.0
    misclassified_ids: list[str] = []
    has_difficulty = any("difficulty" in item for item in items)
    diff_counts: dict[str, dict[str, int]] = (
        {d: {"total": 0, "correct": 0} for d in DIFFICULTIES} if has_difficulty else {}
    )
    lang_counts: dict[str, dict[str, int]] = {
        lang: {"total": 0, "correct": 0} for lang in LANGUAGES
    }
    source_counts: dict[str, dict[str, int]] = {}
    stratum_counts: dict[str, dict[str, int]] = {}

    for item, (pred_label, prob) in zip(items, predictions, strict=True):
        expected = item["expected_label"]
        y_true = 1.0 if expected == "ENTAIL" else 0.0
        brier_sum += (prob - y_true) ** 2
        is_correct = pred_label == expected

        if has_difficulty and "difficulty" in item:
            _increment_bucket(diff_counts, item["difficulty"], is_correct)
        _increment_bucket(lang_counts, item["language"], is_correct)
        _increment_bucket(source_counts, _item_source(item), is_correct)
        _increment_bucket(stratum_counts, _item_stratum(item), is_correct)

        if is_correct:
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

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "brier_score": brier_score,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "by_difficulty": _summarize_breakdown(diff_counts) if has_difficulty else {},
        "by_stratum": _summarize_breakdown(stratum_counts),
        "by_source": _summarize_breakdown(source_counts),
        "by_language": _summarize_breakdown(lang_counts),
        "latency": _compute_latency_stats(latencies_ms, total_ms),
        "misclassified_ids": misclassified_ids,
    }


def _run_tier1_mode(
    items: list[dict[str, Any]], threshold: float
) -> tuple[list[tuple[EntailmentLabel, float]], list[float], float]:
    """Evaluate all pairs using Tier 1 deterministic rules and fallback score only."""
    predictions: list[tuple[EntailmentLabel, float]] = []
    latencies_ms: list[float] = []

    batch_start = time.perf_counter()
    for item in items:
        t0 = time.perf_counter()
        t1 = evaluate_tier1(item["sentence1"], item["sentence2"])
        prob = float(t1.score if t1.decisive else t1.fallback_score)
        label: EntailmentLabel = (
            t1.label if t1.decisive else ("ENTAIL" if prob >= threshold else "NO_ENTAIL")
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        predictions.append((label, prob))
    total_ms = (time.perf_counter() - batch_start) * 1000.0
    return predictions, latencies_ms, total_ms


def _run_tier2_mode(
    items: list[dict[str, Any]],
    scoring_mode: ScoringMode = "mean",
) -> tuple[list[tuple[EntailmentLabel, float]], list[float], float]:
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
        apply_lexical_surface=False,
        scoring_mode=scoring_mode,
    )
    total_ms = (time.perf_counter() - batch_start) * 1000.0

    latencies_ms: list[float] = []
    for pair, t1 in zip(pairs, neutral_t1, strict=True):
        t0 = time.perf_counter()
        nli_engine.score_pairs_batch(
            [pair],
            [t1],
            escalation_judge=_NOOP_ESCALATION_JUDGE,
            apply_lexical_surface=False,
            scoring_mode=scoring_mode,
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    predictions = [(detail.label, float(detail.score)) for detail in batch_details]
    return predictions, latencies_ms, total_ms


def _run_hybrid_mode(
    items: list[dict[str, Any]],
    scoring_mode: ScoringMode = "mean",
) -> tuple[list[tuple[EntailmentLabel, float]], list[float], float]:
    """Evaluate all pairs using judge_service's Tier 1 fast-gate + Tier 2 batch pipeline."""
    pairs = [(item["sentence1"], item["sentence2"]) for item in items]

    batch_start = time.perf_counter()
    outputs = evaluate_uncached_pairs(
        pairs,
        escalation_judge=_NOOP_ESCALATION_JUDGE,
        scoring_mode=scoring_mode,
    )
    total_ms = (time.perf_counter() - batch_start) * 1000.0
    predictions = [(out.label, float(out.score)) for out in outputs]

    latencies_ms: list[float] = []
    for pair in pairs:
        t0 = time.perf_counter()
        evaluate_uncached_pairs(
            [pair],
            escalation_judge=_NOOP_ESCALATION_JUDGE,
            scoring_mode=scoring_mode,
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    return predictions, latencies_ms, total_ms


def _build_dataset_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize pair counts across difficulty, language, labels, source, and stratum."""
    has_difficulty = any("difficulty" in item for item in items)
    by_difficulty = {d: 0 for d in DIFFICULTIES} if has_difficulty else {}
    by_language = {lang: 0 for lang in LANGUAGES}
    by_label = {lbl: 0 for lbl in LABELS}
    by_source: dict[str, int] = {}
    by_stratum: dict[str, int] = {}

    for item in items:
        if has_difficulty and "difficulty" in item:
            diff = item["difficulty"]
            by_difficulty[diff] = by_difficulty.get(diff, 0) + 1
        lang = item["language"]
        lbl = item["expected_label"]
        src = _item_source(item)
        stratum = _item_stratum(item)
        by_language[lang] = by_language.get(lang, 0) + 1
        by_label[lbl] = by_label.get(lbl, 0) + 1
        by_source[src] = by_source.get(src, 0) + 1
        by_stratum[stratum] = by_stratum.get(stratum, 0) + 1

    return {
        "total_pairs": len(items),
        "by_difficulty": by_difficulty,
        "by_language": by_language,
        "by_label": by_label,
        "by_source": by_source,
        "by_stratum": by_stratum,
    }


def _resolve_fixture(dataset: str, custom_fixture: str | None) -> tuple[str, Path]:
    """Resolve dataset choice or explicit fixture path to (dataset_name, fixture_path)."""
    if custom_fixture is not None:
        return "custom", Path(custom_fixture)
    if dataset == "holdout":
        return "holdout", DEFAULT_HOLDOUT_PATH
    return "dev", DEFAULT_GAUNTLET_PATH


def _load_and_validate_fixture(fixture_path: Path) -> list[dict[str, Any]]:
    """Load and validate an evaluation fixture JSON file, raising CommandError on errors."""
    if not fixture_path.exists():
        raise CommandError(f"Evaluation gauntlet fixture not found: {fixture_path}")
    try:
        raw = json.loads(fixture_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CommandError(f"Invalid JSON in evaluation fixture {fixture_path}: {exc}") from exc

    if not isinstance(raw, list) or not raw:
        raise CommandError(f"Evaluation fixture must be a non-empty JSON array: {fixture_path}")

    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise CommandError(f"Fixture item #{idx} must be a JSON object.")
        for key in ("id", "sentence1", "sentence2"):
            val = item.get(key)
            if not isinstance(val, str) or not val.strip():
                raise CommandError(f"Fixture item #{idx} missing non-empty string field '{key}'.")
        if item.get("language") not in LANGUAGES:
            raise CommandError(
                f"Fixture item {item['id']} has invalid language {item.get('language')!r}; "
                f"expected one of {LANGUAGES}."
            )
        if item.get("expected_label") not in LABELS:
            raise CommandError(
                f"Fixture item {item['id']} has invalid expected_label "
                f"{item.get('expected_label')!r}; expected one of {LABELS}."
            )
        if "difficulty" in item and item["difficulty"] is not None:
            if item["difficulty"] not in DIFFICULTIES:
                raise CommandError(
                    f"Fixture item {item['id']} has invalid difficulty "
                    f"{item['difficulty']!r}; expected one of {DIFFICULTIES}."
                )
        for opt_key in ("stratum", "source"):
            if opt_key in item and item[opt_key] is not None:
                if not isinstance(item[opt_key], str) or not item[opt_key].strip():
                    raise CommandError(
                        f"Fixture item {item['id']} field '{opt_key}' must be a non-empty string."
                    )
    return raw


def _get_git_sha() -> str:
    """Return current HEAD commit SHA, or 'unknown' if git metadata is unavailable."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=settings.BASE_DIR,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return proc.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _format_fixture_path(fixture_path: Path) -> str:
    """Return fixture path relative to BASE_DIR when inside repo."""
    base_dir = Path(settings.BASE_DIR).resolve()
    resolved = fixture_path.resolve()
    try:
        return str(resolved.relative_to(base_dir))
    except ValueError:
        return str(resolved)


def _get_model_revision(model_name: str) -> str:
    """Resolve model commit revision via loaded engine, AutoConfig, or HF cache snapshot."""
    rev = nli_engine.get_model_revision()
    if rev:
        return rev
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_name, local_files_only=True)
        rev = _resolve_hf_model_revision(model_name, cfg)
        if rev:
            return rev
    except Exception:
        pass
    rev = _resolve_hf_model_revision(model_name)
    return rev or "unknown"


def _get_hardware_info() -> dict[str, Any]:
    """Collect runtime OS, CPU, PyTorch, and accelerator metadata."""
    torch_version = "unavailable"
    cuda_available = False
    device = "cpu"
    try:
        import torch

        torch_version = torch.__version__
        cuda_available = bool(torch.cuda.is_available())
        device = "cuda" if cuda_available else "cpu"
    except Exception:
        pass
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "device": device,
    }


def _build_provenance(
    dataset_name: str,
    fixture_path: Path,
    scoring_mode: ScoringMode,
    model_name: str,
    threshold: float,
    as_json: bool,
) -> dict[str, Any]:
    """Assemble full reproducible provenance metadata for the evaluation run."""
    fixture_rel = _format_fixture_path(fixture_path)
    cmd_parts = [".venv/bin/python", "manage.py", "run_eval"]
    if dataset_name == "custom":
        cmd_parts.extend(["--fixture", fixture_rel])
    else:
        cmd_parts.extend(["--dataset", dataset_name])
    cmd_parts.extend(["--scoring", scoring_mode])
    if as_json:
        cmd_parts.append("--json")

    return {
        "git_sha": _get_git_sha(),
        "dataset_name": dataset_name,
        "fixture_path": fixture_rel,
        "scoring_mode": scoring_mode,
        "model_name": model_name,
        "model_revision": _get_model_revision(model_name),
        "hardware": _get_hardware_info(),
        "inference_config": {
            "max_length": NLI_MAX_LENGTH,
            "batch_size": NLI_BATCH_SIZE,
            "entail_threshold": threshold,
            "entail_margin": None,
            "contradict_cap": NLI_CONTRADICT_CAP,
            "lexical_contradict_cap": NLI_LEXICAL_CONTRADICT_CAP,
            "directional_hypernym_guard": {
                "max_entail": NLI_DIRECTIONAL_HIGH,
                "min_entail": NLI_DIRECTIONAL_LOW,
            },
            "scoring_mode": scoring_mode,
            "apply_lexical_surface_tier2": False,
            "apply_lexical_surface_hybrid": True,
        },
        "command": " ".join(cmd_parts),
    }


def _append_breakdown_table(
    lines: list[str],
    title: str,
    col_header: str,
    keys: list[str],
    modes: dict[str, Any],
    breakdown_key: str,
) -> None:
    """Append a Markdown accuracy breakdown table across tier1, tier2, and hybrid."""
    if not keys:
        return
    lines.extend(
        [
            "",
            f"## {title}",
            "",
            f"| {col_header} | tier1 | tier2 | hybrid |",
            "|---|---:|---:|---:|",
        ]
    )
    for key in keys:
        cells = []
        for mode_name in ("tier1", "tier2", "hybrid"):
            bucket = modes[mode_name][breakdown_key].get(
                key, {"accuracy": 0.0, "correct": 0, "total": 0}
            )
            cells.append(f"{bucket['accuracy']:.4f} ({bucket['correct']}/{bucket['total']})")
        lines.append(f"| `{key}` | {cells[0]} | {cells[1]} | {cells[2]} |")


def _format_markdown_report(report: dict[str, Any]) -> str:
    """Render the evaluation provenance report as clean Markdown tables."""
    dataset = report["dataset"]
    model = report["model"]
    modes = report["modes"]
    prov = report["provenance"]
    hw = prov["hardware"]

    lang_summary = ", ".join(f"{k}={v}" for k, v in dataset["by_language"].items())
    label_summary = ", ".join(f"{k}={v}" for k, v in dataset["by_label"].items())
    source_summary = ", ".join(f"{k}={v}" for k, v in dataset["by_source"].items())

    lines = [
        "# Evaluation Gauntlet Provenance Report",
        "",
        (
            f"- **Dataset:** `{prov['dataset_name']}` (`{dataset['total_pairs']}` pairs; "
            f"Sources: {source_summary}; Language: {lang_summary}; Labels: {label_summary})"
        ),
        (
            f"- **Model:** `{model['name']}` (revision=`{prov['model_revision']}`, "
            f"threshold=`{model['threshold']}`, scoring=`{prov['scoring_mode']}`)"
        ),
        (
            f"- **Provenance:** `git_sha={prov['git_sha']}`, `fixture={prov['fixture_path']}`, "
            f"`device={hw['device']}` (`torch={hw['torch_version']}`, `{hw['machine']}`, "
            f"`cpus={hw['cpu_count']}`)"
        ),
        f"- **Command:** `{prov['command']}`",
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

    if dataset["by_difficulty"]:
        _append_breakdown_table(
            lines,
            "Per-Difficulty Accuracy",
            "Difficulty",
            list(DIFFICULTIES),
            modes,
            "by_difficulty",
        )
    _append_breakdown_table(
        lines,
        "Per-Stratum Accuracy",
        "Stratum",
        list(dataset["by_stratum"].keys()),
        modes,
        "by_stratum",
    )
    _append_breakdown_table(
        lines,
        "Per-Source Accuracy",
        "Source",
        list(dataset["by_source"].keys()),
        modes,
        "by_source",
    )
    _append_breakdown_table(
        lines,
        "Per-Language Accuracy",
        "Language",
        list(dataset["by_language"].keys()),
        modes,
        "by_language",
    )

    lines.extend(["", "## Misclassified IDs", ""])
    for mode_name in ("tier1", "tier2", "hybrid"):
        failed = modes[mode_name]["misclassified_ids"]
        failed_str = ", ".join(failed) if failed else "None"
        lines.append(f"- **`{mode_name}` ({len(failed)}):** {failed_str}")

    return "\n".join(lines)


class Command(BaseCommand):
    """Benchmark Tier 1, Tier 2, and Hybrid entailment engines on dev or holdout datasets."""

    help = "Run the 60-pair evaluation benchmark across tier1, tier2, and hybrid modes."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--json",
            action="store_true",
            help="Emit structured JSON provenance report instead of Markdown.",
        )
        parser.add_argument(
            "--dataset",
            choices=DATASETS,
            default="dev",
            help="Dataset to evaluate ('dev' or 'holdout'). Ignored when --fixture is given.",
        )
        parser.add_argument(
            "--scoring",
            choices=SCORING_MODES,
            default="mean",
            help="Bidirectional NLI scoring mode ('mean' or 'min').",
        )
        parser.add_argument(
            "--fixture",
            default=None,
            help="Optional path to a custom evaluation JSON fixture.",
        )

    def handle(self, *args, **options) -> None:
        dataset_name, fixture_path = _resolve_fixture(options["dataset"], options["fixture"])
        items = _load_and_validate_fixture(fixture_path)
        scoring_mode: ScoringMode = options["scoring"]
        threshold = float(getattr(settings, "ENTAIL_THRESHOLD", 0.50))
        model_name = getattr(
            settings,
            "NLI_MODEL_NAME",
            "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli",
        )

        # Pre-warm NLI cross-encoder before timing so weight loading does not skew latencies
        nli_engine.warm_up()

        t1_preds, t1_lats, t1_total = _run_tier1_mode(items, threshold)
        t2_preds, t2_lats, t2_total = _run_tier2_mode(items, scoring_mode=scoring_mode)
        hy_preds, hy_lats, hy_total = _run_hybrid_mode(items, scoring_mode=scoring_mode)

        report = {
            "provenance": _build_provenance(
                dataset_name=dataset_name,
                fixture_path=fixture_path,
                scoring_mode=scoring_mode,
                model_name=model_name,
                threshold=threshold,
                as_json=bool(options["json"]),
            ),
            "dataset": _build_dataset_summary(items),
            "model": {
                "name": model_name,
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
