"""
Management command to benchmark Tier 1, Tier 2, and Hybrid NLI modes across the
60-pair dev gauntlet or independent held-out benchmark without mutating DB tables.
"""

from __future__ import annotations

import json
import math
import os
import platform
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from assignment.services.judge_service import SCORING_POLICY_VERSION
from assignment.services.nli_engine import (
    NLI_BATCH_SIZE,
    NLI_MAX_LENGTH,
    HuggingFaceNLIEngine,
    NLIScoreDetail,
    _resolve_hf_model_revision,
    nli_engine,
)
from assignment.services.normalizer import (
    Tier1Result,
    evaluate_tier1,
    normalize_sentence,
)
from assignment.types import EntailmentLabel

AblationMode = Literal["exact_plus_nli", "nli_only"]

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures"
DEFAULT_GAUNTLET_PATH = FIXTURES_DIR / "eval_gauntlet.json"
DEFAULT_HOLDOUT_PATH = FIXTURES_DIR / "eval_holdout.json"
DEFAULT_FINAL_HOLDOUT_PATH = FIXTURES_DIR / "eval_final_holdout.json"
DIFFICULTIES = ("easy", "medium", "hard", "edge")
LANGUAGES = ("ja", "en")
LABELS: tuple[EntailmentLabel, ...] = ("ENTAIL", "NO_ENTAIL")
DATASETS = ("dev", "dev_diagnostic", "holdout", "final_holdout")
DATASET_PATHS: dict[str, Path] = {
    "dev": DEFAULT_GAUNTLET_PATH,
    "dev_diagnostic": DEFAULT_HOLDOUT_PATH,
    "holdout": DEFAULT_HOLDOUT_PATH,
    "final_holdout": DEFAULT_FINAL_HOLDOUT_PATH,
}
ABLATION_MODES: tuple[AblationMode, ...] = ("exact_plus_nli", "nli_only")


@dataclass(frozen=True)
class _PairDiagnostic:
    origin_tier: str
    tier1_reason: str
    directional_nli: dict[str, float] | None


def _directional_nli_payload(detail: NLIScoreDetail) -> dict[str, float]:
    """Serialize directional entailment and contradiction probabilities from NLIScoreDetail."""
    return {
        "p_entail_fwd": detail.forward_entail,
        "p_entail_rev": detail.reverse_entail,
        "p_contra_fwd": detail.forward_contradict,
        "p_contra_rev": detail.reverse_contradict,
    }


def _wilson_ci_95(correct: int, total: int, z: float = 1.95996) -> list[float]:
    """Compute clamped 95% Wilson score confidence interval [lo, hi] for binomial accuracy."""
    if total <= 0:
        return [0.0, 0.0]
    p = max(0.0, min(1.0, correct / total))
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    half_width = (z * math.sqrt((p * (1.0 - p)) / total + z2 / (4.0 * total * total))) / denom
    lo = max(0.0, min(1.0, center - half_width))
    hi = max(0.0, min(1.0, center + half_width))
    return [round(lo, 4), round(hi, 4)]


def _classify_error_category(item: dict[str, Any]) -> str:
    """Deterministically map fixture metadata (stratum/phenomenon/source) to an error category."""
    tag = str(item.get("stratum") or item.get("phenomenon") or "").strip().lower()
    source = str(item.get("source") or "").strip().lower()

    if tag in {
        "adversarial_swap",
        "role_inversion",
        "word_order_or_particle_swap",
        "word_order",
        "particle_swap",
        "scrambling",
        "voice_inversion",
    } or any(tok in tag for tok in ("swap", "inversion", "scram")):
        return "word_order_or_particle_swap"

    if tag in {
        "oneway_entail",
        "oneway_specificity",
        "directional_hypernym",
        "hypernym",
        "hyponym",
    } or any(tok in tag for tok in ("oneway", "hypernym", "hyponym", "specificity")):
        return "oneway_specificity"

    if tag in {
        "contradict",
        "antonym_contradiction",
        "negation_flip",
        "double_negation",
        "restrictive_particle",
        "negation_or_antonym",
    } or any(tok in tag for tok in ("negat", "antonym", "contradict")):
        return "negation_or_antonym"

    if tag == "caption_codescription" or "codescription" in tag or "caption" in tag:
        return "caption_codescription"
    if tag in {"ja_jnli_bidir_entail", "ja_jnli_neutral"} or (
        source == "jglue_jnli" and tag in {"bidir_entail", "neutral"}
    ):
        return "caption_codescription"

    if tag in {
        "lexical_synonym_gap",
        "lexical_paraphrase",
        "paraphrase",
        "ja_paws_paraphrase",
        "en_paws_paraphrase",
        "bidir_entail",
        "register_shift",
        "pro_drop",
        "colloquial_contraction",
        "loanword_mixed",
        "yakuwarigo",
    }:
        return "lexical_synonym_gap"

    return "other"


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
    """Convert raw total/correct bucket counts into accuracy and Wilson CI summaries."""
    return {
        key: {
            "total": bucket["total"],
            "correct": bucket["correct"],
            "accuracy": (round(bucket["correct"] / bucket["total"], 4) if bucket["total"] else 0.0),
            "accuracy_ci95": _wilson_ci_95(bucket["correct"], bucket["total"]),
        }
        for key, bucket in counts.items()
    }


def _compute_mode_metrics(
    items: list[dict[str, Any]],
    predictions: list[tuple[EntailmentLabel, float]],
    latencies_ms: list[float],
    total_ms: float,
    diagnostics: list[_PairDiagnostic] | None = None,
    include_diagnostics: bool = False,
) -> dict[str, Any]:
    """Compute classification, calibration, stratified breakdowns, and latency metrics."""
    tp = fp = tn = fn = 0
    brier_sum = 0.0
    misclassified_ids: list[str] = []
    misclassifications: list[dict[str, Any]] = []
    has_difficulty = any(item.get("difficulty") is not None for item in items)
    diff_counts: dict[str, dict[str, int]] = (
        {d: {"total": 0, "correct": 0} for d in DIFFICULTIES} if has_difficulty else {}
    )
    lang_counts: dict[str, dict[str, int]] = {
        lang: {"total": 0, "correct": 0} for lang in LANGUAGES
    }
    source_counts: dict[str, dict[str, int]] = {}
    stratum_counts: dict[str, dict[str, int]] = {}

    diag_seq = (
        diagnostics
        if diagnostics is not None
        else [_PairDiagnostic("tier1_rule", "unknown", None) for _ in items]
    )

    for item, (pred_label, prob), diag in zip(items, predictions, diag_seq, strict=True):
        expected = item["expected_label"]
        y_true = 1.0 if expected == "ENTAIL" else 0.0
        brier_sum += (prob - y_true) ** 2
        is_correct = pred_label == expected

        if has_difficulty and item.get("difficulty") is not None:
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
            if include_diagnostics:
                misclassifications.append(
                    {
                        "id": item["id"],
                        "source": _item_source(item),
                        "source_id": item.get("source_id"),
                        "language": item["language"],
                        "stratum": _item_stratum(item),
                        "sentence1": item["sentence1"],
                        "sentence2": item["sentence2"],
                        "expected_label": expected,
                        "predicted_label": pred_label,
                        "score": round(float(prob), 4),
                        "origin_tier": diag.origin_tier,
                        "tier1_reason": diag.tier1_reason,
                        "directional_nli": diag.directional_nli,
                        "error_category": _classify_error_category(item),
                    }
                )

    total = len(items)
    correct_total = tp + tn
    accuracy = round(correct_total / total, 4) if total else 0.0
    precision = round(tp / (tp + fp), 4) if (tp + fp) else 0.0
    recall = round(tp / (tp + fn), 4) if (tp + fn) else 0.0
    f1 = round((2 * tp) / (2 * tp + fp + fn), 4) if (2 * tp + fp + fn) else 0.0
    specificity = round(tn / (tn + fp), 4) if (tn + fp) else 0.0
    brier_score = round(brier_sum / total, 6) if total else 0.0

    result: dict[str, Any] = {
        "accuracy": accuracy,
        "accuracy_ci95": _wilson_ci_95(correct_total, total),
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
    if include_diagnostics:
        result["misclassifications"] = misclassifications
    return result


def _run_tier1_mode_with_diagnostics(
    items: list[dict[str, Any]], threshold: float
) -> tuple[list[tuple[EntailmentLabel, float]], list[float], float, list[_PairDiagnostic]]:
    """Evaluate all pairs using Tier 1 deterministic rules and collect pair diagnostics."""
    predictions: list[tuple[EntailmentLabel, float]] = []
    latencies_ms: list[float] = []
    diagnostics: list[_PairDiagnostic] = []

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
        origin_tier = (
            "tier1_exact"
            if (t1.decisive and t1.reason == "exact_normalized_match")
            else "tier1_rule"
        )
        diagnostics.append(
            _PairDiagnostic(
                origin_tier=origin_tier,
                tier1_reason=t1.reason,
                directional_nli=None,
            )
        )
    total_ms = (time.perf_counter() - batch_start) * 1000.0
    return predictions, latencies_ms, total_ms, diagnostics


def _run_tier1_mode(
    items: list[dict[str, Any]], threshold: float
) -> tuple[list[tuple[EntailmentLabel, float]], list[float], float]:
    """Evaluate all pairs using Tier 1 deterministic rules and fallback score only."""
    preds, latencies_ms, total_ms, _ = _run_tier1_mode_with_diagnostics(items, threshold)
    return preds, latencies_ms, total_ms


def _neutral_tier1_result(s1: str, s2: str, reason: str) -> Tier1Result:
    """Build a non-decisive zero-fallback Tier1Result for raw NLI evaluation."""
    return Tier1Result(
        decisive=False,
        label="NO_ENTAIL",
        score=0.0,
        reason=reason,
        canonical_s1=normalize_sentence(s1),
        canonical_s2=normalize_sentence(s2),
        fallback_score=0.0,
    )


def _run_tier2_mode_with_diagnostics(
    items: list[dict[str, Any]],
    engine: HuggingFaceNLIEngine | None = None,
) -> tuple[list[tuple[EntailmentLabel, float]], list[float], float, list[_PairDiagnostic]]:
    """Evaluate all pairs using raw Tier 2 NLI cross-encoder and collect pair diagnostics."""
    active_engine = engine if engine is not None else nli_engine
    pairs = [(item["sentence1"], item["sentence2"]) for item in items]
    neutral_t1 = [_neutral_tier1_result(s1, s2, "tier2_only") for s1, s2 in pairs]

    batch_start = time.perf_counter()
    batch_details = active_engine.score_pairs_batch(pairs, neutral_t1)
    total_ms = (time.perf_counter() - batch_start) * 1000.0

    latencies_ms: list[float] = []
    for pair, t1 in zip(pairs, neutral_t1, strict=True):
        t0 = time.perf_counter()
        active_engine.score_pairs_batch([pair], [t1])
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    predictions = [(detail.label, float(detail.score)) for detail in batch_details]
    diagnostics = [
        _PairDiagnostic(
            origin_tier="tier2_nli",
            tier1_reason="tier2_only",
            directional_nli=_directional_nli_payload(detail),
        )
        for detail in batch_details
    ]
    return predictions, latencies_ms, total_ms, diagnostics


def _run_tier2_mode(
    items: list[dict[str, Any]],
    engine: HuggingFaceNLIEngine | None = None,
) -> tuple[list[tuple[EntailmentLabel, float]], list[float], float]:
    """Evaluate all pairs using raw Tier 2 NLI cross-encoder without Tier 1 rules."""
    preds, latencies_ms, total_ms, _ = _run_tier2_mode_with_diagnostics(items, engine=engine)
    return preds, latencies_ms, total_ms


def _hybrid_fast_gate(
    s1: str, s2: str
) -> tuple[tuple[EntailmentLabel, float] | None, _PairDiagnostic | None, Tier1Result]:
    """Resolve Tier 1 exact-match fast-gate behavior matching production exact_plus_nli."""
    t1 = evaluate_tier1(s1, s2)
    if t1.decisive:
        return (
            (t1.label, round(float(t1.score), 4)),
            _PairDiagnostic(
                origin_tier="tier1_exact",
                tier1_reason=t1.reason,
                directional_nli=None,
            ),
            t1,
        )
    return None, None, t1


def _run_hybrid_mode_with_diagnostics(
    items: list[dict[str, Any]],
    ablation: AblationMode = "exact_plus_nli",
    engine: HuggingFaceNLIEngine | None = None,
) -> tuple[list[tuple[EntailmentLabel, float]], list[float], float, list[_PairDiagnostic]]:
    """Evaluate pairs under the selected offline ablation mode and collect diagnostics."""
    active_engine = engine if engine is not None else nli_engine
    if ablation == "nli_only":
        return _run_tier2_mode_with_diagnostics(items, engine=active_engine)

    pairs = [(item["sentence1"], item["sentence2"]) for item in items]

    predictions: list[tuple[EntailmentLabel, float] | None] = [None] * len(pairs)
    diagnostics: list[_PairDiagnostic | None] = [None] * len(pairs)
    unresolved: list[tuple[int, str, str, Tier1Result]] = []

    batch_start = time.perf_counter()
    for idx, (s1, s2) in enumerate(pairs):
        pred, diag, t1_input = _hybrid_fast_gate(s1, s2)
        if pred is not None and diag is not None:
            predictions[idx] = pred
            diagnostics[idx] = diag
        else:
            unresolved.append((idx, s1, s2, t1_input))

    if unresolved:
        nli_inputs = [(s1, s2) for _, s1, s2, _ in unresolved]
        t1_list = [t1_input for _, _, _, t1_input in unresolved]
        nli_details = active_engine.score_pairs_batch(nli_inputs, t1_list)
        for (idx, _s1, _s2, t1_input), detail in zip(unresolved, nli_details, strict=True):
            predictions[idx] = (detail.label, round(float(detail.score), 4))
            diagnostics[idx] = _PairDiagnostic(
                origin_tier="tier2_nli",
                tier1_reason=t1_input.reason,
                directional_nli=_directional_nli_payload(detail),
            )
    total_ms = (time.perf_counter() - batch_start) * 1000.0

    latencies_ms: list[float] = []
    for s1, s2 in pairs:
        t0 = time.perf_counter()
        pred, _diag, t1_input = _hybrid_fast_gate(s1, s2)
        if pred is None:
            active_engine.score_pairs_batch([(s1, s2)], [t1_input])
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    resolved_preds: list[tuple[EntailmentLabel, float]] = []
    resolved_diags: list[_PairDiagnostic] = []
    for pred, diag in zip(predictions, diagnostics, strict=True):
        if pred is None or diag is None:
            raise RuntimeError("Missing hybrid ablation prediction or diagnostic")
        resolved_preds.append(pred)
        resolved_diags.append(diag)
    return resolved_preds, latencies_ms, total_ms, resolved_diags


def _run_hybrid_mode(
    items: list[dict[str, Any]],
    ablation: AblationMode = "exact_plus_nli",
    engine: HuggingFaceNLIEngine | None = None,
) -> tuple[list[tuple[EntailmentLabel, float]], list[float], float]:
    """Evaluate all pairs using Tier 1 exact-match fast-gate + Tier 2 batch pipeline."""
    preds, latencies_ms, total_ms, _ = _run_hybrid_mode_with_diagnostics(
        items, ablation=ablation, engine=engine
    )
    return preds, latencies_ms, total_ms


def _build_dataset_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize pair counts across difficulty, language, labels, source, and stratum."""
    has_difficulty = any(item.get("difficulty") is not None for item in items)
    by_difficulty = {d: 0 for d in DIFFICULTIES} if has_difficulty else {}
    by_language = {lang: 0 for lang in LANGUAGES}
    by_label = {lbl: 0 for lbl in LABELS}
    by_source: dict[str, int] = {}
    by_stratum: dict[str, int] = {}

    for item in items:
        if has_difficulty and item.get("difficulty") is not None:
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
    if dataset in DATASET_PATHS:
        return dataset, DATASET_PATHS[dataset]
    return "dev", DEFAULT_GAUNTLET_PATH


def _load_and_validate_fixture(fixture_path: Path) -> list[dict[str, Any]]:
    """Load and validate an evaluation fixture JSON file, raising CommandError on errors."""
    if not fixture_path.is_file():
        raise CommandError(f"Evaluation gauntlet fixture not found: {fixture_path}")
    try:
        raw_text = fixture_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise CommandError(f"Unable to read evaluation fixture {fixture_path}: {exc}") from exc
    try:
        raw = json.loads(raw_text)
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
        if "source_id" in item and item["source_id"] is not None:
            sid = item["source_id"]
            if isinstance(sid, bool) or not (
                isinstance(sid, int) or (isinstance(sid, str) and sid.strip())
            ):
                raise CommandError(
                    f"Fixture item {item['id']} field 'source_id' must be a "
                    "non-empty string or integer."
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


def _get_model_revision(model_name: str, engine: HuggingFaceNLIEngine | None = None) -> str:
    """Resolve model commit revision from a matching engine, AutoConfig, or HF cache snapshot."""
    active_engine = engine if engine is not None else nli_engine
    if active_engine.model_name == model_name:
        rev = active_engine.get_model_revision()
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
    model_name: str,
    threshold: float,
    as_json: bool,
    ablation: AblationMode = "exact_plus_nli",
    diagnostics: bool = False,
    custom_model_name: str | None = None,
    engine: HuggingFaceNLIEngine | None = None,
) -> dict[str, Any]:
    """Assemble full reproducible provenance metadata for the evaluation run."""
    fixture_rel = _format_fixture_path(fixture_path)
    cmd_parts = [".venv/bin/python", "manage.py", "run_eval"]
    if dataset_name == "custom":
        cmd_parts.extend(["--fixture", fixture_rel])
    else:
        cmd_parts.extend(["--dataset", dataset_name])
    cmd_parts.extend(["--ablation", ablation])
    if custom_model_name is not None:
        cmd_parts.extend(["--model-name", custom_model_name])
    if diagnostics:
        cmd_parts.append("--diagnostics")
    if as_json:
        cmd_parts.append("--json")

    scoring_rule = "min(p_entail_forward, p_entail_reverse)"
    return {
        "git_sha": _get_git_sha(),
        "dataset_name": dataset_name,
        "fixture_path": fixture_rel,
        "scoring_policy": SCORING_POLICY_VERSION,
        "scoring_rule": scoring_rule,
        "ablation": ablation,
        "model_name": model_name,
        "model_revision": _get_model_revision(model_name, engine=engine),
        "hardware": _get_hardware_info(),
        "inference_config": {
            "max_length": NLI_MAX_LENGTH,
            "batch_size": NLI_BATCH_SIZE,
            "entail_threshold": threshold,
            "entail_margin": None,
            "contradict_cap": None,
            "ablation": ablation,
            "scoring_policy": SCORING_POLICY_VERSION,
            "scoring_rule": scoring_rule,
        },
        "command": shlex.join(cmd_parts),
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
            f"threshold=`{model['threshold']}`, policy=`{prov['scoring_policy']}`, "
            f"ablation=`{prov.get('ablation', 'exact_plus_nli')}`)"
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
            "| Mode | Accuracy | 95% CI | Precision | Recall | F1 | Specificity | Brier Score | "
            "Batch Total (ms) | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | CM (TP/FP/TN/FN) |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for mode_name in ("tier1", "tier2", "hybrid"):
        m = modes[mode_name]
        cm = m["confusion_matrix"]
        lat = m["latency"]
        ci_lo, ci_hi = m.get("accuracy_ci95", [0.0, 0.0])
        cm_str = f"{cm['tp']}/{cm['fp']}/{cm['tn']}/{cm['fn']}"
        lines.append(
            f"| `{mode_name}` | {m['accuracy']:.4f} | [{ci_lo:.4f}, {ci_hi:.4f}] | "
            f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['specificity']:.4f} | "
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

    if any("misclassifications" in modes[mode_name] for mode_name in ("tier1", "tier2", "hybrid")):
        lines.extend(["", "## Misclassification Diagnostics", ""])
        for mode_name in ("tier1", "tier2", "hybrid"):
            records = modes[mode_name].get("misclassifications", [])
            lines.append(f"### `{mode_name}` ({len(records)})")
            if not records:
                lines.append("None")
                continue
            for rec in records:
                lines.append(
                    f"- `{rec['id']}` [{rec['language']}/{rec['stratum']}] "
                    f"expected=`{rec['expected_label']}` pred=`{rec['predicted_label']}` "
                    f"score=`{rec['score']:.4f}` origin=`{rec['origin_tier']}` "
                    f"reason=`{rec['tier1_reason']}` category=`{rec['error_category']}`"
                )

    return "\n".join(lines)


class Command(BaseCommand):
    """Benchmark Tier 1, Tier 2, and Hybrid entailment engines on dev or holdout datasets."""

    help = "Run the evaluation benchmark across tier1, tier2, and hybrid modes."

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
            help=(
                "Dataset to evaluate ('dev', 'dev_diagnostic', 'holdout', or 'final_holdout'). "
                "Ignored when --fixture is given."
            ),
        )
        parser.add_argument(
            "--ablation",
            choices=ABLATION_MODES,
            default="exact_plus_nli",
            help="Offline hybrid ablation mode ('exact_plus_nli' or 'nli_only').",
        )
        parser.add_argument(
            "--model-name",
            default=None,
            help="Optional HuggingFace NLI model ID scoped strictly to this offline eval run.",
        )
        parser.add_argument(
            "--diagnostics",
            action="store_true",
            help="Include per-misclassification error taxonomy and directional NLI details.",
        )
        parser.add_argument(
            "--fixture",
            default=None,
            help="Optional path to a custom evaluation JSON fixture.",
        )

    def handle(self, *args, **options) -> None:
        dataset_name, fixture_path = _resolve_fixture(options["dataset"], options["fixture"])
        items = _load_and_validate_fixture(fixture_path)
        ablation: AblationMode = options["ablation"]
        include_diagnostics = bool(options["diagnostics"])
        threshold = float(getattr(settings, "ENTAIL_THRESHOLD", 0.50))

        raw_custom_model: str | None = options["model_name"]
        custom_model_name: str | None = None
        if raw_custom_model is not None:
            custom_model_name = raw_custom_model.strip()
            if not custom_model_name:
                raise CommandError("--model-name must be a non-empty string.")
            if custom_model_name == nli_engine.model_name:
                engine = nli_engine
                model_name = nli_engine.model_name
            else:
                engine = HuggingFaceNLIEngine(model_name=custom_model_name)
                model_name = custom_model_name
        else:
            engine = nli_engine
            model_name = getattr(
                settings,
                "NLI_MODEL_NAME",
                "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli",
            )

        # Pre-warm NLI cross-encoder before timing so weight loading does not skew latencies
        if not engine.warm_up():
            raise CommandError(
                f"Failed to load or warm up NLI model '{model_name}'. "
                "Ensure NLI_ENABLE_MODEL=true and model weights/dependencies are available."
            )

        t1_preds, t1_lats, t1_total, t1_diags = _run_tier1_mode_with_diagnostics(items, threshold)
        t2_preds, t2_lats, t2_total, t2_diags = _run_tier2_mode_with_diagnostics(
            items, engine=engine
        )
        hy_preds, hy_lats, hy_total, hy_diags = _run_hybrid_mode_with_diagnostics(
            items, ablation=ablation, engine=engine
        )

        report = {
            "provenance": _build_provenance(
                dataset_name=dataset_name,
                fixture_path=fixture_path,
                model_name=model_name,
                threshold=threshold,
                as_json=bool(options["json"]),
                ablation=ablation,
                diagnostics=include_diagnostics,
                custom_model_name=custom_model_name,
                engine=engine,
            ),
            "dataset": _build_dataset_summary(items),
            "model": {
                "name": model_name,
                "threshold": threshold,
            },
            "modes": {
                "tier1": _compute_mode_metrics(
                    items,
                    t1_preds,
                    t1_lats,
                    t1_total,
                    diagnostics=t1_diags,
                    include_diagnostics=include_diagnostics,
                ),
                "tier2": _compute_mode_metrics(
                    items,
                    t2_preds,
                    t2_lats,
                    t2_total,
                    diagnostics=t2_diags,
                    include_diagnostics=include_diagnostics,
                ),
                "hybrid": _compute_mode_metrics(
                    items,
                    hy_preds,
                    hy_lats,
                    hy_total,
                    diagnostics=hy_diags,
                    include_diagnostics=include_diagnostics,
                ),
            },
        }

        if options["json"]:
            self.stdout.write(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            self.stdout.write(_format_markdown_report(report))
