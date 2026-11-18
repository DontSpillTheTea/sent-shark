"""
Tier 2 Bidirectional HuggingFace NLI Cross-Encoder.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path

from django.conf import settings

from assignment.services.normalizer import Tier1Result, evaluate_tier1, normalize_sentence
from assignment.types import EntailmentLabel, ScoringMode, TierName

logger = logging.getLogger(__name__)

NLI_MAX_LENGTH = 256
NLI_BATCH_SIZE = 64  # Up to 64 directional premise/hypothesis sequences per pass

_NEGATED_ENTAIL_RE = re.compile(r"(?:^|[^a-z])(?:non|not|no)[\s_-]*entail")
_NEGATED_CONTRADICT_RE = re.compile(r"(?:^|[^a-z])(?:non|not|no)[\s_-]*contradict")


def _validate_scoring_mode(scoring_mode: ScoringMode) -> None:
    if scoring_mode not in ("mean", "min"):
        raise ValueError(f"Unsupported scoring_mode: {scoring_mode!r}")


def _aggregate_entailment(
    p_entail_fwd: float,
    p_entail_rev: float,
    scoring_mode: ScoringMode,
) -> float:
    """Aggregate bidirectional entailment probabilities via mean or min."""
    _validate_scoring_mode(scoring_mode)
    if scoring_mode == "min":
        return min(p_entail_fwd, p_entail_rev)
    return 0.5 * (p_entail_fwd + p_entail_rev)


def _resolve_hf_model_revision(model_name: str, model_config=None) -> str | None:
    """Resolve model commit revision from Transformers config or local HF cache snapshot."""
    commit_hash = getattr(model_config, "_commit_hash", None) if model_config is not None else None
    if isinstance(commit_hash, str) and commit_hash:
        return commit_hash
    try:
        if Path(model_name).is_dir():
            return None
        from huggingface_hub import snapshot_download

        snapshot_path = snapshot_download(model_name, local_files_only=True)
        rev = Path(snapshot_path).name
        return rev or None
    except Exception:
        return None


@dataclass(frozen=True)
class NLIScoreDetail:
    score: float
    label: EntailmentLabel
    tier_used: TierName
    forward_entail: float
    reverse_entail: float
    max_contradiction: float
    forward_contradict: float = 0.0
    reverse_contradict: float = 0.0


class HuggingFaceNLIEngine:
    """
    Singleton wrapper around a pretrained HuggingFace NLI Cross-Encoder.
    Evaluates forward (s1 -> s2) and reverse (s2 -> s1) entailment in batched tensor passes.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or getattr(
            settings,
            "NLI_MODEL_NAME",
            "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli",
        )
        self._lock = threading.Lock()
        self._tokenizer = None
        self._model = None
        self._entail_idx: int | None = None
        self._contradict_idx: int | None = None
        self.model_revision: str | None = None
        self._load_failed = False

    def get_model_revision(self) -> str | None:
        """Return the loaded HuggingFace model revision commit hash, if available."""
        if self.model_revision:
            return self.model_revision
        if self._model is not None:
            self.model_revision = _resolve_hf_model_revision(
                self.model_name, getattr(self._model, "config", None)
            )
        return self.model_revision

    @staticmethod
    def _parse_override_index(value, setting_name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{setting_name} must be a non-negative integer, got {value!r}.")
        return value

    def _validate_resolved_indices(
        self,
        entail_idx: int,
        contradict_idx: int,
        num_labels: int | None,
        source_desc: str,
    ) -> tuple[int, int]:
        if entail_idx == contradict_idx or entail_idx < 0 or contradict_idx < 0:
            raise ValueError(
                f"Invalid NLI label indices ({entail_idx}, {contradict_idx}) "
                f"from {source_desc} for model '{self.model_name}'."
            )
        if num_labels is not None and (entail_idx >= num_labels or contradict_idx >= num_labels):
            raise ValueError(
                f"Resolved NLI label indices ({entail_idx}, {contradict_idx}) "
                f"out of range for num_labels={num_labels} on model '{self.model_name}'."
            )
        return entail_idx, contradict_idx

    def _resolve_label_indices(self, model_config) -> tuple[int, int]:
        id2label = getattr(model_config, "id2label", None)
        if not isinstance(id2label, dict):
            id2label = {}

        raw_num_labels = getattr(model_config, "num_labels", None)
        if (
            isinstance(raw_num_labels, int)
            and not isinstance(raw_num_labels, bool)
            and raw_num_labels > 0
        ):
            num_labels: int | None = raw_num_labels
        elif id2label:
            try:
                num_labels = max(int(k) for k in id2label) + 1
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid id2label keys on model '{self.model_name}': {id2label!r}"
                ) from exc
        else:
            num_labels = None

        entail_override = getattr(settings, "NLI_ENTAIL_INDEX", None)
        contradict_override = getattr(settings, "NLI_CONTRADICT_INDEX", None)

        if (entail_override is None) != (contradict_override is None):
            raise ValueError(
                "Both NLI_ENTAIL_INDEX and NLI_CONTRADICT_INDEX must be configured together."
            )

        if entail_override is not None and contradict_override is not None:
            entail_idx = self._parse_override_index(entail_override, "NLI_ENTAIL_INDEX")
            contradict_idx = self._parse_override_index(contradict_override, "NLI_CONTRADICT_INDEX")
            return self._validate_resolved_indices(
                entail_idx, contradict_idx, num_labels, "explicit settings overrides"
            )

        if not id2label:
            raise ValueError(
                f"Model '{self.model_name}' config.id2label is empty; cannot infer "
                "entailment/contradiction indices. Set NLI_ENTAIL_INDEX and NLI_CONTRADICT_INDEX."
            )

        entail_matches: list[int] = []
        contradict_matches: list[int] = []
        for idx_raw, lbl in id2label.items():
            try:
                idx = int(idx_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Non-integer id2label key {idx_raw!r} on model '{self.model_name}'."
                ) from exc
            low = str(lbl).strip().lower()
            has_entail = "entail" in low and not _NEGATED_ENTAIL_RE.search(low)
            has_contradict = "contradict" in low and not _NEGATED_CONTRADICT_RE.search(low)
            if has_entail and has_contradict:
                raise ValueError(
                    f"Ambiguous id2label entry {idx}: {lbl!r} on model '{self.model_name}'."
                )
            if has_entail:
                entail_matches.append(idx)
            elif has_contradict:
                contradict_matches.append(idx)

        if len(entail_matches) != 1 or len(contradict_matches) != 1:
            raise ValueError(
                f"Unable to resolve entailment/contradiction labels from "
                f"id2label={id2label!r} for model '{self.model_name}'. "
                "Configure NLI_ENTAIL_INDEX and NLI_CONTRADICT_INDEX explicitly."
            )

        return self._validate_resolved_indices(
            entail_matches[0],
            contradict_matches[0],
            num_labels,
            f"id2label={id2label!r}",
        )

    def _ensure_loaded(self) -> bool:
        if not getattr(settings, "NLI_ENABLE_MODEL", True):
            return False
        if self._model is not None:
            return True
        if self._load_failed:
            return False

        with self._lock:
            if self._model is not None:
                return True
            if self._load_failed:
                return False

            try:
                import torch
                from transformers import (
                    AutoModelForSequenceClassification,
                    AutoTokenizer,
                )

                # Guard against multi-thread CPU oversubscription during parallel bursts
                torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))

                logger.info("Loading HuggingFace NLI model: %s", self.model_name)
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                model.eval()

                entail_idx, contradict_idx = self._resolve_label_indices(model.config)
                model_revision = _resolve_hf_model_revision(self.model_name, model.config)

                self._tokenizer = tokenizer
                self._model = model
                self._entail_idx = entail_idx
                self._contradict_idx = contradict_idx
                self.model_revision = model_revision
                return True
            except ValueError:
                raise
            except Exception as exc:
                logger.warning(
                    "Unable to load NLI model '%s'; falling back to Tier 1 scores: %s",
                    self.model_name,
                    exc,
                )
                self._load_failed = True
                return False

    def reset_for_tests(self) -> None:
        with self._lock:
            self._load_failed = False

    def warm_up(self) -> bool:
        """Load the NLI model and run a minimal inference pass so subsequent timings are warm."""
        if not self._ensure_loaded():
            return False
        warm_pair = ("warm up sentence", "warm up sentence")
        self.score_pairs_batch([warm_pair], [evaluate_tier1(*warm_pair)])
        return True

    def score_pairs_batch(
        self,
        pairs: list[tuple[str, str]],
        tier1_results: list[Tier1Result] | None = None,
        scoring_mode: ScoringMode = "mean",
    ) -> list[NLIScoreDetail]:
        """
        Score a batch of sentence pairs using raw normalized bidirectional NLI inference.
        Deduplicates identical directional (premise, hypothesis) pairs before tokenization,
        and falls back cleanly to Tier 1 scores if the transformer model is unavailable.
        """
        _validate_scoring_mode(scoring_mode)
        if not pairs:
            return []

        resolved_t1 = (
            tier1_results
            if tier1_results is not None
            else [evaluate_tier1(s1, s2) for s1, s2 in pairs]
        )
        if len(resolved_t1) != len(pairs):
            raise ValueError("tier1_results length must match pairs length.")

        threshold = float(getattr(settings, "ENTAIL_THRESHOLD", 0.50))

        if not self._ensure_loaded():
            return [
                NLIScoreDetail(
                    score=round(t1.fallback_score, 4),
                    label="ENTAIL" if t1.fallback_score >= threshold else "NO_ENTAIL",
                    tier_used="tier1_rule",
                    forward_entail=t1.fallback_score,
                    reverse_entail=t1.fallback_score,
                    max_contradiction=round(1.0 - t1.fallback_score, 4),
                    forward_contradict=round(1.0 - t1.fallback_score, 4),
                    reverse_contradict=round(1.0 - t1.fallback_score, 4),
                )
                for t1 in resolved_t1
            ]

        import torch

        unique_dir_pairs: list[tuple[str, str]] = []
        dir_index_by_pair: dict[tuple[str, str], int] = {}
        pair_dir_indices: list[tuple[int, int]] = []

        for s1, s2 in pairs:
            n1 = normalize_sentence(s1)
            n2 = normalize_sentence(s2)
            fwd_key = (n1, n2)
            rev_key = (n2, n1)
            fwd_idx = dir_index_by_pair.get(fwd_key)
            if fwd_idx is None:
                fwd_idx = len(unique_dir_pairs)
                dir_index_by_pair[fwd_key] = fwd_idx
                unique_dir_pairs.append(fwd_key)
            rev_idx = dir_index_by_pair.get(rev_key)
            if rev_idx is None:
                rev_idx = len(unique_dir_pairs)
                dir_index_by_pair[rev_key] = rev_idx
                unique_dir_pairs.append(rev_key)
            pair_dir_indices.append((fwd_idx, rev_idx))

        probs_all: list[list[float]] = []
        with torch.inference_mode():
            for start in range(0, len(unique_dir_pairs), NLI_BATCH_SIZE):
                chunk = unique_dir_pairs[start : start + NLI_BATCH_SIZE]
                encoded = self._tokenizer(
                    [p for p, _ in chunk],
                    [h for _, h in chunk],
                    padding=True,
                    truncation=True,
                    max_length=NLI_MAX_LENGTH,
                    return_tensors="pt",
                )
                logits = self._model(**encoded).logits
                probs_all.extend(torch.softmax(logits, dim=-1).tolist())

        results: list[NLIScoreDetail] = []
        for fwd_idx, rev_idx in pair_dir_indices:
            fwd = probs_all[fwd_idx]
            rev = probs_all[rev_idx]
            p_entail_fwd = float(fwd[self._entail_idx])
            p_entail_rev = float(rev[self._entail_idx])
            p_contra_fwd = float(fwd[self._contradict_idx])
            p_contra_rev = float(rev[self._contradict_idx])
            p_contra_max = max(p_contra_fwd, p_contra_rev)
            entail_signal = _aggregate_entailment(p_entail_fwd, p_entail_rev, scoring_mode)
            nli_score = entail_signal * (1.0 - p_contra_max)
            score = round(max(0.0, min(1.0, nli_score)), 4)
            label: EntailmentLabel = "ENTAIL" if score >= threshold else "NO_ENTAIL"

            results.append(
                NLIScoreDetail(
                    score=score,
                    label=label,
                    tier_used="tier2_nli",
                    forward_entail=round(p_entail_fwd, 4),
                    reverse_entail=round(p_entail_rev, 4),
                    max_contradiction=round(p_contra_max, 4),
                    forward_contradict=round(p_contra_fwd, 4),
                    reverse_contradict=round(p_contra_rev, 4),
                )
            )
        return results


nli_engine = HuggingFaceNLIEngine()
