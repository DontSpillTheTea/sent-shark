"""
Tier 2 Bidirectional HuggingFace NLI Cross-Encoder and Tier 3 Optional Cloud LLM
Escalation Adapter.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from django.conf import settings

from assignment.services.normalizer import (
    Tier1Result,
    evaluate_tier1,
    normalize_lexical_surface,
    normalize_sentence,
)
from assignment.types import EntailmentLabel, ScoringMode, TierName

logger = logging.getLogger(__name__)

NLI_MAX_LENGTH = 256
NLI_BATCH_SIZE = 64  # 32 directional sentence pairs (64 directional inputs) per pass
NLI_CONTRADICT_CAP = 0.35
NLI_LEXICAL_CONTRADICT_CAP = 0.25
NLI_DIRECTIONAL_HIGH = 0.70
NLI_DIRECTIONAL_LOW = 0.15


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


class EscalationJudgeProtocol(Protocol):
    def classify(self, sentence1: str, sentence2: str) -> tuple[EntailmentLabel, float] | None: ...


class NoOpEscalationJudge:
    """Disable Tier 3 external LLM calls for warm-up and deterministic offline benchmarks."""

    def classify(self, sentence1: str, sentence2: str) -> tuple[EntailmentLabel, float] | None:
        return None


class OpenAIEscalationJudge:
    """Optional Tier 3 LLM classifier for grey-zone entailment pairs."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        timeout_sec: float = 2.0,
    ):
        self.api_key = api_key
        self.model = model
        self.timeout_sec = timeout_sec

    def classify(self, sentence1: str, sentence2: str) -> tuple[EntailmentLabel, float] | None:
        if not self.api_key:
            return None

        payload = {
            "model": self.model,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Determine whether two same-language sentences convey the same "
                        "meaning (bidirectional entailment). Reply strictly in JSON: "
                        '{"label": "ENTAIL" or "NO_ENTAIL", "score": float in [0.0, 1.0]}.'
                    ),
                },
                {
                    "role": "user",
                    "content": f"Sentence 1: {sentence1}\nSentence 2: {sentence2}",
                },
            ],
        }
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            label = str(parsed.get("label", "")).upper()
            score = float(parsed.get("score", 0.5))
            if label == "ENTAIL":
                return "ENTAIL", max(0.0, min(1.0, score))
            if label == "NO_ENTAIL":
                return "NO_ENTAIL", max(0.0, min(1.0, score))
        except (
            urllib.error.URLError,
            TimeoutError,
            KeyError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            logger.warning("Tier 3 LLM escalation failed; using Tier 2 score: %s", exc)
        return None


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
            if entail_idx == contradict_idx:
                raise ValueError(
                    f"NLI_ENTAIL_INDEX and NLI_CONTRADICT_INDEX must differ, got {entail_idx}."
                )
            if num_labels is not None and (
                entail_idx >= num_labels or contradict_idx >= num_labels
            ):
                raise ValueError(
                    f"Configured NLI label indices ({entail_idx}, {contradict_idx}) "
                    f"out of range for num_labels={num_labels} on model '{self.model_name}'."
                )
            return entail_idx, contradict_idx

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
            low = str(lbl).lower()
            has_entail = "entail" in low
            has_contradict = "contradict" in low
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

        entail_idx = entail_matches[0]
        contradict_idx = contradict_matches[0]
        if entail_idx == contradict_idx or entail_idx < 0 or contradict_idx < 0:
            raise ValueError(
                f"Resolved invalid NLI label indices ({entail_idx}, {contradict_idx}) "
                f"from id2label={id2label!r} for model '{self.model_name}'."
            )
        return entail_idx, contradict_idx

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
                    "Unable to load NLI model '%s'; falling back to Tier 1 heuristics: %s",
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
        self.score_pairs_batch(
            [warm_pair],
            [evaluate_tier1(*warm_pair)],
            escalation_judge=NoOpEscalationJudge(),
        )
        return True

    def _pair_metrics(
        self,
        probs_all: list[list[float]],
        pair_idx: int,
        scoring_mode: ScoringMode = "mean",
    ) -> tuple[float, float, float, float]:
        """Compute forward entailment, reverse entailment, max contradiction, and base score."""
        fwd = probs_all[2 * pair_idx]
        rev = probs_all[2 * pair_idx + 1]
        p_entail_fwd = float(fwd[self._entail_idx])
        p_entail_rev = float(rev[self._entail_idx])
        p_contra_max = max(
            float(fwd[self._contradict_idx]),
            float(rev[self._contradict_idx]),
        )
        entail_signal = _aggregate_entailment(p_entail_fwd, p_entail_rev, scoring_mode)
        base_score = entail_signal * (1.0 - p_contra_max)
        return p_entail_fwd, p_entail_rev, p_contra_max, base_score

    @staticmethod
    def _calibrate_score(
        raw_metrics: tuple[float, float, float, float],
        canon_metrics: tuple[float, float, float, float] | None,
        fallback_score: float,
        scoring_mode: ScoringMode = "mean",
    ) -> tuple[float, float, float, float]:
        """
        Fuse raw and canonical NLI passes and apply non-contradiction lexical calibration.
        Guards against boosting directional hypernym pairs (strong forward/reverse asymmetry).
        """
        _validate_scoring_mode(scoring_mode)
        raw_fwd, raw_rev, raw_contra, raw_score = raw_metrics
        if canon_metrics is None:
            p_entail_fwd, p_entail_rev, p_contra_max, nli_score = raw_metrics
        else:
            can_fwd, can_rev, can_contra, can_score = canon_metrics
            if can_contra >= NLI_CONTRADICT_CAP and can_score < raw_score:
                # Canonicalization exposed a contradiction masked by phrasal variants
                p_entail_fwd, p_entail_rev, p_contra_max, nli_score = canon_metrics
            elif (
                can_contra < NLI_DIRECTIONAL_LOW
                and can_score >= 0.75
                and min(can_fwd, can_rev) >= 0.50
                and raw_score < 0.50
                and raw_contra < NLI_CONTRADICT_CAP
            ):
                # Canonicalization resolved lexical synonyms where raw MiniLM misfired
                p_entail_fwd, p_entail_rev, p_contra_max, nli_score = canon_metrics
            else:
                p_entail_fwd = 0.5 * (raw_fwd + can_fwd)
                p_entail_rev = 0.5 * (raw_rev + can_rev)
                p_contra_max = max(raw_contra, can_contra)
                entail_signal = _aggregate_entailment(p_entail_fwd, p_entail_rev, scoring_mode)
                nli_score = entail_signal * (1.0 - p_contra_max)

        # Strong one-way entailment (e.g., Siamese cat -> animal) must not be boosted
        is_directional_hypernym = (
            max(p_entail_fwd, p_entail_rev) >= NLI_DIRECTIONAL_HIGH
            and min(p_entail_fwd, p_entail_rev) < NLI_DIRECTIONAL_LOW
        )
        if not is_directional_hypernym:
            if p_contra_max < NLI_DIRECTIONAL_LOW and fallback_score >= 0.50:
                lexical_entail = (1.0 - p_contra_max) * fallback_score
                nli_score = max(nli_score, 0.50 * nli_score + 0.50 * lexical_entail)
            elif p_contra_max < NLI_LEXICAL_CONTRADICT_CAP and fallback_score >= 0.65:
                nli_score = max(nli_score, 0.70 * nli_score + 0.30 * fallback_score)

        return p_entail_fwd, p_entail_rev, p_contra_max, nli_score

    def score_pairs_batch(
        self,
        pairs: list[tuple[str, str]],
        tier1_results: list[Tier1Result],
        escalation_judge: EscalationJudgeProtocol | None = None,
        apply_lexical_surface: bool = True,
        scoring_mode: ScoringMode = "mean",
    ) -> list[NLIScoreDetail]:
        """
        Score a batch of sentence pairs using bidirectional NLI inference.
        Evaluates both normalized surface inputs (applying safe lexical surface
        normalization when ``apply_lexical_surface=True``, or raw ``normalize_sentence``
        when ``False``) and canonicalized Tier 1 representations when they differ,
        then applies non-contradiction calibration.
        Falls back cleanly to Tier 1 scores if the transformer model is unavailable.
        """
        _validate_scoring_mode(scoring_mode)
        if not pairs:
            return []

        threshold = float(getattr(settings, "ENTAIL_THRESHOLD", 0.50))
        esc_low = float(getattr(settings, "LLM_ESCALATION_LOW", 0.40))
        esc_high = float(getattr(settings, "LLM_ESCALATION_HIGH", 0.65))

        if escalation_judge is None:
            api_key = getattr(settings, "OPENAI_API_KEY", "")
            if api_key:
                escalation_judge = OpenAIEscalationJudge(
                    api_key=api_key,
                    model=getattr(settings, "OPENAI_MODEL", "gpt-4o-mini"),
                )

        if not self._ensure_loaded():
            return [
                NLIScoreDetail(
                    score=round(t1.fallback_score, 4),
                    label="ENTAIL" if t1.fallback_score >= threshold else "NO_ENTAIL",
                    tier_used="tier1_rule",
                    forward_entail=t1.fallback_score,
                    reverse_entail=t1.fallback_score,
                    max_contradiction=round(1.0 - t1.fallback_score, 4),
                )
                for t1 in tier1_results
            ]

        import torch

        # Build directional inputs for raw normalized pairs and (when distinct) canonical pairs
        directional_premises: list[str] = []
        directional_hypotheses: list[str] = []
        pair_index_map: list[tuple[int, int | None]] = []

        surface_fn = normalize_lexical_surface if apply_lexical_surface else normalize_sentence
        for (s1, s2), t1 in zip(pairs, tier1_results, strict=True):
            n1 = surface_fn(s1)
            n2 = surface_fn(s2)
            raw_idx = len(directional_premises) // 2
            directional_premises.extend((n1, n2))
            directional_hypotheses.extend((n2, n1))

            c1 = (t1.canonical_s1 or n1).replace("<SELF>", "私") or n1
            c2 = (t1.canonical_s2 or n2).replace("<SELF>", "私") or n2
            if (c1, c2) != (n1, n2):
                canon_idx = len(directional_premises) // 2
                directional_premises.extend((c1, c2))
                directional_hypotheses.extend((c2, c1))
            else:
                canon_idx = None

            pair_index_map.append((raw_idx, canon_idx))

        probs_all: list[list[float]] = []
        with torch.inference_mode():
            for start in range(0, len(directional_premises), NLI_BATCH_SIZE):
                end = start + NLI_BATCH_SIZE
                encoded = self._tokenizer(
                    directional_premises[start:end],
                    directional_hypotheses[start:end],
                    padding=True,
                    truncation=True,
                    max_length=NLI_MAX_LENGTH,
                    return_tensors="pt",
                )
                logits = self._model(**encoded).logits
                probs = torch.softmax(logits, dim=-1).tolist()
                probs_all.extend(probs)

        results: list[NLIScoreDetail] = []
        for (s1, s2), t1, (raw_idx, canon_idx) in zip(
            pairs, tier1_results, pair_index_map, strict=True
        ):
            raw_metrics = self._pair_metrics(probs_all, raw_idx, scoring_mode=scoring_mode)
            canon_metrics = (
                self._pair_metrics(probs_all, canon_idx, scoring_mode=scoring_mode)
                if canon_idx is not None
                else None
            )
            p_entail_fwd, p_entail_rev, p_contra_max, nli_score = self._calibrate_score(
                raw_metrics,
                canon_metrics,
                t1.fallback_score,
                scoring_mode=scoring_mode,
            )

            score = round(max(0.0, min(1.0, nli_score)), 4)
            label: EntailmentLabel = "ENTAIL" if score >= threshold else "NO_ENTAIL"
            tier_used: TierName = "tier2_nli"

            # Optional Tier 3 LLM escalation in uncertain grey zone
            if escalation_judge is not None and esc_low <= score <= esc_high:
                escalated = escalation_judge.classify(s1, s2)
                if escalated is not None:
                    label, esc_score = escalated
                    score = round(esc_score, 4)
                    tier_used = "tier3_llm"

            results.append(
                NLIScoreDetail(
                    score=score,
                    label=label,
                    tier_used=tier_used,
                    forward_entail=round(p_entail_fwd, 4),
                    reverse_entail=round(p_entail_rev, 4),
                    max_contradiction=round(p_contra_max, 4),
                )
            )
        return results


nli_engine = HuggingFaceNLIEngine()
