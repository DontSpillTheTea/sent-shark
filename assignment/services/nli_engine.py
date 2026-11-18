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
from typing import Protocol

from django.conf import settings

from assignment.services.normalizer import Tier1Result, normalize_sentence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NLIScoreDetail:
    score: float
    label: str
    tier_used: str
    forward_entail: float
    reverse_entail: float
    max_contradiction: float


class EscalationJudgeProtocol(Protocol):
    def classify(self, sentence1: str, sentence2: str) -> tuple[str, float] | None: ...


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

    def classify(self, sentence1: str, sentence2: str) -> tuple[str, float] | None:
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
            if label in ("ENTAIL", "NO_ENTAIL"):
                return label, max(0.0, min(1.0, score))
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
        self._load_failed = False

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

                id2label = getattr(model.config, "id2label", None) or {}
                entail_idx = None
                contradict_idx = None
                for idx_raw, lbl in id2label.items():
                    idx = int(idx_raw)
                    low = str(lbl).lower()
                    if "entail" in low:
                        entail_idx = idx
                    elif "contradict" in low:
                        contradict_idx = idx

                if entail_idx is None or contradict_idx is None:
                    # Fallback standard MNLI/XNLI ordering: 0=entailment, 1=neutral, 2=contradiction
                    entail_idx = 0 if entail_idx is None else entail_idx
                    contradict_idx = 2 if contradict_idx is None else contradict_idx

                self._tokenizer = tokenizer
                self._model = model
                self._entail_idx = entail_idx
                self._contradict_idx = contradict_idx
                return True
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

    def _pair_metrics(
        self, probs_all: list[list[float]], pair_idx: int
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
        base_score = 0.5 * (p_entail_fwd + p_entail_rev) * (1.0 - p_contra_max)
        return p_entail_fwd, p_entail_rev, p_contra_max, base_score

    @staticmethod
    def _calibrate_score(
        raw_metrics: tuple[float, float, float, float],
        canon_metrics: tuple[float, float, float, float] | None,
        fallback_score: float,
    ) -> tuple[float, float, float, float]:
        """
        Fuse raw and canonical NLI passes and apply non-contradiction lexical calibration.
        Guards against boosting directional hypernym pairs (strong forward/reverse asymmetry).
        """
        raw_fwd, raw_rev, raw_contra, raw_score = raw_metrics
        if canon_metrics is None:
            p_entail_fwd, p_entail_rev, p_contra_max, nli_score = raw_metrics
        else:
            can_fwd, can_rev, can_contra, can_score = canon_metrics
            if can_contra >= 0.35 and can_score < raw_score:
                # Canonicalization exposed a contradiction masked by phrasal variants
                p_entail_fwd, p_entail_rev, p_contra_max, nli_score = canon_metrics
            elif (
                can_contra < 0.15
                and can_score >= 0.75
                and min(can_fwd, can_rev) >= 0.50
                and raw_score < 0.50
            ):
                # Canonicalization resolved lexical synonyms where raw MiniLM misfired
                p_entail_fwd, p_entail_rev, p_contra_max, nli_score = canon_metrics
            else:
                p_entail_fwd = 0.5 * (raw_fwd + can_fwd)
                p_entail_rev = 0.5 * (raw_rev + can_rev)
                p_contra_max = max(raw_contra, can_contra)
                nli_score = 0.5 * (raw_score + can_score)

        # Strong one-way entailment (e.g., Siamese cat -> animal) must not be boosted
        is_directional_hypernym = (
            max(p_entail_fwd, p_entail_rev) >= 0.70 and min(p_entail_fwd, p_entail_rev) < 0.05
        )
        if not is_directional_hypernym:
            if p_contra_max < 0.15 and fallback_score >= 0.50:
                lexical_entail = (1.0 - p_contra_max) * fallback_score
                nli_score = max(nli_score, 0.50 * nli_score + 0.50 * lexical_entail)
            elif p_contra_max < 0.25 and fallback_score >= 0.65:
                nli_score = max(nli_score, 0.70 * nli_score + 0.30 * fallback_score)

        return p_entail_fwd, p_entail_rev, p_contra_max, nli_score

    def score_pairs_batch(
        self,
        pairs: list[tuple[str, str]],
        tier1_results: list[Tier1Result],
        escalation_judge: EscalationJudgeProtocol | None = None,
    ) -> list[NLIScoreDetail]:
        """
        Score a batch of sentence pairs using bidirectional NLI inference.
        Evaluates both normalized surface inputs and canonicalized Tier 1 representations
        when they differ, then applies non-contradiction calibration.
        Falls back cleanly to Tier 1 scores if the transformer model is unavailable.
        """
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

        for (s1, s2), t1 in zip(pairs, tier1_results):
            n1 = normalize_sentence(s1)
            n2 = normalize_sentence(s2)
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
        chunk_size = 64  # 32 directional sentence pairs (64 directional inputs) per pass
        with torch.inference_mode():
            for start in range(0, len(directional_premises), chunk_size):
                end = start + chunk_size
                encoded = self._tokenizer(
                    directional_premises[start:end],
                    directional_hypotheses[start:end],
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                logits = self._model(**encoded).logits
                probs = torch.softmax(logits, dim=-1).tolist()
                probs_all.extend(probs)

        results: list[NLIScoreDetail] = []
        for (s1, s2), t1, (raw_idx, canon_idx) in zip(pairs, tier1_results, pair_index_map):
            raw_metrics = self._pair_metrics(probs_all, raw_idx)
            canon_metrics = (
                self._pair_metrics(probs_all, canon_idx) if canon_idx is not None else None
            )
            p_entail_fwd, p_entail_rev, p_contra_max, nli_score = self._calibrate_score(
                raw_metrics,
                canon_metrics,
                t1.fallback_score,
            )

            score = round(max(0.0, min(1.0, nli_score)), 4)
            label = "ENTAIL" if score >= threshold else "NO_ENTAIL"
            tier_used = "tier2_nli"

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
