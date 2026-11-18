"""Shared domain Literal type aliases for entailment labels, tiers, and scoring modes."""

from __future__ import annotations

from typing import Literal

EntailmentLabel = Literal["ENTAIL", "NO_ENTAIL"]
TierName = Literal["tier1_rule", "tier2_nli"]
ScoringMode = Literal["mean", "min"]
