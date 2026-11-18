"""Shared domain Literal type aliases for entailment labels and evaluation tiers."""

from __future__ import annotations

from typing import Literal

EntailmentLabel = Literal["ENTAIL", "NO_ENTAIL"]
TierName = Literal["tier1_rule", "tier2_nli", "tier3_llm"]
