"""
Tier 1 exact normalized surface identity and same-language validation for
Japanese and English sentence pairs.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from assignment.types import EntailmentLabel

_KANA_RE = re.compile(r"[\u3040-\u309f\u30a0-\u30ff]")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_EDGE_RE = re.compile(
    r"^[\s。．.！!？?、,「」『』\"'()（）]+|[\s。．.！!？?、,「」『』\"'()（）]+$"
)


@dataclass(frozen=True)
class Tier1Result:
    decisive: bool
    label: EntailmentLabel
    score: float
    reason: str
    canonical_s1: str
    canonical_s2: str
    fallback_score: float


def normalize_sentence(text: str) -> str:
    """Apply Unicode NFKC normalization and trim edge punctuation/whitespace."""
    nfkc = unicodedata.normalize("NFKC", text)
    collapsed = _WHITESPACE_RE.sub(" ", nfkc).strip()
    return _PUNCT_EDGE_RE.sub("", collapsed).strip()


def has_substantive_content(text: str) -> bool:
    """
    Return True when normalized text contains at least one letter, number, or symbol
    (rejecting empty, whitespace-, punctuation-, or invisible-only inputs without
    mutating legitimate internal Unicode formatting characters such as emoji ZWJ).
    """
    if not isinstance(text, str):
        return False
    norm = normalize_sentence(text)
    return bool(norm) and any(unicodedata.category(ch)[0] in ("L", "N", "S") for ch in norm)


def contains_japanese(text: str) -> bool:
    """Return True if text contains Japanese Kana or CJK characters."""
    return bool(_KANA_RE.search(text) or _CJK_RE.search(text))


def is_same_language_pair(s1: str, s2: str) -> bool:
    """
    Verify both sentences belong to the same language family without rejecting
    sentences that legitimately embed loanwords/acronyms (e.g., 'AI' inside Japanese).
    """
    has_jp_script1 = contains_japanese(s1)
    has_jp_script2 = contains_japanese(s2)

    if has_jp_script1 != has_jp_script2:
        if (_LATIN_RE.search(s1) and not has_jp_script1) or (
            _LATIN_RE.search(s2) and not has_jp_script2
        ):
            return False
    return True


def evaluate_tier1(sentence1: str, sentence2: str) -> Tier1Result:
    """
    Run Tier 1 exact normalized match check only.
    Non-identical pairs stay non-decisive for Tier 2 bidirectional NLI inference.
    """
    norm1 = normalize_sentence(sentence1)
    norm2 = normalize_sentence(sentence2)
    is_jp = contains_japanese(norm1) or contains_japanese(norm2)

    if norm1 == norm2 or (not is_jp and norm1.lower() == norm2.lower()):
        return Tier1Result(
            decisive=True,
            label="ENTAIL",
            score=0.99,
            reason="exact_normalized_match",
            canonical_s1=norm1,
            canonical_s2=norm2,
            fallback_score=0.99,
        )

    return Tier1Result(
        decisive=False,
        label="NO_ENTAIL",
        score=0.0,
        reason="non_identical",
        canonical_s1=norm1,
        canonical_s2=norm2,
        fallback_score=0.0,
    )
