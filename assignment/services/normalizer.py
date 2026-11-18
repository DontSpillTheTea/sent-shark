"""
Tier 1 deterministic linguistic normalization, polarity verification, and fast-gate
contradiction detection for Japanese and English same-language sentence pairs.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Script detection regexes
_KANA_RE = re.compile(r"[\u3040-\u309f\u30a0-\u30ff]")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_EDGE_RE = re.compile(
    r"^[\s。．.！!？?、,「」『』\"'()（）]+|[\s。．.！!？?、,「」『』\"'()（）]+$"
)
_PUNCT_ALL_RE = re.compile(r"[。．.！!？?、,「」『』\"'()（）\s]+")

# Japanese first-person pronouns (frequently omitted in conversational/manga Japanese)
_JP_FIRST_PERSON_PREFIX_RE = re.compile(
    r"^(?:私|わたし|ワタシ|僕|ぼく|ボク|俺|おれ|オレ|あたし|わし|うち|吾輩|おいら)([はがも])"
)
_JP_OTHER_SUBJECT_RE = re.compile(r"([ぁ-んァ-ヶ一-龥A-Za-z0-9]+?)(?:は|が)")

# Japanese sentence-final copulas and conversational particles (affirmative only;
# negatives like ない/ません are preserved for polarity checking)
_JP_FINAL_PARTICLES_RE = re.compile(
    r"(?:だよ|だね|だな|ですよ|ですね|かな|かも|よ|ね|な|わ|ぞ|ぜ|さ)+$"
)
_JP_COPULA_END_RE = re.compile(r"(?:でございます|であります|である|です|だ)$")

# Japanese colloquial contractions & tense normalization
_JP_CONTRACTIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"ちゃいました"), "てしまった"),
    (re.compile(r"じゃいました"), "でしまった"),
    (re.compile(r"ちゃった"), "てしまった"),
    (re.compile(r"じゃった"), "でしまった"),
    (re.compile(r"てしまいました"), "てしまった"),
    (re.compile(r"でしまいました"), "でしまった"),
    (re.compile(r"ちゃう(?=$|[よねなぞぜ])"), "てしまう"),
    (re.compile(r"じゃう(?=$|[よねなぞぜ])"), "でしまう"),
    (re.compile(r"([ぁ-ん一-龥])てた(?=$|[よねなぞぜ])"), r"\1ていた"),
    (re.compile(r"([ぁ-ん一-龥])でた(?=$|[よねなぞぜ])"), r"\1でいた"),
    (re.compile(r"([ぁ-ん一-龥])てる(?=$|[よねなぞぜ])"), r"\1ている"),
    (re.compile(r"([ぁ-ん一-龥])でる(?=$|[よねなぞぜ])"), r"\1でいる"),
]

_JP_EVENT_SYNONYMS: list[tuple[str, str]] = [
    ("ミーティング", "会議"),
    ("先送りになりました", "延期された"),
    ("先送りになった", "延期された"),
    ("先送りされました", "延期された"),
    ("先送りされた", "延期された"),
    ("延期になりました", "延期された"),
    ("延期になった", "延期された"),
    ("中止になりました", "中止された"),
    ("中止になった", "中止された"),
    ("足を運びました", "行った"),
    ("足を運んだ", "行った"),
]

_JP_LEXICAL_VERBS: list[tuple[str, str]] = [
    ("乗車した", "乗った"),
    ("購入した", "買った"),
    ("終了した", "終わった"),
]

_JP_SYNONYM_MAP: list[tuple[str, str]] = [
    ("大好", "好"),
    ("とても好", "好"),
    ("すごく好", "好"),
    ("めっちゃ好", "好"),
    ("超好", "好"),
    ("非常に好", "好"),
    ("大嫌", "嫌"),
    ("とても嫌", "嫌"),
    ("すでに", "もう"),
    ("既に", "もう"),
    ("が好", "は好"),
    ("が嫌", "は嫌"),
    ("が欲", "は欲"),
    ("が分か", "は分か"),
    ("ができ", "はでき"),
    *_JP_EVENT_SYNONYMS,
    ("されました", "された"),
    ("しました", "した"),
    ("読みました", "読んだ"),
    ("買いました", "買った"),
    ("行きました", "行った"),
    *_JP_LEXICAL_VERBS,
]

_EN_CONTRACTIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bdon't\b"), "do not"),
    (re.compile(r"\bdoesn't\b"), "does not"),
    (re.compile(r"\bdidn't\b"), "did not"),
    (re.compile(r"\bisn't\b"), "is not"),
    (re.compile(r"\baren't\b"), "are not"),
    (re.compile(r"\bwasn't\b"), "was not"),
    (re.compile(r"\bweren't\b"), "were not"),
    (re.compile(r"\bcan't\b"), "cannot"),
    (re.compile(r"\bcannot\b"), "can not"),
    (re.compile(r"\bwon't\b"), "will not"),
    (re.compile(r"\bi'm\b"), "i am"),
    (re.compile(r"\bi've\b"), "i have"),
    (re.compile(r"\bi'll\b"), "i will"),
    (re.compile(r"\bit's\b"), "it is"),
    (re.compile(r"\bthat's\b"), "that is"),
]

_EN_ARTICLES_RE = re.compile(r"\b(?:a|an|the)\b")
_EN_SURFACE_SYNONYMS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bgave up\b", re.IGNORECASE), "quit"),
    (re.compile(r"\bgive up\b", re.IGNORECASE), "quit"),
    (
        re.compile(
            r"(?<!\bhalf )(?<!\bnearly )(?<!\balmost )"
            r"(?<!\babout )(?<!\bover )(?<!\bunder )"
            r"(?<!\bmore than )(?<!\bless than )(?<!\bjust )(?<!\bonly )(?<!\bnot )"
            r"\ba year ago\b",
            re.IGNORECASE,
        ),
        "last year",
    ),
]
_EN_SYNONYM_MAP: list[tuple[re.Pattern[str], str]] = [
    *_EN_SURFACE_SYNONYMS,
    (re.compile(r"\bpurchased\b"), "bought"),
    (re.compile(r"\bautomobile\b"), "car"),
    (re.compile(r"\bfilm\b"), "movie"),
    (re.compile(r"\bdull\b"), "boring"),
    (re.compile(r"\bextremely\b"), "very"),
    (re.compile(r"\breally like\b"), "like"),
    (re.compile(r"\badore\b"), "like"),
    (re.compile(r"\blove\b"), "like"),
    (re.compile(r"\bfeline\b"), "cat"),
    (re.compile(r"\bcanine\b"), "dog"),
    (re.compile(r"\bcarnivorous\b"), "carnivore"),
    (re.compile(r"\bherbivorous\b"), "herbivore"),
]

# Negation markers for polarity parity checking
_JP_NEGATION_RE = re.compile(
    r"(?:ません|ませんでした|ない|なかった|なく|ぬ|ん(?=$|[よね])|ず|じゃない|ではない|じゃありません|ではありません|嫌い|嫌|無理|不可能|禁止|反対)"
)
_EN_NEGATION_RE = re.compile(
    r"\b(?:not|never|no|none|neither|nobody|nothing|nowhere|hardly|scarcely|dislike|hate)\b"
)

# Restrictive particle marker in Japanese (e.g. 猫しか好きじゃない = ONLY cats)
_JP_RESTRICTIVE_RE = re.compile(r"しか")

# Antonym pairs across Japanese and English
_ANTONYM_PAIRS: list[tuple[str, str]] = [
    ("好き", "嫌い"),
    ("大好き", "大嫌い"),
    ("肉食", "草食"),
    ("肉食", "菜食"),
    ("増える", "減る"),
    ("増加", "減少"),
    ("上がる", "下がる"),
    ("上昇", "下降"),
    ("勝つ", "負ける"),
    ("勝利", "敗北"),
    ("生きる", "死ぬ"),
    ("同じ", "違う"),
    ("同じ", "異なる"),
    ("本物", "偽物"),
    ("賛成", "反対"),
    ("愛する", "憎む"),
    ("開ける", "閉める"),
    ("開く", "閉じる"),
    ("大きい", "小さい"),
    ("高い", "低い"),
    ("多い", "少ない"),
    ("ある", "ない"),
    ("いる", "いない"),
    ("carnivore", "vegetarian"),
    ("carnivore", "herbivore"),
    ("meat", "vegetarian"),
    ("like", "hate"),
    ("like", "dislike"),
    ("love", "hate"),
    ("always", "never"),
    ("true", "false"),
    ("hot", "cold"),
    ("alive", "dead"),
    ("buy", "sell"),
    ("win", "lose"),
    ("increase", "decrease"),
    ("same", "different"),
    ("agree", "disagree"),
]

# Role & voice inversion patterns
_JP_SOV_RE = re.compile(r"^(.+?)(?:が|は)(.+?)を(.+)$")
_EN_SVO_RE = re.compile(r"^([a-z]+)\s+([a-z]+(?:s|ed)?)\s+([a-z]+)$")
_JP_VOICE_PASSIVE_RE = re.compile(r"^(.*?)([一-龥ァ-ヶー]+)(?:され|られ)(?:た|る|ている|ていた)$")
_JP_VOICE_ACTIVE_RE = re.compile(r"^(.*?)([一-龥ァ-ヶー]+)(?:し(?:た|ている|ていた)|する)$")


@dataclass(frozen=True)
class Tier1Result:
    decisive: bool
    label: str
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


def normalize_lexical_surface(text: str) -> str:
    """Normalize NFKC/whitespace and apply unambiguous lexical synonyms preserving register."""
    s = normalize_sentence(text)
    if contains_japanese(s):
        for src, dst in (*_JP_EVENT_SYNONYMS, *_JP_LEXICAL_VERBS):
            s = s.replace(src, dst)
        return s
    for pattern, repl in _EN_SURFACE_SYNONYMS:
        s = pattern.sub(
            lambda m, r=repl: r.capitalize() if m.group(0)[:1].isupper() else r,
            s,
        )
    return s


def contains_japanese(text: str) -> bool:
    return bool(_KANA_RE.search(text) or _CJK_RE.search(text))


def is_same_language_pair(s1: str, s2: str) -> bool:
    """
    Verify both sentences belong to the same language family without rejecting
    sentences that legitimately embed loanwords/acronyms (e.g., 'AI' inside Japanese).
    """
    jp1 = bool(_KANA_RE.search(s1))
    jp2 = bool(_KANA_RE.search(s2))
    cjk1 = bool(_CJK_RE.search(s1))
    cjk2 = bool(_CJK_RE.search(s2))
    has_jp_script1 = jp1 or cjk1
    has_jp_script2 = jp2 or cjk2

    if has_jp_script1 != has_jp_script2:
        # One uses Japanese/CJK characters and the other is strictly non-CJK (e.g., English)
        if (_LATIN_RE.search(s1) and not has_jp_script1) or (
            _LATIN_RE.search(s2) and not has_jp_script2
        ):
            return False
    return True


def _extract_subject_class(text: str) -> tuple[str, str | None]:
    """
    Classify leading subject for Japanese sentences:
    - ('self', particle) when starting with first-person pronoun (私/僕/俺/etc.)
    - ('other', noun) when starting with an explicit non-first-person topic/subject
    - ('none', None) when subject is omitted
    """
    m_self = _JP_FIRST_PERSON_PREFIX_RE.match(text)
    if m_self:
        return ("self", m_self.group(1))
    m_other = _JP_OTHER_SUBJECT_RE.match(text)
    if m_other:
        subj = m_other.group(1)
        # Do not treat object/stative noun before が好/が嫌 as third-person agent
        if any(
            text[m_other.end() :].startswith(x)
            for x in ("好き", "嫌い", "欲しい", "分かる", "できる")
        ):
            return ("none", None)
        return ("other", subj)
    return ("none", None)


def _apply_jp_synonyms(text: str) -> str:
    """Apply ordered Japanese lexical, passive, and past-ending canonicalizations."""
    for src, dst in _JP_SYNONYM_MAP:
        text = text.replace(src, dst)
    return text


def canonicalize_japanese(text: str, drop_first_person: bool = False) -> str:
    s = normalize_sentence(text)
    s = _PUNCT_ALL_RE.sub("", s)

    if drop_first_person:
        s = _JP_FIRST_PERSON_PREFIX_RE.sub("", s)
    else:
        s = _JP_FIRST_PERSON_PREFIX_RE.sub(r"<SELF>\1", s)

    for pattern, repl in _JP_CONTRACTIONS:
        s = pattern.sub(repl, s)

    # Apply lexical synonyms before copula stripping so past-tense -んだ verbs
    # (e.g., 足を運んだ) are canonicalized before だ$ truncation.
    s = _apply_jp_synonyms(s)

    # Strip sentence-final register/politeness markers iteratively
    prev = None
    while prev != s:
        prev = s
        s = _JP_FINAL_PARTICLES_RE.sub("", s)
        s = _JP_COPULA_END_RE.sub("", s)

    return s


def canonicalize_english(text: str) -> str:
    s = normalize_sentence(text).lower()
    for pattern, repl in _EN_CONTRACTIONS:
        s = pattern.sub(repl, s)
    for pattern, repl in _EN_SYNONYM_MAP:
        s = pattern.sub(repl, s)
    s = _EN_ARTICLES_RE.sub("", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return _WHITESPACE_RE.sub(" ", s).strip()


def canonicalize_for_comparison(text: str, drop_first_person: bool = False) -> str:
    if contains_japanese(text):
        return canonicalize_japanese(text, drop_first_person=drop_first_person)
    return canonicalize_english(text)


def _count_negations(text: str) -> int:
    norm = normalize_sentence(text).lower()
    if contains_japanese(norm):
        return len(_JP_NEGATION_RE.findall(norm))
    norm_en = canonicalize_english(norm)
    return len(_EN_NEGATION_RE.findall(norm_en))


def _has_antonym_collision(s1: str, s2: str) -> bool:
    l1 = normalize_sentence(s1).lower()
    l2 = normalize_sentence(s2).lower()
    if not contains_japanese(l1):
        l1 = canonicalize_english(l1)
        l2 = canonicalize_english(l2)

    for a, b in _ANTONYM_PAIRS:
        if (a in l1 and b in l2) or (b in l1 and a in l2):
            # Ensure it's a genuine opposition (neither sentence already contains both)
            if not ((a in l1 and b in l1) or (a in l2 and b in l2)):
                return True
    return False


def _has_role_inversion(s1: str, s2: str) -> bool:
    c1 = normalize_sentence(s1)
    c2 = normalize_sentence(s2)
    if contains_japanese(c1):
        c1_p = _PUNCT_ALL_RE.sub("", c1)
        c2_p = _PUNCT_ALL_RE.sub("", c2)
        m1 = _JP_SOV_RE.match(c1_p)
        m2 = _JP_SOV_RE.match(c2_p)
        if m1 and m2:
            subj1, obj1, verb1 = m1.groups()
            subj2, obj2, verb2 = m2.groups()
            if subj1 == obj2 and obj1 == subj2 and subj1 != obj1 and verb1[:1] == verb2[:1]:
                return True
    else:
        e1 = canonicalize_english(c1)
        e2 = canonicalize_english(c2)
        m1 = _EN_SVO_RE.match(e1)
        m2 = _EN_SVO_RE.match(e2)
        if m1 and m2:
            s_a, v_a, o_a = m1.groups()
            s_b, v_b, o_b = m2.groups()
            if s_a == o_b and o_a == s_b and s_a != o_a and v_a == v_b:
                return True
    return False


def _has_jp_voice_inversion(s1: str, s2: str) -> bool:
    """
    Detect transitive Japanese passive-vs-active voice flips sharing the same subject
    (e.g., '彼が逮捕された。' vs '彼が逮捕した。'), while allowing true passive/active
    object-promoted paraphrases and event-noun canonicalizations.
    """
    if not (contains_japanese(s1) and contains_japanese(s2)):
        return False

    c1 = _JP_FINAL_PARTICLES_RE.sub("", _apply_jp_synonyms(_PUNCT_ALL_RE.sub("", s1)))
    c2 = _JP_FINAL_PARTICLES_RE.sub("", _apply_jp_synonyms(_PUNCT_ALL_RE.sub("", s2)))

    for passive_cand, active_cand in ((c1, c2), (c2, c1)):
        mp = _JP_VOICE_PASSIVE_RE.match(passive_cand)
        ma = _JP_VOICE_ACTIVE_RE.match(active_cand)
        if not (mp and ma):
            continue
        p_pref, p_stem = mp.groups()
        a_pref, a_stem = ma.groups()
        if p_stem != a_stem:
            continue
        if not p_pref and not a_pref:
            return True
        subj_p_type, subj_p_val = _extract_subject_class(p_pref)
        subj_a_type, subj_a_val = _extract_subject_class(a_pref)
        if subj_p_type != "none" and subj_p_type == subj_a_type:
            if subj_p_type == "self":
                return True
            canon_p = _apply_jp_synonyms(subj_p_val) if subj_p_val is not None else None
            canon_a = _apply_jp_synonyms(subj_a_val) if subj_a_val is not None else None
            if canon_p == canon_a:
                return True
    return False


def _char_ngram_jaccard(a: str, b: str, n: int = 2) -> float:
    if not a or not b:
        return 0.0
    if len(a) < n or len(b) < n:
        return 1.0 if a == b else 0.0
    grams_a = {a[i : i + n] for i in range(len(a) - n + 1)}
    grams_b = {b[i : i + n] for i in range(len(b) - n + 1)}
    union = len(grams_a | grams_b)
    return (len(grams_a & grams_b) / union) if union else 0.0


def _token_jaccard(a: str, b: str) -> float:
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def evaluate_tier1(sentence1: str, sentence2: str) -> Tier1Result:
    """
    Run Tier 1 deterministic fast-gate and fallback similarity computation.
    """
    norm1 = normalize_sentence(sentence1)
    norm2 = normalize_sentence(sentence2)

    is_jp = contains_japanese(norm1) or contains_japanese(norm2)

    # Exact surface match after NFKC + punctuation trim
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

    # Contradiction / polarity / role-inversion gates
    antonym_hit = _has_antonym_collision(norm1, norm2)
    neg1 = _count_negations(norm1)
    neg2 = _count_negations(norm2)
    polarity_mismatch = (neg1 % 2) != (neg2 % 2)
    restrictive_mismatch = bool(_JP_RESTRICTIVE_RE.search(norm1)) != bool(
        _JP_RESTRICTIVE_RE.search(norm2)
    )
    role_inverted = _has_role_inversion(norm1, norm2) or _has_jp_voice_inversion(norm1, norm2)

    if antonym_hit or role_inverted or restrictive_mismatch:
        c1 = canonicalize_for_comparison(norm1)
        c2 = canonicalize_for_comparison(norm2)
        return Tier1Result(
            decisive=True,
            label="NO_ENTAIL",
            score=0.06,
            reason="contradiction_veto",
            canonical_s1=c1,
            canonical_s2=c2,
            fallback_score=0.06,
        )

    # Subject compatibility check for Japanese pro-drop canonicalization
    if is_jp:
        subj1_type, subj1_val = _extract_subject_class(norm1)
        subj2_type, subj2_val = _extract_subject_class(norm2)
        allow_drop_self = (subj1_type == "self" and subj2_type in ("self", "none")) or (
            subj2_type == "self" and subj1_type in ("self", "none")
        )
        canon_subj1 = _apply_jp_synonyms(subj1_val) if subj1_val is not None else None
        canon_subj2 = _apply_jp_synonyms(subj2_val) if subj2_val is not None else None
        subjects_conflict = (
            subj1_type == "other" and subj2_type == "other" and canon_subj1 != canon_subj2
        ) or ({subj1_type, subj2_type} == {"self", "other"})
        c1 = canonicalize_japanese(norm1, drop_first_person=allow_drop_self)
        c2 = canonicalize_japanese(norm2, drop_first_person=allow_drop_self)
        raw_sim = _char_ngram_jaccard(c1, c2, n=2)
    else:
        subjects_conflict = False
        c1 = canonicalize_english(norm1)
        c2 = canonicalize_english(norm2)
        raw_sim = 0.5 * _token_jaccard(c1, c2) + 0.5 * _char_ngram_jaccard(
            c1.replace(" ", ""), c2.replace(" ", ""), n=3
        )

    # Fallback score applies heavy penalty on polarity mismatch or conflicting subjects
    fallback_score = raw_sim
    if polarity_mismatch or subjects_conflict:
        fallback_score = min(fallback_score * 0.25, 0.20)

    if polarity_mismatch and raw_sim >= 0.30:
        return Tier1Result(
            decisive=True,
            label="NO_ENTAIL",
            score=0.08,
            reason="negation_polarity_flip",
            canonical_s1=c1,
            canonical_s2=c2,
            fallback_score=0.08,
        )

    # Fast-path Japanese register equivalence in Tier 1; English canonical collapses
    # (which drop articles and replace phrasal synonyms) stay non-decisive for Tier 2 verification.
    if is_jp and c1 == c2 and not polarity_mismatch and not subjects_conflict:
        return Tier1Result(
            decisive=True,
            label="ENTAIL",
            score=0.92,
            reason="canonical_register_match",
            canonical_s1=c1,
            canonical_s2=c2,
            fallback_score=0.92,
        )

    return Tier1Result(
        decisive=False,
        label="ENTAIL" if fallback_score >= 0.55 else "NO_ENTAIL",
        score=round(fallback_score, 4),
        reason="tier1_heuristic",
        canonical_s1=c1,
        canonical_s2=c2,
        fallback_score=round(fallback_score, 4),
    )
