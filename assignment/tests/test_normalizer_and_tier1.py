import pytest

from assignment.services import normalizer
from assignment.services.normalizer import (
    evaluate_tier1,
    is_same_language_pair,
    normalize_sentence,
)


def test_nfkc_normalization_and_punctuation_trim():
    assert normalize_sentence("　Ｈｅｌｌｏ　ｗｏｒｌｄ！　") == "Hello world"
    assert normalize_sentence("私は猫が大好きだ。") == "私は猫が大好きだ"


def test_same_language_detection_allows_loanwords_and_rejects_cross_lang():
    assert is_same_language_pair("私は猫が好きだ。", "猫が好きです。") is True
    assert is_same_language_pair("猫とAIが好きです。", "AIと猫が大好きだ。") is True
    assert is_same_language_pair("A cat is a carnivore.", "Cats eat meat.") is True
    assert is_same_language_pair("私は猫が大好きだ。", "I love cats.") is False


def test_tier1_exact_normalized_match_only_and_non_identical_fallback():
    res_jp_exact = evaluate_tier1("　私は猫が大好きだ！　", "私は猫が大好きだ。")
    assert res_jp_exact.decisive is True
    assert res_jp_exact.label == "ENTAIL"
    assert res_jp_exact.score == 0.99
    assert res_jp_exact.reason == "exact_normalized_match"
    assert res_jp_exact.fallback_score == 0.99

    res_en_exact = evaluate_tier1("Hello,  World!", "hello, world.")
    assert res_en_exact.decisive is True
    assert res_en_exact.label == "ENTAIL"
    assert res_en_exact.score == 0.99
    assert res_en_exact.reason == "exact_normalized_match"

    for s1, s2 in (
        ("私は猫が大好きだ。", "猫が好きです。"),
        ("私は猫が大好きだ。", "彼は猫が大好きだ。"),
        ("A cat is a carnivore.", "A cat is a vegetarian."),
        ("猫が魚を食べた。", "魚が猫を食べた。"),
    ):
        res = evaluate_tier1(s1, s2)
        assert res.decisive is False
        assert res.label == "NO_ENTAIL"
        assert res.score == 0.0
        assert res.reason == "non_identical"
        assert res.fallback_score == 0.0


def test_removed_tier1_heuristic_helpers_absent():
    for removed_name in (
        "canonicalize_japanese",
        "canonicalize_english",
        "canonicalize_for_comparison",
        "normalize_lexical_surface",
        "_has_antonym_collision",
        "_has_role_inversion",
        "_has_jp_voice_inversion",
    ):
        assert not hasattr(normalizer, removed_name)


@pytest.mark.integration
@pytest.mark.django_db
@pytest.mark.parametrize(
    ("sentence1", "sentence2", "expected"),
    [
        ("私は猫が大好きだ。", "猫が好きです。", "ENTAIL"),
        ("The movie was extremely boring.", "The film was very dull.", "ENTAIL"),
        ("雨が降っているので、傘を持っていく。", "雨天のため、傘を持参する。", "ENTAIL"),
        ("A cat is a carnivore.", "A cat is a vegetarian.", "NO_ENTAIL"),
        ("I adopted a Siamese cat.", "I adopted an animal.", "NO_ENTAIL"),
        ("私は柴犬を飼っている。", "私は動物を飼っている。", "NO_ENTAIL"),
        ("会議は来週の月曜日に延期された。", "会議は今週の金曜日に前倒しされた。", "NO_ENTAIL"),
        ("The car is red.", "The automobile is blue.", "NO_ENTAIL"),
        ("A wolf is a canine.", "A wolf is a dog.", "NO_ENTAIL"),
    ],
)
def test_nli_end_to_end_paraphrases_contradictions_and_hypernyms(sentence1, sentence2, expected):
    from assignment.services.judge_service import judge_single

    res = judge_single(sentence1, sentence2)
    assert res.label == expected


@pytest.mark.integration
def test_nli_batch_deduplicates_identical_and_swapped_directional_pairs(monkeypatch):
    from assignment.services.nli_engine import nli_engine

    assert nli_engine.warm_up() is True
    captured_batches: list[tuple[list[str], list[str]]] = []
    orig_tokenizer = nli_engine._tokenizer

    def recording_tokenizer(premises, hypotheses, **kwargs):
        captured_batches.append((list(premises), list(hypotheses)))
        return orig_tokenizer(premises, hypotheses, **kwargs)

    monkeypatch.setattr(nli_engine, "_tokenizer", recording_tokenizer)

    s1, s2 = "私は猫が大好きだ。", "猫が好きです。"
    details = nli_engine.score_pairs_batch([(s1, s2), (s2, s1), (s1, s2)])
    assert len(details) == 3
    # Only 2 unique directional sequences (s1->s2 and s2->s1) should be tokenized/inferred
    assert captured_batches == [
        (["私は猫が大好きだ", "猫が好きです"], ["猫が好きです", "私は猫が大好きだ"])
    ]
    assert details[0].score == details[1].score == details[2].score
    assert details[0].label == details[1].label == details[2].label == "ENTAIL"
    assert details[0].forward_entail == details[1].reverse_entail
    assert details[0].reverse_entail == details[1].forward_entail


def test_nli_pure_min_scoring_and_unrounded_threshold_boundary(monkeypatch):
    from types import SimpleNamespace

    import torch

    from assignment.services.nli_engine import HuggingFaceNLIEngine

    engine = HuggingFaceNLIEngine("mock/pure-min")
    with pytest.raises(TypeError):
        engine.score_pairs_batch([("A", "B")], scoring_mode="min")  # type: ignore[call-arg]

    monkeypatch.setattr(engine, "_ensure_loaded", lambda: True)
    engine._entail_idx = 0
    engine._contradict_idx = 2
    engine._tokenizer = lambda premises, hypotheses, **_kw: {
        "input_ids": torch.zeros((len(premises), 2), dtype=torch.long)
    }

    def run_mock_pair(fwd_probs: list[float], rev_probs: list[float]):
        probs = torch.tensor([fwd_probs, rev_probs], dtype=torch.float32)
        monkeypatch.setattr(
            engine, "_model", lambda **_enc: SimpleNamespace(logits=torch.log(probs))
        )
        return engine.score_pairs_batch([("Alpha", "Beta")])[0]

    # Asymmetric directional entailment uses pure min -> NO_ENTAIL
    asym = run_mock_pair([0.95, 0.04, 0.01], [0.20, 0.78, 0.02])
    assert asym.score == 0.2
    assert asym.label == "NO_ENTAIL"

    # Strong one-way entailment (0.99 / 0.01) -> 0.01 NO_ENTAIL
    oneway = run_mock_pair([0.99, 0.005, 0.005], [0.01, 0.98, 0.01])
    assert oneway.score == 0.01
    assert oneway.label == "NO_ENTAIL"

    # Moderate contradiction (0.35) does NOT apply (1 - p_contra_max) penalty, but is still recorded
    moderate_contra = run_mock_pair([0.64, 0.01, 0.35], [0.62, 0.03, 0.35])
    assert moderate_contra.score == 0.62
    assert moderate_contra.label == "ENTAIL"
    assert moderate_contra.max_contradiction == 0.35

    # Unrounded threshold check MUST happen before 4-decimal rounding
    below_thresh = run_mock_pair([0.80, 0.19, 0.01], [0.49996, 0.49004, 0.01])
    assert below_thresh.label == "NO_ENTAIL"
    assert below_thresh.score == 0.5

    at_or_above_thresh = run_mock_pair([0.80, 0.19, 0.01], [0.50004, 0.48996, 0.01])
    assert at_or_above_thresh.label == "ENTAIL"
    assert at_or_above_thresh.score == 0.5


def test_nli_score_pairs_batch_raises_token_limit_exceeded_without_truncation(monkeypatch):
    import torch

    from assignment.services.nli_engine import (
        NLI_MAX_LENGTH,
        HuggingFaceNLIEngine,
        TokenLimitExceededError,
    )

    engine = HuggingFaceNLIEngine("mock/token-limit")
    monkeypatch.setattr(engine, "_ensure_loaded", lambda: True)
    engine._entail_idx = 0
    engine._contradict_idx = 2

    captured_kwargs: dict[str, object] = {}

    def overlong_tokenizer(premises, hypotheses, **kwargs):
        captured_kwargs.update(kwargs)
        return {"input_ids": torch.zeros((len(premises), NLI_MAX_LENGTH + 1), dtype=torch.long)}

    model_called = False

    def fail_if_model_called(**_enc):
        nonlocal model_called
        model_called = True
        raise AssertionError("_model must not run when token limit is exceeded")

    engine._tokenizer = overlong_tokenizer
    monkeypatch.setattr(engine, "_model", fail_if_model_called)

    with pytest.raises(TokenLimitExceededError):
        engine.score_pairs_batch([("Alpha", "Beta")])

    assert captured_kwargs.get("truncation") is False
    assert "max_length" not in captured_kwargs
    assert model_called is False


def test_nli_ensure_loaded_reuses_matching_tokenizer_and_offline_unrounded_threshold(monkeypatch):
    from types import SimpleNamespace

    import transformers

    from assignment.services.judge_service import evaluate_uncached_pairs
    from assignment.services.nli_engine import HuggingFaceNLIEngine
    from assignment.services.normalizer import Tier1Result

    with pytest.raises(TypeError):
        evaluate_uncached_pairs([("A", "B")], scoring_mode="min")  # type: ignore[call-arg]

    # Offline fallback compares unrounded fallback_score against threshold before 4-dp rounding
    offline_engine = HuggingFaceNLIEngine("mock/offline")
    monkeypatch.setattr(offline_engine, "_ensure_loaded", lambda: False)
    t1_below = Tier1Result(
        decisive=False,
        label="NO_ENTAIL",
        score=0.0,
        reason="mock",
        canonical_s1="alpha",
        canonical_s2="beta",
        fallback_score=0.49996,
    )
    t1_above = Tier1Result(
        decisive=False,
        label="NO_ENTAIL",
        score=0.0,
        reason="mock",
        canonical_s1="alpha",
        canonical_s2="beta",
        fallback_score=0.50004,
    )
    res_below, res_above = offline_engine.score_pairs_batch(
        [("alpha", "beta"), ("alpha", "beta")],
        [t1_below, t1_above],
    )
    assert res_below.label == "NO_ENTAIL" and res_below.score == 0.5
    assert res_above.label == "ENTAIL" and res_above.score == 0.5

    # _ensure_loaded reuses pre-loaded _tokenizer when _tokenizer_model_name matches model_name
    tokenizer_calls = {"count": 0}
    preloaded_tokenizer = object()

    def fake_from_pretrained(_name):
        tokenizer_calls["count"] += 1
        return object()

    class FakeModel:
        config = SimpleNamespace(
            id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
            num_labels=3,
            _commit_hash="rev-reuse-1",
        )

        def eval(self):
            return self

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(
        transformers.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda _name: FakeModel(),
    )

    reuse_engine = HuggingFaceNLIEngine("mock/reuse-tok")
    reuse_engine._tokenizer = preloaded_tokenizer
    reuse_engine._tokenizer_model_name = "mock/reuse-tok"
    assert reuse_engine._ensure_loaded() is True
    assert tokenizer_calls["count"] == 0
    assert reuse_engine._tokenizer is preloaded_tokenizer

    # Mismatched _tokenizer_model_name triggers a fresh tokenizer load
    mismatch_engine = HuggingFaceNLIEngine("mock/new-tok")
    mismatch_engine._tokenizer = preloaded_tokenizer
    mismatch_engine._tokenizer_model_name = "mock/old-tok"
    assert mismatch_engine._ensure_loaded() is True
    assert tokenizer_calls["count"] == 1
    assert mismatch_engine._tokenizer is not preloaded_tokenizer
    assert mismatch_engine._tokenizer_model_name == "mock/new-tok"


def test_settings_optional_int_env_error_context(monkeypatch):
    from assignment.settings import _optional_int_env

    monkeypatch.setenv("TEST_NLI_INDEX", "not-an-int")
    with pytest.raises(ValueError, match="TEST_NLI_INDEX"):
        _optional_int_env("TEST_NLI_INDEX")


@pytest.mark.integration
@pytest.mark.django_db
def test_evaluate_uncached_pairs_and_nli_warm_up_without_db_mutation():
    from assignment.models import JudgmentRecord
    from assignment.services.judge_service import evaluate_uncached_pairs
    from assignment.services.nli_engine import nli_engine

    assert JudgmentRecord.objects.count() == 0
    assert nli_engine.warm_up() is True

    outputs = evaluate_uncached_pairs(
        [
            ("　私は猫が大好きだ！　", "私は猫が大好きだ。"),
            ("私は猫が大好きだ。", "猫が好きです。"),
            ("A cat is a carnivore.", "A cat is a vegetarian."),
        ]
    )
    assert [out.label for out in outputs] == ["ENTAIL", "ENTAIL", "NO_ENTAIL"]
    assert [out.tier_used for out in outputs] == ["tier1_rule", "tier2_nli", "tier2_nli"]
    assert outputs[0].score == 0.99
    assert all(out.cached is False for out in outputs)
    assert JudgmentRecord.objects.count() == 0


def test_has_substantive_content_and_internal_zwj_preservation():
    from assignment.services.normalizer import has_substantive_content

    for empty_or_non_substantive in (
        "",
        "   ",
        "\u3000\t\n",
        "...",
        "!!!",
        "。。。",
        "！？",
        "---",
        "\u200b",
        "\ufeff",
        "\u200c\u200d",
        "\u200b。。。",
    ):
        assert has_substantive_content(empty_or_non_substantive) is False

    for valid in ("456", "猫", "Hello!", "👨‍👩‍👧", "👨‍👩‍👧が笑う", "👨‍👩‍👧が微笑む"):
        assert has_substantive_content(valid) is True

    # Internal Unicode ZWJ (\u200d) must NOT be stripped from meaningful text
    norm_emoji = normalize_sentence("　👨‍👩‍👧が笑う！　")
    assert "\u200d" in norm_emoji
    assert norm_emoji == "👨‍👩‍👧が笑う"


def test_nli_ensure_tokenizer_caching_and_offline_bypass(monkeypatch, settings):
    import transformers

    from assignment.services.nli_engine import HuggingFaceNLIEngine

    load_calls: list[tuple[str, bool | None]] = []

    def fake_from_pretrained(name, **kwargs):
        load_calls.append((name, kwargs.get("local_files_only")))
        if kwargs.get("local_files_only"):
            raise OSError("not in local cache")
        return lambda premises, hypotheses, **_kw: {"input_ids": [[1, 2, 3]] * len(premises)}

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fake_from_pretrained)

    # Disabled model must never load or require HF tokenizer files
    settings.NLI_ENABLE_MODEL = False
    disabled_engine = HuggingFaceNLIEngine("mock/disabled-tok")
    assert disabled_engine.ensure_tokenizer() is None
    assert disabled_engine.count_pair_tokens("alpha", "beta") == 0
    assert load_calls == []

    # Enabled model caches tokenizer once across calls (trying local_files_only=True then hub)
    settings.NLI_ENABLE_MODEL = True
    enabled_engine = HuggingFaceNLIEngine("mock/cached-tok")
    tok1 = enabled_engine.ensure_tokenizer()
    tok2 = enabled_engine.ensure_tokenizer()
    assert tok1 is not None
    assert tok1 is tok2
    assert load_calls == [("mock/cached-tok", True), ("mock/cached-tok", None)]

    # Disabling model after tokenizer is cached still bypasses tokenization
    settings.NLI_ENABLE_MODEL = False
    assert enabled_engine.ensure_tokenizer() is None
    assert enabled_engine.count_pair_tokens("alpha", "beta") == 0


def test_nli_count_pair_tokens_checks_both_directions_without_truncation(monkeypatch, settings):
    from assignment.services.judge_service import validate_pairs_token_limits
    from assignment.services.nli_engine import (
        NLI_MAX_LENGTH,
        HuggingFaceNLIEngine,
        TokenLimitExceededError,
        nli_engine,
    )

    settings.NLI_ENABLE_MODEL = True
    engine = HuggingFaceNLIEngine("mock/asymmetric-tok")
    captured_kwargs: list[dict[str, object]] = []

    def asymmetric_tokenizer(premises, hypotheses, **kwargs):
        captured_kwargs.append(dict(kwargs))
        lengths = [
            NLI_MAX_LENGTH + 1 if p == "short" and h == "reverse-overflow" else 10
            for p, h in zip(premises, hypotheses, strict=True)
        ]
        return {"input_ids": [[0] * length for length in lengths]}

    engine._tokenizer = asymmetric_tokenizer
    engine._tokenizer_model_name = engine.model_name

    assert engine.count_pair_tokens("reverse-overflow", "short") == NLI_MAX_LENGTH + 1
    assert captured_kwargs and captured_kwargs[0].get("truncation") is False

    # Also verify judge_service.validate_pairs_token_limits raises on reverse-direction overflow
    monkeypatch.setattr(nli_engine, "_tokenizer", asymmetric_tokenizer)
    monkeypatch.setattr(nli_engine, "_tokenizer_model_name", nli_engine.model_name)
    with pytest.raises(TokenLimitExceededError):
        validate_pairs_token_limits([("reverse-overflow", "short")])


def test_nli_offline_fallback_rounds_directional_entail_and_ensure_loaded_orders_model_last(
    monkeypatch,
):
    from types import SimpleNamespace

    import transformers

    from assignment.services.nli_engine import HuggingFaceNLIEngine
    from assignment.services.normalizer import Tier1Result

    offline_engine = HuggingFaceNLIEngine("mock/offline-round")
    monkeypatch.setattr(offline_engine, "_ensure_loaded", lambda: False)
    t1 = Tier1Result(
        decisive=False,
        label="NO_ENTAIL",
        score=0.0,
        reason="mock",
        canonical_s1="alpha",
        canonical_s2="beta",
        fallback_score=0.49996,
    )
    detail = offline_engine.score_pairs_batch([("alpha", "beta")], [t1])[0]
    assert detail.score == 0.5
    assert detail.forward_entail == 0.5
    assert detail.reverse_entail == 0.5

    # Verify _ensure_loaded populates metadata fields before _model becomes non-None
    class OrderedEngine(HuggingFaceNLIEngine):
        def __init__(self, model_name: str):
            self.set_order: list[str] = []
            super().__init__(model_name)

        def __setattr__(self, name, value):
            if hasattr(self, "set_order") and name in {
                "_tokenizer",
                "_tokenizer_model_name",
                "_entail_idx",
                "_contradict_idx",
                "model_revision",
                "_model",
            }:
                if value is not None:
                    self.set_order.append(name)
            super().__setattr__(name, value)

    class FakeModel:
        config = SimpleNamespace(
            id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
            num_labels=3,
            _commit_hash="rev-order-1",
        )

        def eval(self):
            return self

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda _name, **_kw: object()
    )
    monkeypatch.setattr(
        transformers.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda _name: FakeModel(),
    )
    ordered = OrderedEngine("mock/order-check")
    assert ordered._ensure_loaded() is True
    assert ordered.set_order[-1] == "_model"
    assert {"_entail_idx", "_contradict_idx", "model_revision"}.issubset(ordered.set_order[:-1])
