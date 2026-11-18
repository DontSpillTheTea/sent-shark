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


def test_nli_scoring_mode_validation_and_min_vs_mean(monkeypatch):
    from types import SimpleNamespace

    import torch

    from assignment.services.nli_engine import HuggingFaceNLIEngine

    engine = HuggingFaceNLIEngine("mock/min-vs-mean")
    with pytest.raises(ValueError, match="Unsupported scoring_mode"):
        engine.score_pairs_batch([("A", "B")], scoring_mode="max")  # type: ignore[arg-type]

    monkeypatch.setattr(engine, "_ensure_loaded", lambda: True)
    engine._entail_idx = 0
    engine._contradict_idx = 2
    engine._tokenizer = lambda premises, hypotheses, **_kw: {
        "input_ids": torch.zeros((len(premises), 2), dtype=torch.long)
    }

    # First sequence (A->B): ~0.95 entail; second sequence (B->A): ~0.20 entail
    probs = torch.tensor([[0.95, 0.04, 0.01], [0.20, 0.78, 0.02]], dtype=torch.float32)
    monkeypatch.setattr(engine, "_model", lambda **_enc: SimpleNamespace(logits=torch.log(probs)))

    mean_res = engine.score_pairs_batch([("Alpha", "Beta")], scoring_mode="mean")[0]
    min_res = engine.score_pairs_batch([("Alpha", "Beta")], scoring_mode="min")[0]
    assert mean_res.label == "ENTAIL"
    assert min_res.label == "NO_ENTAIL"
    assert min_res.score < mean_res.score


def test_settings_optional_int_env_error_context(monkeypatch):
    from assignment.settings import _optional_int_env

    monkeypatch.setenv("TEST_NLI_INDEX", "not-an-int")
    with pytest.raises(ValueError, match="TEST_NLI_INDEX"):
        _optional_int_env("TEST_NLI_INDEX")


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
