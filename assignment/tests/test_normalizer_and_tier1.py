from assignment.services.normalizer import (
    canonicalize_english,
    canonicalize_japanese,
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


def test_japanese_canonical_register_and_pronoun_prodrop():
    c1 = canonicalize_japanese("私は猫が大好きだ。", drop_first_person=True)
    c2 = canonicalize_japanese("猫が好きです。", drop_first_person=True)
    assert c1 == c2 == "猫は好き"

    res = evaluate_tier1("私は猫が大好きだ。", "猫が好きです。")
    assert res.decisive is True
    assert res.label == "ENTAIL"
    assert res.score >= 0.85


def test_third_person_subjects_do_not_falsely_match():
    res = evaluate_tier1("私は猫が大好きだ。", "彼は猫が大好きだ。")
    assert res.decisive is False
    assert res.fallback_score < 0.50


def test_english_contractions_and_article_normalization():
    assert canonicalize_english("He doesn't like the dog.") == "he does not like dog"


def test_antonym_and_negation_contradiction_vetoes():
    res_en = evaluate_tier1("A cat is a carnivore.", "A cat is a vegetarian.")
    assert res_en.decisive is True
    assert res_en.label == "NO_ENTAIL"
    assert res_en.score <= 0.15

    res_jp = evaluate_tier1("私は猫が大好きだ。", "私は猫が大嫌いだ。")
    assert res_jp.decisive is True
    assert res_jp.label == "NO_ENTAIL"
    assert res_jp.score <= 0.15

    res_neg = evaluate_tier1("私は犬が好きだ。", "私は犬が好きじゃない。")
    assert res_neg.decisive is True
    assert res_neg.label == "NO_ENTAIL"
    assert res_neg.score <= 0.15


def test_japanese_and_english_role_inversion_veto():
    res_jp = evaluate_tier1("猫が魚を食べた。", "魚が猫を食べた。")
    assert res_jp.decisive is True
    assert res_jp.label == "NO_ENTAIL"
    assert res_jp.score <= 0.10

    res_en = evaluate_tier1("dog chased cat", "cat chased dog")
    assert res_en.decisive is True
    assert res_en.label == "NO_ENTAIL"
    assert res_en.score <= 0.10
