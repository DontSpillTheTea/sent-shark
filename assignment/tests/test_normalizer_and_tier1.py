import pytest

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


@pytest.mark.django_db
def test_synonym_and_temporal_paraphrases_resolve_to_entail():
    from assignment.services.judge_service import judge_single

    en_res = judge_single("He gave up smoking last year.", "He quit smoking a year ago.")
    assert en_res.label == "ENTAIL"
    assert en_res.score >= 0.50

    jp_res = judge_single(
        "会議は来週の月曜日に延期された。", "ミーティングは来週月曜へ先送りになった。"
    )
    assert jp_res.label == "ENTAIL"
    assert jp_res.score >= 0.50


def test_lexical_paraphrase_and_verb_ending_canonicalizations():
    assert canonicalize_english("She purchased an automobile.") == "she bought car"
    assert canonicalize_english("The film was extremely dull.") == "movie was very boring"
    assert canonicalize_english("Do not give up.") == "do not quit"

    assert canonicalize_japanese("電車に乗車しました。") == "電車に乗った"
    assert canonicalize_japanese("新しい車を購入した。") == "新しい車を買った"
    assert canonicalize_japanese("会議は終了しました。") == "会議は終わった"
    assert canonicalize_japanese("ミーティングは先送りされた。") == "会議は延期した"
    assert canonicalize_japanese("昨日東京へ足を運んだ。") == "昨日東京へ行った"


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("sentence1", "sentence2"),
    [
        ("I adopted a Siamese cat.", "I adopted an animal."),
        ("私は柴犬を飼っている。", "私は動物を飼っている。"),
        ("He gave up smoking last year.", "He started smoking last year."),
        ("会議は来週の月曜日に延期された。", "会議は今週の金曜日に前倒しされた。"),
        ("The car is red.", "The automobile is blue."),
    ],
)
def test_directional_hypernyms_and_lexical_contradictions_remain_no_entail(sentence1, sentence2):
    from assignment.services.judge_service import judge_single

    res = judge_single(sentence1, sentence2)
    assert res.label == "NO_ENTAIL"
