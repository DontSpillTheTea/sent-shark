import pytest

from assignment.services.normalizer import (
    canonicalize_english,
    canonicalize_japanese,
    evaluate_tier1,
    is_same_language_pair,
    normalize_lexical_surface,
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
    assert canonicalize_japanese("ミーティングは先送りされた。") == "会議は延期された"
    assert canonicalize_japanese("昨日東京へ足を運んだ。") == "昨日東京へ行った"
    assert canonicalize_japanese("図書館で本を読みました。") == canonicalize_japanese(
        "図書館で本を読んだ。"
    )


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


@pytest.mark.django_db
def test_passive_vs_active_japanese_voice_not_collapsed_to_entail():
    from assignment.services.judge_service import judge_single

    assert canonicalize_japanese("彼が逮捕された。") != canonicalize_japanese("彼が逮捕した。")
    assert canonicalize_japanese("昨日東京へ行きました。") == "昨日東京へ行った"
    assert canonicalize_japanese("会議は延期になりました。") == canonicalize_japanese(
        "会議は延期された。"
    )
    assert canonicalize_japanese("イベントは中止になった。") == canonicalize_japanese(
        "イベントは中止された。"
    )

    t1 = evaluate_tier1("彼が逮捕された。", "彼が逮捕した。")
    assert not (t1.decisive and t1.label == "ENTAIL")

    res = judge_single("彼が逮捕された。", "彼が逮捕した。")
    assert res.label == "NO_ENTAIL"


def test_temporal_a_year_ago_preserves_modifiers():
    assert "last year" not in canonicalize_english("half a year ago")
    assert normalize_lexical_surface("He left half a year ago.") == "He left half a year ago"
    assert canonicalize_english("He left half a year ago.") == "he left half year ago"
    assert canonicalize_english("He left nearly a year ago.") == "he left nearly year ago"
    assert canonicalize_english("He quit smoking a year ago.") == "he quit smoking last year"


def test_nli_calibrate_score_guards_surface_contradictions_and_fuses_penalty():
    from assignment.services.nli_engine import HuggingFaceNLIEngine

    # Surface contradiction (raw_contra >= 0.35) must block canonical synonym rescue
    raw_metrics = (0.10, 0.12, 0.45, 0.06)
    canon_metrics = (0.90, 0.88, 0.02, 0.87)
    _, _, p_contra, score = HuggingFaceNLIEngine._calibrate_score(
        raw_metrics, canon_metrics, fallback_score=0.20
    )
    assert p_contra == pytest.approx(0.45)
    assert score == pytest.approx(0.275)
    assert score < 0.50

    # Fused else branch must apply (1.0 - max_contra) penalty to averaged directional entailment
    raw_metrics_else = (0.90, 0.80, 0.05, 0.8075)
    canon_metrics_else = (0.85, 0.85, 0.30, 0.595)
    _, _, p_contra_else, score_else = HuggingFaceNLIEngine._calibrate_score(
        raw_metrics_else, canon_metrics_else, fallback_score=0.50
    )
    assert p_contra_else == pytest.approx(0.30)
    assert score_else == pytest.approx(0.595)

    # Widened directional hypernym guard (< 0.15) must prevent lexical boosting
    raw_hyp = (0.90, 0.12, 0.02, 0.50)
    canon_hyp = (0.90, 0.10, 0.02, 0.49)
    _, _, _, score_hyp = HuggingFaceNLIEngine._calibrate_score(
        raw_hyp, canon_hyp, fallback_score=0.90
    )
    assert score_hyp == pytest.approx(0.4949)
    assert score_hyp < 0.50


@pytest.mark.django_db
def test_evaluate_uncached_pairs_and_nli_warm_up_without_db_mutation():
    from assignment.models import JudgmentRecord
    from assignment.services.judge_service import evaluate_uncached_pairs
    from assignment.services.nli_engine import NoOpEscalationJudge, nli_engine

    assert JudgmentRecord.objects.count() == 0
    assert nli_engine.warm_up() is True

    outputs = evaluate_uncached_pairs(
        [
            ("私は猫が大好きだ。", "猫が好きです。"),
            ("He gave up smoking last year.", "He quit smoking a year ago."),
            ("彼が逮捕された。", "彼が逮捕した。"),
        ],
        escalation_judge=NoOpEscalationJudge(),
    )
    assert [out.label for out in outputs] == ["ENTAIL", "ENTAIL", "NO_ENTAIL"]
    assert all(out.cached is False for out in outputs)
    assert JudgmentRecord.objects.count() == 0
