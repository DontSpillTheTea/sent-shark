import pytest
from rest_framework import status
from rest_framework.test import APIClient

from assignment.rate_limit import rate_limiter


@pytest.mark.integration
@pytest.mark.django_db
class TestMangaDialogueRegisters:
    """
    Domain-specific tests for Japanese manga dialogue registers (yakuwarigo),
    colloquial contractions, idiom paraphrasing, and negation/antonym traps
    under the Phase 4 exact_plus_nli + MiniLMv2 pipeline.

    Note: Under exact_plus_nli (after removing dev-tailored Tier 1 SOV regexes),
    MiniLMv2 handles register, contraction, idiom, restrictive-particle, negation,
    and antonym pairs natively; pure Japanese SOV case-particle swaps such as
    ("猫が魚を食べた。", "魚が猫を食べた。") remain a known model limitation.
    """

    def setup_method(self):
        self.client = APIClient()
        rate_limiter.reset()

    def teardown_method(self):
        rate_limiter.reset()

    @pytest.mark.parametrize(
        ("sentence1", "sentence2", "expected"),
        [
            # Masculine casual yakuwarigo vs polite keigo
            ("俺が守る！", "私が守ります。", "ENTAIL"),
            # Colloquial contraction (~ちゃった) vs polite (~てしまいました)
            ("もう食べちゃったよ", "すでに食べてしまいました", "ENTAIL"),
            # Idiomatic paraphrase
            ("彼は昨日東京に行った。", "昨日、彼は東京へ足を運んだ。", "ENTAIL"),
            # Restrictive particle trap (しか〜ない vs も)
            ("猫しか好きじゃない。", "猫も犬も好きだ。", "NO_ENTAIL"),
            # Verbal negation polarity flip
            ("猫は魚を食べない。", "猫は魚を食べる。", "NO_ENTAIL"),
            # Direct emotion polarity flip
            ("私は猫が大好きだ。", "私は猫が大嫌いだ。", "NO_ENTAIL"),
            # Exam outcome antonym
            ("彼は試験に合格した。", "彼は試験に落ちた。", "NO_ENTAIL"),
        ],
    )
    def test_manga_register_and_trap_pairs(self, sentence1, sentence2, expected):
        rate_limiter.reset()
        response = self.client.post(
            "/judge",
            {"sentence1": sentence1, "sentence2": sentence2},
            format="json",
        )
        assert response.status_code == status.HTTP_200_OK
        assert response.data["label"] == expected
