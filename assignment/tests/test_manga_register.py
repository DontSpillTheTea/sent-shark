import pytest
from rest_framework import status
from rest_framework.test import APIClient

from assignment.rate_limit import rate_limiter


@pytest.mark.integration
@pytest.mark.django_db
class TestMangaDialogueRegisters:
    """
    Domain-specific tests for Japanese manga dialogue registers (yakuwarigo),
    colloquial contractions, direct synonyms, active/passive equivalence,
    grammatical-role reversals, one-way hypernym asymmetry, and negation/numerical
    contradictions under the exact_plus_nli:v2:min + mDeBERTa-v3-base-mnli-xnli pipeline.

    Note: Certain cross-script culinary loanword pairs (such as 牛肉/ビーフ) remain
    a documented zero-shot model limitation rather than being patched via rules.
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
            # Direct Japanese synonym equivalence
            ("母は夕食を作る。", "母は晩ご飯を作る。", "ENTAIL"),
            ("彼は自動車を運転する。", "彼は車を運転する。", "ENTAIL"),
            (
                "午後の会議はオンラインで行います。",
                "午後のミーティングはオンラインで行います。",
                "ENTAIL",
            ),
            # Active / passive voice equivalence
            ("先生がその学生の努力を褒めた。", "その学生は先生に努力を褒められた。", "ENTAIL"),
            ("警察が容疑者の男を逮捕した。", "容疑者の男は警察に逮捕された。", "ENTAIL"),
            # Japanese grammatical-role reversals (subject/object & dative swaps)
            ("猫が庭でネズミを追いかけた。", "ネズミが庭で猫を追いかけた。", "NO_ENTAIL"),
            ("先生が教室で学生を褒めた。", "学生が教室で先生を褒めた。", "NO_ENTAIL"),
            ("母が娘に誕生日の本をあげた。", "娘が母に誕生日の本をあげた。", "NO_ENTAIL"),
            # One-way entailment (hyponym/specificity) vs symmetric equivalence
            ("彼女は犬を飼っている。", "彼女は動物を飼っている。", "NO_ENTAIL"),
            ("姉はステーキを食べる。", "姉は牛肉のステーキを食べる。", "NO_ENTAIL"),
            # Restrictive particle trap (しか〜ない vs も)
            ("猫しか好きじゃない。", "猫も犬も好きだ。", "NO_ENTAIL"),
            # Verbal negation polarity flip & numerical contradiction
            ("猫は魚を食べない。", "猫は魚を食べる。", "NO_ENTAIL"),
            ("テーブルの上にりんごが三つある。", "テーブルの上にりんごが五つある。", "NO_ENTAIL"),
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
