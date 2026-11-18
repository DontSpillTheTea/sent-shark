import pytest
from rest_framework import status
from rest_framework.test import APIClient

from assignment.models import JudgmentRecord, RequestAuditLog
from assignment.rate_limit import rate_limiter


@pytest.mark.django_db
class TestJudgeAPI:
    def setup_method(self):
        self.client = APIClient()
        rate_limiter.reset()

    def teardown_method(self):
        rate_limiter.reset()

    def test_f1_prompt_example_entailment(self):
        response = self.client.post(
            "/judge",
            {"sentence1": "私は猫が大好きだ。", "sentence2": "猫が好きです。"},
            format="json",
        )
        assert response.status_code == status.HTTP_200_OK
        assert response.data["label"] == "ENTAIL"
        assert 0.70 <= response.data["score"] <= 1.0
        assert JudgmentRecord.objects.count() == 1
        assert RequestAuditLog.objects.filter(status_code=200).count() == 1

    def test_f1_english_paraphrase_and_contradiction_and_neutral(self):
        entail = self.client.post(
            "/judge",
            {
                "sentence1": "The movie was extremely boring.",
                "sentence2": "The film was very dull.",
            },
            format="json",
        )
        assert entail.status_code == status.HTTP_200_OK
        assert entail.data["label"] == "ENTAIL"
        assert entail.data["score"] >= 0.70

        contra = self.client.post(
            "/judge",
            {
                "sentence1": "A cat is a carnivore.",
                "sentence2": "A cat is a vegetarian.",
            },
            format="json",
        )
        assert contra.status_code == status.HTTP_200_OK
        assert contra.data["label"] == "NO_ENTAIL"
        assert contra.data["score"] <= 0.25

        neutral = self.client.post(
            "/judge",
            {
                "sentence1": "The Earth orbits the Sun.",
                "sentence2": "Humans are capable of thought.",
            },
            format="json",
        )
        assert neutral.status_code == status.HTTP_200_OK
        assert neutral.data["label"] == "NO_ENTAIL"
        assert neutral.data["score"] <= 0.25

    def test_f1_validation_rejects_empty_and_cross_language(self):
        empty_resp = self.client.post(
            "/judge", {"sentence1": "   ", "sentence2": "猫が好きです。"}, format="json"
        )
        assert empty_resp.status_code == status.HTTP_400_BAD_REQUEST

        cross_resp = self.client.post(
            "/judge",
            {"sentence1": "私は猫が大好きだ。", "sentence2": "I love cats."},
            format="json",
        )
        assert cross_resp.status_code == status.HTTP_400_BAD_REQUEST

    def test_f2_bulk_raw_list_and_wrapped_payload_up_to_100(self):
        pairs = [
            {"sentence1": "私は猫が大好きだ。", "sentence2": "猫が好きです。"},
            {"sentence1": "A cat is a carnivore.", "sentence2": "A cat is a vegetarian."},
            {
                "sentence1": "雨が降っているので、傘を持っていく。",
                "sentence2": "雨天のため、傘を持参する。",
            },
            {
                "sentence1": "The Earth orbits the Sun.",
                "sentence2": "Humans are capable of thought.",
            },
        ]
        resp_wrapped = self.client.post("/judge/bulk", {"pairs": pairs}, format="json")
        assert resp_wrapped.status_code == status.HTTP_200_OK
        assert isinstance(resp_wrapped.data, list)
        assert len(resp_wrapped.data) == 4
        assert [row["label"] for row in resp_wrapped.data] == [
            "ENTAIL",
            "NO_ENTAIL",
            "ENTAIL",
            "NO_ENTAIL",
        ]

        # Test raw JSON array of exactly 100 pairs (mixed unique + repeated)
        hundred_pairs = [
            {
                "sentence1": f"テスト文{i % 10}です。",
                "sentence2": f"テスト文{i % 10}だ。",
            }
            for i in range(100)
        ]
        resp_array = self.client.post("/judge/bulk", hundred_pairs, format="json")
        assert resp_array.status_code == status.HTTP_200_OK
        assert len(resp_array.data) == 100
        assert all(item["label"] == "ENTAIL" for item in resp_array.data)

    def test_f2_bulk_rejects_empty_and_over_100_pairs(self):
        resp_empty = self.client.post("/judge/bulk", {"pairs": []}, format="json")
        assert resp_empty.status_code == status.HTTP_400_BAD_REQUEST

        too_many = [
            {"sentence1": f"Sentence {i}", "sentence2": f"Sentence {i}"} for i in range(101)
        ]
        resp_over = self.client.post("/judge/bulk", {"pairs": too_many}, format="json")
        assert resp_over.status_code == status.HTTP_400_BAD_REQUEST

    def test_f3_evaluate_and_stats_accuracy_calculation(self):
        eval_payload = {
            "pairs": [
                {
                    "sentence1": "私は猫が大好きだ。",
                    "sentence2": "猫が好きです。",
                    "expected_label": "ENTAIL",
                },
                {
                    "sentence1": "雨が降っているので、傘を持っていく。",
                    "sentence2": "雨天のため、傘を持参する。",
                    "expected_label": "ENTAIL",
                },
                {
                    "sentence1": "A cat is a carnivore.",
                    "sentence2": "A cat is a vegetarian.",
                    "expected_label": "NO_ENTAIL",
                },
                {
                    "sentence1": "彼女は毎朝コーヒーを飲む。",
                    "sentence2": "彼女は毎朝紅茶を飲む。",
                    "expected_label": "NO_ENTAIL",
                },
            ]
        }
        eval_resp = self.client.post("/judge/evaluate", eval_payload, format="json")
        assert eval_resp.status_code == status.HTTP_200_OK
        assert eval_resp.data["total"] == 4
        assert eval_resp.data["correct"] == 4
        assert eval_resp.data["accuracy"] == 1.0
        assert eval_resp.data["f1"] == 1.0

        stats_resp = self.client.get("/judge/stats")
        assert stats_resp.status_code == status.HTTP_200_OK
        assert stats_resp.data["total_judgments"] == 4
        assert stats_resp.data["stored_ground_truth_count"] == 4
        assert stats_resp.data["stored_accuracy"] == 1.0
        assert stats_resp.data["audit"]["total_requests"] >= 1
        assert RequestAuditLog.objects.count() >= 2

    def test_api_v1_route_aliases_work(self):
        resp = self.client.post(
            "/api/v1/judge",
            {"sentence1": "猫が好きです。", "sentence2": "私は猫が大好きだ。"},
            format="json",
        )
        assert resp.status_code == status.HTTP_200_OK
        assert resp.data["label"] == "ENTAIL"

    def test_tier1_offline_fallback_when_model_disabled(self, settings):
        settings.NLI_ENABLE_MODEL = False
        resp_entail = self.client.post(
            "/judge",
            {"sentence1": "私は猫が大好きだ。", "sentence2": "猫が好きです。"},
            format="json",
        )
        assert resp_entail.status_code == status.HTTP_200_OK
        assert resp_entail.data["label"] == "ENTAIL"

        resp_contra = self.client.post(
            "/judge",
            {"sentence1": "A cat is a carnivore.", "sentence2": "A cat is a vegetarian."},
            format="json",
        )
        assert resp_contra.status_code == status.HTTP_200_OK
        assert resp_contra.data["label"] == "NO_ENTAIL"

    def test_tier3_escalation_gateway_on_grey_zone_and_timeout_fallback(self):
        from assignment.services.nli_engine import nli_engine
        from assignment.services.normalizer import evaluate_tier1

        class MockLLM:
            def classify(self, sentence1: str, sentence2: str):
                return ("ENTAIL", 0.88)

        class FailingLLM:
            def classify(self, sentence1: str, sentence2: str):
                return None

        # ("She bought a brand new car.", "She purchased a new automobile.") lands in [0.40, 0.65]
        pair = ("She bought a brand new car.", "She purchased a new automobile.")
        t1 = [evaluate_tier1(*pair)]
        escalated = nli_engine.score_pairs_batch([pair], t1, escalation_judge=MockLLM())[0]
        assert escalated.tier_used == "tier3_llm"
        assert escalated.label == "ENTAIL"
        assert escalated.score == 0.88

        fallback = nli_engine.score_pairs_batch([pair], t1, escalation_judge=FailingLLM())[0]
        assert fallback.tier_used == "tier2_nli"
        assert fallback.label == "ENTAIL"
