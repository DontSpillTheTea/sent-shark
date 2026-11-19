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

    @pytest.mark.integration
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

    @pytest.mark.integration
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

    @pytest.mark.integration
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

    def test_unrequired_stats_evaluate_and_v1_judge_aliases_removed(self, monkeypatch):
        from assignment.services import nli_engine as nli_mod

        monkeypatch.setattr(
            nli_mod.nli_engine,
            "score_pairs_batch",
            lambda pairs, tier1_results=None: [
                nli_mod.NLIScoreDetail(
                    score=0.92,
                    label="ENTAIL",
                    tier_used="tier2_nli",
                    forward_entail=0.93,
                    reverse_entail=0.91,
                    max_contradiction=0.02,
                    forward_contradict=0.02,
                    reverse_contradict=0.01,
                )
                for _ in pairs
            ],
        )
        assert self.client.get("/judge/stats").status_code == status.HTTP_404_NOT_FOUND
        assert self.client.get("/judge/stats/").status_code == status.HTTP_404_NOT_FOUND
        assert (
            self.client.post("/judge/evaluate", {"pairs": []}, format="json").status_code
            == status.HTTP_404_NOT_FOUND
        )
        assert (
            self.client.post("/judge/evaluate/", {"pairs": []}, format="json").status_code
            == status.HTTP_404_NOT_FOUND
        )
        for path in (
            "/api/v1/judge",
            "/api/v1/judge/",
            "/api/v1/judge/bulk",
            "/api/v1/judge/bulk/",
        ):
            assert (
                self.client.post(
                    path,
                    {"sentence1": "猫が好きです。", "sentence2": "私は猫が大好きだ。"},
                    format="json",
                ).status_code
                == status.HTTP_404_NOT_FOUND
            )
        assert self.client.get("/api/v1/judge/stats").status_code == status.HTTP_404_NOT_FOUND
        assert self.client.get("/api/v1/judge/stats/").status_code == status.HTTP_404_NOT_FOUND
        assert (
            self.client.post("/api/v1/judge/evaluate", {"pairs": []}, format="json").status_code
            == status.HTTP_404_NOT_FOUND
        )
        assert (
            self.client.post("/api/v1/judge/evaluate/", {"pairs": []}, format="json").status_code
            == status.HTTP_404_NOT_FOUND
        )

        # Kept routes still route properly
        assert self.client.get("/health").status_code == status.HTTP_200_OK
        assert self.client.get("/playground").status_code == status.HTTP_200_OK
        assert self.client.get("/api/v1/user/me/").status_code == status.HTTP_401_UNAUTHORIZED
        assert (
            self.client.post(
                "/judge",
                {"sentence1": "私は猫が大好きだ。", "sentence2": "猫が好きです。"},
                format="json",
            ).status_code
            == status.HTTP_200_OK
        )
        assert (
            self.client.post(
                "/judge/bulk",
                {"pairs": [{"sentence1": "私は猫が大好きだ。", "sentence2": "猫が好きです。"}]},
                format="json",
            ).status_code
            == status.HTTP_200_OK
        )

    def test_tier1_offline_fallback_when_model_disabled(self, settings):
        settings.NLI_ENABLE_MODEL = False
        resp_entail = self.client.post(
            "/judge",
            {"sentence1": "私は猫が大好きだ。", "sentence2": "　私は猫が大好きだ！　"},
            format="json",
        )
        assert resp_entail.status_code == status.HTTP_200_OK
        assert resp_entail.data["label"] == "ENTAIL"
        assert resp_entail.data["score"] == 0.99

        resp_contra = self.client.post(
            "/judge",
            {"sentence1": "A cat is a carnivore.", "sentence2": "A cat is a vegetarian."},
            format="json",
        )
        assert resp_contra.status_code == status.HTTP_200_OK
        assert resp_contra.data["label"] == "NO_ENTAIL"
        assert resp_contra.data["score"] == 0.0


def test_unknown_nli_id2label_raises_explicit_value_error(monkeypatch, settings):
    from types import SimpleNamespace

    import torch
    import transformers

    from assignment.services.nli_engine import HuggingFaceNLIEngine
    from assignment.services.normalizer import evaluate_tier1

    current_id2label = {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}
    current_num_labels = 3

    def fake_tokenizer(
        premises,
        hypotheses,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    ):
        return {"input_ids": torch.zeros((len(premises), 4), dtype=torch.long)}

    class FakeModel:
        def __init__(self, id2label, num_labels):
            self.config = SimpleNamespace(
                id2label=id2label,
                num_labels=num_labels,
                _commit_hash="deadbeef1234",
            )

        def eval(self):
            return self

        def __call__(self, **encoded):
            batch_n = encoded["input_ids"].shape[0]
            # High logit at index 2 (custom entailment when overridden), low at index 0
            logits = torch.tensor([[-2.0, 0.0, 4.0]] * batch_n, dtype=torch.float32)
            return SimpleNamespace(logits=logits)

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda _name: fake_tokenizer,
    )
    monkeypatch.setattr(
        transformers.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda _name: FakeModel(current_id2label, current_num_labels),
    )

    pair = ("私は猫が大好きだ。", "猫が好きです。")
    t1 = [evaluate_tier1(*pair)]

    # 1. Unknown id2label with no explicit override must raise ValueError explicitly (twice too!)
    settings.NLI_ENTAIL_INDEX = None
    settings.NLI_CONTRADICT_INDEX = None
    engine = HuggingFaceNLIEngine("mock/unknown-id2label")
    with pytest.raises(ValueError, match="id2label"):
        engine.score_pairs_batch([pair], t1)
    with pytest.raises(ValueError, match="id2label"):
        engine.score_pairs_batch([pair], t1)

    # 2. Ambiguous / negated id2label ("entailment" + "not_entailment", or "non-contradiction")
    for bad_map in (
        {0: "entailment", 1: "not_entailment", 2: "neutral"},
        {0: "entailment", 1: "non-contradiction"},
    ):
        current_id2label = bad_map
        current_num_labels = len(bad_map)
        ambiguous_engine = HuggingFaceNLIEngine("mock/ambiguous-id2label")
        with pytest.raises(ValueError):
            ambiguous_engine.score_pairs_batch([pair], t1)

    # 3. Inferred index >= num_labels must raise ValueError instead of failing at inference
    current_id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
    current_num_labels = 2
    oob_engine = HuggingFaceNLIEngine("mock/oob-inferred-id2label")
    with pytest.raises(ValueError, match="out of range"):
        oob_engine.score_pairs_batch([pair], t1)

    # 4. Model with non-contradiction + contradiction resolves contradiction cleanly
    current_id2label = {0: "entailment", 1: "non-contradiction", 2: "contradiction"}
    current_num_labels = 3
    clean_engine = HuggingFaceNLIEngine("mock/non-contradiction-plus-contradiction")
    clean_res = clean_engine.score_pairs_batch([pair], t1)
    assert len(clean_res) == 1
    assert clean_res[0].label == "NO_ENTAIL"

    # 5. Invalid override indices (out-of-range, negative, equal, partial) raise ValueError
    current_id2label = {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}
    current_num_labels = 3
    for bad_entail, bad_contradict in [(3, 0), (-1, 0), (1, 1), (0, None)]:
        settings.NLI_ENTAIL_INDEX = bad_entail
        settings.NLI_CONTRADICT_INDEX = bad_contradict
        bad_engine = HuggingFaceNLIEngine("mock/invalid-override")
        with pytest.raises(ValueError):
            bad_engine.score_pairs_batch([pair], t1)

    # 6. Valid explicit override works and captures model_revision
    settings.NLI_ENTAIL_INDEX = 2
    settings.NLI_CONTRADICT_INDEX = 0
    override_engine = HuggingFaceNLIEngine("mock/explicit-override")
    res = override_engine.score_pairs_batch([pair], t1)
    assert len(res) == 1
    assert res[0].tier_used == "tier2_nli"
    assert res[0].label == "ENTAIL"
    assert res[0].forward_entail > 0.90
    assert override_engine.model_revision == "deadbeef1234"


@pytest.mark.django_db
class TestInputValidationAndTokenLimits:
    def setup_method(self):
        self.client = APIClient()
        rate_limiter.reset()

    def teardown_method(self):
        rate_limiter.reset()

    @staticmethod
    def _mock_nli(monkeypatch):
        from assignment.services import nli_engine as nli_mod

        monkeypatch.setattr(
            nli_mod.nli_engine,
            "score_pairs_batch",
            lambda pairs, tier1_results=None: [
                nli_mod.NLIScoreDetail(
                    score=0.92,
                    label="ENTAIL",
                    tier_used="tier2_nli",
                    forward_entail=0.93,
                    reverse_entail=0.91,
                    max_contradiction=0.02,
                    forward_contradict=0.02,
                    reverse_contradict=0.01,
                )
                for _ in pairs
            ],
        )

    @pytest.mark.parametrize(
        ("s1", "s2"),
        [
            (123, "456"),
            ("456", 123),
            (12.5, "12.5"),
            (True, False),
            (["a"], "a"),
            ({}, "a"),
        ],
    )
    def test_rejects_non_string_json_types_on_single_and_bulk(self, s1, s2):
        r_single = self.client.post(
            "/judge",
            {"sentence1": s1, "sentence2": s2},
            format="json",
            REMOTE_ADDR="198.51.100.41",
        )
        assert r_single.status_code == status.HTTP_400_BAD_REQUEST

        r_bulk_wrapped = self.client.post(
            "/judge/bulk",
            {"pairs": [{"sentence1": s1, "sentence2": s2}]},
            format="json",
            REMOTE_ADDR="198.51.100.42",
        )
        assert r_bulk_wrapped.status_code == status.HTTP_400_BAD_REQUEST

        r_bulk_list = self.client.post(
            "/judge/bulk",
            [{"sentence1": s1, "sentence2": s2}],
            format="json",
            REMOTE_ADDR="198.51.100.43",
        )
        assert r_bulk_list.status_code == status.HTTP_400_BAD_REQUEST
        assert JudgmentRecord.objects.count() == 0
        assert RequestAuditLog.objects.filter(status_code=400).count() == 3

    def test_accepts_numeric_strings_while_rejecting_numeric_json_primitives(self):
        resp_str = self.client.post(
            "/judge",
            {"sentence1": "123", "sentence2": "123"},
            format="json",
            REMOTE_ADDR="198.51.100.40",
        )
        assert resp_str.status_code == status.HTTP_200_OK
        assert resp_str.json() == {"label": "ENTAIL", "score": 0.99}

    @pytest.mark.parametrize(
        ("s1", "s2"),
        [
            ("", "猫が好きです。"),
            ("   ", "猫が好きです。"),
            ("\u3000\u3000", "猫が好きです。"),
            ("\u200b", "\u200b"),
            ("\ufeff", "猫が好きです。"),
            ("\u200c\u200d", "\u200b"),
            ("...", "!!!"),
            ("。。。", "！？"),
            ("\u200b。。。", "\ufeff"),
            ("hello\x00world", "hello world"),
        ],
    )
    def test_rejects_empty_invisible_punctuation_only_and_nul_strings(self, s1, s2):
        resp = self.client.post(
            "/judge",
            {"sentence1": s1, "sentence2": s2},
            format="json",
            REMOTE_ADDR="198.51.100.44",
        )
        assert resp.status_code == status.HTTP_400_BAD_REQUEST
        assert JudgmentRecord.objects.count() == 0
        assert RequestAuditLog.objects.filter(status_code=400).count() == 1

    def test_preserves_internal_unicode_zwj_in_meaningful_text(self, monkeypatch):
        from assignment.services.normalizer import normalize_sentence

        self._mock_nli(monkeypatch)
        s1 = "👨‍👩‍👧が笑う"
        s2 = "👨‍👩‍👧が微笑む"
        assert "\u200d" in normalize_sentence(s1)

        resp = self.client.post(
            "/judge",
            {"sentence1": s1, "sentence2": s2},
            format="json",
            REMOTE_ADDR="198.51.100.45",
        )
        assert resp.status_code == status.HTTP_200_OK
        rec = JudgmentRecord.objects.get()
        assert "\u200d" in rec.sentence1
        assert "\u200d" in rec.sentence2

    def test_token_limit_256_passes_and_257_rejected_atomically_across_single_and_bulk(
        self, monkeypatch, settings
    ):
        from assignment.services import nli_engine as nli_mod
        from assignment.services.nli_engine import NLI_MAX_LENGTH, nli_engine

        settings.NLI_ENABLE_MODEL = True
        tok = nli_engine.ensure_tokenizer()
        assert tok is not None

        short = "cat"
        words = ["cat"] * 240
        while True:
            candidate = " ".join(words)
            tok_count = max(
                len(tok([candidate], [short], padding=False, truncation=False)["input_ids"][0]),
                len(tok([short], [candidate], padding=False, truncation=False)["input_ids"][0]),
            )
            if tok_count == NLI_MAX_LENGTH:
                break
            assert tok_count < NLI_MAX_LENGTH
            words.append("cat")
        boundary_256 = " ".join(words)
        overflow_257 = " ".join([*words, "cat"])
        assert len(overflow_257) <= 2000

        # Exact 256-token pair succeeds (200)
        self._mock_nli(monkeypatch)
        ok_resp = self.client.post(
            "/judge",
            {"sentence1": boundary_256, "sentence2": short},
            format="json",
            REMOTE_ADDR="198.51.100.46",
        )
        assert ok_resp.status_code == status.HTTP_200_OK
        assert JudgmentRecord.objects.count() == 1
        JudgmentRecord.objects.all().delete()

        # Any 257-token overflow (forward or reverse) must fail with 400 before NLI or DB writes
        def fail_if_nli_called(*_args, **_kwargs):
            raise AssertionError("score_pairs_batch must not run when token limit is exceeded")

        monkeypatch.setattr(nli_mod.nli_engine, "score_pairs_batch", fail_if_nli_called)

        for s1, s2 in ((overflow_257, short), (short, overflow_257)):
            bad_resp = self.client.post(
                "/judge",
                {"sentence1": s1, "sentence2": s2},
                format="json",
                REMOTE_ADDR="198.51.100.47",
            )
            assert bad_resp.status_code == status.HTTP_400_BAD_REQUEST
            assert "error" in bad_resp.json()
            assert JudgmentRecord.objects.count() == 0

        # Bulk atomicity: 1 valid pair (+ Tier-1 exact match) + 1 257-token pair -> 0 writes
        bulk_resp = self.client.post(
            "/judge/bulk",
            {
                "pairs": [
                    {"sentence1": "猫が好きです。", "sentence2": "猫が好きです。"},
                    {"sentence1": "私は猫が大好きだ。", "sentence2": "猫が好きです。"},
                    {"sentence1": short, "sentence2": overflow_257},
                ]
            },
            format="json",
            REMOTE_ADDR="198.51.100.48",
        )
        assert bulk_resp.status_code == status.HTTP_400_BAD_REQUEST
        assert "error" in bulk_resp.json()
        assert JudgmentRecord.objects.count() == 0
        assert RequestAuditLog.objects.filter(status_code=400).count() == 3

    def test_offline_disabled_model_allows_2000_char_input_without_tokenizer_files(
        self, monkeypatch, settings
    ):
        import transformers

        from assignment.services.nli_engine import nli_engine

        settings.NLI_ENABLE_MODEL = False
        monkeypatch.setattr(nli_engine, "_tokenizer", None)
        monkeypatch.setattr(nli_engine, "_tokenizer_model_name", None)

        def fail_if_tokenizer_loaded(*_args, **_kwargs):
            raise RuntimeError(
                "HF tokenizer files must not be required when NLI_ENABLE_MODEL=False"
            )

        monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fail_if_tokenizer_loaded)

        long_s1 = ("cat " * 400).strip()  # 1599 chars <= 2000, > 256 tokens
        long_s2 = ("dog " * 400).strip()
        resp = self.client.post(
            "/judge",
            {"sentence1": long_s1, "sentence2": long_s2},
            format="json",
            REMOTE_ADDR="198.51.100.49",
        )
        assert resp.status_code == status.HTTP_200_OK
        assert resp.json() == {"label": "NO_ENTAIL", "score": 0.0}

        too_long = "a" * 2001
        resp_over_char = self.client.post(
            "/judge",
            {"sentence1": too_long, "sentence2": "dog"},
            format="json",
            REMOTE_ADDR="198.51.100.49",
        )
        assert resp_over_char.status_code == status.HTTP_400_BAD_REQUEST
