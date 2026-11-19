import re

import pytest
from rest_framework import status
from rest_framework.test import APIClient

from assignment.models import JudgmentRecord, RequestAuditLog


@pytest.mark.django_db
def test_playground_renders_minimal_dependency_free_html():
    client = APIClient()
    resp = client.get("/playground")
    assert resp.status_code == status.HTTP_200_OK
    assert "text/html" in resp["Content-Type"]
    body = resp.content.decode("utf-8")
    assert 'name="sentence1"' in body
    assert 'name="sentence2"' in body
    assert "POST" in body
    assert "/judge" in body
    assert "X-Request-Id" in body
    assert "X-RateLimit-Limit" in body
    assert "X-RateLimit-Remaining" in body
    assert "Retry-After" in body
    assert "performance.now" in body
    assert "textContent" in body
    # Zero external scripts, stylesheets, or CDN URLs
    assert not re.search(r"<script[^>]+src\s*=", body, re.IGNORECASE)
    assert not re.search(r"<link[^>]+href\s*=", body, re.IGNORECASE)
    assert "http://" not in body
    assert "https://" not in body
    # Rendering playground does not execute judgments or rate-limit
    assert JudgmentRecord.objects.count() == 0
    assert RequestAuditLog.objects.count() == 0


@pytest.mark.django_db
def test_playground_not_rate_limited_and_trailing_slash_works():
    client = APIClient()
    for _ in range(7):
        resp = client.get("/playground/", REMOTE_ADDR="198.51.100.55")
        assert resp.status_code == status.HTTP_200_OK


@pytest.mark.django_db
def test_playground_rejects_non_get_methods():
    client = APIClient()
    for path in ("/playground", "/playground/"):
        resp = client.post(path, data={"sentence1": "a", "sentence2": "b"}, format="json")
        assert resp.status_code == status.HTTP_405_METHOD_NOT_ALLOWED


@pytest.mark.django_db
def test_judge_single_still_works_after_playground(monkeypatch):
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
    client = APIClient()
    get_resp = client.get("/playground")
    assert get_resp.status_code == status.HTTP_200_OK

    judge_resp = client.post(
        "/judge",
        data={
            "sentence1": "A cat is sleeping on the sofa.",
            "sentence2": "An animal is resting on a couch.",
        },
        format="json",
    )
    assert judge_resp.status_code == status.HTTP_200_OK
    payload = judge_resp.json()
    assert payload["label"] in ("ENTAIL", "NO_ENTAIL")
    assert 0.0 <= payload["score"] <= 1.0
    assert "X-Request-Id" in judge_resp
    assert "X-RateLimit-Limit" in judge_resp
    assert "X-RateLimit-Remaining" in judge_resp
