"""
Service layer orchestrating cache lookup, Tier 1/2/3 NLI evaluation, atomic
deduplicated persistence (safe under 30+ parallel writes), and accuracy stats.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass

from django.conf import settings
from django.db import IntegrityError, connection, transaction
from django.db.models import Avg, Count

from assignment.models import JudgmentRecord, RequestAuditLog
from assignment.services.nli_engine import nli_engine
from assignment.services.normalizer import evaluate_tier1, normalize_sentence

# Serialize writes when running on SQLite so 30 concurrent writer threads
# never collide on SQLite's single-writer file lock. PostgreSQL handles
# concurrent row/index locking natively via MVCC + unique constraint.
_SQLITE_WRITE_LOCK = threading.RLock()


@dataclass(frozen=True)
class JudgmentOutput:
    label: str
    score: float
    tier_used: str
    cached: bool


def compute_pair_hash(sentence1: str, sentence2: str) -> tuple[str, str, str]:
    norm1 = normalize_sentence(sentence1)
    norm2 = normalize_sentence(sentence2)
    digest = hashlib.sha256(f"{norm1}\0{norm2}".encode("utf-8")).hexdigest()
    return digest, norm1, norm2


def _prune_table(model_cls, max_rows: int) -> None:
    """Delete oldest rows by (created_at, id) when table size exceeds max_rows."""
    if max_rows < 0:
        return
    total = model_cls.objects.count()
    if total <= max_rows:
        return
    if max_rows == 0:
        model_cls.objects.all().delete()
        return

    excess = total - max_rows
    while excess > 0:
        batch_size = min(excess, 500)
        oldest_ids = list(
            model_cls.objects.order_by("created_at", "id").values_list("id", flat=True)[:batch_size]
        )
        if not oldest_ids:
            break
        model_cls.objects.filter(id__in=oldest_ids).delete()
        excess -= len(oldest_ids)


def prune_storage_if_needed() -> None:
    """Enforce bounded storage limits on JudgmentRecord and RequestAuditLog."""
    max_judgments = int(getattr(settings, "MAX_STORED_JUDGMENTS", 10000))
    max_audit_logs = int(getattr(settings, "MAX_AUDIT_LOGS", 50000))

    def _do_prune():
        _prune_table(JudgmentRecord, max_judgments)
        _prune_table(RequestAuditLog, max_audit_logs)

    if connection.vendor == "sqlite":
        with _SQLITE_WRITE_LOCK:
            _do_prune()
    else:
        _do_prune()


def _persist_records_safely(records_to_create: list[JudgmentRecord]) -> None:
    """
    Persist newly computed JudgmentRecord objects idempotently under concurrent writes.
    Duplicate pair_hashes created by racing requests are safely ignored.
    """
    if not records_to_create:
        return

    max_judgments = int(getattr(settings, "MAX_STORED_JUDGMENTS", 10000))

    def _do_write():
        for rec in records_to_create:
            try:
                with transaction.atomic():
                    JudgmentRecord.objects.get_or_create(
                        pair_hash=rec.pair_hash,
                        defaults={
                            "sentence1": rec.sentence1,
                            "sentence2": rec.sentence2,
                            "label": rec.label,
                            "score": rec.score,
                            "tier_used": rec.tier_used,
                            "expected_label": rec.expected_label,
                        },
                    )
            except IntegrityError:
                # Another concurrent transaction inserted this exact pair_hash first.
                pass
        _prune_table(JudgmentRecord, max_judgments)

    if connection.vendor == "sqlite":
        with _SQLITE_WRITE_LOCK:
            _do_write()
    else:
        _do_write()


def judge_single(sentence1: str, sentence2: str) -> JudgmentOutput:
    """Judge a single same-language sentence pair and persist deduplicated result."""
    return judge_bulk([(sentence1, sentence2)])[0]


def judge_bulk(pairs: list[tuple[str, str]]) -> list[JudgmentOutput]:
    """
    Judge a batch of up to 100 sentence pairs with:
    1. In-batch deduplication by canonical SHA-256 pair_hash
    2. Single-query DB cache lookup
    3. Tier 1 fast-gate filtering
    4. Single batched HuggingFace NLI inference pass for unresolved pairs
    5. Concurrency-safe atomic DB persistence
    """
    if not pairs:
        return []

    hashed_items: list[tuple[str, str, str, str, str]] = []
    unique_hashes: list[str] = []
    seen_hashes: set[str] = set()

    for s1, s2 in pairs:
        pair_hash, norm1, norm2 = compute_pair_hash(s1, s2)
        hashed_items.append((pair_hash, s1, s2, norm1, norm2))
        if pair_hash not in seen_hashes:
            seen_hashes.add(pair_hash)
            unique_hashes.append(pair_hash)

    existing_by_hash: dict[str, JudgmentRecord] = {
        rec.pair_hash: rec for rec in JudgmentRecord.objects.filter(pair_hash__in=unique_hashes)
    }

    computed_by_hash: dict[str, JudgmentOutput] = {}
    for pair_hash, rec in existing_by_hash.items():
        computed_by_hash[pair_hash] = JudgmentOutput(
            label=rec.label,
            score=round(float(rec.score), 4),
            tier_used=rec.tier_used,
            cached=True,
        )

    # Evaluate unique uncached pairs
    unresolved_for_tier2: list[tuple[str, str, str, str, str, object]] = []
    new_records: list[JudgmentRecord] = []

    for pair_hash, s1, s2, norm1, norm2 in hashed_items:
        if pair_hash in computed_by_hash:
            continue

        t1 = evaluate_tier1(s1, s2)
        if t1.decisive:
            out = JudgmentOutput(
                label=t1.label,
                score=round(t1.score, 4),
                tier_used=JudgmentRecord.TIER_RULE,
                cached=False,
            )
            computed_by_hash[pair_hash] = out
            new_records.append(
                JudgmentRecord(
                    pair_hash=pair_hash,
                    sentence1=s1,
                    sentence2=s2,
                    label=out.label,
                    score=out.score,
                    tier_used=out.tier_used,
                )
            )
        else:
            # Mark placeholder in computed_by_hash only after batch NLI completes;
            # prevent double-queuing identical hashes within the same bulk request
            if not any(item[0] == pair_hash for item in unresolved_for_tier2):
                unresolved_for_tier2.append((pair_hash, s1, s2, norm1, norm2, t1))

    if unresolved_for_tier2:
        nli_inputs = [(item[1], item[2]) for item in unresolved_for_tier2]
        t1_list = [item[5] for item in unresolved_for_tier2]
        nli_details = nli_engine.score_pairs_batch(nli_inputs, t1_list)

        for (pair_hash, s1, s2, norm1, norm2, _), detail in zip(unresolved_for_tier2, nli_details):
            out = JudgmentOutput(
                label=detail.label,
                score=round(detail.score, 4),
                tier_used=detail.tier_used,
                cached=False,
            )
            computed_by_hash[pair_hash] = out
            new_records.append(
                JudgmentRecord(
                    pair_hash=pair_hash,
                    sentence1=s1,
                    sentence2=s2,
                    label=out.label,
                    score=out.score,
                    tier_used=out.tier_used,
                )
            )

    _persist_records_safely(new_records)

    return [computed_by_hash[pair_hash] for pair_hash, _, _, _, _ in hashed_items]


def compute_stats() -> dict:
    """Aggregate judgment statistics, tier usage, and audit log metrics (F3)."""
    total = JudgmentRecord.objects.count()
    label_counts = {
        row["label"]: row["cnt"]
        for row in JudgmentRecord.objects.values("label").annotate(cnt=Count("id"))
    }
    tier_counts = {
        row["tier_used"]: row["cnt"]
        for row in JudgmentRecord.objects.values("tier_used").annotate(cnt=Count("id"))
    }
    avg_score = JudgmentRecord.objects.aggregate(avg=Avg("score"))["avg"]

    total_requests = RequestAuditLog.objects.count()
    rate_limited_requests = RequestAuditLog.objects.filter(rate_limited=True).count()
    avg_latency_ms = RequestAuditLog.objects.aggregate(avg=Avg("latency_ms"))["avg"]

    labeled_qs = JudgmentRecord.objects.exclude(expected_label__isnull=True).exclude(
        expected_label=""
    )
    labeled_count = labeled_qs.count()
    stored_accuracy = None
    if labeled_count > 0:
        correct = sum(
            1
            for rec in labeled_qs.only("label", "expected_label")
            if rec.label == rec.expected_label
        )
        stored_accuracy = round(correct / labeled_count, 4)

    return {
        "total_judgments": total,
        "label_distribution": {
            "ENTAIL": label_counts.get(JudgmentRecord.LABEL_ENTAIL, 0),
            "NO_ENTAIL": label_counts.get(JudgmentRecord.LABEL_NO_ENTAIL, 0),
        },
        "tier_distribution": tier_counts,
        "average_score": round(float(avg_score), 4) if avg_score is not None else None,
        "stored_ground_truth_count": labeled_count,
        "stored_accuracy": stored_accuracy,
        "audit": {
            "total_requests": total_requests,
            "rate_limited_requests": rate_limited_requests,
            "average_latency_ms": (
                round(float(avg_latency_ms), 2) if avg_latency_ms is not None else None
            ),
        },
    }


def evaluate_ground_truth(
    labeled_pairs: list[tuple[str, str, str]],
) -> dict:
    """
    Evaluate predicted entailment labels against ground-truth expected labels and
    compute accuracy, precision, recall, F1, and confusion matrix (F3 deliverable).
    """
    raw_pairs = [(s1, s2) for s1, s2, _ in labeled_pairs]
    predictions = judge_bulk(raw_pairs)

    tp = fp = tn = fn = 0
    results = []

    for (s1, s2, expected), pred in zip(labeled_pairs, predictions):
        pair_hash, _, _ = compute_pair_hash(s1, s2)
        JudgmentRecord.objects.filter(pair_hash=pair_hash).update(expected_label=expected)
        is_correct = pred.label == expected
        if expected == "ENTAIL" and pred.label == "ENTAIL":
            tp += 1
        elif expected == "NO_ENTAIL" and pred.label == "ENTAIL":
            fp += 1
        elif expected == "NO_ENTAIL" and pred.label == "NO_ENTAIL":
            tn += 1
        elif expected == "ENTAIL" and pred.label == "NO_ENTAIL":
            fn += 1

        results.append(
            {
                "sentence1": s1,
                "sentence2": s2,
                "expected_label": expected,
                "predicted_label": pred.label,
                "score": pred.score,
                "correct": is_correct,
                "tier_used": pred.tier_used,
            }
        )

    total = len(labeled_pairs)
    accuracy = round((tp + tn) / total, 4) if total else 0.0
    precision = round(tp / (tp + fp), 4) if (tp + fp) else 0.0
    recall = round(tp / (tp + fn), 4) if (tp + fn) else 0.0
    f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) else 0.0

    return {
        "total": total,
        "correct": tp + tn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "results": results,
    }
