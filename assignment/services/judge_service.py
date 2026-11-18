"""
Service layer orchestrating cache lookup, Tier 1/2 NLI evaluation, and atomic
deduplicated persistence (safe under 30+ parallel writes).
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from django.conf import settings
from django.db import connection, transaction

from assignment.models import JudgmentRecord
from assignment.services.nli_engine import _validate_scoring_mode, nli_engine
from assignment.services.normalizer import Tier1Result, evaluate_tier1, normalize_sentence
from assignment.types import EntailmentLabel, ScoringMode, TierName

SCORING_POLICY_VERSION = "exact_plus_nli:v1:mean"

# Serialize writes when running on SQLite so 30 concurrent writer threads
# never collide on SQLite's single-writer file lock. PostgreSQL handles
# concurrent row/index locking natively via MVCC + unique constraint.
_SQLITE_WRITE_LOCK = threading.RLock()

_HOT_CACHE_MAX_ENTRIES = 2048
_HOT_CACHE_LOCK = threading.Lock()
_HOT_CACHE: OrderedDict[str, JudgmentOutput] = OrderedDict()


def run_sqlite_write(fn: Callable[[], None]) -> None:
    """Serialize write operations on SQLite to avoid single-writer lock contention."""
    if connection.vendor == "sqlite":
        with _SQLITE_WRITE_LOCK:
            fn()
    else:
        fn()


@dataclass(frozen=True)
class JudgmentOutput:
    label: EntailmentLabel
    score: float
    tier_used: TierName
    cached: bool


@dataclass
class _InflightCall:
    """Coordination handle shared by concurrent callers for the same pair_hash."""

    event: threading.Event
    result: JudgmentOutput | None = None
    error: BaseException | None = None


class SingleflightGroup:
    """
    Per-key in-flight coalescer preventing duplicate inference/writes on cold-cache bursts.

    Callers claim ownership of uncached, non-in-flight keys atomically under one lock,
    compute and finalize all owned keys before waiting on peer-owned keys, guaranteeing
    deadlock freedom even for overlapping bulk batches.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._inflight: dict[str, _InflightCall] = {}

    def acquire_many(
        self,
        keys: list[str],
        cache_lookup: Callable[[str], JudgmentOutput | None] | None = None,
    ) -> tuple[
        dict[str, JudgmentOutput],
        dict[str, _InflightCall],
        dict[str, _InflightCall],
    ]:
        unique_keys = list(dict.fromkeys(keys))
        cache_hits: dict[str, JudgmentOutput] = {}
        owned_calls: dict[str, _InflightCall] = {}
        awaited_calls: dict[str, _InflightCall] = {}

        with self._lock:
            for key in unique_keys:
                if cache_lookup is not None:
                    cached = cache_lookup(key)
                    if cached is not None:
                        cache_hits[key] = cached
                        continue
                existing = self._inflight.get(key)
                if existing is not None:
                    awaited_calls[key] = existing
                else:
                    call = _InflightCall(event=threading.Event())
                    self._inflight[key] = call
                    owned_calls[key] = call

        return cache_hits, owned_calls, awaited_calls

    def complete_many(
        self,
        owned_calls: dict[str, _InflightCall],
        results: dict[str, JudgmentOutput],
        error: BaseException | None = None,
    ) -> None:
        if not owned_calls:
            return
        with self._lock:
            for key, call in owned_calls.items():
                if error is not None:
                    call.error = error
                elif key in results:
                    call.result = results[key]
                else:
                    call.error = RuntimeError(f"Missing singleflight result for key {key}")
                if self._inflight.get(key) is call:
                    self._inflight.pop(key, None)
                call.event.set()


_SINGLEFLIGHT = SingleflightGroup()
_INFLIGHT_LOCK = _SINGLEFLIGHT._lock
_INFLIGHT = _SINGLEFLIGHT._inflight


def _cache_get(pair_hash: str) -> JudgmentOutput | None:
    with _HOT_CACHE_LOCK:
        item = _HOT_CACHE.get(pair_hash)
        if item is not None:
            _HOT_CACHE.move_to_end(pair_hash)
        return item


def _cache_put_many(
    entries: dict[str, JudgmentOutput],
    max_entries: int | None = None,
) -> None:
    if not entries:
        return
    limit = _HOT_CACHE_MAX_ENTRIES if max_entries is None else max_entries
    with _HOT_CACHE_LOCK:
        if limit <= 0:
            _HOT_CACHE.clear()
            return
        for pair_hash, out in entries.items():
            cached_out = JudgmentOutput(
                label=out.label,
                score=out.score,
                tier_used=out.tier_used,
                cached=True,
            )
            if pair_hash in _HOT_CACHE:
                _HOT_CACHE[pair_hash] = cached_out
                _HOT_CACHE.move_to_end(pair_hash)
            else:
                while len(_HOT_CACHE) >= limit:
                    _HOT_CACHE.popitem(last=False)
                _HOT_CACHE[pair_hash] = cached_out


def reset_cache() -> None:
    """Clear the bounded in-memory hot LRU cache."""
    with _HOT_CACHE_LOCK:
        _HOT_CACHE.clear()


def compute_pair_hash(
    sentence1: str,
    sentence2: str,
    model_name: str | None = None,
    policy_version: str = SCORING_POLICY_VERSION,
) -> tuple[str, str, str]:
    """Compute canonical, order-independent, model- and policy-versioned SHA-256 pair hash."""
    resolved_model = model_name or getattr(
        settings,
        "NLI_MODEL_NAME",
        "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli",
    )
    norm1 = normalize_sentence(sentence1)
    norm2 = normalize_sentence(sentence2)
    a, b = sorted((norm1, norm2))
    digest = hashlib.sha256(
        f"{resolved_model}\0{policy_version}\0{a}\0{b}".encode("utf-8")
    ).hexdigest()
    return digest, norm1, norm2


def _persist_records_safely(records_to_create: list[JudgmentRecord]) -> dict[str, JudgmentRecord]:
    """
    Persist newly computed JudgmentRecord objects idempotently under concurrent writes.
    Sorts records deterministically by pair_hash before bulk_create(..., ignore_conflicts=True)
    and re-checks persisted rows so concurrent writers remain conflict- and deadlock-free.
    """
    if not records_to_create:
        return {}

    ordered_records = sorted(
        {rec.pair_hash: rec for rec in records_to_create}.values(),
        key=lambda rec: rec.pair_hash,
    )
    pair_hashes = [rec.pair_hash for rec in ordered_records]

    def _do_write() -> None:
        with transaction.atomic():
            JudgmentRecord.objects.bulk_create(ordered_records, ignore_conflicts=True)

    run_sqlite_write(_do_write)
    return {rec.pair_hash: rec for rec in JudgmentRecord.objects.filter(pair_hash__in=pair_hashes)}


def judge_single(sentence1: str, sentence2: str) -> JudgmentOutput:
    """Judge a single same-language sentence pair and persist deduplicated result."""
    return judge_bulk([(sentence1, sentence2)])[0]


def evaluate_uncached_pairs(
    pairs: list[tuple[str, str]],
    scoring_mode: ScoringMode = "mean",
) -> list[JudgmentOutput]:
    """
    Pure Tier 1 exact-match gate + batched Tier 2 NLI evaluation without cache or DB side effects.
    Shared by both judge_bulk (after cache/singleflight filtering) and offline evaluation.
    """
    _validate_scoring_mode(scoring_mode)
    if not pairs:
        return []

    outputs: list[JudgmentOutput | None] = [None] * len(pairs)
    unresolved_for_tier2: list[tuple[int, str, str, Tier1Result]] = []

    for idx, (s1, s2) in enumerate(pairs):
        t1 = evaluate_tier1(s1, s2)
        if t1.decisive:
            outputs[idx] = JudgmentOutput(
                label=t1.label,
                score=round(t1.score, 4),
                tier_used=JudgmentRecord.TIER_RULE,
                cached=False,
            )
        else:
            unresolved_for_tier2.append((idx, s1, s2, t1))

    if unresolved_for_tier2:
        nli_inputs = [(s1, s2) for _, s1, s2, _ in unresolved_for_tier2]
        t1_list = [t1 for _, _, _, t1 in unresolved_for_tier2]
        score_kwargs: dict[str, object] = {}
        if scoring_mode != "mean":
            score_kwargs["scoring_mode"] = scoring_mode
        nli_details = nli_engine.score_pairs_batch(nli_inputs, t1_list, **score_kwargs)

        for (idx, _s1, _s2, _), detail in zip(unresolved_for_tier2, nli_details, strict=True):
            outputs[idx] = JudgmentOutput(
                label=detail.label,
                score=round(detail.score, 4),
                tier_used=detail.tier_used,
                cached=False,
            )

    resolved: list[JudgmentOutput] = []
    for out in outputs:
        if out is None:
            raise RuntimeError("Missing uncached evaluation output")
        resolved.append(out)
    return resolved


def _compute_uncached_pairs(
    owned_pairs: list[tuple[str, str, str]],
) -> tuple[dict[str, JudgmentOutput], list[JudgmentRecord]]:
    """Run Tier 1 exact-match gate + batched Tier 2 NLI inference for owned uncached pairs."""
    outputs = evaluate_uncached_pairs([(s1, s2) for _, s1, s2 in owned_pairs])
    owned_results: dict[str, JudgmentOutput] = {}
    new_records: list[JudgmentRecord] = []

    for (pair_hash, s1, s2), out in zip(owned_pairs, outputs, strict=True):
        owned_results[pair_hash] = out
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
    return owned_results, new_records


def judge_bulk(pairs: list[tuple[str, str]]) -> list[JudgmentOutput]:
    """
    Judge a batch of up to 100 sentence pairs with:
    1. In-batch deduplication by canonical order-independent SHA-256 pair_hash
    2. Bounded in-memory LRU hot cache + single-query DB cache lookup
    3. Per-hash singleflight coalescing across concurrent cold-cache requests
    4. Tier 1 exact-match gate + single batched HuggingFace NLI inference pass
    5. Concurrency-safe atomic DB persistence
    """
    if not pairs:
        return []

    hashed_items: list[tuple[str, str, str]] = []
    pair_by_hash: dict[str, tuple[str, str]] = {}

    for s1, s2 in pairs:
        pair_hash, _, _ = compute_pair_hash(s1, s2)
        hashed_items.append((pair_hash, s1, s2))
        if pair_hash not in pair_by_hash:
            pair_by_hash[pair_hash] = (s1, s2)

    computed_by_hash: dict[str, JudgmentOutput] = {}
    db_lookup_hashes: list[str] = []
    for pair_hash in pair_by_hash:
        cached_out = _cache_get(pair_hash)
        if cached_out is not None:
            computed_by_hash[pair_hash] = cached_out
        else:
            db_lookup_hashes.append(pair_hash)

    missing_hashes: list[str] = []
    if db_lookup_hashes:
        existing_by_hash: dict[str, JudgmentRecord] = {
            rec.pair_hash: rec
            for rec in JudgmentRecord.objects.filter(pair_hash__in=db_lookup_hashes)
        }
        db_hits: dict[str, JudgmentOutput] = {}
        for pair_hash in db_lookup_hashes:
            rec = existing_by_hash.get(pair_hash)
            if rec is not None:
                out = JudgmentOutput(
                    label=rec.label,
                    score=round(float(rec.score), 4),
                    tier_used=rec.tier_used,
                    cached=True,
                )
                computed_by_hash[pair_hash] = out
                db_hits[pair_hash] = out
            else:
                missing_hashes.append(pair_hash)
        if db_hits:
            _cache_put_many(db_hits)

    if missing_hashes:
        rechecked_hits, owned_calls, awaited_calls = _SINGLEFLIGHT.acquire_many(
            missing_hashes,
            cache_lookup=_cache_get,
        )
        computed_by_hash.update(rechecked_hits)

        if owned_calls:
            owned_pairs = [(pair_hash, *pair_by_hash[pair_hash]) for pair_hash in owned_calls]
            owned_results: dict[str, JudgmentOutput] = {}
            owned_error: BaseException | None = None
            try:
                owned_results, new_records = _compute_uncached_pairs(owned_pairs)
                persisted_by_hash = _persist_records_safely(new_records)
                for pair_hash, rec in persisted_by_hash.items():
                    if pair_hash in owned_results:
                        owned_results[pair_hash] = JudgmentOutput(
                            label=rec.label,
                            score=round(float(rec.score), 4),
                            tier_used=rec.tier_used,
                            cached=False,
                        )
                if owned_results:
                    _cache_put_many(owned_results)
                computed_by_hash.update(owned_results)
            except BaseException as exc:
                owned_error = exc
                raise
            finally:
                _SINGLEFLIGHT.complete_many(owned_calls, owned_results, error=owned_error)

        for pair_hash, call in awaited_calls.items():
            call.event.wait()
            if call.error is not None:
                raise call.error
            if call.result is None:
                raise RuntimeError(f"Singleflight call completed without result for {pair_hash}")
            computed_by_hash[pair_hash] = JudgmentOutput(
                label=call.result.label,
                score=call.result.score,
                tier_used=call.result.tier_used,
                cached=True,
            )

    return [computed_by_hash[pair_hash] for pair_hash, _, _ in hashed_items]
