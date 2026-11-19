"""
Service layer orchestrating cache lookup, Tier 1/2 NLI evaluation, and atomic
deduplicated persistence (safe under 30+ parallel writes).
"""

from __future__ import annotations

import hashlib
import math
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from django.conf import settings
from django.db import connection, transaction

from assignment.models import JudgmentRecord
from assignment.services import nli_engine as nli_mod
from assignment.services.nli_engine import NLI_MAX_LENGTH, TokenLimitExceededError, nli_engine
from assignment.services.normalizer import Tier1Result, evaluate_tier1, normalize_sentence
from assignment.types import EntailmentLabel, TierName

SCORING_POLICY_VERSION = "exact_plus_nli:v2:min"
_UNSET = object()

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


@dataclass(frozen=True)
class _CacheIdentityConfig:
    model_name: str
    model_revision: str
    model_enabled: bool
    entail_index: int | None
    contradict_index: int | None
    policy_version: str
    threshold_value: float
    threshold_key: str


def _canonical_threshold(value: float) -> tuple[float, str]:
    """Validate and canonicalize ENTAIL_THRESHOLD so equivalent floats share a key."""
    val = float(value)
    if math.isnan(val) or math.isinf(val):
        raise ValueError(f"ENTAIL_THRESHOLD must be a finite float, got {value!r}.")
    if val == 0.0:
        val = 0.0
    return val, repr(val)


def _resolve_model_revision_for_cache(resolved_model: str, model_enabled: bool) -> str:
    """Resolve active model revision for cache identity without loading model weights."""
    if not model_enabled:
        return "disabled"
    if resolved_model == nli_engine.model_name and nli_engine.model_revision:
        return nli_engine.model_revision
    return nli_mod._resolve_hf_model_revision(resolved_model) or "unknown"


def _resolve_cache_config(
    *,
    model_name: str | None = None,
    policy_version: str = SCORING_POLICY_VERSION,
    entail_threshold: float | None = None,
    model_revision: str | None = None,
    model_enabled: bool | None = None,
    entail_index: int | None | object = _UNSET,
    contradict_index: int | None | object = _UNSET,
) -> _CacheIdentityConfig:
    """Resolve all prediction-affecting settings for cache keying."""
    resolved_model = model_name or getattr(
        settings,
        "NLI_MODEL_NAME",
        "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli",
    )
    resolved_enabled = (
        bool(getattr(settings, "NLI_ENABLE_MODEL", True))
        if model_enabled is None
        else bool(model_enabled)
    )
    resolved_revision = (
        model_revision
        if model_revision is not None
        else _resolve_model_revision_for_cache(resolved_model, resolved_enabled)
    )
    raw_threshold = (
        getattr(settings, "ENTAIL_THRESHOLD", 0.50)
        if entail_threshold is None
        else entail_threshold
    )
    threshold_value, threshold_key = _canonical_threshold(raw_threshold)
    resolved_entail_idx = (
        getattr(settings, "NLI_ENTAIL_INDEX", None) if entail_index is _UNSET else entail_index
    )
    resolved_contradict_idx = (
        getattr(settings, "NLI_CONTRADICT_INDEX", None)
        if contradict_index is _UNSET
        else contradict_index
    )
    return _CacheIdentityConfig(
        model_name=resolved_model,
        model_revision=resolved_revision,
        model_enabled=resolved_enabled,
        entail_index=resolved_entail_idx,  # type: ignore[arg-type]
        contradict_index=resolved_contradict_idx,  # type: ignore[arg-type]
        policy_version=policy_version,
        threshold_value=threshold_value,
        threshold_key=threshold_key,
    )


def _hash_normalized_pair(norm1: str, norm2: str, cfg: _CacheIdentityConfig) -> str:
    a, b = sorted((norm1, norm2))
    payload = "\0".join(
        [
            cfg.model_name,
            cfg.model_revision,
            "1" if cfg.model_enabled else "0",
            str(cfg.entail_index),
            str(cfg.contradict_index),
            cfg.policy_version,
            cfg.threshold_key,
            a,
            b,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_pair_hash(
    sentence1: str,
    sentence2: str,
    model_name: str | None = None,
    policy_version: str = SCORING_POLICY_VERSION,
    entail_threshold: float | None = None,
    model_revision: str | None = None,
    model_enabled: bool | None = None,
    entail_index: int | None | object = _UNSET,
    contradict_index: int | None | object = _UNSET,
) -> tuple[str, str, str]:
    """
    Compute canonical, order-independent SHA-256 pair hash scoped by every setting
    that affects predictions (model name/revision, enable flag, label index overrides,
    scoring policy version, and entailment threshold).
    """
    cfg = _resolve_cache_config(
        model_name=model_name,
        policy_version=policy_version,
        entail_threshold=entail_threshold,
        model_revision=model_revision,
        model_enabled=model_enabled,
        entail_index=entail_index,
        contradict_index=contradict_index,
    )
    norm1 = normalize_sentence(sentence1)
    norm2 = normalize_sentence(sentence2)
    return _hash_normalized_pair(norm1, norm2, cfg), norm1, norm2


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
) -> list[JudgmentOutput]:
    """
    Pure Tier 1 exact-match gate + batched Tier 2 NLI evaluation without cache or DB side effects.
    Shared by both judge_bulk (after cache/singleflight filtering) and offline evaluation.
    """
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
        nli_details = nli_engine.score_pairs_batch(nli_inputs, t1_list)

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


def validate_pairs_token_limits(pairs: list[tuple[str, str]]) -> None:
    """
    Validate that all normalized sentence pairs fit within NLI_MAX_LENGTH combined
    tokens across both directions when NLI_ENABLE_MODEL is enabled.
    When NLI_ENABLE_MODEL is False, skip tokenization and rely on character bounds.
    """
    if not pairs or not getattr(settings, "NLI_ENABLE_MODEL", True):
        return
    max_tokens = nli_engine.max_pair_tokens_batch(pairs)
    if max_tokens > NLI_MAX_LENGTH:
        if len(pairs) == 1:
            raise TokenLimitExceededError(
                f"Sentence pair exceeds maximum NLI sequence length of "
                f"{NLI_MAX_LENGTH} tokens (got {max_tokens} tokens)."
            )
        raise TokenLimitExceededError(
            f"Batch contains a sentence pair exceeding maximum NLI sequence length of "
            f"{NLI_MAX_LENGTH} tokens (max {max_tokens} tokens)."
        )


def judge_bulk(pairs: list[tuple[str, str]]) -> list[JudgmentOutput]:
    """
    Judge a batch of up to 100 sentence pairs with:
    1. Upfront 256-token bound validation across all pairs (zero partial writes)
    2. In-batch deduplication by canonical order-independent SHA-256 pair_hash
    3. Bounded in-memory LRU hot cache + single-query DB cache lookup
    4. Per-hash singleflight coalescing across concurrent cold-cache requests
    5. Tier 1 exact-match gate + single batched HuggingFace NLI inference pass
    6. Concurrency-safe atomic DB persistence
    """
    if not pairs:
        return []

    validate_pairs_token_limits(pairs)
    cache_cfg = _resolve_cache_config()
    hashed_items: list[tuple[str, str, str]] = []
    pair_by_hash: dict[str, tuple[str, str]] = {}

    for s1, s2 in pairs:
        norm1 = normalize_sentence(s1)
        norm2 = normalize_sentence(s2)
        pair_hash = _hash_normalized_pair(norm1, norm2, cache_cfg)
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
                store_hash_by_owned = {rec.pair_hash: rec.pair_hash for rec in new_records}
                if (
                    cache_cfg.model_enabled
                    and cache_cfg.model_revision == "unknown"
                    and cache_cfg.model_name == nli_engine.model_name
                    and nli_engine.model_revision
                ):
                    post_cfg = _resolve_cache_config(
                        model_name=cache_cfg.model_name,
                        policy_version=cache_cfg.policy_version,
                        entail_threshold=cache_cfg.threshold_value,
                        model_revision=nli_engine.model_revision,
                        model_enabled=cache_cfg.model_enabled,
                        entail_index=cache_cfg.entail_index,
                        contradict_index=cache_cfg.contradict_index,
                    )
                    for rec in new_records:
                        old_hash = rec.pair_hash
                        new_hash = _hash_normalized_pair(
                            normalize_sentence(rec.sentence1),
                            normalize_sentence(rec.sentence2),
                            post_cfg,
                        )
                        store_hash_by_owned[old_hash] = new_hash
                        rec.pair_hash = new_hash

                persisted_by_hash = _persist_records_safely(new_records)
                cache_entries: dict[str, JudgmentOutput] = {}
                for old_hash, store_hash in store_hash_by_owned.items():
                    rec = persisted_by_hash.get(store_hash)
                    if rec is not None and old_hash in owned_results:
                        out = JudgmentOutput(
                            label=rec.label,
                            score=round(float(rec.score), 4),
                            tier_used=rec.tier_used,
                            cached=False,
                        )
                        owned_results[old_hash] = out
                        cache_entries[store_hash] = out
                if cache_entries:
                    _cache_put_many(cache_entries)
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
