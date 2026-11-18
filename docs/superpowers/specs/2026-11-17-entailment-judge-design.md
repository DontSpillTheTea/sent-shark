# Same-Language Entailment Judge API — Design Specification

**Date:** 2026-11-17  
**Project:** Mantra Inc. — Langaku Backend Engineer Coding Assignment (Option 3)  
**Author:** Anthony Ma  

## 1. Overview
The **Same-Language Entailment Judge API** evaluates whether two sentences written in the same language convey the same meaning (`ENTAIL`) or do not (`NO_ENTAIL`), returning a confidence score in `[0.0, 1.0]`.

The architecture balances latency, accuracy, concurrency safety, and operational simplicity using a **3-Tier Hybrid NLI Pipeline**:
1. **Tier 1 — Deterministic Fast-Gate (`<1ms`):** NFKC normalization, Japanese politeness/register canonicalization (e.g., `だ`/`です`/`ます`, dropped first-person topic pronouns, colloquial contractions `〜てる`/`〜ちゃう`), negation polarity parity checks, subject-object inversion detection, and bilingual antonym contradiction veto (`好き↔嫌い`, `肉食↔草食`, `carnivore↔vegetarian`).
2. **Tier 2 — Pretrained HuggingFace NLI Cross-Encoder (`~15–40ms` batched CPU):** Default `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` (configurable via `NLI_MODEL_NAME`), evaluating **bidirectional entailment** $(s_1 \Rightarrow s_2)$ and $(s_2 \Rightarrow s_1)$ in a single vectorized tensor batch and applying a contradiction penalty:
   $$\text{score}_{\text{nli}} = \frac{p_E(s_1 \Rightarrow s_2) + p_E(s_2 \Rightarrow s_1)}{2} \times \left(1 - \max\big(p_C(s_1 \Rightarrow s_2),\, p_C(s_2 \Rightarrow s_1)\big)\right)$$
3. **Tier 3 — Optional Cloud LLM Escalation Adapter:** Protocol-based escalation for borderline scores (`0.40 <= score <= 0.65`) when `OPENAI_API_KEY` is configured, with a strict timeout and seamless fallback to Tier 2.

## 2. Requirements Coverage
- **F1 (`POST /judge` & `POST /api/v1/judge`):** Validates non-empty same-language sentence pairs and returns `{"label": "ENTAIL" | "NO_ENTAIL", "score": float}`.
- **F2 (`POST /judge/bulk` & `POST /api/v1/judge/bulk`):** Evaluates 1–100 sentence pairs in a single batched pass (accepting either a JSON array `[...]` or `{"pairs": [...]}`).
- **F3 (`GET /judge/stats` & `POST /judge/evaluate`):** Exposes aggregate statistics (total judgments, label split, tier usage, average latency) and computes accuracy/precision/recall/F1 against labeled ground-truth datasets.
- **F4 (Rate Limiting):** Sliding-window per-`client_ip` limiter (`5 req/sec`) returning `HTTP 429 Too Many Requests` with `Retry-After` when exceeded.
- **NFR 1 (Latency vs. Accuracy):** DB hash cache (`O(1)` lookup) + Tier 1 `<1ms` deterministic gate + vectorized Tier 2 batching.
- **NFR 2 (Concurrent Writes):** Deterministic `SHA-256` `pair_hash` with `UNIQUE` DB constraint, atomic `get_or_create` / `IntegrityError` recovery, and SQLite WAL mode / PostgreSQL `ON CONFLICT` safety across 30+ parallel writers.
- **NFR 3 (Security / Auditability):** `RequestAuditLog` records every incoming API request (`client_ip`, `method`, `path`, `status_code`, `payload_chars`, `latency_ms`, `rate_limited`, `created_at`).
