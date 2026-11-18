# `sent-shark` — Same-Language Entailment Judge API

> *Sentence-Same (鮫 / same)* — evaluates whether two sentences in the same language convey the same meaning (`ENTAIL`) or not (`NO_ENTAIL`).

MVP backend service for **Mantra Inc. — Langaku Backend Engineer Coding Assignment (Option 3: Entailment Judge API)**.

---

## 1. Quick Start & Model Caching

### Option A: Docker Compose (Recommended — Pre-Cached Model + PostgreSQL 15)

The `Dockerfile` downloads and caches `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` during image build (`HF_HOME=/app/.cache/huggingface`) and launches PostgreSQL 15 + Django/Gunicorn:

```bash
docker compose up --build
```

Verify the running service:

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

### Option B: Local Virtual Environment (`uv` or `.venv`)

```bash
# Using uv:
uv sync --all-groups
uv run manage.py migrate
uv run manage.py runserver

# Or using Python venv (pip >= 25.1 required for PEP 735 --group):
python3 -m venv .venv
.venv/bin/pip install --upgrade pip && .venv/bin/pip install -e . --group dev
.venv/bin/python manage.py migrate
.venv/bin/python manage.py runserver
```

**Model caching & runtime configuration:**
- On the first Tier 2 inference call (if not built via Docker), `transformers` downloads `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` once into `~/.cache/huggingface` (`HF_HOME`). Subsequent runs load from local disk.
- Key environment variables (defaults in `assignment/settings.py`):
  - `NLI_MODEL_NAME` (`MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli`)
  - `NLI_ENABLE_MODEL` (`true`; set `false` for offline Tier 1 heuristic mode)
  - `NLI_ENTAIL_INDEX` / `NLI_CONTRADICT_INDEX` (`None`; optional non-negative integers set together when a model's `config.id2label` cannot be inferred)
  - `ENTAIL_THRESHOLD` (`0.50`)
  - `OPENAI_API_KEY` (optional; enables Tier 3 grey-zone escalation in `[0.40, 0.65]` with a 2.0s timeout)
  - `RATE_LIMIT_REQUESTS` (`5`) and `RATE_LIMIT_WINDOW_SECONDS` (`1.0`)
  - `TRUST_PROXY_HEADERS` (`false`; enables right-to-left `X-Forwarded-For` parsing when `REMOTE_ADDR` is trusted)
  - `TRUSTED_PROXY_IPS` (`""`; comma-separated trusted proxy IPs or CIDRs)
  - `MAX_STORED_JUDGMENTS` (`10000`) and `MAX_AUDIT_LOGS` (`50000`)
  - `POSTGRES_HOST` (unset uses SQLite WAL; set to use PostgreSQL 15)

### Verification, Linting & Benchmark Commands

```bash
.venv/bin/pytest -v            # 78 tests
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python manage.py check
.venv/bin/python manage.py run_eval --dataset dev --scoring mean --json
.venv/bin/python manage.py run_eval --dataset holdout --scoring mean --json
```

---

## 2. Complete API Reference

Full OpenAPI 3.0.3 schema: [`openapi.yaml`](openapi.yaml).
All `/judge*` routes are mounted both at root (`/judge*`) and under `/api/v1/judge*`, with or without a trailing slash.

### Endpoints Overview

| Method | Path (and `/api/v1` alias) | Purpose | Rate Limited |
|---|---|---|---|
| `POST` | `/judge` | Single sentence-pair entailment judgment (`F1`) | Yes (`5 req/s`) |
| `POST` | `/judge/bulk` | Batch judgment for `1..100` pairs (`F2`) | Yes (`5 req/s`) |
| `GET` | `/judge/stats` | Aggregate judgment, tier, accuracy & audit metrics (`F3`) | Yes (`5 req/s`) |
| `POST` | `/judge/evaluate` | Evaluate `1..100` labeled pairs for accuracy & F1 (`F3`) | Yes (`5 req/s`) |
| `GET` | `/playground` | Minimal dependency-free HTML test UI (`POST /judge`, root only) | No |
| `GET` | `/health` | Container/load-balancer health check (root only) | No |

### Common Headers, Proxy IP Resolution & Error Auditing

Every `/judge*` response (`200`, `400`, `405`, `406`, `429`) includes `X-Request-Id` (trimmed printable ASCII up to 128 chars, or generated UUID4). Responses reaching rate-limit/handler execution also return:
- `X-RateLimit-Limit`: Maximum requests allowed per sliding window per `client_ip` (default `"5"`).
- `X-RateLimit-Remaining`: Remaining requests in the current `1.0s` sliding window (`"0"` when rate-limited).
- `Retry-After` (`429` only): Whole seconds to wait before retrying (`max(1, ceil(retry_after))`).

**Proxy-safe client IP extraction (`assignment/rate_limit.py`):** By default (`TRUST_PROXY_HEADERS=False`), `X-Forwarded-For` is ignored and `REMOTE_ADDR` is used directly. When `TRUST_PROXY_HEADERS=True` and `REMOTE_ADDR` matches `TRUSTED_PROXY_IPS`, `X-Forwarded-For` is parsed right-to-left skipping trusted proxy hops and failing closed to `REMOTE_ADDR` on malformed or all-trusted chains.

**Failure & 429 auditing (`RequestAuditLog`):** All `/judge*` outcomes — including `400` validation/malformed JSON (`ParseError`), `405`, `406`, `429`, and unhandled `500` errors — record a `RequestAuditLog` row (`client_ip`, `method`, `path`, `status_code`, `payload_chars`, `latency_ms`, `rate_limited`) once via an idempotency guard without storing any sentence text.

**`429 Too Many Requests` response body:**

```json
{
  "error": "Rate limit exceeded: max 5 requests per 1s per client_ip.",
  "retry_after": 0.842
}
```

### Validation Rules (`JudgePairSerializer`)

- `sentence1` and `sentence2` are required strings (`allow_blank=False`, `trim_whitespace=True`, `max_length=2000`).
- **Same-language check (`is_same_language_pair`):** Rejects pairs where one sentence uses Japanese/CJK script (`\u3040-\u309f`, `\u30a0-\u30ff`, `\u4e00-\u9fff`) and the other is strictly Latin (`[A-Za-z]` without kana/CJK). Japanese sentences containing embedded ASCII loanwords or acronyms (e.g., `"AI"`, `"VR"`, `"SDK"`) are accepted.

---

### `POST /judge` — Single Pair Judgment (`F1`)

**Request Body:**

```json
{
  "sentence1": "私は猫が大好きだ。",
  "sentence2": "猫が好きです。"
}
```

**`200 OK` Response:**

```json
{
  "label": "ENTAIL",
  "score": 0.92
}
```

- `label`: `"ENTAIL"` | `"NO_ENTAIL"`
- `score`: `float` in `[0.0, 1.0]` (rounded to 4 decimal places)

**`400 Bad Request` Response (e.g., cross-language pair):**

```json
{
  "non_field_errors": [
    "sentence1 and sentence2 must be written in the same language."
  ]
}
```

**`curl` Example:**

```bash
curl -i -X POST http://localhost:8000/judge \
  -H "Content-Type: application/json" \
  -H "X-Request-Id: demo-req-001" \
  -d '{"sentence1": "私は猫が大好きだ。", "sentence2": "猫が好きです。"}'
```

---

### `POST /judge/bulk` — Batch Judgment (`F2`)

Accepts **either** a raw JSON array of `1..100` pair objects **or** a wrapper object `{"pairs": [...]}` with `1..100` pair objects. Deduplicates identical canonical pairs within the batch and executes a single vectorized NLI pass over uncached unresolved pairs, returning judgments in the exact input order.

**Request Body (Raw Array or Wrapper):**

```json
{
  "pairs": [
    {"sentence1": "私は猫が大好きだ。", "sentence2": "猫が好きです。"},
    {"sentence1": "A cat is a carnivore.", "sentence2": "A cat is a vegetarian."}
  ]
}
```

**`200 OK` Response:**

```json
[
  {"label": "ENTAIL", "score": 0.92},
  {"label": "NO_ENTAIL", "score": 0.06}
]
```

**`400 Bad Request` Responses:**
- Raw JSON list with `0` or `>100` items:
  ```json
  {"error": "Bulk request must contain between 1 and 100 sentence pairs."}
  ```
- Raw JSON list with per-item validation error: array of DRF error objects (e.g., `[{"sentence1": ["This field may not be blank."]}, {}]`).
- Wrapper object validation failure: `{"pairs": ...}` DRF error object.

**`curl` Example:**

```bash
curl -X POST http://localhost:8000/judge/bulk \
  -H "Content-Type: application/json" \
  -d '[{"sentence1": "私は猫が大好きだ。", "sentence2": "猫が好きです。"}, {"sentence1": "猫が魚を食べた。", "sentence2": "魚が猫を食べた。"}]'
```

---

### `GET /judge/stats` — Aggregate Metrics & Audit Summary (`F3`)

Returns stored judgment counts, label/tier distributions, mean score, stored ground-truth accuracy (populated by `POST /judge/evaluate`), and `RequestAuditLog` metrics.

**`200 OK` Response:**

```json
{
  "total_judgments": 12,
  "label_distribution": {
    "ENTAIL": 7,
    "NO_ENTAIL": 5
  },
  "tier_distribution": {
    "tier1_rule": 4,
    "tier2_nli": 8
  },
  "average_score": 0.6421,
  "stored_ground_truth_count": 4,
  "stored_accuracy": 1.0,
  "audit": {
    "total_requests": 15,
    "rate_limited_requests": 1,
    "average_latency_ms": 18.42
  }
}
```

*(Note: `average_score`, `stored_accuracy`, and `audit.average_latency_ms` are `null` when no corresponding records exist yet.)*

**`curl` Example:**

```bash
curl http://localhost:8000/judge/stats
```

---

### `POST /judge/evaluate` — Ground-Truth Evaluation (`F3`)

Evaluates `1..100` labeled pairs (`expected_label` in `{"ENTAIL", "NO_ENTAIL"}`), updates `JudgmentRecord.expected_label` for each pair, and computes accuracy, precision, recall, F1, confusion matrix, and per-pair details.

**Request Body:**

```json
{
  "pairs": [
    {
      "sentence1": "私は猫が大好きだ。",
      "sentence2": "猫が好きです。",
      "expected_label": "ENTAIL"
    },
    {
      "sentence1": "猫が魚を食べた。",
      "sentence2": "魚が猫を食べた。",
      "expected_label": "NO_ENTAIL"
    }
  ]
}
```

**`200 OK` Response:**

```json
{
  "total": 2,
  "correct": 2,
  "accuracy": 1.0,
  "precision": 1.0,
  "recall": 1.0,
  "f1": 1.0,
  "confusion_matrix": {"tp": 1, "fp": 0, "tn": 1, "fn": 0},
  "results": [
    {
      "sentence1": "私は猫が大好きだ。",
      "sentence2": "猫が好きです。",
      "expected_label": "ENTAIL",
      "predicted_label": "ENTAIL",
      "score": 0.92,
      "correct": true,
      "tier_used": "tier1_rule"
    },
    {
      "sentence1": "猫が魚を食べた。",
      "sentence2": "魚が猫を食べた。",
      "expected_label": "NO_ENTAIL",
      "predicted_label": "NO_ENTAIL",
      "score": 0.06,
      "correct": true,
      "tier_used": "tier1_rule"
    }
  ]
}
```

**`curl` Example:**

```bash
curl -X POST http://localhost:8000/judge/evaluate \
  -H "Content-Type: application/json" \
  -d '{"pairs":[{"sentence1":"私は猫が大好きだ。","sentence2":"猫が好きです。","expected_label":"ENTAIL"}]}'
```

---

### `GET /health` — Health Check

**`200 OK` Response:**

```json
{"status": "ok"}
```

---

### `GET /playground` — Developer Test UI

Serves a minimal, dependency-free HTML/CSS/vanilla-JS page (`assignment/templates/assignment/playground.html`, zero external scripts/styles/CDNs) for interactive manual testing:
- Provides `sentence1` and `sentence2` inputs, four one-click EN/JA presets, and an **Evaluate** button calling `POST /judge` (with optional `csrftoken` header propagation).
- Displays `label`, `score`, HTTP status, client-measured latency (`performance.now()`), `X-Request-Id`, `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `Retry-After`, and raw JSON output.
- Mounted at `/playground` and `/playground/` via `@require_GET`, completely isolated from `_BaseJudgeView` API views, rate limiting, and audit logging.

---

## 3. Algorithm & Architecture (F1–F4 & NFRs 1–3)

### 3-Tier Hybrid NLI Pipeline

Pure embedding cosine similarity struggles with antonyms (`"A cat is a carnivore."` vs. `"A cat is a vegetarian."`) and role swaps (`"猫が魚を食べた。"` vs. `"魚が猫を食べた。"`). This service combines deterministic linguistic gates with bidirectional NLI cross-encoding:

1. **Tier 1 — Deterministic Fast-Gate (`assignment/services/normalizer.py`, `<1 ms`):**
   - Applies Unicode NFKC normalization, whitespace/punctuation trimming, Japanese politeness/copula/contraction/pro-drop canonicalization, and surface lexical synonyms.
   - Immediately resolves **exact normalized matches** (`0.99`), **Japanese register equivalents** (`0.92`), **negation parity flips** (`0.08`), and **contradiction vetoes** (`0.06`: bilingual antonym collisions, Japanese restrictive particle `しか` mismatches, SOV/SVO role inversions, and same-subject passive/active voice flips).
2. **Tier 2 — Bidirectional NLI Cross-Encoder (`assignment/services/nli_engine.py`):**
   - Unresolved pairs run through `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` in chunked `torch.inference_mode()` batches (chunk size `64` directional inputs) evaluating both **forward** $(s_1 \Rightarrow s_2)$ and **reverse** $(s_2 \Rightarrow s_1)$ directions (plus canonicalized representations when distinct):
     $$p_{E,\text{mean}} = \frac{p_E(s_1 \Rightarrow s_2) + p_E(s_2 \Rightarrow s_1)}{2}, \qquad p_{C,\text{max}} = \max\big(p_C(s_1 \Rightarrow s_2),\; p_C(s_2 \Rightarrow s_1)\big)$$
     $$\text{score}_{\text{NLI}} = p_{E,\text{mean}} \times \big(1 - p_{C,\text{max}}\big)$$
   - **Explicit `id2label` resolution:** Label indices are resolved strictly from `model.config.id2label` or explicit `NLI_ENTAIL_INDEX` + `NLI_CONTRADICT_INDEX` overrides; missing, non-integer, or ambiguous `id2label` maps raise an explicit `ValueError` instead of silently falling back to `0`/`2` or swallowing into Tier 1.
   - Non-contradiction lexical calibration fuses raw and canonical NLI probabilities when safe while explicitly guarding against boosting directional hypernyms ($\max(p_E) \ge 0.70$ and $\min(p_E) < 0.15$, e.g., `"Siamese cat"` $\not\leftrightarrow$ `"animal"`).
   - Pairs with $\text{score} \ge 0.50$ classify as `ENTAIL`; otherwise `NO_ENTAIL`.
3. **Tier 3 — Optional LLM Escalation (`OpenAIEscalationJudge`):**
   - If `OPENAI_API_KEY` is set and a Tier 2 score falls in `[0.40, 0.65]`, escalates with a `2.0s` timeout and falls back to Tier 2 on failure.

### Caching, Singleflight Concurrency, Process Model & Bounded Storage (`F4`, `NFR 1–3`)

- **Cache & `SingleflightGroup` Coalescing (`assignment/services/judge_service.py`):**
  Every pair is hashed by canonical normalized `SHA-256` (`pair_hash`). Lookups check the bounded in-memory LRU cache (`_HOT_CACHE`, max `2048` entries) and then `JudgmentRecord`. Concurrent cold-cache bursts for the same `pair_hash` coalesce via `_SINGLEFLIGHT` (`SingleflightGroup`), guaranteeing at most **1** NLI forward pass per unique cold pair.
- **Single-Worker Gunicorn Consistency (`Dockerfile`):**
  The container runs `gunicorn --workers 1 --threads 8` so `_HOT_CACHE`, `_SINGLEFLIGHT`, `SlidingWindowRateLimiter`, and the loaded PyTorch model share one process memory space with zero cross-worker rate-limit/cache drift or duplicate model RAM. Trade-off: throughput scales vertically within 8 threads (with `torch.set_num_threads` capped at `<= 4`); multi-worker or multi-replica deployments require swapping in a shared Redis rate-limiter/cache backend.
- **Atomic Deduplicated Persistence (`NFR 2`):**
  `JudgmentRecord.pair_hash` carries a `UNIQUE` database constraint. Writes run inside `transaction.atomic()` with `get_or_create` + `IntegrityError` handling (plus SQLite WAL pragmas in `assignment/apps.py` and `_SQLITE_WRITE_LOCK` on SQLite), verified under 30 parallel threads with **0 duplicate rows**.
- **Bounded Storage Compaction:**
  `prune_storage_if_needed` enforces `MAX_STORED_JUDGMENTS` (`10000`) and `MAX_AUDIT_LOGS` (`50000`) by deleting oldest rows by `(created_at, id)` and evicting pruned hashes from `_HOT_CACHE`.
- **Rate Limiting & Audit Logging (`F4`, `NFR 3`):**
  `SlidingWindowRateLimiter` (`assignment/rate_limit.py`) maintains a thread-safe `deque` per `client_ip` (LRU-bounded at `4096` keys) enforcing `5 req / 1.0s -> 429`. Every `/judge*` request (including `400`/`405`/`406`/`429`/`500` responses) writes a `RequestAuditLog` row (`client_ip`, `method`, `path`, `status_code`, `payload_chars`, `latency_ms`, `rate_limited`, `created_at`) without storing sentence text.

---

## 4. Benchmarks: Development vs. Independent Held-Out Generalization

All benchmarks run offline with Tier 3 LLM disabled (`NoOpEscalationJudge`) and without mutating persistent database tables. Runs below were captured on the final code commit (`git_sha`: `accc98e7211943da5cb74b0f8d86b35b8a798e6f`) prior to this documentation commit:

- **Model:** `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` (`model_revision`: `0a71e92a985b6e1ad1828cf67ce9c459639c1dca`)
- **Hardware:** `Linux-6.6.87.2-microsoft-standard-WSL2-aarch64-with-glibc2.39` (Linux WSL2 `aarch64` CPU, 10 cores, Python `3.12.3`, `torch 2.15.0+cpu`, `cuda_available=false`)
- **Inference Config:** `max_length=256`, `batch_size=64`, `entail_threshold=0.5`, `entail_margin=null`, `contradict_cap=0.35`, `lexical_contradict_cap=0.25`, `directional_hypernym_guard={"max_entail": 0.70, "min_entail": 0.15}`, `scoring_mode="mean"`, `apply_lexical_surface_tier2=false`, `apply_lexical_surface_hybrid=true`

```bash
.venv/bin/python manage.py run_eval --dataset dev --scoring mean --json
.venv/bin/python manage.py run_eval --dataset dev --scoring min --json
.venv/bin/python manage.py run_eval --dataset holdout --scoring mean --json
```

### 4.1 Development Benchmark (`eval_gauntlet.json`, 60 Pairs)

Stratified internal dev set ([`assignment/tests/fixtures/eval_gauntlet.json`](assignment/tests/fixtures/eval_gauntlet.json)): `60` pairs (`easy=15`, `medium=15`, `hard=15`, `edge=15`; `ja=32`, `en=28`; `ENTAIL=30`, `NO_ENTAIL=30`).

| Mode (`mean`) | Accuracy | Precision | Recall | F1 | Specificity | Brier | Total (ms) | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | CM (TP/FP/TN/FN) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `tier1` | 0.6500 | 0.7143 | 0.5000 | 0.5882 | 0.8000 | 0.245949 | 7.87 | 0.13 | 0.06 | 0.30 | 0.33 | 15/6/24/15 |
| `tier2` | 0.8333 | 0.8333 | 0.8333 | 0.8333 | 0.8333 | 0.165574 | 431.16 | 54.86 | 55.01 | 61.90 | 70.59 | 25/5/25/5 |
| `hybrid` | 0.9167 | 0.9310 | 0.9000 | 0.9153 | 0.9333 | 0.088749 | 531.37 | 39.17 | 49.10 | 65.83 | 103.63 | 27/2/28/3 |

- **Per-Difficulty (`mean`):** `easy` `1.0000` (`tier1/tier2/hybrid`), `medium` `0.8000 / 0.8000 / 0.9333`, `hard` `0.3333 / 0.8667 / 1.0000`, `edge` `0.4667 / 0.6667 / 0.7333`.
- **By Language (`hybrid`):** `ja` `0.9375 (30/32)`, `en` `0.8929 (25/28)`.
- **Misclassified Pairs (`hybrid`, 5 / 60):** `G22` (rare synonym `canine`/`dog`), `G46` & `G50` (double negation vs. universal quantifier), `G55` (3-argument `が`/`に`/`を` swap), `G59` (`Before`/`After` temporal flip).

#### Dev-Only `mean` vs. `min` Formula Selection

Before evaluating the frozen holdout set, `--scoring mean` vs. `--scoring min` were compared strictly on `--dataset dev`:

| Mode & Scoring | Accuracy | F1 | Brier Score | `hard` Stratum | CM (TP/FP/TN/FN) |
|---|---:|---:|---:|---:|---|
| `tier2` (`mean`) | **0.8333** | **0.8333** | **0.165574** | **0.8667 (13/15)** | 25/5/25/5 |
| `tier2` (`min`) | 0.8000 | 0.7931 | 0.168103 | 0.7333 (11/15) | 23/5/25/7 |
| `hybrid` (`mean`) | **0.9167** | **0.9153** | **0.088749** | **1.0000 (15/15)** | 27/2/28/3 |
| `hybrid` (`min`) | 0.8833 | 0.8772 | 0.092126 | 0.8667 (13/15) | 25/2/28/5 |

`mean` outperformed `min` on both `tier2` (`0.8333` vs. `0.8000`) and `hybrid` (`0.9167` / F1 `0.9153` vs. `0.8833` / F1 `0.8772`), so `"mean"` remained locked before touching `--dataset holdout` (zero post-holdout threshold or formula edits).

---

### 4.2 Independent Held-Out Generalization (`eval_holdout.json`, 60 External Pairs)

Frozen external test set ([`assignment/tests/fixtures/eval_holdout.json`](assignment/tests/fixtures/eval_holdout.json), [`eval_holdout_provenance.json`](assignment/tests/fixtures/eval_holdout_provenance.json)), sampled deterministically (`seed=42`, disjoint from `eval_gauntlet.json`, `30 ENTAIL` / `30 NO_ENTAIL`):
- **`jglue_jnli` (`40 ja` pairs):** Yahoo Japan Corp. & Kawahara Lab [`yahoojapan/JGLUE` `jnli-v1.3/test-v1.3.json`](https://github.com/yahoojapan/JGLUE) (`CC BY-SA 4.0`, commit `6f071c09316baae89c3d083a90985b4b1cb9968c`; Kurihara et al., LREC 2022). Bidirectional entailment pairs map to `ENTAIL`; one-way entailment, contradiction, and neutral pairs map to `NO_ENTAIL`.
- **`paws_x` (`20 en` pairs):** Google LLC [`google-research-datasets/paws-x`](https://huggingface.co/datasets/google-research-datasets/paws-x) `en/test` (HF SHA `4cd8187c404b` / `4cd8187c404bda33cb1f62b49b001115862acf37`, permissive free use with Google LLC attribution; Yang et al., EMNLP 2019). `10` paraphrases (`ENTAIL`) + `10` adversarial word-order swaps (`NO_ENTAIL`).

| Mode (`holdout`, `mean`) | Accuracy | Precision | Recall | F1 | Specificity | Brier | Total (ms) | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | CM (TP/FP/TN/FN) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `tier1` | 0.5667 | 0.5625 | 0.6000 | 0.5806 | 0.5333 | 0.279338 | 14.58 | 0.24 | 0.06 | 0.71 | 0.95 | 18/14/16/12 |
| `tier2` | 0.6167 | 0.6061 | 0.6667 | 0.6349 | 0.5667 | 0.283832 | 966.27 | 57.72 | 56.76 | 67.98 | 74.02 | 20/13/17/10 |
| `hybrid` | 0.6000 | 0.5882 | 0.6667 | 0.6250 | 0.5333 | 0.281570 | 1563.40 | 65.66 | 65.13 | 86.86 | 92.21 | 20/14/16/10 |

#### Breakdowns (`by_language`, `by_source`, `by_stratum`)

| Dimension / Bucket (`n`) | `tier1` | `tier2` | `hybrid` |
|---|---:|---:|---:|
| `ja` / `jglue_jnli` (`40`) | 0.6000 (24/40) | 0.7000 (28/40) | 0.6750 (27/40) |
| `en` / `paws_x` (`20`) | 0.5000 (10/20) | 0.4500 (9/20) | 0.4500 (9/20) |
| `bidir_entail` (`20`) | 0.4000 (8/20) | 0.6000 (12/20) | 0.6000 (12/20) |
| `oneway_entail` (`7`) | 0.8571 (6/7) | 0.7143 (5/7) | 0.7143 (5/7) |
| `contradict` (`7`) | 0.5714 (4/7) | 0.7143 (5/7) | 0.5714 (4/7) |
| `neutral` (`6`) | 1.0000 (6/6) | 1.0000 (6/6) | 1.0000 (6/6) |
| `paraphrase` (`10`) | 1.0000 (10/10) | 0.8000 (8/10) | 0.8000 (8/10) |
| `adversarial_swap` (`10`) | 0.0000 (0/10) | 0.1000 (1/10) | 0.1000 (1/10) |

- **Key Diagnostic Finding:** On external `paws_x` `adversarial_swap` pairs (e.g., identical bag of words with swapped syntactic roles), lightweight multilingual `MiniLMv2-L6` scores only `0.1000` (`1/10`), pulling `paws_x` to `0.4500` and overall holdout `hybrid` accuracy to `0.6000` (vs. `0.6750` on `jglue_jnli` alone).
- **`misclassified_ids`:**
  - `tier1` (26): `H03, H04, H05, H07, H09, H10, H11, H12, H14, H15, H16, H17, H24, H28, H31, H33, H51, H52, H53, H54, H55, H56, H57, H58, H59, H60`
  - `tier2` (23): `H01, H02, H03, H05, H07, H09, H10, H12, H23, H24, H28, H29, H41, H42, H51, H52, H54, H55, H56, H57, H58, H59, H60`
  - `hybrid` (24): `H01, H02, H03, H05, H07, H09, H10, H12, H23, H24, H28, H29, H33, H41, H42, H51, H52, H54, H55, H56, H57, H58, H59, H60`

---

## 5. Future Improvements

1. **Japanese-Pretrained NLI Backbone (`cl-tohoku/bert-base-japanese-v3`) Fine-Tuned on Japanese NLI + Manga Dialogue Hard Negatives + Adversarial Swaps:**
   Distilled multilingual XNLI models (`MiniLMv2-L6`) suffer from shallow Japanese subword coverage (`0.6750` on JGLUE JNLI) and bag-of-words sensitivity (`0.1000` on PAWS-X swaps). Fine-tuning a native Japanese-pretrained encoder (`cl-tohoku/bert-base-japanese-v3`) jointly on JGLUE JNLI / JSICK, Langaku manga *yakuwarigo* (役割語) & double-negation pairs, and synthetic word-order swap hard negatives directly targets both the holdout generalization gaps and colloquial manga register mismatches.
2. **Contextual Panel & Speaker Windowing + Dependency Role Guards:**
   Prepending optional `speaker_id` and `previous_bubble` context resolves Japanese zero-pronouns, while lightweight dependency parsing (SudachiPy / GiNZA) catches multi-argument swaps (`が`/`に`/`を`) and temporal flips (`Before`/`After`).
3. **Production Horizontal Scaling (Redis Rate Limiter + Managed PostgreSQL):**
   Swap `SlidingWindowRateLimiter` and `_HOT_CACHE` to a shared Redis backend so multi-worker/multi-replica deployments enforce global rate limits and share hot cache hits alongside managed PostgreSQL 15.

---

## 6. Verified Runtime & Known Environment Limits

- **Verified locally in `.venv`:** All `78` automated tests (`.venv/bin/pytest -v`), `.venv/bin/ruff check .`, `.venv/bin/ruff format --check .`, `.venv/bin/python manage.py check`, real HuggingFace CPU inference (`MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli`), 100-pair bulk evaluation, and 30-thread concurrent SQLite WAL writes (0 duplicate rows) passed.
- **Docker / PostgreSQL 15 environment limit:** Docker Compose + live PostgreSQL 15 integration could **not** be executed in this WSL host because Docker Engine / WSL integration was unavailable; all automated tests and benchmarks executed against SQLite WAL (`POSTGRES_HOST` unset).
