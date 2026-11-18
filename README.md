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

# Or using Python venv:
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python manage.py migrate
.venv/bin/python manage.py runserver
```

**Model caching & runtime configuration:**
- On the first Tier 2 inference call (if not built via Docker), `transformers` downloads `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` once into `~/.cache/huggingface` (`HF_HOME`). Subsequent runs load from local disk.
- Key environment variables (defaults in `assignment/settings.py`):
  - `NLI_MODEL_NAME` (`MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli`)
  - `NLI_ENABLE_MODEL` (`true`; set `false` for offline Tier 1 heuristic mode)
  - `ENTAIL_THRESHOLD` (`0.50`)
  - `OPENAI_API_KEY` (optional; enables Tier 3 grey-zone escalation in `[0.40, 0.65]` with a 2.0s timeout)
  - `RATE_LIMIT_REQUESTS` (`5`) and `RATE_LIMIT_WINDOW_SECONDS` (`1.0`)
  - `MAX_STORED_JUDGMENTS` (`10000`) and `MAX_AUDIT_LOGS` (`50000`)
  - `POSTGRES_HOST` (unset uses SQLite WAL; set to use PostgreSQL 15)

### Verification, Linting & Benchmark Commands

```bash
.venv/bin/pytest -v            # 60 tests
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python manage.py run_eval
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
| `GET` | `/health` | Container/load-balancer health check | No |

### Common Headers & Rate-Limit Behavior

Every response from `/judge`, `/judge/bulk`, `/judge/stats`, and `/judge/evaluate` (`200`, `400`, and `429`) includes:
- `X-Request-Id`: Propagates a valid incoming `X-Request-Id` header (trimmed, max 128 chars, single-line) or generates a UUID4 identifier.
- `X-RateLimit-Limit`: Maximum requests allowed per sliding window per `client_ip` (default `"5"`).
- `X-RateLimit-Remaining`: Remaining requests in the current `1.0s` sliding window (`"0"` when rate-limited).
- `Retry-After` (`429` only): Whole seconds to wait before retrying (`max(1, ceil(retry_after))`).

Client IP resolution checks the left-most entry of `X-Forwarded-For`, falling back to `REMOTE_ADDR`.

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
   - Non-contradiction lexical calibration fuses raw and canonical NLI probabilities when safe while explicitly guarding against boosting directional hypernyms ($\max(p_E) \ge 0.70$ and $\min(p_E) < 0.15$, e.g., `"Siamese cat"` $\not\leftrightarrow$ `"animal"`).
   - Pairs with $\text{score} \ge 0.50$ classify as `ENTAIL`; otherwise `NO_ENTAIL`.
3. **Tier 3 — Optional LLM Escalation (`OpenAIEscalationJudge`):**
   - If `OPENAI_API_KEY` is set and a Tier 2 score falls in `[0.40, 0.65]`, escalates with a `2.0s` timeout and falls back to Tier 2 on failure.

### Caching, Singleflight Concurrency & Bounded Storage (`F4`, `NFR 1–3`)

- **Cache & `SingleflightGroup` Coalescing (`assignment/services/judge_service.py`):**
  Every pair is hashed by canonical normalized `SHA-256` (`pair_hash`). Lookups check the bounded in-memory LRU cache (`_HOT_CACHE`, max `2048` entries) and then `JudgmentRecord`. Concurrent cold-cache bursts for the same `pair_hash` coalesce via `_SINGLEFLIGHT` (`SingleflightGroup`), guaranteeing at most **1** NLI forward pass per unique cold pair.
- **Atomic Deduplicated Persistence (`NFR 2`):**
  `JudgmentRecord.pair_hash` carries a `UNIQUE` database constraint. Writes run inside `transaction.atomic()` with `get_or_create` + `IntegrityError` handling (plus SQLite WAL pragmas in `assignment/apps.py` and `_SQLITE_WRITE_LOCK` on SQLite), verified under 30 parallel threads with **0 duplicate rows**.
- **Bounded Storage Compaction:**
  `prune_storage_if_needed` enforces `MAX_STORED_JUDGMENTS` (`10000`) and `MAX_AUDIT_LOGS` (`50000`) by deleting oldest rows by `(created_at, id)` and evicting pruned hashes from `_HOT_CACHE`.
- **Rate Limiting & Audit Logging (`F4`, `NFR 3`):**
  `SlidingWindowRateLimiter` (`assignment/rate_limit.py`) maintains a thread-safe `deque` per `client_ip` (LRU-bounded at `4096` keys) enforcing `5 req / 1.0s -> 429`. Every `/judge*` request writes a `RequestAuditLog` row (`client_ip`, `method`, `path`, `status_code`, `payload_chars`, `latency_ms`, `rate_limited`, `created_at`).

---

## 4. Reproducible Evaluation Report (60-Pair Stratified Gauntlet)

All numbers below come directly from running the committed evaluation command against [`assignment/tests/fixtures/eval_gauntlet.json`](assignment/tests/fixtures/eval_gauntlet.json) (without mutating persistent DB tables and with Tier 3 LLM disabled for determinism):

```bash
.venv/bin/python manage.py run_eval
# Machine-readable JSON output:
.venv/bin/python manage.py run_eval --json
```

- **Dataset:** `60` pairs (`easy=15`, `medium=15`, `hard=15`, `edge=15`; `ja=32`, `en=28`; `ENTAIL=30`, `NO_ENTAIL=30`)
- **Model:** `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` (`threshold=0.5`)

### Mode Comparison

| Mode | Accuracy | Precision | Recall | F1 | Specificity | Brier Score | Batch Total (ms) | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | CM (TP/FP/TN/FN) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `tier1` | 0.6500 | 0.7143 | 0.5000 | 0.5882 | 0.8000 | 0.245949 | 8.99 | 0.15 | 0.09 | 0.34 | 0.38 | 15/6/24/15 |
| `tier2` | 0.8333 | 0.8333 | 0.8333 | 0.8333 | 0.8333 | 0.165574 | 417.03 | 52.82 | 53.12 | 60.45 | 66.45 | 25/5/25/5 |
| `hybrid` | 0.9167 | 0.9310 | 0.9000 | 0.9153 | 0.9333 | 0.088749 | 538.65 | 41.01 | 53.61 | 70.03 | 75.28 | 27/2/28/3 |

### Per-Difficulty Accuracy

| Difficulty | `tier1` | `tier2` | `hybrid` |
|---|---:|---:|---:|
| `easy` | 1.0000 (15/15) | 1.0000 (15/15) | 1.0000 (15/15) |
| `medium` | 0.8000 (12/15) | 0.8000 (12/15) | 0.9333 (14/15) |
| `hard` | 0.3333 (5/15) | 0.8667 (13/15) | 1.0000 (15/15) |
| `edge` | 0.4667 (7/15) | 0.6667 (10/15) | 0.7333 (11/15) |

### Misclassified Pairs Analysis (`hybrid`: 5 / 60)

- **`hybrid` (`5` errors, `91.67%` accuracy, Brier `0.088749` vs. `0.165574` for raw `tier2`):** `100%` (`30/30`) across `easy` and `hard`, `93.33%` (`14/15`) on `medium`, and `73.33%` (`11/15`) on `edge`:
  - `G22` (`en`, `medium`, `lexical_paraphrase`, `ENTAIL`): `"A canine was sleeping near the front door."` vs. `"A dog was sleeping by the front door."` — rare-word/taxonomic synonym (`canine` vs. `dog`) missed by MiniLMv2-L6 once taxonomic regex rewrites were removed to preserve strict directional-hypernym fidelity.
  - `G46` (`ja`, `double_negation`, `ENTAIL`): `"この規則を知らない人はいない。"` vs. `"全員がこの規則を知っている。"` — double negation (`〜ない人はいない`) vs. universal quantifier (`全員`).
  - `G50` (`en`, `double_negation`, `ENTAIL`): `"No student failed to submit the report."` vs. `"Every student submitted the report."` — double negative (`No ... failed to`) vs. universal (`Every ...`).
  - `G55` (`ja`, `role_inversion`, `NO_ENTAIL`): `"太郎が花子に秘密を打ち明けた。"` vs. `"花子が太郎に秘密を打ち明けた。"` — 3-argument ditransitive (`が`/`に`/`を`) swap where Tier 1's 2-argument `が`/`を` regex does not trigger and MiniLMv2-L6 is insensitive to argument order.
  - `G59` (`en`, `temporal_shift`, `NO_ENTAIL`): `"Before she moved to Tokyo, she worked in Osaka."` vs. `"After she moved to Tokyo, she worked in Osaka."` — temporal preposition flip (`Before` vs. `After`).

---

## 5. Three Ideas for Future Accuracy Improvements

1. **Fine-Tune on Manga Dialogue & *Yakuwarigo* (役割語) Pairs:**
   General NLI corpora (MNLI/XNLI/JNLI) skew toward formal news prose and underperform on manga speech registers (`〜だぜ`, `〜なのじゃ`), contractions, and double-negation idioms (`〜ない人はいない`). Fine-tuning a Japanese Cross-Encoder (`llm-book/bert-base-japanese-v3-jnli` or `mDeBERTa-v3`) on synthetic *yakuwarigo*-to-*hyojungo* and double-negation paraphrase pairs mined from Langaku's bilingual manga dataset would close gaps like `G46`/`G50`.
2. **Contextual Panel & Speaker Windowing for Pro-Drop Resolution:**
   Manga speech bubbles frequently omit subjects (`"行くぞ！"` vs. `"一緒に出発しましょう。"`). Accepting optional `speaker_id` and `previous_bubble` context fields to prepend to the NLI premise (`"[Prev: 敵が来た] [Speaker: Luffy] 俺が守る！"`) resolves ambiguous zero-pronouns without guessing hidden agents.
3. **Morphological Dependency Role Checking + Calibrated Temperature Scaling:**
   Replacing regex-based `が`/`を` matching with lightweight morphological/dependency parsing (e.g., SudachiPy / GiNZA) catches 3-argument ditransitive swaps (`が`/`に`/`を` in `G55`) and temporal-preposition flips (`Before`/`After` in `G59`), while fitting a temperature scalar $T$ via `POST /judge/evaluate` calibrates confidence scores to true empirical probabilities.
