# `sent-shark` — Same-Language Entailment Judge API

> Evaluates whether two sentences in the same language (`ja–ja` or `en–en`) convey the same meaning (`ENTAIL`) or not (`NO_ENTAIL`) — Mantra Inc. Langaku Backend Assignment (Option 3).

---

## 1. Quick Start, Configuration & Verification

### Option A: Docker Compose (Recommended — Pre-Cached Model + PostgreSQL 15)

The [`Dockerfile`](Dockerfile) (with [`.dockerignore`](.dockerignore) excluding `.git/`, `.venv/`, `.local_docs/`, `.superpowers/`, caches, local SQLite DBs, and `.env*` while preserving example templates) pre-caches `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` (`HF_HOME=/app/.cache/huggingface`) and runs migrations before serving Gunicorn (`--workers 1 --threads 8 --timeout 60`) backed by `postgres:15-alpine`:

```bash
docker compose up --build -d
curl http://localhost:8000/health   # {"status":"ok"}
docker compose down -v
```

### Option B: Local Virtual Environment (`uv` or `.venv`)

```bash
# Using uv:
uv sync --all-groups
uv run python manage.py migrate
uv run python manage.py runserver

# Or using Python venv (pip >= 25.1 for PEP 735 --group):
python3 -m venv .venv
.venv/bin/pip install --upgrade pip && .venv/bin/pip install -e . --group dev
.venv/bin/python manage.py migrate
.venv/bin/python manage.py runserver
```

**Model Download & Offline Fallback:** With `NLI_ENABLE_MODEL=true` (default), `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` (~430 MiB snapshot) downloads once from Hugging Face Hub into `HF_HOME` (`~/.cache/huggingface`). To pre-download offline weights:

```bash
.venv/bin/python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; m='MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli'; AutoTokenizer.from_pretrained(m); AutoModelForSequenceClassification.from_pretrained(m)"
```

Set `NLI_ENABLE_MODEL=false` to skip Hugging Face model/tokenizer loading entirely and use exact-match-only character-bound fallback (`0.99` / `0.0`).

### Environment Variables (`assignment/settings.py`)

| Variable | Default | Purpose |
|---|---|---|
| `NLI_MODEL_NAME` | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` | Hugging Face NLI cross-encoder model ID |
| `NLI_ENABLE_MODEL` | `true` | Set `false` for exact-match-only offline character-bound fallback (`0.99` / `0.0`) |
| `NLI_ENTAIL_INDEX` / `NLI_CONTRADICT_INDEX` | unset | Non-negative ints (set together) when `config.id2label` uses generic labels |
| `ENTAIL_THRESHOLD` | `0.50` | Classification cutoff (unrounded `raw_score >= threshold` $\to$ `ENTAIL`) |
| `DATA_UPLOAD_MAX_MEMORY_SIZE` | `2621440` | Max JSON request body in bytes (`2.5 MiB`; exceeded $\to$ HTTP `413` JSON) |
| `RATE_LIMIT_REQUESTS` / `RATE_LIMIT_WINDOW_SECONDS` | `5` / `1.0` | Per-`client_ip` sliding-window rate limit |
| `TRUST_PROXY_HEADERS` / `TRUSTED_PROXY_IPS` | `false` / `""` | Enable right-to-left `X-Forwarded-For` parsing for trusted proxy IPs/CIDRs |
| `POSTGRES_HOST` / `POSTGRES_DB` / `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_PORT` | unset (`langaku_judge`, `postgres`, `postgres`, `5432`) | Unset uses SQLite WAL (`db.sqlite3`); set `POSTGRES_HOST` for PostgreSQL 15 (`CONN_MAX_AGE=60`) |
| `DJANGO_SECRET_KEY` / `DJANGO_DEBUG` / `DJANGO_ALLOWED_HOSTS` | dev key / `true` / `*` | Standard Django runtime configuration |

### Verification & Evaluation Commands

```bash
.venv/bin/pytest -m "not integration" -v        # 88 fast unit/API/concurrency/security tests
.venv/bin/pytest -v                             # All 114 tests (incl. 26 real-model integration tests)
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/python manage.py check && .venv/bin/python manage.py makemigrations --check --dry-run
.venv/bin/python manage.py run_eval --dataset dev --diagnostics --json
.venv/bin/python manage.py run_eval --dataset dev_diagnostic --diagnostics --json
```

---

## 2. Lean API Reference

Full OpenAPI 3.0.3 schema: [`openapi.yaml`](openapi.yaml). Template compatibility routes (`/admin/`, `/init_data/`, `/api/v1/user/me/`) remain intact.

| Method | Path | Purpose | Rate Limited |
|---|---|---|---|
| `POST` | `/judge` | Single same-language pair judgment (`F1`) | Yes (`5 req/s`) |
| `POST` | `/judge/bulk` | Batch judgment for `1..100` pairs (`F2`) | Yes (`5 req/s`) |
| `GET` | `/playground` | Dependency-free HTML UI calling `POST /judge` (4 presets, headers, latency) | No |
| `GET` | `/health` | Container health check (`{"status": "ok"}`) | No |

**Validation, Headers, Proxy Safety & Auditing:**
- **Strict inputs & atomicity:** Sentences must be strict JSON strings (int/float/bool/list/dict rejected with `400`), contain at least one substantive letter/number/symbol (`L`/`N`/`S`) after NFKC normalization (preserving internal Unicode formatting chars like emoji ZWJ `\u200d`), fit within `2000` chars per sentence, belong to the same language family (`ja`/`en`, allowing embedded ASCII loanwords in Japanese), and fit within `NLI_MAX_LENGTH=256` combined tokens per direction (`truncation=False`; excess returns `400`). `/judge/bulk` (`1..100` pairs) validates all pairs atomically before any cache or DB writes (`0` partial writes). Request bodies over `2.5 MiB` return JSON `413`.
- **Headers & rate limiting:** `/judge*` responses emit `X-Request-Id` (sanitized printable ASCII `<= 128` chars or UUID4), `X-RateLimit-Limit`, `X-RateLimit-Remaining`, and `Retry-After` on `429`. Both `400` and `413` responses consume rate-limit quota; `405`/`406` and `/health`, `/playground` do not. `SlidingWindowRateLimiter` tracks up to `4096` client IPs via bounded LRU eviction.
- **Proxy IP safety (`assignment/rate_limit.py`):** Ignores `X-Forwarded-For` unless `TRUST_PROXY_HEADERS=true` and `REMOTE_ADDR` matches `TRUSTED_PROXY_IPS`, walking right-to-left and failing closed on malformed chains.
- **Audit logging (`RequestAuditLog`):** Records every `/judge*` outcome (`200`, `400`, `405`, `406`, `413`, `429`, `500`) once (`client_ip`, `method`, `path`, `status_code`, `payload_chars`, `latency_ms`, `rate_limited`) without storing sentence text.

```bash
# Single judgment (F1) — illustrative v2:min score for this pair (JSON serializes 0.9210 as 0.921)
curl -s -X POST http://localhost:8000/judge -H "Content-Type: application/json" \
  -d '{"sentence1":"私は猫が大好きだ。","sentence2":"猫が好きです。"}'
# -> {"label":"ENTAIL","score":0.921}

# Bulk judgment (F2 — raw JSON list or {"pairs":[...]}, 1..100 pairs; illustrative v2:min scores 0.9210 / 0.0560)
curl -s -X POST http://localhost:8000/judge/bulk -H "Content-Type: application/json" \
  -d '{"pairs":[{"sentence1":"私は猫が大好きだ。","sentence2":"猫が好きです。"},{"sentence1":"A cat is a carnivore.","sentence2":"A cat is a vegetarian."}]}'
# -> [{"label":"ENTAIL","score":0.921},{"label":"NO_ENTAIL","score":0.056}]
```

**Error Responses (`400` Validation/Token Limit, `413` Payload Too Large, `429` Rate Limit):**

```json
// 400 Bad Request (cross-language; token limit returns {"error": "Sentence pair exceeds maximum NLI sequence length of 256 tokens (got 257 tokens)."}; bulk raw list bounds return {"error": "Bulk request must contain between 1 and 100 sentence pairs."}; malformed JSON returns {"detail": "JSON parse error - ..."})
{"non_field_errors": ["sentence1 and sentence2 must be written in the same language."]}

// 413 Payload Too Large (> 2.5 MiB)
{"error": "Request body exceeds maximum allowed size (2.5 MiB)."}

// 429 Too Many Requests
{"error": "Rate limit exceeded: max 5 requests per 1s per client_ip.", "retry_after": 0.842}
```

---

## 3. Architecture & Design Rationale (`exact_plus_nli:v2:min`)

1. **Tier 1 Exact Match (`assignment/services/normalizer.py`):** Applies Unicode `NFKC` normalization, whitespace collapse, and edge punctuation trimming; identical normalized pairs (case-insensitive for English) short-circuit (`ENTAIL`, `0.99`). When `NLI_ENABLE_MODEL=false`, non-identical pairs return `NO_ENTAIL` (`0.0`) using character-bound validation without loading Hugging Face weights or tokenizers.
2. **Tier 2 Bidirectional NLI (`assignment/services/nli_engine.py`):** Non-identical pairs run in deduplicated chunked batches (`batch_size=64`, `NLI_MAX_LENGTH=256`, `truncation=False`) through `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` (`mit`, `6L x 384`, 107M params) with fail-closed `id2label` resolution:
   $$\text{raw\_score} = \operatorname{clamp}\big(\min(p_E(s_1 \Rightarrow s_2),\; p_E(s_2 \Rightarrow s_1)),\; 0.0,\; 1.0\big)$$
   `label = "ENTAIL"` iff unrounded `raw_score >= ENTAIL_THRESHOLD` (`0.50` default), then `score = round(raw_score, 4)`. Using pure minimum guarantees symmetric equivalence requires support in *both* directions and eliminates one-way entailment false positives near `0.50`.
3. **Cache Symmetry, `_HOT_CACHE` & `SingleflightGroup` (`assignment/services/judge_service.py`):**
   - **Versioned `pair_hash`:** `SHA-256` over `model_name`, resolved Hugging Face `model_revision`, `NLI_ENABLE_MODEL`, `NLI_ENTAIL_INDEX`, `NLI_CONTRADICT_INDEX`, `SCORING_POLICY_VERSION` (`"exact_plus_nli:v2:min"`), canonical `ENTAIL_THRESHOLD`, and canonically sorted `(norm1, norm2)`, backed by `JudgmentRecord.pair_hash` (`unique=True`) and sorted `bulk_create(..., ignore_conflicts=True)` for deadlock-free zero-duplicate concurrent writes.
   - **`_HOT_CACHE` (`2048`-entry LRU) & `SingleflightGroup`:** Measured `25.97x` single-pair p50 (`0.0067 ms` vs `0.1740 ms`) and `2.63x` bulk-100 p50 lookup speedups over DB-only, and `30 -> 1` NLI batch calls (`7.30x` wall-clock / `7.78x` p95 speedup) across 30-thread cold bursts.
   - **Controlled Negative Findings (Archived in `.local_docs/audits/`):** Hand-crafted Tier 1 synonym/regex rules overfit synthetic `dev` patterns while fixing `0/60` independent `dev_diagnostic` pairs (`-1.67 pp`, `+60.8%` batch CPU latency) and were removed. Japanese-only 12-layer BERTs (`llm-book` `apache-2.0`, `akiFQC` `cc-by-sa-4.0`) collapsed on non-caption `dev ja` (`0.4375` and `0.5938` vs `MiniLMv2` `0.8750`), broke English (`0/10` and `5/10` PAWS-X paraphrases), ran `~9x` slower on CPU batches, and added `+254–262 MiB` peak RSS + `fugashi`/`unidic-lite`.

---

## 4. `exact_plus_nli:v2:min` Development Evaluation (`0ab315f`)

Offline runs on commit `0ab315f` with `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` (`rev=0a71e92a985b6e1ad1828cf67ce9c459639c1dca`), `ENTAIL_THRESHOLD=0.50`, CPU `torch 2.15.0+cpu` (`0` exact-identical pairs in either set, so `hybrid` matches `tier2` predictions).

| Development Dataset (`n`) | `hybrid` Acc [95% Wilson CI] | F1 | Specificity | Brier | CM (`TP/FP/TN/FN`) | Mean / p95 (ms) | `ja` Primary Acc [95% CI] | `en` Secondary Acc [95% CI] |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| [`dev`](assignment/tests/fixtures/eval_gauntlet.json) (`60`: `32 ja`/`28 en`) | **`0.8167`** `[0.7008, 0.8944]` | `0.8136` | `0.8333` | `0.1583` | `24/5/25/6` | `45.84` / `51.75` | **`0.8750`** (`28/32`) `[0.7193, 0.9503]` | `0.7500` (`21/28`) `[0.5664, 0.8732]` |
| [`dev_diagnostic`](assignment/tests/fixtures/eval_holdout.json) (`60`: `40 ja`/`20 en`) | **`0.6333`** `[0.5068, 0.7438]` | `0.6452` | `0.6000` | `0.3063` | `20/12/18/10` | `51.03` / `60.12` | **`0.7500`** (`30/40`) `[0.5981, 0.8581]` | `0.4000` (`8/20`) `[0.2188, 0.6134]` |

- **Holdout Independence Note:** `assignment/tests/fixtures/eval_final_holdout.json` (`178` pairs) was consumed once under archival `567f8a0` (`exact_plus_nli:v1:mean`) and was intentionally **not** re-evaluated or claimed as an untouched blind holdout after switching to `exact_plus_nli:v2:min`.

---

## 5. Live Docker Compose + PostgreSQL 15 Verification

Verified end-to-end on `feat/mvp-hardening` (`exact_plus_nli:v2:min`) against `postgres:15-alpine` + Gunicorn (`--workers 1 --threads 8 --timeout 60`):
- **`.dockerignore` Build & Image Hygiene:** Initial filtered build context transferred **`267.13kB`** (down from `1.43GB`; `2.98kB` cached sync), and container `/app` (`431M` incl. pre-cached `MiniLMv2`) excluded `.git/`, `.venv/`, `.local_docs/`, `.superpowers/`, and `db.sqlite3*`.
- **`pytest -v` (`POSTGRES_HOST=127.0.0.1`):** **`114 passed in 61.39s`** (`88` fast unit/API/concurrency/security + `26` real-model integration tests).
- **Live Gunicorn + PostgreSQL 15 E2E:** `migrate --check` and `manage.py check` exited `0`; `/health` (`200`), `/playground` (`200`), `/judge` (`0.921`, `ENTAIL`), and `/judge/bulk` (`0.921`, `0.056`) passed; `400` checks (int, `"。。。"`, `"\u200b"`, bulk `306`-token rejection) produced **`0`** partial `JudgmentRecord` writes (`2 -> 2`); `2.7 MB` payload returned JSON **`413`**; 15-thread single-IP burst yielded **`{200: 5, 429: 10}`** with **`7`** `JudgmentRecord` rows and **`22`** sentence-free `RequestAuditLog` rows (`7 x 200`, `4 x 400`, `1 x 413`, `10 x 429`), followed by clean `docker compose down -v`.

---

## 6. Known Limitations & Future Accuracy Improvements

**Known Limitations:**
- **Adversarial word-order/particle swaps:** Distilled 6-layer XNLI struggles on high lexical-overlap swaps (`G39`, `G42`, `G55`, `G58` on `dev`; `1/10` `adversarial_swap` on `dev_diagnostic`).
- **Pure `min` trade-offs:** Rejecting pairs when either direction dips below `0.50` drops asymmetric paraphrases (`G38`, `H15`, `H47`), and removing `(1 - max_contradiction)` improves `H02`/`H10` while letting moderate-contradiction `H33` (`min=0.6044`, `max_contra=0.2408`) pass.
- **Process-local state & corpus noise:** `_HOT_CACHE`, `_SINGLEFLIGHT`, and `SlidingWindowRateLimiter` are single-process (multi-worker scaling requires Redis), and upstream `JNLI`/`PAWS-X` fixtures contain scene co-descriptions (`H03, H05, H07, H09, H12`) and label noise (`H48`).

**Three Future Accuracy Improvements:**
1. **Japanese-Specific Linguistic Fine-Tuning:** Fine-tune a compact bilingual/Japanese cross-encoder on compositional particle-scrambling vs. case-inversion pairs (**Yanaka & Mineshima, 2022**, *TACL* 10:1266–1284, JSICK), adversarial paraphrase swaps (**Yang et al., 2019**, *EMNLP*, PAWS-X), and native Japanese NLI (**Kurihara et al., 2022**, *LREC*, pp. 2957–2966, JGLUE) to solve word-order/particle-swap and lexical misses.
2. **Context-Aware Semantic Judgment:** Incorporate surrounding dialogue turns or panel context to disambiguate omitted Japanese pro-drop subjects, anaphora, and colloquial manga expressions where isolated sentences are under-specified.
3. **Uncertainty-Aware Stronger-Model Escalation:** Selectively escalate borderline or high-disagreement directional predictions (e.g., strong forward/reverse asymmetry or moderate contradiction mass) to a stronger model only when validated against latency budgets on independent development sets.
