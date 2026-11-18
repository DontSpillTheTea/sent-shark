# `sent-shark` — Same-Language Entailment Judge API

> Evaluates whether two sentences in the same language (`ja–ja` or `en–en`) convey the same meaning (`ENTAIL`) or not (`NO_ENTAIL`) — Mantra Inc. Langaku Backend Assignment (Option 3).

---

## 1. Quick Start, Configuration & Verification

### Option A: Docker Compose (Recommended — Pre-Cached Model + PostgreSQL 15)

The [`Dockerfile`](Dockerfile) pre-caches `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` (`HF_HOME=/app/.cache/huggingface`) and serves Gunicorn (`--workers 1 --threads 8`) backed by `postgres:15-alpine`:

```bash
docker compose up --build -d
curl http://localhost:8000/health   # {"status":"ok"}
docker compose down -v
```

### Option B: Local Virtual Environment (`uv` or `.venv`)

```bash
# Using uv:
uv sync --all-groups
uv run manage.py migrate
uv run manage.py runserver

# Or using Python venv (pip >= 25.1 for PEP 735 --group):
python3 -m venv .venv
.venv/bin/pip install --upgrade pip && .venv/bin/pip install -e . --group dev
.venv/bin/python manage.py migrate
.venv/bin/python manage.py runserver
```

### Environment Variables (`assignment/settings.py`)

| Variable | Default | Purpose |
|---|---|---|
| `NLI_MODEL_NAME` | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` | Hugging Face NLI cross-encoder model ID |
| `NLI_ENABLE_MODEL` | `true` | Set `false` for exact-match-only offline fallback (`0.99` / `0.0`) |
| `NLI_ENTAIL_INDEX` / `NLI_CONTRADICT_INDEX` | unset | Non-negative ints (set together) when `config.id2label` uses generic labels |
| `ENTAIL_THRESHOLD` | `0.50` | Classification cutoff (`score >= threshold` $\to$ `ENTAIL`) |
| `RATE_LIMIT_REQUESTS` / `RATE_LIMIT_WINDOW_SECONDS` | `5` / `1.0` | Per-`client_ip` sliding-window rate limit |
| `TRUST_PROXY_HEADERS` / `TRUSTED_PROXY_IPS` | `false` / `""` | Enable right-to-left `X-Forwarded-For` parsing for trusted proxy IPs/CIDRs |
| `POSTGRES_HOST` / `POSTGRES_DB` / `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_PORT` | unset (`langaku_judge`, `postgres`, `5432`) | Unset uses SQLite WAL; set `POSTGRES_HOST` for PostgreSQL 15 |
| `DJANGO_SECRET_KEY` / `DJANGO_DEBUG` / `DJANGO_ALLOWED_HOSTS` | dev key / `true` / `*` | Standard Django runtime configuration |

### Verification & Evaluation Commands

```bash
.venv/bin/pytest -m "not integration" -v        # 53 fast unit/API/concurrency/security tests
.venv/bin/pytest -v                             # All 79 tests (incl. 26 real-model integration tests)
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python manage.py check
.venv/bin/python manage.py makemigrations --check --dry-run
.venv/bin/python manage.py run_eval --dataset dev --diagnostics --json
.venv/bin/python manage.py run_eval --dataset dev_diagnostic --diagnostics --json
.venv/bin/python manage.py run_eval --dataset final_holdout --diagnostics --json
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

**Headers, Proxy Security, Validation & Auditing:**
- **Headers:** Every `/judge*` response carries `X-Request-Id` (sanitized printable ASCII `<= 128` chars or generated UUID4); rate-checked responses include `X-RateLimit-Limit` (`"5"`) and `X-RateLimit-Remaining` (`"0"` on `429` plus `Retry-After` whole seconds).
- **Proxy IP safety (`assignment/rate_limit.py`):** `get_client_ip()` ignores `X-Forwarded-For` unless `TRUST_PROXY_HEADERS=true` and `REMOTE_ADDR` matches `TRUSTED_PROXY_IPS`, walking `X-Forwarded-For` right-to-left and failing closed to `REMOTE_ADDR` on malformed chains.
- **Audit logging (`RequestAuditLog`):** Records every `/judge*` outcome (`200`, `400`, `405`, `406`, `429`, `500`) once via an idempotency guard (`client_ip`, `method`, `path`, `status_code`, `payload_chars`, `latency_ms`, `rate_limited`) **without** storing sentence text.
- **Validation (`JudgePairSerializer`):** Non-blank trimmed strings (`<= 2000` chars); rejects cross-language pairs (`contains_japanese` mismatch vs. strict Latin) while allowing embedded ASCII loanwords (e.g., `"AI"`, `"SDK"`) in Japanese.

```bash
# Single judgment (F1)
curl -s -X POST http://localhost:8000/judge -H "Content-Type: application/json" \
  -d '{"sentence1":"私は猫が大好きだ。","sentence2":"猫が好きです。"}'
# -> {"label":"ENTAIL","score":0.9536}

# Bulk judgment (F2 — raw JSON list or {"pairs":[...]}, 1..100 pairs)
curl -s -X POST http://localhost:8000/judge/bulk -H "Content-Type: application/json" \
  -d '{"pairs":[{"sentence1":"私は猫が大好きだ。","sentence2":"猫が好きです。"},{"sentence1":"A cat is a carnivore.","sentence2":"A cat is a vegetarian."}]}'
# -> [{"label":"ENTAIL","score":0.9536},{"label":"NO_ENTAIL","score":0.0044}]
```

**Error Responses (`400` Validation / `429` Rate Limit):**

```json
// 400 Bad Request (cross-language; bulk raw list bounds return {"error": "Bulk request must contain between 1 and 100 sentence pairs."})
{"non_field_errors": ["sentence1 and sentence2 must be written in the same language."]}

// 429 Too Many Requests
{"error": "Rate limit exceeded: max 5 requests per 1s per client_ip.", "retry_after": 0.842}
```

---

## 3. Evidence-Backed Architecture & Simplification

### Production 2-Tier Pipeline (`exact_plus_nli`)

1. **Tier 1 Exact Match (`assignment/services/normalizer.py`):** Applies Unicode `NFKC` normalization, whitespace collapse, and edge punctuation trimming; identical normalized pairs (case-insensitive for English) short-circuit in `<0.1 ms` (`ENTAIL`, `0.99`).
2. **Tier 2 Bidirectional NLI (`assignment/services/nli_engine.py`):** Non-identical pairs run in deduplicated chunked batches (`batch_size=64`, `max_length=256`) through `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` (`mit`, `6L x 384`, 107M params), scoring both directions symmetrically:
   $$\text{score} = \frac{p_E(s_1 \Rightarrow s_2) + p_E(s_2 \Rightarrow s_1)}{2} \times \Big(1 - \max\big(p_C(s_1 \Rightarrow s_2),\; p_C(s_2 \Rightarrow s_1)\big)\Big)$$
   `id2label` is resolved fail-closed (raises `ValueError` on missing/generic/ambiguous labels unless `NLI_ENTAIL_INDEX` and `NLI_CONTRADICT_INDEX` are both explicitly configured).
3. **Deduplication, Hot Cache & Singleflight (`assignment/services/judge_service.py`):**
   - **Order-independent versioned `pair_hash`:** `SHA-256` over `f"{model_name}\0{SCORING_POLICY_VERSION}\0{a}\0{b}"` (`(a, b) = sorted((norm1, norm2))`), enforced via `JudgmentRecord.pair_hash` (`unique=True`) and sorted `bulk_create(..., ignore_conflicts=True)` with `0` duplicate rows under 30 parallel threads.
   - **`_HOT_CACHE` (`2048`-entry LRU):** `25.97x` single-pair p50 speedup (`0.0067 ms` vs `0.1740 ms` DB-only) and `2.63x` bulk-100 p50 speedup (`0.6011 ms` vs `1.5825 ms`).
   - **`SingleflightGroup`:** Under 30-thread cold-cache bursts, cuts NLI batch calls **`30 -> 1`**, burst wall-clock **`7.30x`** (`123.85 ms` vs `904.67 ms`), and p95 **`7.78x`** (`119.35 ms` vs `928.82 ms`).
   - **Scope simplification:** Removed unrequired aggregate stats/online evaluation endpoints, duplicate versioned judge route aliases, speculative external-API escalation, and hot-path table pruning (eliminating concurrent PostgreSQL lock contention and `expected_label` schema bloat).

### Negative Empirical Findings (Phases 2–3 on `dev` + `dev_diagnostic`)

- **Hand-crafted Tier 1 heuristics rejected:** Synonym/register rewrites, SVO/SOV regex vetoes, and char-2gram calibration (`full_hybrid`) boosted only `dev` (`50/60 -> 55/60`) via synthetic pattern overlap while hitting a substring false-positive veto (`G46`). On independent `dev_diagnostic`, heuristics fixed **`0/60`** pairs, regressed `H33` (`0.4699 -> 0.5335`), lowered accuracy (`0.6167 -> 0.6000`), and increased batch CPU latency **`+60.8%`** (`866.20 ms -> 1392.85 ms`).
- **Japanese-only BERT candidates rejected:** On uncontaminated `dev` Japanese (`32` pairs), `llm-book/bert-base-japanese-v3-jnli` (`apache-2.0`) collapsed to **`0.4375`** (`14/32`, `0/18` `ENTAIL` recall; McNemar $p=0.0013$) due to `JNLI` caption overfitting, and `akiFQC/bert-base-japanese-v3_nli-jsnli-jnli-jsick` (`cc-by-sa-4.0`, `JNLI+JSICK+JSNLI` train overlap) dropped to **`0.5938`** (`19/32`, $p=0.0117$) and misclassified directional hyponym `G40` at `0.9997 ENTAIL`. Both 12-layer BERTs broke English (`llm-book` `0/10` PAWS paraphrases; `akiFQC` `0.50–0.57` EN), ran **`~9.0–9.5x` slower** on 60-pair CPU batches (`3.27–7.77 s` vs `0.35–0.85 s`), used **`+254–262 MiB`** peak RSS, and required `fugashi` + `unidic-lite`.

---

## 4. Frozen Evaluation Results (`567f8a0`)

Evaluated on locked commit `567f8a0` (`MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli`, `model_revision=0a71e92a985b6e1ad1828cf67ce9c459639c1dca`, `exact_plus_nli:v1:mean`, `ENTAIL_THRESHOLD=0.50`, CPU `torch 2.15.0+cpu`, 10 cores). Because all three datasets contain `0` exact-identical pairs, `tier1` predicts `NO_ENTAIL` (`0.5000` acc) and production `hybrid` matches `tier2` predictions identically.

| Dataset (`n`) | `hybrid` Acc [95% Wilson CI] | F1 | Brier | CM (`TP/FP/TN/FN`) | Mean / p95 (ms) | `ja` Primary Acc [95% CI] | `en` Secondary Acc [95% CI] |
|---|---:|---:|---:|---|---:|---:|---:|
| [`dev`](assignment/tests/fixtures/eval_gauntlet.json) (`60`: `32 ja`/`28 en`) | **`0.8333`** `[0.7197, 0.9069]` | `0.8333` | `0.1656` | `25/5/25/5` | `41.73` / `48.23` | **`0.8750`** (`28/32`) `[0.7193, 0.9503]` | `0.7857` (`22/28`) `[0.6046, 0.8979]` |
| [`dev_diagnostic`](assignment/tests/fixtures/eval_holdout.json) (`60`: `40 ja`/`20 en`) | `0.6167` `[0.4902, 0.7291]` | `0.6349` | `0.2838` | `20/13/17/10` | `41.95` / `49.67` | **`0.7000`** (`28/40`) `[0.5457, 0.8193]` | `0.4500` (`9/20`) `[0.2582, 0.6579]` |
| [**`final_holdout`**](assignment/tests/fixtures/eval_final_holdout.json) (`178`: `138 ja`/`40 en`, untouched) | **`0.6292`** `[0.5562, 0.6967]` | `0.6972` | `0.3173` | `76/53/36/13` | `47.61` / `57.95` | **`0.6884`** (`95/138`) `[0.6069, 0.7597]` | `0.4250` (`17/40`) `[0.2851, 0.5780]` |

### Untouched `final_holdout` Source & Stratum Breakdown ([Provenance](assignment/tests/fixtures/eval_final_holdout_provenance.json))

| Source (`n`) | Source Acc [95% Wilson CI] | Strata Accuracy Breakdown |
|---|---|---|
| `jglue_jnli` (`ja`, `58`) | **`0.8276`** (`48/58`) `[0.7109, 0.9036]` | `bidir_entail` `26/29` (`0.8966`), `oneway` `8/10` (`0.8000`), `contradict` `7/10` (`0.7000`), `neutral` `7/9` (`0.7778`) |
| `paws_x_ja` (`ja`, `80`) | `0.5875` (`47/80`) `[0.4780, 0.6889]` | `ja_paws_paraphrase` `34/40` (`0.8500`), `ja_paws_swap` `13/40` (`0.3250`) |
| `paws_x_en` (`en`, `40`) | `0.4250` (`17/40`) `[0.2851, 0.5780]` | `en_paws_paraphrase` `16/20` (`0.8000`), `en_paws_swap` `1/20` (`0.0500`) |

- **Error Taxonomy (`66` errors, all `tier2_nli`):** `46` `word_order_or_particle_swap` (`27 ja`, `19 en`), `10` `lexical_synonym_gap`, `5` `caption_codescription`, `3` `negation_or_antonym`, `2` `oneway_specificity`.

---

## 5. Live Docker Compose + PostgreSQL 15 Verification

Verified end-to-end on `567f8a0` against `postgres:15-alpine`:
- **`pytest -v` (`POSTGRES_HOST=127.0.0.1`):** **`79 passed in 64.14s`** (53 unit/security/concurrency + 26 real-model integration tests).
- **Live Gunicorn + PostgreSQL 15 (`docker compose up --build -d`):** `migrate --check` and `manage.py check` exited `0`; `/health` `200`, `/playground` `200`, `/judge` and `/judge/bulk` verified; 15-thread single-IP burst yielded **`{200: 5, 429: 10}`** with **`7`** deduplicated `JudgmentRecord` rows and **`17`** `RequestAuditLog` rows (`7 x 200`, `10 x 429`) persisted in PostgreSQL.

---

## 6. Known Limitations & Future Improvements

**Known Limitations:**
- **Adversarial word-order/particle swaps:** Distilled 6-layer XNLI struggles when bag-of-words overlap is high (`ja_paws_swap` `0.3250`, `en_paws_swap` `0.0500`, `Specificity 0.4045` on `final_holdout`).
- **Borderline one-way entailments under `mean`:** Averaging $p_{E,\text{fwd}} \approx 0.99$ with $p_{E,\text{rev}} \approx 0.01$ leaves scores just above `0.50` (`F151 0.5001`, `F154 0.5025`; `H24 0.5023`).
- **Single-process concurrency state & corpus noise:** `_HOT_CACHE`, `_SINGLEFLIGHT`, and `SlidingWindowRateLimiter` are process-local (`--workers 1 --threads 8`); `PAWS-X` also contains upstream translation/entity noise (`F032`, `F092`, `H48`).

**Future Improvements:**
1. **Syntax- & Role-Aware Contrastive Fine-Tuning:** Fine-tune a compact bilingual/Japanese cross-encoder on compositional Japanese particle-scrambling vs. case-inversion pairs (**Yanaka & Mineshima, 2022**, *TACL* 10:1266–1284, JSICK), adversarial paraphrase swaps (**Yang et al., 2019**, *EMNLP*, PAWS-X), and native Japanese NLI (**Kurihara et al., 2022**, *LREC*, pp. 2957–2966, JGLUE) alongside manga register pairs.
2. **Directional Asymmetry Guard for One-Way Entailment:** Penalize or calibrate large $|p_{E,\text{fwd}} - p_{E,\text{rev}}|$ gaps against JGLUE one-way splits (**Kurihara et al., 2022**) and JSICK directional predicates (**Yanaka & Mineshima, 2022**) so `0.99 / 0.01` asymmetric pairs do not cross `0.50`.
3. **Shared Redis Rate Limiter & Cache:** Replace in-process `SlidingWindowRateLimiter` and `_HOT_CACHE` with atomic Redis Lua + cache lookups for multi-worker / multi-container horizontal scaling over PostgreSQL 15.
