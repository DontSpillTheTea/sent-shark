FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    NLI_MODEL_NAME=MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install "django>=5.2.4" "djangorestframework>=3.16.0" "gunicorn>=23.0.0" "psycopg[binary]>=3.2.0" "protobuf>=5.28.0" "sentencepiece>=0.2.0" && \
    pip install --extra-index-url https://download.pytorch.org/whl/cpu "torch>=2.4.0" "transformers>=4.46.0"

# Pre-download and cache the NLI Cross-Encoder into the container image
# so docker compose up boots immediately without runtime model downloads.
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
    m = '${NLI_MODEL_NAME}'; \
    AutoTokenizer.from_pretrained(m); \
    AutoModelForSequenceClassification.from_pretrained(m)"

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "python manage.py migrate --noinput && gunicorn assignment.wsgi:application --bind 0.0.0.0:8000 --workers 1 --threads 8 --timeout 60"]
