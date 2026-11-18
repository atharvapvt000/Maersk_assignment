# ---- builder: resolve and install locked dependencies into a venv ---------------------------
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.7.20 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# ---- runtime: slim image, non-root, only what the service needs -----------------------------
FROM python:3.12-slim AS runtime
ARG GIT_SHA=unknown
ARG APP_VERSION=0.1.0
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GIT_SHA=$GIT_SHA \
    APP_VERSION=$APP_VERSION \
    DB_PATH=/app/data/out/supply_chain.duckdb \
    DQ_REPORT_PATH=/app/data/out/dq_report.json \
    MODEL_PATH=/app/models/delay_model.joblib

RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY app ./app
COPY dq ./dq
COPY pipeline ./pipeline
COPY ml ./ml
COPY assistant ./assistant
COPY models ./models
COPY scripts ./scripts
COPY data/raw ./data/raw
RUN mkdir -p /app/data/out /app/logs && chown -R app:app /app/data/out /app/logs
USER app

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"

CMD ["python", "-m", "app"]
