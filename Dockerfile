# syntax=docker/dockerfile:1.7

FROM python:3.12-slim-trixie AS builder
COPY --from=ghcr.io/astral-sh/uv:0.9.30 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /app

# Dependencies first so source edits don't invalidate this layer.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY README.md alembic.ini ./
COPY app ./app
COPY alembic ./alembic
# uv's build cache is keyed on pyproject.toml, not the source, so force a fresh wheel.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --reinstall-package rag-document-qa


# Test/lint image: adds the dev group and installs the project in editable mode.
FROM builder AS dev
ENV PATH=/opt/venv/bin:$PATH \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen
COPY . .
CMD ["pytest"]


FROM python:3.12-slim-trixie AS runtime
RUN useradd --create-home --uid 10001 app
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken
COPY --from=builder /opt/venv /opt/venv
# Fetch the tokenizer file at build time so the container needs no network at runtime.
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

WORKDIR /app
COPY alembic.ini ./
COPY alembic ./alembic
COPY eval ./eval

USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]
