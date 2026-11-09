# syntax=docker/dockerfile:1
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /usr/local/bin/uv
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    HF_HOME=/models PATH=/app/.venv/bin:$PATH
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra ml --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --extra ml
# Model weights are fetched here, at build time, from Hugging Face. The running containers are
# offline (HF_HUB_OFFLINE=1) and network-locked to Sefaria, Hebcal and the Anthropic API.

RUN python -m chavruta.fetch_models --chosen --reranker && chmod -R a+rX /models
COPY alembic.ini ./
COPY alembic ./alembic
COPY corpus ./corpus
COPY eval ./eval
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN useradd --create-home app && mkdir -p /app/data && chown -R app /app/data
USER app
EXPOSE 8000
CMD ["uvicorn", "--factory", "chavruta.app:make", "--host", "0.0.0.0", "--port", "8000"]
