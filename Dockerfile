FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    export PATH="/root/.local/bin:$PATH" && \
    uv sync --frozen

ENV PATH="/root/.local/bin:/app/.venv/bin:$PATH"

CMD ["sleep", "infinity"]
