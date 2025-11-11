FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    export PATH="/root/.local/bin:$PATH" && \
    uv sync --frozen

RUN /app/.venv/bin/pip install --no-cache-dir "zenml[local]==0.91.0"

ENV PATH="/root/.local/bin:/app/.venv/bin:$PATH"
ENV ZENML_CONFIG_PATH="/runpod-volume/.config/zenml"

CMD ["sleep", "infinity"]
