FROM python:3.11-slim

WORKDIR /app
COPY . .

# Install system tools: curl (for uv), SSH server/client, and netcat for tunnel checks
RUN apt-get update && apt-get install -y \
    curl \
    openssh-server \
    openssh-client \
    netcat-openbsd \
 && rm -rf /var/lib/apt/lists/*

# Set up SSH server
RUN mkdir /var/run/sshd

# Install uv and sync dependencies
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    export PATH="/root/.local/bin:$PATH" && \
    uv sync --frozen

# Install ZenML dependencies
RUN /app/.venv/bin/pip install --no-cache-dir "zenml[local]==0.91.0"

# Expose SSH port
EXPOSE 22/tcp

# Environment variables
ENV PATH="/root/.local/bin:/app/.venv/bin:$PATH"
ENV ZENML_CONFIG_PATH="/runpod-volume/.config/zenml"

# Start SSH service and keep container running
CMD service ssh start && sleep infinity
