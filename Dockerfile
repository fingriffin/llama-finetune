FROM runpod/base:0.7.0-ubuntu2404

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install -y \
    curl \
    netcat-openbsd \
 && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    export PATH="/root/.local/bin:$PATH" && \
    uv sync --frozen

RUN mkdir -p /run/sshd && ssh-keygen -A

EXPOSE 22/tcp
ENV PATH="/root/.local/bin:/app/.venv/bin:$PATH"

CMD bash -c 'mkdir -p /root/.ssh && chmod 700 /root/.ssh && \
              if [ -n "$PUBLIC_KEY" ]; then echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys; fi && \
              /usr/sbin/sshd -D'