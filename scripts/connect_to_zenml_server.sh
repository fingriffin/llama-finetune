#!/bin/bash
set -euo pipefail

# Configuration
ZENML_PORT=${ZENML_PORT:-8237}
SSH_USER=${SSH_USER:-root}
SSH_HOST=${SSH_HOST:?Host not set}
SSH_KEY_PATH=${SSH_KEY_PATH:-~/.ssh/id_ed25519}

echo "[INFO] Connecting to ZenML server on ${SSH_HOST}:${ZENML_PORT}"

# Open SSH Tunnel
echo "[INFO] Opening SSH tunnel (localhost:${ZENML_PORT} -> ${SSH_HOST}:${ZENML_PORT})..."
ssh -N -L ${ZENML_PORT}:127.0.0.1:${ZENML_PORT} -i ${SSH_KEY_PATH} ${SSH_USER}@${SSH_HOST} &
SSH_TUNNEL_PID=$!

# Wait for tunnel to establish
sleep 2

# Check connection
if ! nc -z 127.0.0.1 ${ZENML_PORT}; then
  echo "[ERROR] Failed to connect to ZenML server via tunnel. Exiting."
  kill $SSH_TUNNEL_PID 2>/dev/null || true
  exit 1
fi
echo "[SUCCESS] Tunnel established."

# Connect to ZenML
echo "[INFO] Logging in to ZenML..."
zenml login http://127.0.0.1:${ZENML_PORT} --no-verify-ssl || {
  echo "[ERROR] ZenML login failed."
  kill $SSH_TUNNEL_PID 2>/dev/null || true
  exit 1
}

echo "[SUCCESS] Connected to ZenML server."
echo "[INFO] You can now run ZenML commands or fine-tuning scripts."

# To keep tunnel alive
wait $SSH_TUNNEL_PID
