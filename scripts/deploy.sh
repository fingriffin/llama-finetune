# Sync local project to RunPod instance, excluding large folders.
#!/usr/bin/env bash
set -euo pipefail

ZIP_NAME="llama-finetune.zip"
PROJECT_NAME="llama-finetune"

rm -f "$ZIP_NAME" 2>/dev/null || true

echo "Zipping project (excluding large or cached folders)..."
zip -r "$ZIP_NAME" . \
  -x ".git/*" \
     ".idea/*" \
     ".venv/*" \
     "__pycache__/*" \
     "*.pyc" \
     "*.pyo" \
     "*.log" \
     ".DS_Store" \
     "uv.lock" \
     "model-out/*" \
     "models/*" \
     ".mypy_cache/*" \
     ".ruff_cache/*" \
     ".pytest_cache/*" \
     ".env" \
     ".env.*" \
     ".secrets.baseline"

echo "Uploading $ZIP_NAME to RunPod..."
runpodctl send "$ZIP_NAME"

rm -f "$ZIP_NAME"

cat <<EOF

In runpod terminal run:

mv -f llama-finetune.zip /workspace/ && \
cd /workspace && \
unzip -qo llama-finetune && \
rm -f llama-finetune.zip

EOF