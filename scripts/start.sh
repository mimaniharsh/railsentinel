#!/usr/bin/env bash
# Start the API (internal) and the Streamlit UI (public) in one container.
# PORT is the public port (8501 by default; platforms such as Hugging Face set their own).
set -euo pipefail

PORT="${PORT:-8501}"
API_PORT="${API_PORT:-8000}"
export RS_API_URL="${RS_API_URL:-http://127.0.0.1:${API_PORT}}"

uvicorn railsentinel.api.main:app --host 127.0.0.1 --port "${API_PORT}" &

for _ in $(seq 1 180); do
  if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${API_PORT}/health', timeout=2)" 2>/dev/null; then
    break
  fi
  sleep 1
done

exec streamlit run ui/Home.py --server.port "${PORT}" --server.address 0.0.0.0
