#!/usr/bin/env bash
# Start LocalScholar for local use: API on :8000, UI on :5173.
# Ollama must already be running (`ollama serve`).
set -euo pipefail
cd "$(dirname "$0")/.."

if ! curl -sf --max-time 3 http://localhost:11434/api/tags >/dev/null; then
  echo "Ollama is not reachable on http://localhost:11434."
  echo "Start it in another terminal with:  ollama serve"
  echo "Retrieval will still work; answering will not."
  echo
fi

trap 'kill 0' EXIT
.venv/bin/python -m uvicorn backend.main:app --port 8000 &
npm --prefix frontend run dev &
echo
echo "LocalScholar starting — open http://localhost:5173"
wait
