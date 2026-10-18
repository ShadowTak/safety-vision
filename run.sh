#!/bin/sh
# Start Safety Vision at http://127.0.0.1:8765
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
fi
# secrets such as TELEGRAM_TOKEN live in .env (not in the code)
[ -f .env ] && set -a && . ./.env && set +a
exec .venv/bin/python -m uvicorn server:app --host 127.0.0.1 --port "${PORT:-8765}"
