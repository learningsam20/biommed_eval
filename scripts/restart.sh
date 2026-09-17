#!/usr/bin/env bash
# Kill listeners on the .env app ports, then restart backend + frontend.
# Usage:
#   ./scripts/restart.sh           # kill + start
#   ./scripts/restart.sh --kill    # kill only
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT/.env"
KILL_ONLY=0
if [[ "${1:-}" == "--kill" || "${1:-}" == "-k" ]]; then
  KILL_ONLY=1
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "missing $ENV_FILE" >&2
  exit 1
fi

env_get() {
  local key="$1" default="${2:-}"
  local line val
  line="$(grep -E "^${key}=" "$ENV_FILE" | tail -n 1 || true)"
  val="${line#*=}"
  val="${val%\"}"
  val="${val#\"}"
  val="${val%\'}"
  val="${val#\'}"
  printf '%s' "${val:-$default}"
}

url_port() {
  python3 - "$1" "$2" <<'PY'
import sys, urllib.parse
url, default = sys.argv[1], sys.argv[2]
p = urllib.parse.urlparse(url if "://" in url else f"http://{url}")
print(p.port or default)
PY
}

BACKEND_HOST="$(env_get BACKEND_HOST 0.0.0.0)"
BACKEND_PORT="$(env_get BACKEND_PORT 5268)"
FRONTEND_URL="$(env_get FRONTEND_URL http://localhost:5269)"
FRONTEND_PORT="$(url_port "$FRONTEND_URL" 5269)"

PYTHON_BIN="$(command -v python3 || command -v python)"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "python3 not found" >&2
  exit 1
fi

kill_port() {
  local port="$1"
  local pids
  pids="$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    echo "kill :$port -> $(echo "$pids" | tr '\n' ' ')"
    # shellcheck disable=SC2086
    kill -9 $pids 2>/dev/null || true
  else
    echo "free :$port"
  fi
}

echo "ports from .env  backend=${BACKEND_HOST}:${BACKEND_PORT}  frontend=${FRONTEND_PORT}"
kill_port "$BACKEND_PORT"
kill_port "$FRONTEND_PORT"

# leftover project processes that may not still hold the port
pkill -9 -f "${ROOT}/backend.*app.main" 2>/dev/null || true
pkill -9 -f "uvicorn app.main:app" 2>/dev/null || true
pkill -9 -f "vite --host .* --port ${FRONTEND_PORT}" 2>/dev/null || true
sleep 0.4

if [[ "$KILL_ONLY" -eq 1 ]]; then
  echo "killed. not restarting (--kill)"
  exit 0
fi

mkdir -p "$ROOT/logs"
: > "$ROOT/logs/backend.log"
: > "$ROOT/logs/frontend.log"

echo "start backend  ${PYTHON_BIN} -m app.main"
(
  cd "$ROOT/backend"
  nohup "$PYTHON_BIN" -m app.main >> "$ROOT/logs/backend.log" 2>&1 &
  echo $! > "$ROOT/logs/backend.pid"
)

echo "start frontend npm run dev -- --host 0.0.0.0 --port ${FRONTEND_PORT}"
(
  cd "$ROOT/frontend"
  nohup npm run dev -- --host 0.0.0.0 --port "$FRONTEND_PORT" >> "$ROOT/logs/frontend.log" 2>&1 &
  echo $! > "$ROOT/logs/frontend.pid"
)

wait_http() {
  local name="$1" url="$2" n
  for n in $(seq 1 40); do
    if curl -sf -o /dev/null --max-time 1 "$url"; then
      echo "ready ${name}  ${url}  (${n}s)"
      return 0
    fi
    sleep 1
  done
  echo "timeout waiting for ${name} at ${url}" >&2
  echo "---- ${name} log ----" >&2
  tail -n 40 "$ROOT/logs/${name}.log" >&2 || true
  return 1
}

wait_http backend "http://127.0.0.1:${BACKEND_PORT}/api/health"
wait_http frontend "http://127.0.0.1:${FRONTEND_PORT}/"

echo "backend  http://127.0.0.1:${BACKEND_PORT}  (pid $(cat "$ROOT/logs/backend.pid"))"
echo "frontend http://127.0.0.1:${FRONTEND_PORT}  (pid $(cat "$ROOT/logs/frontend.pid"))"
echo "logs     $ROOT/logs/{backend,frontend}.log"
