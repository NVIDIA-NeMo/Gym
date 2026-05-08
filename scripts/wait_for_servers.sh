#!/usr/bin/env bash
# Wait for ng_run head server and all child servers to become ready.
#
# Usage: ./scripts/wait_for_servers.sh <ng_run_pid> [head_server_port] [max_wait_seconds]
#
# Arguments:
#   ng_run_pid          PID of the background ng_run process
#   head_server_port    Port of the head server (default: 11000)
#   max_wait_seconds    Max seconds to wait for child servers (default: 180)

set -euo pipefail

NG_RUN_PID="${1:?Usage: wait_for_servers.sh <ng_run_pid> [head_server_port] [max_wait_seconds]}"
HEAD_PORT="${2:-11000}"
MAX_WAIT="${3:-180}"

HEAD_URL="http://127.0.0.1:${HEAD_PORT}"
POLL_INTERVAL=2

check_pid() {
  if ! kill -0 "$NG_RUN_PID" 2>/dev/null; then
    echo "ng_run process (PID $NG_RUN_PID) exited unexpectedly"
    exit 1
  fi
}

# Phase 1: Wait for head server to respond on /server_instances (max 60s)
# The head server has no root route, but /server_instances returns 200.
echo "Waiting for head server on port ${HEAD_PORT}..."
HEAD_READY="false"
for i in $(seq 1 $((60 / POLL_INTERVAL))); do
  if curl -sf "${HEAD_URL}/server_instances" > /dev/null 2>&1; then
    echo "Head server up after $((i * POLL_INTERVAL))s"
    HEAD_READY="true"
    break
  fi
  check_pid
  sleep "$POLL_INTERVAL"
done
if [ "$HEAD_READY" != "true" ]; then
  echo "Head server did not respond within 60s"
  exit 1
fi

# Phase 2: Poll child servers on /docs (returns 200 via FastAPI) until all ready.
echo "Waiting for all child servers..."
ITERATIONS=$((MAX_WAIT / POLL_INTERVAL))
ALL_READY="false"

for i in $(seq 1 "$ITERATIONS"); do
  RAW_RESPONSE=$(curl -s "${HEAD_URL}/server_instances" 2>&1) || true
  URLS=$(echo "$RAW_RESPONSE" | python3 -c "
import json, sys
try:
    instances = json.load(sys.stdin)
    if not isinstance(instances, list):
        print('ERROR: expected list, got ' + type(instances).__name__, file=sys.stderr)
        sys.exit(1)
    urls = [inst['url'] for inst in instances if inst.get('url')]
    if not urls:
        print('ERROR: no urls found in server_instances response', file=sys.stderr)
        sys.exit(1)
    print(' '.join(urls))
except (json.JSONDecodeError, KeyError) as e:
    print(f'ERROR: failed to parse /server_instances: {e}', file=sys.stderr)
    sys.exit(1)" 2>&1)

  PARSE_EXIT=$?
  if [ $PARSE_EXIT -ne 0 ]; then
    # Only warn on first few iterations — server may still be starting up
    if [ $(( (i * POLL_INTERVAL) % 30 )) -eq 0 ]; then
      echo "Warning: $URLS"
    fi
    check_pid
    sleep "$POLL_INTERVAL"
    continue
  fi

  READY=0
  TOTAL=0
  NOT_READY=""
  for url in $URLS; do
    TOTAL=$((TOTAL + 1))
    if curl -sf "$url/docs" > /dev/null 2>&1; then
      READY=$((READY + 1))
    else
      NOT_READY="${NOT_READY}  - ${url}\n"
    fi
  done

  if [ "$READY" -eq "$TOTAL" ] && [ "$TOTAL" -gt 0 ]; then
    echo "All $TOTAL servers ready after $((i * POLL_INTERVAL))s"
    ALL_READY="true"
    break
  fi

  # Progress update every 30s
  if [ $(( (i * POLL_INTERVAL) % 30 )) -eq 0 ]; then
    echo "$READY / $TOTAL servers ready..."
  fi

  check_pid
  sleep "$POLL_INTERVAL"
done

if [ "$ALL_READY" != "true" ]; then
  echo "FAILED: Servers did not become ready within ${MAX_WAIT}s"
  echo "Servers still not responding:"
  printf "$NOT_READY"
  echo ""
  echo "Last /server_instances response:"
  echo "$RAW_RESPONSE"
  exit 1
fi
