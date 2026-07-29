#!/usr/bin/env bash
# In-container PinchBench entrypoint, baked into the image at /opt/run_task.sh (see
# Dockerfile.benchmark). Runs the stock PinchBench benchmark.py for a SINGLE task
# through OpenClaw and tars its results to <BASE>/out/out.tgz so the host can pull
# them back via the Sandbox API. The skill lives at /opt/pinchbench-skill (cloned at
# a pinned tag + NVIDIA-patched at image build).
#
# One sandbox per task = the isolation boundary (own filesystem -> own ~/.openclaw ->
# own gateway), so the gateway never shares a workspace across tasks (no
# WorkspaceVanishedError cliff). Provider-neutral (apptainer / opensandbox).
#
# WHY everything lives under $BASE (the Sandbox API working mount, default /sandbox):
# under the apptainer provider the image rootfs is READ-ONLY and /tmp + $HOME are the
# *host's* dirs SHARED across concurrent instances -- only the per-sandbox bind at the
# provider mount_point (/sandbox) is both writable AND isolated. So we point the skill
# copy, OpenClaw's $HOME, $TMPDIR and benchmark.py's run-root all under $BASE. (Under
# opensandbox the rootfs is writable anyway, so $BASE is just a private working dir.)
#
# Required env: TASK_ID MODEL_NAME MODEL_BASE_URL MODEL_API_KEY
#               JUDGE_MODEL JUDGE_BASE_URL JUDGE_API_KEY OPENCLAW_GATEWAY_TOKEN
# Optional env: PINCHBENCH_WORK_BASE (default /sandbox) PINCHBENCH_WEB_SEARCH_PROVIDER
#               BRAVE_API_KEY TAVILY_API_KEY PINCHBENCH_MAX_TOKENS
#               PINCHBENCH_CONTEXT_WINDOW TIMEOUT_MULT
#               PINCHBENCH_GATEWAY_READY_TIMEOUT_SECONDS
#               PINCHBENCH_GATEWAY_START_JITTER_S
#               PINCHBENCH_GATEWAY_PORT_LOW PINCHBENCH_GATEWAY_PORT_HIGH
set -uo pipefail

SKILL=/opt/pinchbench-skill
BASE="${PINCHBENCH_WORK_BASE:-/sandbox}"
WORK="$BASE/work"
OUT="$BASE/out"
# Redirect every writable target into the per-sandbox isolated mount (see header).
export HOME="$BASE/home"
export TMPDIR="$BASE/tmp"
export PINCHBENCH_RUN_ROOT="$BASE/pinchbench"
mkdir -p "$WORK" "$OUT" "$HOME" "$TMPDIR" "$PINCHBENCH_RUN_ROOT"

# Detach this script AND all its descendants from the exec's stdout/stderr pipe by
# pointing them at a file. OpenClaw spawns tool subprocesses that outlive run_task.sh;
# if any keeps the exec pipe open, the host's `apptainer exec` (asyncio communicate())
# never sees EOF and hangs forever. Redirecting here closes the pipe for the whole tree.
# run.log lands in $OUT, so it still ships back inside out.tgz for debugging.
exec >"$OUT/run.log" 2>&1

GW_PID=""
GWPORT=""

gateway_log_tail() {
  if [ -f "$OUT/gateway.log" ]; then
    echo "[run_task] gateway.log tail:"
    tail -80 "$OUT/gateway.log" || true
  fi
}

cleanup_gateway() {
  if [ -z "${GW_PID:-}" ]; then
    return
  fi

  # Prefer killing the process group created by setsid, but fall back to the PID for
  # environments where setsid is unavailable or the process group has already exited.
  kill -TERM -- "-$GW_PID" 2>/dev/null || kill "$GW_PID" 2>/dev/null || true
  for _ in $(seq 1 20); do
    kill -0 "$GW_PID" 2>/dev/null || break
    sleep 0.2
  done
  kill -KILL -- "-$GW_PID" 2>/dev/null || kill -9 "$GW_PID" 2>/dev/null || true
  wait "$GW_PID" 2>/dev/null || true
}

finish() {
  rc=$?
  trap - EXIT
  cleanup_gateway
  # Package $OUT so the host can download it (Sandbox API pulls one file). Tar to a
  # temp path then move in, so the archive never tries to include itself.
  tar czf "$TMPDIR/out.tgz" -C "$OUT" . 2>/dev/null || true
  mv -f "$TMPDIR/out.tgz" "$OUT/out.tgz" 2>/dev/null || true
  exit "$rc"
}

trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

select_gateway_port() {
  python3 - <<'PY'
import hashlib
import os
import socket
import sys

try:
    low = int(os.environ.get("PINCHBENCH_GATEWAY_PORT_LOW", "20000") or 20000)
    high = int(os.environ.get("PINCHBENCH_GATEWAY_PORT_HIGH", "60000") or 60000)
except ValueError:
    low, high = 20000, 60000
if low > high:
    low, high = high, low

span = max(1, high - low + 1)
try:
    limit = int(os.environ.get("PINCHBENCH_GATEWAY_PORT_SCAN_LIMIT", "4096") or 4096)
except ValueError:
    limit = 4096
limit = max(1, min(span, limit))

seed = ":".join(
    [
        os.environ.get("TASK_ID", ""),
        os.environ.get("PINCHBENCH_WORK_BASE", ""),
        str(os.getpid()),
    ]
)
start = int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16) % span
for offset in range(limit):
    port = low + ((start + offset) % span)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            continue
    print(port)
    sys.exit(0)

print(f"no free loopback port found in {low}-{high} after {limit} attempts", file=sys.stderr)
sys.exit(1)
PY
}

gateway_start_delay() {
  python3 - <<'PY'
import hashlib
import os

try:
    jitter = int(os.environ.get("PINCHBENCH_GATEWAY_START_JITTER_S", "0") or 0)
except ValueError:
    jitter = 0
if jitter <= 0:
    print(0)
else:
    seed = ":".join(
        [
            os.environ.get("TASK_ID", ""),
            os.environ.get("PINCHBENCH_WORK_BASE", ""),
            str(os.getpid()),
        ]
    )
    print(int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16) % (jitter + 1))
PY
}

# Copy the (read-only) skill to the writable working tree.
cp -a "$SKILL"/. "$WORK"/
cd "$WORK"

export OPENAI_API_KEY="${OPENAI_API_KEY:-$MODEL_API_KEY}"

# Per-task OpenClaw gateway (token auth + loopback bind). At openclaw 2026.6.5
# `openclaw agent` routes through a gateway to persist session transcripts; the
# per-task sandbox keeps this gateway isolated to one task.
: "${OPENCLAW_GATEWAY_TOKEN:?run_task.sh needs OPENCLAW_GATEWAY_TOKEN}"
echo "[run_task] starting gateway (token auth, loopback)"
# apptainer shares the host network -> a fixed gateway port collides across concurrent
# sandboxes. Pick a currently-free loopback port from a hashed per-task scan; the
# in-sandbox client reads gateway.port from $HOME/.openclaw (HOME is per-sandbox), so
# gateway + client agree.
DELAY=$(gateway_start_delay)
if [ "$DELAY" -gt 0 ] 2>/dev/null; then
  echo "[run_task] gateway start jitter: ${DELAY}s"
  sleep "$DELAY"
fi
if ! GWPORT=$(select_gateway_port); then
  echo "[run_task] failed to select OpenClaw gateway port"
  exit 69
fi
openclaw config set gateway.port "$GWPORT" >/dev/null 2>&1
if command -v setsid >/dev/null 2>&1; then
  setsid openclaw gateway --auth token --bind loopback --allow-unconfigured --port "$GWPORT" >"$OUT/gateway.log" 2>&1 &
else
  openclaw gateway --auth token --bind loopback --allow-unconfigured --port "$GWPORT" >"$OUT/gateway.log" 2>&1 &
fi
GW_PID=$!
echo "[run_task] gateway pid=$GW_PID port=$GWPORT"

READY_TIMEOUT="${PINCHBENCH_GATEWAY_READY_TIMEOUT_SECONDS:-300}"
case "$READY_TIMEOUT" in
  ''|*[!0-9]*) READY_TIMEOUT=300 ;;
esac
if [ "$READY_TIMEOUT" -lt 1 ]; then
  READY_TIMEOUT=1
fi

# Port-agnostic readiness: the gateway binds a per-task port, so wait on its log
# marker. Fail fast if the gateway process dies or the bind race is lost.
GATEWAY_READY=0
for _ in $(seq 1 "$READY_TIMEOUT"); do
  if ! kill -0 "$GW_PID" 2>/dev/null; then
    echo "[run_task] gateway exited before readiness"
    gateway_log_tail
    exit 70
  fi
  if grep -Eiq 'EADDRINUSE|address already in use|port .*already in use' "$OUT/gateway.log" 2>/dev/null; then
    echo "[run_task] gateway port bind failure on port $GWPORT"
    gateway_log_tail
    exit 71
  fi
  if grep -q 'plugins pre-warmed' "$OUT/gateway.log" 2>/dev/null; then
    GATEWAY_READY=1
    break
  fi
  sleep 1
done
if [ "$GATEWAY_READY" -ne 1 ]; then
  echo "[run_task] gateway readiness timed out after ${READY_TIMEOUT}s"
  gateway_log_tail
  exit 72
fi
sleep 2

echo "[run_task] task=$TASK_ID model=$MODEL_NAME base=$BASE"
uv run --no-project --with pyyaml python scripts/benchmark.py \
  --model "$MODEL_NAME" \
  --base-url "$MODEL_BASE_URL" \
  --api-key "$MODEL_API_KEY" \
  --judge "$JUDGE_MODEL" \
  --suite "$TASK_ID" \
  --no-upload --no-fail-fast \
  --timeout-multiplier "${TIMEOUT_MULT:-3}" \
  --output-dir "$OUT"
rc=$?
exit "$rc"
