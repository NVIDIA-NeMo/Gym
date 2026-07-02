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
# sandboxes. Give each sandbox a unique port; the in-sandbox client reads gateway.port
# from $HOME/.openclaw (HOME is per-sandbox), so gateway + client agree.
GWPORT=$(shuf -i 20000-60000 -n1)
openclaw config set gateway.port "$GWPORT" >/dev/null 2>&1
openclaw gateway --auth token --bind loopback --allow-unconfigured --port "$GWPORT" >"$OUT/gateway.log" 2>&1 &
GW_PID=$!
# Port-agnostic readiness: the gateway binds an ephemeral port, so wait on its log
# marker ("plugins pre-warmed", ~15s) rather than probing a fixed port.
for i in $(seq 1 120); do grep -q 'plugins pre-warmed' "$OUT/gateway.log" 2>/dev/null && break; sleep 1; done
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

# benchmark.py is done; stop the gateway (and its tool subprocesses). The per-task
# instance is torn down right after, so a hard kill is safe, and it keeps teardown fast.
kill "$GW_PID" 2>/dev/null || true
pkill -9 -f openclaw 2>/dev/null || true

# Package $OUT so the host can download it (Sandbox API pulls one file). Tar to a temp
# path then move in, so the archive never tries to include itself.
tar czf "$TMPDIR/out.tgz" -C "$OUT" . 2>/dev/null || true
mv -f "$TMPDIR/out.tgz" "$OUT/out.tgz"
exit "$rc"
