#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# lexmount_browser — one-script, reproducible walk-through for reviewers.
#
# Three explicit, independently runnable stages:
#
#   A. rollout   (no GPU, ~minutes)  — install deps, run the backend test, start
#                                       the Gym serving stack, collect rollouts
#                                       over data/example.jsonl against a policy
#                                       endpoint. THIS IS THE DEFAULT.
#   B. train     (1 GPU)             — GRPO training smoke via NeMo-RL.
#   C. rollout --backend lexmount    — same as A, but drives the Lexmount cloud
#                                       browser instead of local Playwright.
#
# Usage:
#   bash example.sh rollout                     # Stage A (Playwright backend)
#   bash example.sh test                        # just the standalone backend test
#   bash example.sh train                       # Stage B (needs 1 GPU + NeMo-RL)
#   bash example.sh rollout --backend lexmount  # Stage C (needs Lexmount creds+SDK)
#
# Policy endpoint (Stage A / C) — pick ONE:
#   (1) Generic OpenAI-compatible endpoint (no local GPU needed). Point at any
#       server that implements the Responses API (/v1/responses):
#         export POLICY_BASE_URL=https://your-endpoint/v1
#         export POLICY_API_KEY=sk-...
#         export POLICY_MODEL=your-model-name
#   (2) A locally served vLLM model (needs a GPU):
#         export POLICY_KIND=vllm            # serve responses_api_models/vllm_model
#         export POLICY_MODEL=Qwen/Qwen3-4B  # HF id / local checkpoint path
#
# Everything is idempotent and fail-fast (set -Eeuo pipefail). Re-running is safe.
# ---------------------------------------------------------------------------
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_REL="resources_servers/lexmount_browser"

# Resolve the Gym repo root (the dir whose child is nemo_gym/ + pyproject.toml).
find_repo_root() {
  local d="$SCRIPT_DIR"
  while [[ "$d" != "/" ]]; do
    if [[ -f "$d/pyproject.toml" && -d "$d/nemo_gym" ]]; then echo "$d"; return 0; fi
    d="$(dirname "$d")"
  done
  return 1
}
REPO_ROOT="$(find_repo_root)" || {
  echo "ERROR: could not locate the NeMo-Gym repo root above $SCRIPT_DIR" >&2
  exit 1
}

LIMIT="${LIMIT:-2}"                    # rollouts to collect in Stage A/C (override with LIMIT=)
POLICY_KIND="${POLICY_KIND:-openai}"   # openai (generic endpoint) | vllm (local GPU)
BACKEND="playwright"                   # browser backend; Stage C flips this to lexmount

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

require_gym() {
  have gym || die "the 'gym' CLI is not on PATH. Install NeMo-Gym first:
    cd $REPO_ROOT && uv venv --python 3.12 && source .venv/bin/activate && uv pip install -e ."
}

# --- Stage: standalone backend test (no GPU, no serving stack) -------------- #
run_backend_test() {
  log "Backend test — headless Chromium against the bundled offline site/"
  have uv || die "uv is required for the standalone backend test (https://docs.astral.sh/uv/)."
  ( cd "$SCRIPT_DIR"
    uv run --no-project --with playwright python -m playwright install chromium
    # Chromium needs system libraries (libnss3, libgbm1, ...). On a bare
    # container/VM the install above warns 'Host system is missing dependencies
    # to run browsers' and the test below fails to launch. Install them via apt
    # when we can (root, or passwordless sudo); otherwise leave a clear hint.
    if [[ "$(id -u)" == "0" ]]; then
      uv run --no-project --with playwright python -m playwright install-deps chromium
    elif have sudo && sudo -n true 2>/dev/null; then
      sudo env "PATH=$PATH" "$(command -v uv)" run --no-project --with playwright python -m playwright install-deps chromium
    else
      echo "NOTE: skipping browser system deps (need root/sudo). If the test below fails" >&2
      echo "      with 'Host system is missing dependencies', run:" >&2
      echo "      sudo \$(command -v uv) run --no-project --with playwright python -m playwright install-deps chromium" >&2
    fi
    uv run --no-project --with playwright --with pytest --with pytest-asyncio \
      python -m pytest tests/test_backend.py -q )
}

# --- Stage A / C: serving-stack rollout ------------------------------------- #
SERVER_PID=""
cleanup() {
  [[ -n "$SERVER_PID" ]] || return 0
  # `gym env start` spawns one subprocess per server (plus Ray infra); they do
  # not reliably die with the parent, so sweep a few levels of descendants.
  local pids="$SERVER_PID" frontier="$SERVER_PID" next="" p
  for _ in 1 2 3; do
    next=""
    for p in $frontier; do
      next="$next $(pgrep -P "$p" 2>/dev/null || true)"
    done
    frontier="$next"
    pids="$pids $next"
  done
  # shellcheck disable=SC2086
  kill $pids 2>/dev/null || true
}
trap cleanup EXIT

policy_flags() {
  case "$POLICY_KIND" in
    openai)
      [[ -n "${POLICY_BASE_URL:-}" ]] || die "POLICY_BASE_URL is unset. Export POLICY_BASE_URL / POLICY_API_KEY / POLICY_MODEL, or set POLICY_KIND=vllm."
      [[ -n "${POLICY_MODEL:-}"    ]] || die "POLICY_MODEL is unset (e.g. gpt-4.1, or your served model name)."
      printf -- '--model-type openai_model --model %q --model-url %q --model-api-key %q' \
        "$POLICY_MODEL" "$POLICY_BASE_URL" "${POLICY_API_KEY:-EMPTY}"
      ;;
    vllm)
      [[ -n "${POLICY_MODEL:-}" ]] || die "POLICY_MODEL is unset (HF id or local checkpoint path to serve with vLLM)."
      printf -- '--model-type vllm_model --model %q' "$POLICY_MODEL"
      ;;
    *) die "unknown POLICY_KIND=$POLICY_KIND (want: openai | vllm)";;
  esac
}

run_rollout() {
  require_gym
  run_backend_test

  local backend_override=()
  local input="$ENV_REL/data/example.jsonl"
  local out="$ENV_REL/data/example_rollouts.jsonl"
  if [[ "$BACKEND" == "lexmount" ]]; then
    for v in LEXMOUNT_API_KEY LEXMOUNT_PROJECT_ID LEXMOUNT_BASE_URL; do
      [[ -n "${!v:-}" ]] || die "backend: lexmount needs $v exported (see README 'Using the Lexmount cloud backend'). The SDK must also be installed in the server venv: pip install lexmount."
    done
    # The Lexmount SDK is imported by the RESOURCES-SERVER process, which runs in
    # its own venv at resources_servers/lexmount_browser/.venv (built by the first
    # `gym env start`) — NOT the repo-root venv. Fail fast with the exact fix
    # instead of a mid-rollout HTTP 500.
    local server_venv="$SCRIPT_DIR/.venv"
    if [[ -x "$server_venv/bin/python" ]]; then
      "$server_venv/bin/python" -c 'import lexmount' 2>/dev/null || die "the Lexmount SDK is not installed in the SERVER venv. Run:
    uv pip install --python $server_venv/bin/python 'lexmount>=0.5.13'"
    else
      die "the per-server venv ($server_venv) does not exist yet. Run Stage A once first
    (bash example.sh rollout) so 'gym env start' builds it, then install the SDK into it:
    uv pip install --python $server_venv/bin/python 'lexmount>=0.5.13'"
    fi
    # One Hydra override flips the browser backend; tools/observation/reward are
    # identical. The leading '+' is required: CLI overrides are parsed against an
    # empty Hydra struct, so a bare key fails with "Key 'lexmount_browser' is not
    # in struct"; '+' appends it and it then wins the merge with the yaml config.
    backend_override=("+lexmount_browser.resources_servers.lexmount_browser.backend=lexmount")
    # The bundled offline tasks are file:// URIs on THIS machine — the cloud
    # browser runs elsewhere and cannot load them (net::ERR_BLOCKED_BY_ADMINISTRATOR).
    # Stage C therefore rolls out on the 3 bundled real-web WebVoyager sample
    # tasks, which already ship in this env's input format (conservative
    # url_contains reward; see README "Data").
    input="$ENV_REL/data/webvoyager_sample.jsonl"
    out="$ENV_REL/data/webvoyager_rollouts.jsonl"
    log "Stage C — rollout with the Lexmount cloud backend (real-web sample tasks)"
  else
    log "Stage A — rollout with the local Playwright backend"
  fi
  local server_log; server_log="$(mktemp -t lexmount_browser_env_start.XXXXXX.log)"
  # (out/input were chosen above; Stage C writes webvoyager_rollouts.jsonl)
  # NOTE: run in the main shell (not a subshell) so SERVER_PID is visible to the
  # EXIT trap — otherwise the serving stack outlives the script and a re-run
  # (e.g. Stage A then Stage C) talks to the stale stack with the old backend.
  cd "$REPO_ROOT"
  log "Starting the Gym serving stack (resources server + policy model)"
  # Word-splitting of policy_flags is intended (they are CLI flags). First run
  # builds a per-server venv (a few minutes); reruns reuse it.
  # shellcheck disable=SC2046
  gym env start --resources-server lexmount_browser $(policy_flags) \
    ${backend_override[@]+"${backend_override[@]}"} >"$server_log" 2>&1 &
  SERVER_PID=$!

  log "Waiting for servers to report ready (log: $server_log)"
  # env start prints 'All N / N servers ready!' once every server is up. The
  # pattern must NOT match the interim '0 / 3 servers ready. Waiting...' lines,
  # or eval starts before the stack is up and spins on connection errors.
  local ready=""
  for _ in $(seq 1 180); do
    kill -0 "$SERVER_PID" 2>/dev/null || { tail -30 "$server_log" >&2; die "server process exited before becoming ready"; }
    if grep -qE "All [0-9]+ / [0-9]+ servers ready" "$server_log" 2>/dev/null; then ready=1; break; fi
    sleep 5
  done
  [[ -n "$ready" ]] || die "timed out waiting for servers (see $server_log)"

  log "Collecting $LIMIT rollout(s) over $input"
  gym eval run --no-serve --agent lexmount_browser_simple_agent \
    --input "$input" \
    --output "$out" --limit "$LIMIT"
  log "Rollouts written to $out"
  echo "First rollout reward:"
  cd "$REPO_ROOT" && head -1 "$out" | python3 -c 'import sys,json; print("  reward =", json.loads(sys.stdin.readline()).get("reward"))' || true
}

# --- Stage B: GRPO training smoke (1 GPU, via NeMo-RL) ---------------------- #
run_train() {
  cat >&2 <<EOF

Stage B runs a single-GPU GRPO training smoke through NeMo-RL (not from this repo).
It is intentionally NOT auto-invoked, because it needs a GPU and a NeMo-RL checkout.

  1. Clone NeMo-RL and mount this Gym checkout at 3rdparty/Gym-workspace/Gym
     (see https://docs.nvidia.com/nemo/gym/latest/ training tutorials).
  2. Download the smoke policy model (Qwen/Qwen3-4B) into your HF cache.
  3. Launch with the committed smoke config:

     CUDA_VISIBLE_DEVICES=0 HF_HOME=\$PWD/.cache/ \\
       uv run python examples/nemo_gym/run_grpo_nemo_gym.py \\
         --config=3rdparty/Gym-workspace/Gym/$ENV_REL/configs/grpo_lexmount_browser_smoke.yaml \\
         ++grpo.max_num_steps=3 \\
         cluster.gpus_per_node=1

The smoke config ports SXH's validated 0721 Qwen3-8B 2x8-NPU hyperparameters
(reward 0.10 -> 0.29 over 60 steps) scaled to one GPU; each value is annotated
with its provenance in configs/grpo_lexmount_browser_smoke.yaml.
EOF
}

# --- arg parsing ------------------------------------------------------------ #
CMD="${1:-rollout}"; shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND="${2:?--backend needs a value (playwright|lexmount)}"; shift 2;;
    --limit)   LIMIT="${2:?--limit needs a value}"; shift 2;;
    *) die "unknown argument: $1";;
  esac
done

case "$CMD" in
  test)    run_backend_test;;
  rollout) run_rollout;;
  train)   run_train;;
  *) die "unknown command: $CMD (want: rollout | test | train)";;
esac

log "Done: $CMD"
