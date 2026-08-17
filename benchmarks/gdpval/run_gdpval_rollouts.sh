#!/usr/bin/env bash
set -euo pipefail

# Smoke, collect, and validate a GDPVal-format batch without judging. The dataset
# is selected by a profile under benchmarks/gdpval/datasets/ (task-id pattern,
# row count, reference locator mode, ...). Run from a cluster checkout with a
# live model endpoint and the GDPVal container image.

umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -n "${GDPVAL_ENV_FILE:-}" ]]; then
  if [[ ! -r "${GDPVAL_ENV_FILE}" ]]; then
    echo "ERROR: GDPVAL_ENV_FILE is not readable: ${GDPVAL_ENV_FILE}" >&2
    exit 2
  fi
  set -a
  # shellcheck disable=SC1090
  source "${GDPVAL_ENV_FILE}"
  set +a
fi

ACTION="${1:-local-check}"
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  DEFAULT_PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
else
  DEFAULT_PYTHON_BIN="python3"
fi
if [[ -x "${REPO_ROOT}/.venv/bin/gym" ]]; then
  DEFAULT_GYM_BIN="${REPO_ROOT}/.venv/bin/gym"
else
  DEFAULT_GYM_BIN="gym"
fi
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"
GYM_BIN="${GYM_BIN:-${DEFAULT_GYM_BIN}}"
APPTAINER_BIN="${GDPVAL_APPTAINER_BIN:-}"
SOURCE_INPUT="${GDPVAL_INPUT_JSONL:-${REPO_ROOT}/benchmarks/gdpval/data/gdpval_benchmark.jsonl}"
# `-` not `:-`: a dataset with no prompt-backed repairs sets this
# to the empty string, which must be honoured rather than falling back.
REFERENCE_OVERRIDES="${GDPVAL_REFERENCE_OVERRIDES-}"
# Dataset profile. Nothing here has a site-specific default: a plausible-looking
# one would silently validate a run against the wrong batch. Source a
# GDPVAL_ENV_FILE (see datasets/example.env) to describe your batch. The SHA and
# count pins are optional -- unset means "do not check".
EXPECTED_SOURCE_SHA256="${GDPVAL_EXPECTED_SOURCE_SHA256-}"
EXPECTED_LAUNCH_SHA256="${GDPVAL_EXPECTED_LAUNCH_SHA256-}"
EXPECTED_COUNT="${GDPVAL_EXPECTED_COUNT-}"
TASK_ID_PATTERN="${GDPVAL_TASK_ID_PATTERN-}"
REFERENCE_MODE="${GDPVAL_REFERENCE_MODE:-https}"
REQUIRE_PRIVATE_FILES="${GDPVAL_REQUIRE_PRIVATE_FILES:-true}"
EXPECTED_GDPVAL_CONTAINER_SHA256="${GDPVAL_EXPECTED_CONTAINER_SHA256-}"
RUN_TAG="${GDPVAL_RUN_TAG:-run}"
STAGING_ROOT="${GDPVAL_SLURM_STAGING_ROOT:-${SCRATCH:-$HOME}/gdpval-rollouts}"
RUN_ROOT="${GDPVAL_RUN_ROOT:-${STAGING_ROOT}/${RUN_TAG}}"
INPUT_DIR="${RUN_ROOT}/input"
RESULTS_DIR="${RUN_ROOT}/results"
LOG_DIR="${RUN_ROOT}/logs"
DELIVERABLES_DIR="${PERSIST_DELIVERABLES_DIR:-${RUN_ROOT}/deliverables}"
SMOKE_DELIVERABLES_DIR="${GDPVAL_SMOKE_DELIVERABLES_DIR:-${RUN_ROOT}/smoke_deliverables}"
LAUNCH_INPUT="${GDPVAL_LAUNCH_INPUT:-${INPUT_DIR}/$(basename "${SOURCE_INPUT}" .jsonl).launch.jsonl}"
SMOKE_TASK_ID="${GDPVAL_SMOKE_TASK_ID-}"
SMOKE_INPUT="${INPUT_DIR}/smoke_${SMOKE_TASK_ID}.jsonl"
SMOKE_OUTPUT="${RESULTS_DIR}/smoke_${SMOKE_TASK_ID}.jsonl"
SMOKE_MATERIALIZED_INPUT="${RESULTS_DIR}/smoke_${SMOKE_TASK_ID}_materialized_inputs.jsonl"
FULL_OUTPUT="${RESULTS_DIR}/rollouts.jsonl"
FULL_MATERIALIZED_INPUT="${RESULTS_DIR}/rollouts_materialized_inputs.jsonl"
FULL_FAILURES="${RESULTS_DIR}/rollouts_failures.jsonl"
CONCURRENCY="${GDPVAL_CONCURRENCY:-16}"
HEAD_PORT="${GDPVAL_HEAD_PORT:-22135}"
PORT_RANGE_LOW="${GDPVAL_PORT_RANGE_LOW:-22136}"
PORT_RANGE_HIGH="${GDPVAL_PORT_RANGE_HIGH:-22999}"
SERVER_WAIT_SECONDS="${GDPVAL_SERVER_WAIT_SECONDS:-900}"
# scripts/wait_for_servers.sh waits in two phases and only the second one takes
# SERVER_WAIT_SECONDS. The first - head server up, which is gated on Ray's API
# server - is capped by HEAD_MAX_WAIT, whose own default is 180s, so raising
# GDPVAL_SERVER_WAIT_SECONDS did nothing for the phase that actually times out.
# On a node already serving a large policy model Ray can exceed 180s, and the
# whole allocation is then discarded after a two-line log. Give both phases the
# same budget by default, and keep a separate knob for tuning this one alone.
HEAD_WAIT_SECONDS="${GDPVAL_HEAD_WAIT_SECONDS:-${SERVER_WAIT_SECONDS}}"
MODEL_TYPE="${GDPVAL_MODEL_TYPE:-vllm_model}"
MODEL_PATH="${GDPVAL_MODEL_PATH-}"
# 250 matches the GDPVal config default (benchmarks/gdpval/config.yaml); the
# previous 100 was a batch-specific narrowing. Per-turn token settings already
# match: --max-output-tokens is the full 262144 context, and
# max_completion_tokens_cap=64000 is the agent default.
AGENT_MAX_TURNS="${GDPVAL_AGENT_MAX_TURNS:-250}"
COMPLETION_TOKEN_BUFFER="${GDPVAL_COMPLETION_TOKEN_BUFFER:-5000}"
MAX_COMPLETION_TOKENS_CAP="${GDPVAL_MAX_COMPLETION_TOKENS_CAP:-64000}"
CONTEXT_WINDOW="${GDPVAL_CONTEXT_WINDOW:-262144}"
TEMPERATURE="${GDPVAL_TEMPERATURE:-1.0}"
TOP_P="${GDPVAL_TOP_P:-0.95}"
ENABLE_THINKING="${GDPVAL_ENABLE_THINKING:-true}"
# A prebuilt component-venv tree with a READY marker, laid down out of band. Point
# GDPVAL_COMPONENT_VENV_ROOT and GDPVAL_RUNTIME_READY_MARKER at yours; the runner only
# reads them, it never creates them.
BOOT_ROOT="${GDPVAL_BOOT_ROOT:-${STAGING_ROOT}/bootstrap}"
COMPONENT_VENV_ROOT="${GDPVAL_COMPONENT_VENV_ROOT:-${BOOT_ROOT}/venvs}"
UV_CACHE_DIR="${GDPVAL_UV_CACHE_DIR:-${BOOT_ROOT}/uv-cache}"
RUNTIME_READY_MARKER="${GDPVAL_RUNTIME_READY_MARKER:-${BOOT_ROOT}/READY}"
VALIDATOR="${REPO_ROOT}/benchmarks/gdpval/validate_gdpval_batch.py"
SERVER_PID=""
LOCK_DIR=""

# Emit only the profile knobs the dataset actually set. Passing an empty value
# is NOT the same as omitting the flag: `--expected-count ""` is an argparse
# error, and `--reference-overrides ""` resolves to the cwd.
profile_args() {
  local -a a=()
  [[ -n "${EXPECTED_COUNT}" ]] && a+=(--expected-count "${EXPECTED_COUNT}")
  [[ -n "${TASK_ID_PATTERN}" ]] && a+=(--task-id-pattern "${TASK_ID_PATTERN}")
  [[ -n "${REFERENCE_MODE}" ]] && a+=(--reference-mode "${REFERENCE_MODE}")
  [[ -n "${REFERENCE_OVERRIDES}" ]] && a+=(--reference-overrides "${REFERENCE_OVERRIDES}")
  printf '%s\n' ${a[@]+"${a[@]}"}
}

usage() {
  cat <<'EOF'
Usage: run_gdpval_rollouts.sh ACTION

Actions:
  local-check  Validate the immutable local source and build no remote state.
  preflight    Validate runtime variables, stage launch/smoke inputs, and check URLs.
  smoke        Start Gym servers and run only GDPVAL_SMOKE_TASK_ID.
  run          Run a validated smoke, then every task in the dataset.
  full         Run every task directly, skipping the serial smoke.
  resume       Resume the full output/cache after an interrupted run.
  validate     Validate any existing smoke/full artifacts without launching models.

Set GDPVAL_ENV_FILE to a private (chmod 600) environment file based on
benchmarks/gdpval/datasets/example.env. No judge credentials are used:
the server is always started with EXECUTE_ONLY=true.
EOF
}

cleanup() {
  stop_servers
  if [[ -n "${LOCK_DIR}" && -d "${LOCK_DIR}" ]]; then
    rmdir "${LOCK_DIR}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

stop_servers() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Stopping gym env start (PID ${SERVER_PID})"
    # Gym handles SIGINT by shutting down its model, agent, and resource
    # children. A plain SIGTERM can orphan them and leave the fixed ports busy.
    kill -INT "${SERVER_PID}" 2>/dev/null || true
    local waited
    for waited in $(seq 1 30); do
      kill -0 "${SERVER_PID}" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "WARNING: Gym did not exit after SIGINT; sending SIGTERM" >&2
      kill -TERM "${SERVER_PID}" 2>/dev/null || true
      for waited in $(seq 1 10); do
        kill -0 "${SERVER_PID}" 2>/dev/null || break
        sleep 1
      done
    fi
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "WARNING: Gym did not exit after SIGTERM; sending SIGKILL" >&2
      kill -KILL "${SERVER_PID}" 2>/dev/null || true
    fi
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}

wait_for_head_port_free() {
  local waited
  for waited in $(seq 1 30); do
    if ! (echo >/dev/tcp/127.0.0.1/"${HEAD_PORT}") 2>/dev/null; then
      return
    fi
    sleep 1
  done
  echo "ERROR: head port ${HEAD_PORT} is still accepting connections after Gym shutdown" >&2
  return 1
}

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "ERROR: required variable ${name} is unset" >&2
    return 1
  fi
}

validate_integer() {
  local name="$1"
  local value="${!name}"
  if [[ ! "${value}" =~ ^[0-9]+$ ]] || (( value < 1 )); then
    echo "ERROR: ${name} must be a positive integer (got ${value})" >&2
    return 1
  fi
}

configure_apptainer() {
  if [[ -z "${APPTAINER_BIN}" ]]; then
    APPTAINER_BIN="$(command -v apptainer 2>/dev/null || true)"
  fi
  if [[ -z "${APPTAINER_BIN}" ]]; then
    echo "ERROR: set GDPVAL_APPTAINER_BIN to the absolute Apptainer executable path" >&2
    return 1
  fi
  if [[ "${APPTAINER_BIN}" != /* ]]; then
    echo "ERROR: GDPVAL_APPTAINER_BIN must be absolute (got ${APPTAINER_BIN})" >&2
    return 1
  fi
  if [[ ! -x "${APPTAINER_BIN}" ]]; then
    echo "ERROR: GDPVAL_APPTAINER_BIN is not executable: ${APPTAINER_BIN}" >&2
    return 1
  fi

  # The Stirrup provider invokes the literal command `apptainer`. Prepending
  # this directory makes the private binary available to every Gym child.
  export GDPVAL_APPTAINER_BIN="${APPTAINER_BIN}"
  export PATH="$(dirname "${APPTAINER_BIN}"):${PATH}"
}

local_check() {

  local lc_extra=() lc_priv=()
  if [[ -n "${REFERENCE_OVERRIDES}" ]]; then
    lc_extra=(--reference-overrides "${REFERENCE_OVERRIDES}")
  fi
  if [[ "${REQUIRE_PRIVATE_FILES}" == "true" ]]; then
    lc_priv=(--require-private-files)
  fi
  local -a lc_profile=()
  lc_profile=()
  while IFS= read -r _pa_line; do lc_profile+=("$_pa_line"); done < <(profile_args)
  local -a lc_sha=()
  [[ -n "${EXPECTED_SOURCE_SHA256}" ]] && lc_sha=(--expected-sha256 "${EXPECTED_SOURCE_SHA256}")
  "${PYTHON_BIN}" "${VALIDATOR}" \
    --input "${SOURCE_INPUT}" \
    ${lc_profile[@]+"${lc_profile[@]}"} \
    ${lc_sha[@]+"${lc_sha[@]}"} \
    ${lc_priv[@]+"${lc_priv[@]}"}
}

acquire_launch_lock() {
  mkdir -p "${RUN_ROOT}"
  local candidate_lock="${RUN_ROOT}/.launch.lock"
  if ! mkdir "${candidate_lock}" 2>/dev/null; then
    echo "ERROR: another launch may own ${candidate_lock}; inspect it before retrying" >&2
    exit 2
  fi
  LOCK_DIR="${candidate_lock}"
}

directory_has_entries() {
  local path="$1"
  [[ -d "${path}" && -n "$(find "${path}" -mindepth 1 -maxdepth 1 -print -quit)" ]]
}

require_fresh_run_state() {
  if directory_has_entries "${RESULTS_DIR}"; then
    echo "ERROR: fresh run requires an empty results directory: ${RESULTS_DIR}" >&2
    return 1
  fi
  if directory_has_entries "${DELIVERABLES_DIR}"; then
    echo "ERROR: fresh run requires an empty deliverables directory: ${DELIVERABLES_DIR}" >&2
    return 1
  fi
  if directory_has_entries "${SMOKE_DELIVERABLES_DIR}"; then
    echo "ERROR: fresh run requires an empty smoke deliverables directory: ${SMOKE_DELIVERABLES_DIR}" >&2
    return 1
  fi
}

require_fresh_smoke_state() {
  if [[ -e "${SMOKE_OUTPUT}" ]]; then
    echo "ERROR: smoke output already exists; validate it or use a fresh GDPVAL_RUN_ROOT: ${SMOKE_OUTPUT}" >&2
    return 1
  fi
  if directory_has_entries "${SMOKE_DELIVERABLES_DIR}"; then
    echo "ERROR: smoke deliverables already exist; use a fresh GDPVAL_RUN_ROOT: ${SMOKE_DELIVERABLES_DIR}" >&2
    return 1
  fi
}

runtime_preflight() {
  require_var POLICY_MODEL_NAME
  require_var POLICY_BASE_URL
  require_var POLICY_API_KEY
  require_var GDPVAL_CONTAINER_PATH
  if [[ "${GDPVAL_ALLOW_NO_TAVILY:-0}" != "1" ]]; then
    require_var TAVILY_API_KEY
    if [[ "${TAVILY_API_KEY}" == REPLACE_WITH_* ]]; then
      echo "ERROR: TAVILY_API_KEY still contains the template placeholder" >&2
      return 1
    fi
  fi
  validate_integer CONCURRENCY
  validate_integer HEAD_PORT
  validate_integer PORT_RANGE_LOW
  validate_integer PORT_RANGE_HIGH
  validate_integer AGENT_MAX_TURNS
  validate_integer COMPLETION_TOKEN_BUFFER
  validate_integer MAX_COMPLETION_TOKENS_CAP
  validate_integer CONTEXT_WINDOW
  if (( PORT_RANGE_LOW > PORT_RANGE_HIGH )); then
    echo "ERROR: GDPVAL_PORT_RANGE_LOW must not exceed GDPVAL_PORT_RANGE_HIGH" >&2
    return 1
  fi
  if [[ "${ENABLE_THINKING}" != "true" && "${ENABLE_THINKING}" != "false" ]]; then
    echo "ERROR: GDPVAL_ENABLE_THINKING must be true or false (got ${ENABLE_THINKING})" >&2
    return 1
  fi
  if [[ "${RUN_ROOT}" != /* || "${DELIVERABLES_DIR}" != /* || "${SMOKE_DELIVERABLES_DIR}" != /* ]]; then
    echo "ERROR: GDPVAL_RUN_ROOT and both deliverables directories must be absolute paths" >&2
    return 1
  fi
  if [[ ! -e "${MODEL_PATH}" ]]; then
    echo "ERROR: GLM tokenizer/checkpoint path not found: ${MODEL_PATH}" >&2
    return 1
  fi
  if [[ ! -f "${GDPVAL_CONTAINER_PATH}" ]]; then
    echo "ERROR: GDPVal Apptainer image not found: ${GDPVAL_CONTAINER_PATH}" >&2
    return 1
  fi
  if [[ ! -f "${RUNTIME_READY_MARKER}" ]]; then
    echo "ERROR: isolated component runtime is not ready: ${RUNTIME_READY_MARKER}" >&2
    return 1
  fi
  configure_apptainer
  command -v "${PYTHON_BIN}" >/dev/null
  command -v "${GYM_BIN}" >/dev/null
  command -v curl >/dev/null
  command -v sha256sum >/dev/null
  local container_sha
  container_sha="$(sha256sum "${GDPVAL_CONTAINER_PATH}" | awk '{print $1}')"
  if [[ -n "${EXPECTED_GDPVAL_CONTAINER_SHA256}" && "${container_sha}" != "${EXPECTED_GDPVAL_CONTAINER_SHA256}" ]]; then
    echo "ERROR: GDPVal container SHA-256 mismatch: expected ${EXPECTED_GDPVAL_CONTAINER_SHA256}, found ${container_sha}" >&2
    return 1
  fi
  "${APPTAINER_BIN}" inspect "${GDPVAL_CONTAINER_PATH}" >/dev/null

  mkdir -p \
    "${INPUT_DIR}" \
    "${RESULTS_DIR}" \
    "${LOG_DIR}" \
    "${DELIVERABLES_DIR}" \
    "${SMOKE_DELIVERABLES_DIR}"
  local -a validator_args=(--input "${SOURCE_INPUT}")
  local -a vp=()
  vp=()
  while IFS= read -r _pa_line; do vp+=("$_pa_line"); done < <(profile_args)
  validator_args+=(${vp[@]+"${vp[@]}"})
  [[ -n "${EXPECTED_SOURCE_SHA256}" ]] && validator_args+=(--expected-sha256 "${EXPECTED_SOURCE_SHA256}")
  validator_args+=(--write-launch-input "${LAUNCH_INPUT}")
  # A smoke id is optional: without one the launch input is still written, there
  # is just no one-row smoke file to write.
  if [[ -n "${SMOKE_TASK_ID}" ]]; then
    validator_args+=(--smoke-task-id "${SMOKE_TASK_ID}" --write-smoke-input "${SMOKE_INPUT}")
  fi
  validator_args+=(
    --check-model-endpoint
    --model-base-url "${POLICY_BASE_URL}"
    --model-name "${POLICY_MODEL_NAME}"
  )
  if [[ "${REQUIRE_PRIVATE_FILES}" == "true" ]]; then
    validator_args+=(--require-private-files)
  fi
  if [[ "${GDPVAL_SKIP_URL_CHECK:-0}" != "1" ]]; then
    validator_args+=(--check-reference-urls)
  fi
  "${PYTHON_BIN}" "${VALIDATOR}" "${validator_args[@]}"

  local launch_sha git_revision runner_sha validator_sha overrides_sha
  local fingerprint fingerprint_file manifest_path manifest_timestamp
  launch_sha="$(sha256sum "${LAUNCH_INPUT}" | awk '{print $1}')"
  if [[ -n "${EXPECTED_LAUNCH_SHA256}" && "${launch_sha}" != "${EXPECTED_LAUNCH_SHA256}" ]]; then
    echo "ERROR: launch SHA-256 mismatch: expected ${EXPECTED_LAUNCH_SHA256}, found ${launch_sha}" >&2
    return 1
  fi
  git_revision="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
  runner_sha="$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')"
  validator_sha="$(sha256sum "${VALIDATOR}" | awk '{print $1}')"
  # Datasets without prompt-backed repairs leave this unset; the
  # fingerprint records the absence explicitly rather than hashing "".
  if [[ -n "${REFERENCE_OVERRIDES}" ]]; then
    overrides_sha="$(sha256sum "${REFERENCE_OVERRIDES}" | awk '{print $1}')"
  else
    overrides_sha="none"
  fi
  fingerprint="$(
    {
      echo "source_sha256=${EXPECTED_SOURCE_SHA256}"
      echo "launch_sha256=${launch_sha}"
      echo "reference_overrides_sha256=${overrides_sha}"
      echo "runner_sha256=${runner_sha}"
      echo "validator_sha256=${validator_sha}"
      echo "policy_model_name=${POLICY_MODEL_NAME}"
      echo "model_type=${MODEL_TYPE}"
      echo "model_path=${MODEL_PATH}"
      echo "gdpval_container_sha256=${container_sha}"
      echo "agent_max_turns=${AGENT_MAX_TURNS}"
      echo "completion_token_buffer=${COMPLETION_TOKEN_BUFFER}"
      echo "max_completion_tokens_cap=${MAX_COMPLETION_TOKENS_CAP}"
      echo "context_window=${CONTEXT_WINDOW}"
      echo "temperature=${TEMPERATURE}"
      echo "top_p=${TOP_P}"
      echo "enable_thinking=${ENABLE_THINKING}"
      echo "num_repeats=1"
      echo "execute_only=true"
    } | sha256sum | awk '{print $1}'
  )"
  fingerprint_file="${RUN_ROOT}/run_config.sha256"
  if [[ -f "${fingerprint_file}" ]]; then
    if [[ "$(<"${fingerprint_file}")" != "${fingerprint}" ]]; then
      echo "ERROR: launch configuration changed for ${RUN_ROOT}; use a fresh run root" >&2
      return 1
    fi
  elif [[ "${ACTION}" == "resume" ]]; then
    echo "ERROR: resume requires the immutable launch fingerprint: ${fingerprint_file}" >&2
    return 1
  else
    printf '%s\n' "${fingerprint}" >"${fingerprint_file}"
    chmod 600 "${fingerprint_file}"
  fi

  manifest_timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  if [[ "${ACTION}" == "resume" ]]; then
    manifest_path="${RUN_ROOT}/resume_manifest_${manifest_timestamp}_$$.txt"
  else
    manifest_path="${RUN_ROOT}/run_manifest.txt"
  fi
  {
    echo "created_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "run_config_sha256=${fingerprint}"
    echo "git_revision=${git_revision}"
    echo "source_input=${SOURCE_INPUT}"
    echo "source_sha256=${EXPECTED_SOURCE_SHA256}"
    echo "launch_input=${LAUNCH_INPUT}"
    echo "launch_sha256=${launch_sha}"
    echo "reference_overrides=${REFERENCE_OVERRIDES}"
    echo "reference_overrides_sha256=${overrides_sha}"
    echo "runner_sha256=${runner_sha}"
    echo "validator_sha256=${validator_sha}"
    echo "policy_model_name=${POLICY_MODEL_NAME}"
    echo "policy_base_url=${POLICY_BASE_URL}"
    echo "server_job_id=${GDPVAL_SERVER_JOB_ID:-unknown}"
    echo "server_root=${GDPVAL_SERVER_ROOT:-unknown}"
    echo "model_type=${MODEL_TYPE}"
    echo "model_path=${MODEL_PATH}"
    echo "gdpval_container_path=${GDPVAL_CONTAINER_PATH}"
    echo "gdpval_container_sha256=${container_sha}"
    echo "apptainer_bin=${APPTAINER_BIN}"
    echo "smoke_deliverables_dir=${SMOKE_DELIVERABLES_DIR}"
    echo "full_deliverables_dir=${DELIVERABLES_DIR}"
    echo "component_venv_root=${COMPONENT_VENV_ROOT}"
    echo "uv_cache_dir=${UV_CACHE_DIR}"
    echo "runtime_ready_marker=${RUNTIME_READY_MARKER}"
    echo "execute_only=true"
    echo "rerun_incomplete=$([[ "${ACTION}" == "resume" ]] && echo true || echo false)"
    echo "num_repeats=1"
    echo "concurrency=${CONCURRENCY}"
    echo "agent_max_turns=${AGENT_MAX_TURNS}"
    echo "completion_token_buffer=${COMPLETION_TOKEN_BUFFER}"
    echo "max_completion_tokens_cap=${MAX_COMPLETION_TOKENS_CAP}"
    echo "context_window=${CONTEXT_WINDOW}"
    echo "temperature=${TEMPERATURE}"
    echo "top_p=${TOP_P}"
    echo "enable_thinking=${ENABLE_THINKING}"
    if [[ -n "${TAVILY_API_KEY:-}" ]]; then
      echo "tavily_configured=true"
    else
      echo "tavily_configured=false"
    fi
    echo "head_port=${HEAD_PORT}"
    echo "port_range=${PORT_RANGE_LOW}-${PORT_RANGE_HIGH}"
  } >"${manifest_path}"
  chmod 600 "${manifest_path}"

  echo "Preflight passed"
  echo "  run root: ${RUN_ROOT}"
  echo "  model: ${POLICY_MODEL_NAME}"
  echo "  model URL: ${POLICY_BASE_URL}"
  echo "  launch input: ${LAUNCH_INPUT}"
  echo "  deliverables: ${DELIVERABLES_DIR}"
  echo "  concurrency: ${CONCURRENCY}"
}

start_servers() {
  local persisted_deliverables_dir="$1"
  local rerun_incomplete="$2"
  export EXECUTE_ONLY=true
  export JUDGE_ONLY=false
  export RERUN_INCOMPLETE="${rerun_incomplete}"
  export PERSIST_DELIVERABLES_DIR="${persisted_deliverables_dir}"
  export GDPVAL_CONTAINER_PATH
  export GDPVAL_APPTAINER_BIN="${APPTAINER_BIN}"
  export PATH="$(dirname "${APPTAINER_BIN}"):${PATH}"
  export NEMO_GYM_MAX_ROLLOUT_ATTEMPTS="${NEMO_GYM_MAX_ROLLOUT_ATTEMPTS:-3}"
  export UV_CACHE_DIR
  # Read by scripts/wait_for_servers.sh for its head-server phase; see the
  # HEAD_WAIT_SECONDS definition above for why the 180s default is not enough.
  export HEAD_MAX_WAIT="${HEAD_WAIT_SECONDS}"
  if [[ "${GYM_BIN}" == */* ]]; then
    export PATH="$(dirname "${GYM_BIN}"):${PATH}"
  fi
  export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  # This delivery uses tokenized Firebase URLs plus two public paper URLs. The
  # downloader attaches HF_TOKEN to every HTTP host when it is present, so keep
  # unrelated Hugging Face credentials out of these requests by default.
  if [[ "${GDPVAL_KEEP_HF_TOKEN:-0}" != "1" ]]; then
    unset HF_TOKEN
  fi

  echo "Starting isolated Gym servers on head port ${HEAD_PORT} with deliverables ${persisted_deliverables_dir}"
  local policy_args=(
    '++policy_base_url=${oc.env:POLICY_BASE_URL}'
    '++policy_model_name=${oc.env:POLICY_MODEL_NAME}'
    '++policy_api_key=${oc.env:POLICY_API_KEY}'
  )
  # Non-interactive shells start asynchronous children with SIGINT ignored.
  # Reset it in the child before exec so stop_servers reaches Gym's graceful
  # KeyboardInterrupt cleanup path.
  (
    trap - INT
    exec "${GYM_BIN}" env start \
      --benchmark gdpval \
      --model-type "${MODEL_TYPE}" \
      "${policy_args[@]}" \
      ++head_server.host=127.0.0.1 \
      "++head_server.port=${HEAD_PORT}" \
      "++port_range_low=${PORT_RANGE_LOW}" \
      "++port_range_high=${PORT_RANGE_HIGH}" \
      "++uv_venv_dir=${COMPONENT_VENV_ROOT}" \
      "++uv_cache_dir=${UV_CACHE_DIR}" \
      ++skip_venv_if_present=true \
      ++uv_pip_set_python=true \
      "++gdpval_stirrup_agent.responses_api_agents.stirrup_agent.concurrency=${CONCURRENCY}" \
      "++gdpval_stirrup_agent.responses_api_agents.stirrup_agent.agent_max_turns=${AGENT_MAX_TURNS}" \
      "++gdpval_stirrup_agent.responses_api_agents.stirrup_agent.model_id=${MODEL_PATH}" \
      "++gdpval_stirrup_agent.responses_api_agents.stirrup_agent.completion_token_buffer=${COMPLETION_TOKEN_BUFFER}" \
      "++gdpval_stirrup_agent.responses_api_agents.stirrup_agent.max_completion_tokens_cap=${MAX_COMPLETION_TOKENS_CAP}" \
      "++gdpval_stirrup_agent.responses_api_agents.stirrup_agent.temperature=${TEMPERATURE}" \
      "++gdpval_stirrup_agent.responses_api_agents.stirrup_agent.top_p=${TOP_P}" \
      "++gdpval_stirrup_agent.responses_api_agents.stirrup_agent.enable_thinking=${ENABLE_THINKING}"
  ) >"${LOG_DIR}/gym_env_start.log" 2>&1 &
  SERVER_PID=$!
  "${REPO_ROOT}/scripts/wait_for_servers.sh" "${SERVER_PID}" "${HEAD_PORT}" "${SERVER_WAIT_SECONDS}"
}

collect() {
  local input_path="$1"
  local output_path="$2"
  local concurrency="$3"
  local resume="$4"
  local eval_args=(
    eval run --no-serve
    --agent gdpval_stirrup_agent
    --input "${input_path}"
    --output "${output_path}"
    --num-repeats 1
    --concurrency "${concurrency}"
    --temperature "${TEMPERATURE}"
    --max-output-tokens "${CONTEXT_WINDOW}"
  )
  if [[ "${resume}" == "true" ]]; then
    eval_args+=(--resume)
  fi
  eval_args+=(
    ++head_server.host=127.0.0.1
    "++head_server.port=${HEAD_PORT}"
    ++upload_rollouts_to_wandb=false
  )
  "${GYM_BIN}" "${eval_args[@]}"
}

validate_smoke() {
  local -a priv=() prof=() lsha=()
  prof=()
  while IFS= read -r _pa_line; do prof+=("$_pa_line"); done < <(profile_args)
  [[ -n "${EXPECTED_LAUNCH_SHA256}" ]] && lsha=(--expected-sha256 "${EXPECTED_LAUNCH_SHA256}")
  if [[ "${REQUIRE_PRIVATE_FILES}" == "true" ]]; then
    priv=(--require-private-files)
  fi
  "${PYTHON_BIN}" "${VALIDATOR}" \
    --input "${LAUNCH_INPUT}" \
    ${prof[@]+"${prof[@]}"} \
    ${lsha[@]+"${lsha[@]}"} \
    ${priv[@]+"${priv[@]}"} \
    ${SMOKE_TASK_ID:+--smoke-task-id "${SMOKE_TASK_ID}"} \
    --rollouts "${SMOKE_OUTPUT}" \
    --deliverables-dir "${SMOKE_DELIVERABLES_DIR}" \
    --expected-response-model "${GDPVAL_EXPECTED_RESPONSE_MODEL:-${POLICY_MODEL_NAME}}" \
    --require-deliverable
}

validate_full() {
  local -a priv=() prof=() lsha=()
  prof=()
  while IFS= read -r _pa_line; do prof+=("$_pa_line"); done < <(profile_args)
  [[ -n "${EXPECTED_LAUNCH_SHA256}" ]] && lsha=(--expected-sha256 "${EXPECTED_LAUNCH_SHA256}")
  if [[ "${REQUIRE_PRIVATE_FILES}" == "true" ]]; then
    priv=(--require-private-files)
  fi
  "${PYTHON_BIN}" "${VALIDATOR}" \
    --input "${LAUNCH_INPUT}" \
    ${prof[@]+"${prof[@]}"} \
    ${lsha[@]+"${lsha[@]}"} \
    ${priv[@]+"${priv[@]}"} \
    --rollouts "${FULL_OUTPUT}" \
    --deliverables-dir "${DELIVERABLES_DIR}" \
    --expected-response-model "${GDPVAL_EXPECTED_RESPONSE_MODEL:-${POLICY_MODEL_NAME}}"
}

run_smoke() {
  if [[ -f "${SMOKE_OUTPUT}" ]]; then
    echo "ERROR: refusing to reuse smoke output from an earlier launch: ${SMOKE_OUTPUT}" >&2
    return 1
  fi
  collect "${SMOKE_INPUT}" "${SMOKE_OUTPUT}" 1 false
  validate_smoke
}

prepare_resume() {
  if [[ -f "${FULL_OUTPUT}" && -f "${FULL_MATERIALIZED_INPUT}" ]]; then
    RESUME_FROM_CACHE=true
    return
  fi
  if [[ -e "${FULL_OUTPUT}" || -e "${FULL_MATERIALIZED_INPUT}" ]]; then
    echo "ERROR: resume requires both ${FULL_OUTPUT} and ${FULL_MATERIALIZED_INPUT}" >&2
    return 1
  fi
  if [[ -e "${FULL_FAILURES}" ]]; then
    echo "ERROR: refusing a fresh full collection with unmatched failure history: ${FULL_FAILURES}" >&2
    return 1
  fi
  if [[ ! -f "${SMOKE_OUTPUT}" || ! -f "${SMOKE_MATERIALIZED_INPUT}" ]]; then
    echo "ERROR: resume without full rollout state requires a completed smoke run" >&2
    return 1
  fi
  validate_smoke
  if directory_has_entries "${DELIVERABLES_DIR}"; then
    echo "ERROR: refusing a fresh full collection over an unmatched deliverables cache: ${DELIVERABLES_DIR}" >&2
    return 1
  fi
  RESUME_FROM_CACHE=false
}

case "${ACTION}" in
  local-check)
    local_check
    ;;
  preflight)
    runtime_preflight
    ;;
  smoke)
    acquire_launch_lock
    require_fresh_smoke_state
    runtime_preflight
    start_servers "${SMOKE_DELIVERABLES_DIR}" false
    run_smoke
    ;;
  run)
    acquire_launch_lock
    require_fresh_run_state
    runtime_preflight
    start_servers "${SMOKE_DELIVERABLES_DIR}" false
    run_smoke
    stop_servers
    wait_for_head_port_free
    start_servers "${DELIVERABLES_DIR}" false
    echo "Smoke passed; starting ${EXPECTED_COUNT:-all} tasks with concurrency ${CONCURRENCY}"
    collect "${LAUNCH_INPUT}" "${FULL_OUTPUT}" "${CONCURRENCY}" false
    validate_full
    ;;
  full)
    # Same fresh-state guards as `run`, without the serial smoke. The smoke
    # collects a single task at concurrency 1, which is the worst shape for this
    # MoE (memory-bandwidth bound, ~4-5 tok/s) and consumed two entire 16-node
    # allocations before any task finished. Collecting all tasks at
    # ${CONCURRENCY} batches the server properly and lets short tasks land even
    # when long ones do not; an interrupted run stays resumable via `resume`.
    acquire_launch_lock
    require_fresh_run_state
    runtime_preflight
    start_servers "${DELIVERABLES_DIR}" false
    echo "Skipping serial smoke; starting ${EXPECTED_COUNT:-all} tasks with concurrency ${CONCURRENCY}"
    collect "${LAUNCH_INPUT}" "${FULL_OUTPUT}" "${CONCURRENCY}" false
    validate_full
    ;;
  resume)
    acquire_launch_lock
    runtime_preflight
    prepare_resume
    start_servers "${DELIVERABLES_DIR}" "${RESUME_FROM_CACHE}"
    collect "${LAUNCH_INPUT}" "${FULL_OUTPUT}" "${CONCURRENCY}" "${RESUME_FROM_CACHE}"
    validate_full
    ;;
  validate)
    local_check
    if [[ -f "${SMOKE_OUTPUT}" || -f "${FULL_OUTPUT}" ]]; then
      require_var POLICY_MODEL_NAME
    fi
    if [[ -f "${SMOKE_OUTPUT}" ]]; then
      validate_smoke
    fi
    if [[ -f "${FULL_OUTPUT}" ]]; then
      validate_full
    fi
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
