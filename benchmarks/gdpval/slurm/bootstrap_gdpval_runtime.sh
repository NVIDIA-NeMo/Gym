#!/usr/bin/env bash
set -euo pipefail

# Build isolated, reproducible component environments for the AfterQuery
# GDPVal rollout without modifying the concurrent gym-lj checkout.

umask 077

CHECKOUT="${GDPVAL_SLURM_CHECKOUT:-${GDPVAL_SLURM_STAGING_ROOT:-${SCRATCH:-$HOME}/gdpval-rollouts}/checkout}"
BOOT_ROOT="${GDPVAL_BOOT_ROOT:-${GDPVAL_SLURM_STAGING_ROOT:-${SCRATCH:-$HOME}/gdpval-rollouts}/bootstrap}"
REFERENCE="${GDPVAL_REFERENCE_CHECKOUT:-}"
UV_BIN="${GDPVAL_UV_BIN:-${CHECKOUT}/.venv/bin/uv}"
PYTHON_BIN="${GDPVAL_BOOTSTRAP_PYTHON:-/cm/local/apps/python3/bin/python3.12}"
CONSTRAINTS_DIR="${BOOT_ROOT}/constraints"
VENV_ROOT="${BOOT_ROOT}/venvs"
UV_CACHE_DIR="${BOOT_ROOT}/uv-cache"
READY_MARKER="${BOOT_ROOT}/READY"

COMPONENTS=(
  responses_api_agents/stirrup_agent
  responses_api_models/vllm_model
  responses_api_models/openai_model
  resources_servers/gdpval
)

usage() {
  cat <<'EOF'
Usage: bootstrap_gdpval_runtime.sh ACTION

Actions:
  bootstrap  Snapshot known-good constraints and build four isolated venvs.
  validate   Verify imports and the recorded READY marker.
EOF
}

preflight() {
  [[ -x "${UV_BIN}" ]] || { echo "ERROR: uv not executable: ${UV_BIN}" >&2; return 1; }
  [[ -x "${PYTHON_BIN}" ]] || { echo "ERROR: Python not executable: ${PYTHON_BIN}" >&2; return 1; }
  [[ -f "${CHECKOUT}/pyproject.toml" ]] || { echo "ERROR: invalid checkout: ${CHECKOUT}" >&2; return 1; }
  local component
  for component in "${COMPONENTS[@]}"; do
    [[ -d "${CHECKOUT}/${component}" ]] || { echo "ERROR: missing checkout component: ${component}" >&2; return 1; }
    [[ -x "${REFERENCE}/${component}/.venv/bin/python" ]] || {
      echo "ERROR: missing reference component venv: ${REFERENCE}/${component}/.venv" >&2
      return 1
    }
  done
}

constraint_path() {
  local component="$1"
  printf '%s/%s.txt' "${CONSTRAINTS_DIR}" "${component//\//__}"
}

target_python() {
  local component="$1"
  printf '%s/%s/.venv/bin/python' "${VENV_ROOT}" "${component}"
}

snapshot_constraints() {
  local component constraints
  for component in "${COMPONENTS[@]}"; do
    constraints="$(constraint_path "${component}")"
    "${UV_BIN}" pip freeze \
      --exclude-editable \
      --python "${REFERENCE}/${component}/.venv/bin/python" \
      >"${constraints}"
    [[ -s "${constraints}" ]] || { echo "ERROR: empty constraints for ${component}" >&2; return 1; }
  done
}

create_component_venv() {
  local component="$1"
  local target="${VENV_ROOT}/${component}/.venv"
  mkdir -p "$(dirname "${target}")"
  UV_CACHE_DIR="${UV_CACHE_DIR}" "${UV_BIN}" venv \
    --seed \
    --allow-existing \
    --python "${PYTHON_BIN}" \
    "${target}"
}

install_requirements_component() {
  local component="$1"
  local python constraints
  python="$(target_python "${component}")"
  constraints="$(constraint_path "${component}")"
  create_component_venv "${component}"
  (
    cd "${CHECKOUT}/${component}"
    UV_CACHE_DIR="${UV_CACHE_DIR}" "${UV_BIN}" pip install \
      --python "${python}" \
      --constraints "${constraints}" \
      -r requirements.txt \
      'ray[default]==2.55.1' \
      'openai==2.7.2'
  )
}

install_vllm_proxy() {
  local component="responses_api_models/vllm_model"
  local python constraints
  python="$(target_python "${component}")"
  constraints="$(constraint_path "${component}")"
  create_component_venv "${component}"
  (
    cd "${CHECKOUT}/${component}"
    UV_CACHE_DIR="${UV_CACHE_DIR}" "${UV_BIN}" pip install \
      --python "${python}" \
      --constraints "${constraints}" \
      -e . \
      'ray[default]==2.55.1' \
      'openai==2.7.2'
  )
}

write_ready_marker() {
  local tmp="${READY_MARKER}.tmp.$$"
  {
    echo "created_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "checkout_revision=$(git -C "${CHECKOUT}" rev-parse HEAD)"
    echo "uv_version=$("${UV_BIN}" --version)"
    echo "python_version=$("${PYTHON_BIN}" --version 2>&1)"
    echo "venv_root=${VENV_ROOT}"
    echo "uv_cache_dir=${UV_CACHE_DIR}"
    sha256sum "${CONSTRAINTS_DIR}"/*.txt
  } >"${tmp}"
  chmod 600 "${tmp}"
  mv "${tmp}" "${READY_MARKER}"
}

validate_runtime() {
  [[ -f "${READY_MARKER}" ]] || { echo "ERROR: runtime marker not found: ${READY_MARKER}" >&2; return 1; }
  local agent_python proxy_python judge_python resource_python
  agent_python="$(target_python responses_api_agents/stirrup_agent)"
  proxy_python="$(target_python responses_api_models/vllm_model)"
  judge_python="$(target_python responses_api_models/openai_model)"
  resource_python="$(target_python resources_servers/gdpval)"

  "${agent_python}" -c 'import nemo_gym, ray, stirrup, transformers'
  "${proxy_python}" -c 'import nemo_gym, openai, ray'
  "${judge_python}" -c 'import nemo_gym, openai, ray'
  "${resource_python}" -c 'import nemo_gym, openai, ray'
  "${agent_python}" -c \
    'import importlib.metadata as m; print("stirrup=" + m.version("stirrup")); print("transformers=" + m.version("transformers")); print("ray=" + m.version("ray")); print("openai=" + m.version("openai"))'
  echo "Runtime validation passed: ${READY_MARKER}"
}

bootstrap() {
  preflight
  mkdir -p "${CONSTRAINTS_DIR}" "${VENV_ROOT}" "${UV_CACHE_DIR}"
  export UV_PYTHON_DOWNLOADS=never
  export UV_LINK_MODE=copy
  snapshot_constraints
  install_requirements_component responses_api_agents/stirrup_agent
  install_vllm_proxy
  install_requirements_component responses_api_models/openai_model
  install_requirements_component resources_servers/gdpval
  write_ready_marker
  validate_runtime
}

case "${1:-validate}" in
  bootstrap) bootstrap ;;
  validate) preflight; validate_runtime ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
