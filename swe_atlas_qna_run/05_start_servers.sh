#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

cd "${GYM_DIR}"
mkdir -p "${RUN_DIR}" results
mkdir -p "${UV_CACHE_DIR}" "${TMPDIR}"
export RAY_TMPDIR UV_CACHE_DIR TMPDIR POLICY_BASE_URL POLICY_API_KEY POLICY_MODEL_NAME SWE_ATLAS_QNA_JUDGE_BASE_URL SWE_ATLAS_QNA_JUDGE_API_KEY SWE_ATLAS_QNA_JUDGE_MODEL OPENAI_API_KEY

if ! curl -fsS --max-time 5 "${POLICY_BASE_URL}/models" >/dev/null; then
  echo "Policy vLLM endpoint is not ready at ${POLICY_BASE_URL}/models." >&2
  echo "Run ${SCRIPT_DIR}/05_start_vllm.sh first, then retry." >&2
  exit 2
fi

python - <<'PY'
from pathlib import Path

config_path = Path("resources_servers/swe_atlas_qna/configs/swe_atlas_qna.yaml")
text = config_path.read_text()
text = text.replace(
    "judge_system_prompt_path: resources_servers/swe_atlas_qna/prompts/judge_system.txt",
    "judge_system_prompt_path: prompts/judge_system.txt",
)
text = text.replace(
    "judge_user_template_path: resources_servers/swe_atlas_qna/prompts/judge_user_template.txt",
    "judge_user_template_path: prompts/judge_user_template.txt",
)
config_path.write_text(text)

agent_config_path = Path("responses_api_agents/mini_swe_agent_qna/configs/mini_swe_agent_qna.yaml")
text = agent_config_path.read_text()
old_create = """          create:
            mount_point: /sandbox
"""
new_create = """          create:
            mount_point: /sandbox
            apply_resource_limits: false
"""
if "apply_resource_limits:" not in text:
    text = text.replace(old_create, new_create)
agent_config_path.write_text(text)

mini_swe_config_path = Path("responses_api_agents/mini_swe_agent_qna/configs/mswea_qa_config.yaml")
text = mini_swe_config_path.read_text()
text = text.replace(
    "{{system}} {{release}} {{version}} {{machine}}",
    '{{system | default("Linux")}} {{release | default("")}} {{version | default("")}} {{machine | default("")}}',
)
mini_swe_config_path.write_text(text)
PY

if [[ -f "${SERVER_PID_FILE}" ]] && kill -0 "$(<"${SERVER_PID_FILE}")" 2>/dev/null; then
  echo "Servers already appear to be running with PID $(<"${SERVER_PID_FILE}")."
  echo "Log: ${SERVER_LOG}"
  exit 0
fi

if [[ "${RUN_GYM_IN_CONTAINER}" == "true" ]]; then
  START_BENCHMARK_CONFIG="${BENCHMARK_CONFIG_PATH}"
  START_MINI_SWE_CONFIG="${MINI_SWE_CONFIG_PATH}"
  START_WORKDIR="${GYM_CONTAINER_DIR}"
  START_HOME="${GYM_CONTAINER_HOME}"
  START_APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR}"
  START_APPTAINER_TMPDIR="${APPTAINER_TMPDIR}"
else
  START_BENCHMARK_CONFIG="${HOST_BENCHMARK_CONFIG_PATH}"
  START_MINI_SWE_CONFIG="${HOST_MINI_SWE_CONFIG_PATH}"
  START_WORKDIR="${GYM_DIR}"
  START_HOME="${WORKSPACE_DIR}"
  START_APPTAINER_CACHEDIR="${HOST_APPTAINER_CACHEDIR}"
  START_APPTAINER_TMPDIR="${HOST_APPTAINER_TMPDIR}"
  if [[ -x "${HOST_APPTAINER_BIN}" ]]; then
    export PATH="$(dirname "${HOST_APPTAINER_BIN}"):${PATH}"
  else
    echo "Warning: HOST_APPTAINER_BIN=${HOST_APPTAINER_BIN} is not executable; falling back to PATH." >&2
  fi
fi

args=(
  gym env start
  --config "${START_BENCHMARK_CONFIG}"
  --config "${POLICY_CONFIG_PATH}"
  --model "${POLICY_MODEL_NAME}"
  "+${AGENT_CONFIG_PATH}.image_template=\"${IMAGES_DIR}/{sif_basename}\""
  "++${AGENT_CONFIG_PATH}.mini_swe_config_path=${START_MINI_SWE_CONFIG}"
  "++${AGENT_CONFIG_PATH}.concurrency=${CONCURRENCY}"
)

if [[ "${SANDBOX_USER_NULL}" == "true" ]]; then
  args+=("++${AGENT_CONFIG_PATH}.sandbox_environment_kwargs.user=null")
fi
if [[ "${APPTAINER_WRITABLE_TMPFS}" == "true" ]]; then
  args+=("++${AGENT_CONFIG_PATH}.sandbox_provider.apptainer.exec={default_timeout_s:900,extra_exec_args:[--writable-tmpfs]}")
fi

echo "Starting Gym servers. Log: ${SERVER_LOG}"
debug_cmd="$(printf 'python -c %q' "import os, sys; from pathlib import Path; sys.path.insert(0, ${START_WORKDIR@Q}); print('pwd=' + os.getcwd()); print('sys_path_head=' + repr(sys.path[:5])); import responses_api_agents.mini_swe_agent_qna.app as app; p=Path(app.__file__); text=p.read_text(errors='replace'); print(f'mini_swe_agent_qna_app={p}'); print('mini_swe_agent_qna_has_verify_result_keys=' + str('verify_result_keys' in text)); print('mini_swe_agent_qna_has_minimal_verify_payload=' + str('NeMoGymResponseCreateParamsNonStreaming(input=[]).model_dump(mode=\"json\")' in text and 'NeMoGymResponse.model_validate(answer_response).model_dump(mode=\"json\")' in text))")"
prefix="$(printf 'mkdir -p %q %q %q/.apptainer/instances && export HOME=%q APPTAINER_CACHEDIR=%q APPTAINER_TMPDIR=%q PATH=%q && cd %q && source .venv/bin/activate && export PYTHONPATH=%q${PYTHONPATH:+:${PYTHONPATH}} && echo "HOME=${HOME}" && echo "PYTHONPATH=${PYTHONPATH}" && echo "apptainer=$(command -v apptainer)" && apptainer --version && ' "${START_APPTAINER_CACHEDIR}" "${START_APPTAINER_TMPDIR}" "${START_HOME}" "${START_HOME}" "${START_APPTAINER_CACHEDIR}" "${START_APPTAINER_TMPDIR}" "${PATH}" "${START_WORKDIR}" "${START_WORKDIR}")"
cmd="${prefix}${debug_cmd} && env PYTHONPATH=\"${START_WORKDIR}:\${PYTHONPATH:-}\" $(printf '%q ' "${args[@]}")"
if [[ "${RUN_GYM_IN_CONTAINER}" == "true" ]]; then
  srun \
    --overlap \
    --container-image "${GYM_IMAGE}" \
    --container-mounts "${CONTAINER_MOUNTS}" \
    --container-workdir "${GYM_CONTAINER_DIR}" \
    bash -lc "${cmd}" \
    > "${SERVER_LOG}" 2>&1 &
else
  srun --overlap bash -lc "${cmd}" > "${SERVER_LOG}" 2>&1 &
fi
echo $! > "${SERVER_PID_FILE}"
echo "Started PID $(<"${SERVER_PID_FILE}")"
echo "Watch with: ${SCRIPT_DIR}/06_check_status.sh --tail"
