#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

cd "${GYM_DIR}"
export RAY_TMPDIR SWE_ATLAS_QNA_JUDGE_BASE_URL SWE_ATLAS_QNA_JUDGE_API_KEY SWE_ATLAS_QNA_JUDGE_MODEL OPENAI_API_KEY

echo "== Node =="
hostname

echo "== Gym container =="
echo "GYM_IMAGE=${GYM_IMAGE}"
echo "VLLM_IMAGE=${VLLM_IMAGE}"
echo "POLICY_BASE_URL=${POLICY_BASE_URL}"
echo "VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS}"

echo "== Slurm =="
command -v srun

echo "== Host Apptainer smoke test =="
srun --overlap bash -s <<EOF
set -euo pipefail
export HOME="${WORKSPACE_DIR}"
export TMPDIR="${TMPDIR}"
export APPTAINER_CACHEDIR="${HOST_APPTAINER_CACHEDIR}"
export APPTAINER_TMPDIR="${HOST_APPTAINER_TMPDIR}"
mkdir -p "\${HOME}/.apptainer/instances" "\${TMPDIR}" "\${APPTAINER_CACHEDIR}" "\${APPTAINER_TMPDIR}"
if [[ -x "${HOST_APPTAINER_BIN}" ]]; then
  export PATH="$(dirname "${HOST_APPTAINER_BIN}"):\${PATH}"
else
  echo "HOST_APPTAINER_BIN=${HOST_APPTAINER_BIN} is not executable; using PATH apptainer." >&2
fi
echo "HOME=\${HOME}"
echo "apptainer=\$(command -v apptainer || true)"
apptainer --version
mapfile -t SIFS < <(compgen -G "${IMAGES_DIR}/*.sif")
if [[ "\${#SIFS[@]}" -eq 0 ]]; then
  echo "No .sif images found under ${IMAGES_DIR}" >&2
  exit 2
fi
TEST_SIF="\${SIFS[0]}"
echo "test_sif=\${TEST_SIF}"
echo "-- direct exec --"
apptainer exec "\${TEST_SIF}" sh -lc 'id && printf "apptainer-direct-ok\n"'
echo "-- instance exec --"
STAGING_DIR="\$(mktemp -d -p "\${TMPDIR}" nemo-gym-apptainer-smoke-XXXXXX)"
INSTANCE_NAME="nemo-gym-smoke-\${USER:-user}-\$\$"
cleanup() {
  apptainer instance stop "\${INSTANCE_NAME}" >/dev/null 2>&1 || true
  rm -rf "\${STAGING_DIR}"
}
trap cleanup EXIT
apptainer instance start --bind "\${STAGING_DIR}:/sandbox" "\${TEST_SIF}" "\${INSTANCE_NAME}"
apptainer exec "instance://\${INSTANCE_NAME}" sh -lc 'printf "apptainer-sandbox-ready\n"'
EOF

echo "== Gym container =="
srun --overlap \
  --container-image "${GYM_IMAGE}" \
  --container-mounts "${CONTAINER_MOUNTS}" \
  --container-workdir "${GYM_CONTAINER_DIR}" \
  bash -lc 'cd "${GYM_CONTAINER_DIR}" && source .venv/bin/activate && echo "python=$(python --version)" && pwd && if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; else echo "nvidia-smi is not available in GYM_IMAGE; this is OK because vLLM runs in VLLM_IMAGE."; fi && apptainer --version'

echo "== vLLM container GPU visibility =="
srun --overlap \
  --container-image "${VLLM_IMAGE}" \
  --container-mounts "${CONTAINER_MOUNTS}" \
  --container-workdir "${GYM_CONTAINER_DIR}" \
  nvidia-smi

echo "== Required paths =="
python - <<'PY2'
import os
from pathlib import Path
for name in ["POLICY_CKPT", "IMAGES_DIR", "SMOKE_PATH"]:
    p = Path(os.environ[name])
    print(f"{name}: {p} exists={p.exists()}")
    if not p.exists():
        raise SystemExit(f"Missing required path: {name}={p}")
for name in ["GYM_IMAGE", "VLLM_IMAGE"]:
    p = Path(os.environ[name])
    print(f"{name}: {p} exists={p.exists()}")
    if not p.exists():
        raise SystemExit(f"Missing required image: {name}={p}")
PY2

if [[ -z "${SWE_ATLAS_QNA_JUDGE_API_KEY}" ]]; then
  echo "SWE_ATLAS_QNA_JUDGE_API_KEY is empty. Export NVIDIA_API_KEY or edit config.env." >&2
  exit 2
fi

echo "== Judge connectivity =="
curl -sS --max-time 30 "${SWE_ATLAS_QNA_JUDGE_BASE_URL}/chat/completions" \
  -H "Authorization: Bearer ${SWE_ATLAS_QNA_JUDGE_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${SWE_ATLAS_QNA_JUDGE_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: OK\"}],\"max_tokens\":16,\"temperature\":0}"
echo
