#!/usr/bin/env bash
# Build the qualified vLLM 0.28 eval SQSH entirely from pinned source inputs.
# CI: changing this file rebuilds and publishes the eval SQSH from main.
#
# The script has two phases:
#   1. On the login node, fetch and pin the vLLM and NeMo Gym source trees.
#   2. Under Slurm, enter the official vLLM image, install those sources, and
#      save the resulting container as an SQSH.
#
# Keeping the network-facing source fetch outside the container means the
# allocated compute node does not need Git credentials.
set -Eeuo pipefail

BASE_IMAGE="${BASE_IMAGE:-vllm/vllm-openai:v0.28.0}"

# Tomer's fork is public; HTTPS keeps CI and local builds credential-free.
VLLM_REPO=https://github.com/TomerBN-Nvidia/vllm.git
VLLM_BRANCH=super_vl_evals_v0.28.0
VLLM_SHA=13d33858944ae8db0f364837c3ec6a17fbc154f7
VLLM_BINARY_SHA=2cf0a6915ce544dc493a0990f2ea38d81601128a
VLLM_VERSION=0.28.0

RAY_VERSION=2.55.1
VLLM_ROUTER_VERSION=0.1.15
UV_VERSION=0.12.3

GYM_REPO=https://github.com/NVIDIA-NeMo/Gym.git
GYM_SHA=a2b40cb9916c21801fa9a08ff9b9697b3227bb03
GYM_PYTHON_VERSION=3.13.14
OPENAI_VERSION=2.44.0

BUILD_ROOT=/opt/deci-vllm

# NeMo Gym launches each resource server, agent, and model adapter from its own
# virtual environment. Prebuilding those environments makes the saved SQSH
# self-contained and avoids dependency downloads when an evaluation starts.
install_gym_venv() {
    local project=$1
    local project_dir="/opt/Gym/${project}"
    local venv="/opt/uv_venvs/${project}/.venv"

    uv venv --seed --python "${GYM_PYTHON}" "${venv}"
    if [[ -f "${project_dir}/requirements.txt" ]]; then
        (
            echo '-e /opt/Gym'
            # An all-filtered file is valid; do not let grep's status trip
            # set -o pipefail after uv successfully installs /opt/Gym.
            grep -v -F '../..' "${project_dir}/requirements.txt" || true
        ) | uv pip install --python "${venv}/bin/python" -r /dev/stdin
    else
        uv pip install --python "${venv}/bin/python" -e /opt/Gym -e "${project_dir}"
    fi
    uv pip install --python "${venv}/bin/python" "openai==${OPENAI_VERSION}"
}

# This block runs inside BASE_IMAGE through the srun invocation below.
if [[ "${1:-}" == __inside_build ]]; then
    # Some Gym environments install dependencies directly from Git repositories.
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends git ca-certificates
    rm -rf /var/lib/apt/lists/*

    python3 -m pip install --no-cache-dir "uv==${UV_VERSION}"
    export UV_PYTHON_INSTALL_DIR=/opt/uv/python
    uv python install "${GYM_PYTHON_VERSION}"
    GYM_PYTHON=$(uv python find --managed-python "${GYM_PYTHON_VERSION}")

    mkdir -p "${BUILD_ROOT}"
    tar -xf "${VLLM_SOURCE_ARCHIVE:?}" -C "${BUILD_ROOT}"
    tar -xf "${GYM_SOURCE_ARCHIVE:?}" -C /opt

    cd "${BUILD_ROOT}/vllm"
    export VLLM_USE_PRECOMPILED=1
    export VLLM_PRECOMPILED_WHEEL_COMMIT="${VLLM_BINARY_SHA}"
    export VLLM_PRECOMPILED_WHEEL_VARIANT=cu130
    export VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX=1
    export VLLM_VERSION_OVERRIDE="${VLLM_VERSION}"
    uv pip install --system --reinstall --no-deps .
    uv pip install --system \
        "ray==${RAY_VERSION}" \
        "vllm-router==${VLLM_ROUTER_VERSION}"

    UV_PROJECT_ENVIRONMENT=/opt/Gym_venv uv sync \
        --directory /opt/Gym --python "${GYM_PYTHON}" --frozen --no-dev
    uv pip install --python /opt/Gym_venv/bin/python \
        "openai==${OPENAI_VERSION}"

    # These are the Gym components used by the Eval team. This is a Bash loop:
    # install_gym_venv creates one isolated Python environment per component.
    for project in \
        resources_servers/aalcr \
        resources_servers/deepswe \
        resources_servers/equivalence_llm_judge \
        resources_servers/mcqa \
        resources_servers/omniscience \
        resources_servers/scicode \
        resources_servers/swebench \
        responses_api_agents/opencode_sandboxed_agent \
        responses_api_agents/scicode_agent \
        responses_api_agents/simple_agent \
        responses_api_agents/tau2 \
        responses_api_models/openai_model \
        responses_api_models/vllm_model; do
        install_gym_venv "${project}"
    done

    # Leave the source tree before deleting it; uv inspects the current
    # directory while cleaning its cache.
    cd /
    rm -rf -- "${BUILD_ROOT}/vllm"
    uv cache clean

    # Record the exact inputs in the image for later provenance checks.
    cat > "${BUILD_ROOT}/build.env" <<EOF
base_image=${BASE_IMAGE}
vllm_repo=${VLLM_REPO}
vllm_branch=${VLLM_BRANCH}
vllm_sha=${VLLM_SHA}
vllm_binary_sha=${VLLM_BINARY_SHA}
vllm_version=${VLLM_VERSION}
ray_version=${RAY_VERSION}
vllm_router_version=${VLLM_ROUTER_VERSION}
uv_version=${UV_VERSION}
gym_repo=${GYM_REPO}
gym_sha=${GYM_SHA}
gym_python_version=${GYM_PYTHON_VERSION}
openai_version=${OPENAI_VERSION}
EOF
    exit 0
fi

# Everything below runs on the login node and prepares the Slurm build.
OUT_SQSH="${1:-}"
if [[ -z "${OUT_SQSH}" || -z "${SLURM_ACCOUNT:-}" ]]; then
    echo "Usage: SLURM_ACCOUNT=<account> $0 <output.sqsh>" >&2
    exit 2
fi

mkdir -p "$(dirname "${OUT_SQSH}")"
OUT_SQSH="$(cd "$(dirname "${OUT_SQSH}")" && pwd)/$(basename "${OUT_SQSH}")"
if [[ -e "${OUT_SQSH}" ]]; then
    echo "ERROR: output already exists: ${OUT_SQSH}" >&2
    exit 1
fi

build_dir=$(dirname "${OUT_SQSH}")
build_name=$(basename "${OUT_SQSH}" .sqsh)
snapshot="${build_dir}/.build-${build_name}.sh"
vllm_archive="${build_dir}/.build-${build_name}-vllm.tar"
gym_archive="${build_dir}/.build-${build_name}-gym.tar"
source_root=$(mktemp -d /tmp/deci-vllm028-source.XXXXXX)
cleanup() {
    rm -rf -- "${source_root}"
    rm -f -- "${snapshot}" "${vllm_archive}" "${gym_archive}"
}
trap cleanup EXIT

# Run an immutable copy so editing the checked-out script cannot affect an
# in-progress build.
cp "$(cd "$(dirname "$0")" && pwd)/$(basename "$0")" "${snapshot}"

# Archive pinned sources beside the output. Only that directory is mounted into
# the container, so the build does not depend on a personal checkout or Lustre
# source tree.
git clone --depth=1 --branch "${VLLM_BRANCH}" "${VLLM_REPO}" "${source_root}/vllm"
test "$(git -C "${source_root}/vllm" rev-parse HEAD)" = "${VLLM_SHA}"
git -C "${source_root}/vllm" archive --format=tar --prefix=vllm/ HEAD > "${vllm_archive}"

git clone --filter=blob:none --no-checkout "${GYM_REPO}" "${source_root}/Gym"
git -C "${source_root}/Gym" fetch --depth=1 origin "${GYM_SHA}"
git -C "${source_root}/Gym" archive --format=tar --prefix=Gym/ FETCH_HEAD > "${gym_archive}"

export VLLM_SOURCE_ARCHIVE="${vllm_archive}"
export GYM_SOURCE_ARCHIVE="${gym_archive}"

echo "Base       ${BASE_IMAGE}"
echo "vLLM       ${VLLM_REPO}:${VLLM_BRANCH} @ ${VLLM_SHA}"
echo "NeMo Gym   ${GYM_REPO} @ ${GYM_SHA}"
echo "Output     ${OUT_SQSH}"

if ! srun --account="${SLURM_ACCOUNT}" \
    --partition="${SLURM_PARTITION:-batch}" \
    --qos="${SLURM_QOS:-interactive}" \
    --job-name=vllm028-eval-sqsh \
    --nodes=1 --ntasks=1 --gpus-per-node=4 \
    --time="${SLURM_TIME:-04:00:00}" \
    --container-image="${BASE_IMAGE}" \
    --container-mounts="${build_dir}:${build_dir}" \
    --container-save="${OUT_SQSH}" --export=ALL \
    bash "${snapshot}" __inside_build; then
    rm -f -- "${OUT_SQSH}"
    exit 1
fi

ls -lh "${OUT_SQSH}"
sha256sum "${OUT_SQSH}"
echo "Contents: srun --container-image=${OUT_SQSH} cat ${BUILD_ROOT}/build.env"
