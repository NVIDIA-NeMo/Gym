#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Layer a FlashInfer source build onto a vLLM image using Slurm/Pyxis.
# Example (run on a login node; the output directory must be shared):
#   SLURM_ACCOUNT=my_account BASE_IMAGE=/lustre/images/custom-vllm.sqsh \
#     FLASHINFER_BRANCH=my-branch bash custom_flashinfer_version.sh /lustre/images/custom-flashinfer.sqsh
# Optional: FLASHINFER_REPO, EXPECTED_FLASHINFER_SHA, SLURM_PARTITION,
# SLURM_QOS, SLURM_TIME (default 04:00:00), MAX_JOBS (default 4),
# FLASHINFER_NVCC_THREADS (default 1), FLASHINFER_CUDA_ARCH_LIST (10.0a or 10.0f).
# FLASHINFER_BUILD_MODE=auto (default), reuse (require compatible official
# binaries), or source (rebuild everything); FLASHINFER_CUBIN_DOWNLOAD_THREADS=32.
# Reuse official binaries for checked ReplaySSM-only changes; otherwise build
# the fork's full SM100 provider. Always rebuild the custom FP16 ReplaySSM variants.
# Keep the CUDA toolkit in the image for other kernels' runtime JIT.
set -Eeuo pipefail

export FLASHINFER_REPO="${FLASHINFER_REPO:-https://github.com/bxyu-nvidia/flashinfer.git}"
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-10.0a}"
export FLASHINFER_BUILD_MODE="${FLASHINFER_BUILD_MODE:-auto}"
export FLASHINFER_CUBIN_DOWNLOAD_THREADS="${FLASHINFER_CUBIN_DOWNLOAD_THREADS:-32}"
case "${FLASHINFER_BUILD_MODE}" in
    auto|reuse|source) ;;
    *) echo "ERROR: FLASHINFER_BUILD_MODE must be auto, reuse, or source." >&2; exit 2 ;;
esac
case "${FLASHINFER_CUDA_ARCH_LIST}" in
    10.0a|10.0f) ;;
    *) echo "ERROR: FLASHINFER_CUDA_ARCH_LIST must be 10.0a or 10.0f for this SM100 image." >&2; exit 2 ;;
esac
BUILD_ROOT=/opt/super-vl-evals

###############################################################################
# Inside the container (this script re-execs itself here).
###############################################################################
if [[ "${1:-}" == __inside_build ]]; then
    : "${BASE_IMAGE:?}" "${FLASHINFER_BRANCH:?}" "${FLASHINFER_HEAD_SHA:?}"
    echo ">>> FlashInfer ${FLASHINFER_BRANCH} @ ${FLASHINFER_HEAD_SHA} on $(hostname)"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y
    apt-get install -y --no-install-recommends git ca-certificates build-essential
    python3 -m pip install uv
    command -v nvcc >/dev/null || {
        echo "ERROR: BASE_IMAGE needs the CUDA toolkit (nvcc) for FlashInfer JIT." >&2
        exit 1
    }

    mkdir -p "${BUILD_ROOT}"
    BUILD_DIR=$(mktemp -d "${BUILD_ROOT}/flashinfer-build.XXXXXX")
    trap 'rm -rf "${BUILD_DIR}"' EXIT
    git init "${BUILD_DIR}/source"
    cd "${BUILD_DIR}/source"
    git remote add origin "${FLASHINFER_REPO}"
    # Fetch the resolved commit, even if the branch moves while the job queues.
    git fetch --depth=1 origin "${FLASHINFER_HEAD_SHA}"
    git checkout --detach FETCH_HEAD
    test "$(git rev-parse HEAD)" = "${FLASHINFER_HEAD_SHA}"
    git submodule update --init --recursive --depth=1
    if [[ ! -f flashinfer-jit-cache-provider/pyproject.toml ]]; then
        echo "ERROR: this FlashInfer branch does not support architecture-specific JIT-cache provider builds." >&2
        exit 1
    fi

    # Reuse requires a diff against the actual upstream release, not a tag from
    # the fork. These source files affect only the two Mamba module families
    # rebuilt below. Any other runtime/build change selects a full source build.
    export FLASHINFER_RELEASE_VERSION="$(cat version.txt)"
    export FLASHINFER_RELEASE_SHA=""
    export FLASHINFER_INSTALL_MODE=source
    if [[ "${FLASHINFER_BUILD_MODE}" != source && "${FLASHINFER_CUDA_ARCH_LIST}" == 10.0a ]]; then
        if git fetch --depth=1 https://github.com/flashinfer-ai/flashinfer.git \
            "refs/tags/v${FLASHINFER_RELEASE_VERSION}"; then
            FLASHINFER_RELEASE_SHA=$(git rev-parse 'FETCH_HEAD^{commit}')
            git diff --name-only --no-renames -z "${FLASHINFER_RELEASE_SHA}" HEAD -- > "${BUILD_DIR}/changed-files"
            reuse_compatible=1
            while IFS= read -r -d '' changed_file; do
                case "${changed_file}" in
                    csrc/replayssm_materialize.cu|\
                    include/flashinfer/mamba/kernel_checkpointing_ssu.cuh|\
                    include/flashinfer/mamba/kernel_checkpointing_ssu_common.cuh|\
                    include/flashinfer/mamba/kernel_checkpointing_ssu_main.cuh|tests/*) ;;
                    *) echo "Source build required by change: ${changed_file}"; reuse_compatible=0 ;;
                esac
            done < "${BUILD_DIR}/changed-files"
            if [[ "${reuse_compatible}" == 1 ]]; then
                FLASHINFER_INSTALL_MODE=reuse
            fi
        else
            echo "Could not resolve the upstream release; selecting a full source build."
        fi
    fi
    if [[ "${FLASHINFER_BUILD_MODE}" == reuse && "${FLASHINFER_INSTALL_MODE}" != reuse ]]; then
        echo "ERROR: official binary reuse requires SM100a and a compatible diff against the upstream release." >&2
        exit 1
    fi
    echo ">>> FlashInfer binary installation mode: ${FLASHINFER_INSTALL_MODE}"
    export FLASHINFER_CACHE_VERSION="${FLASHINFER_RELEASE_VERSION}+$(python3 -c \
        'import torch; print("cu" + "".join(torch.version.cuda.split(".")[:2]))')"

    # Prevent dependency resolution from replacing the image's CUDA/PyTorch stack.
    # Other FlashInfer dependencies can be installed/upgraded as the branch requires.
    python3 - "${BUILD_DIR}" <<'PY'
import importlib.metadata as metadata
import os
import re
import shutil
import sys
import tomllib
from pathlib import Path

import torch
from build_utils import get_build_dependency_requirements

build_dir = Path(sys.argv[1])
constraints = []
remove = []
preserve = {"torch", "torchvision", "torchaudio", "vllm", "triton", "numpy", "cuda-python", "cuda-bindings"}
for dist in metadata.distributions():
    name = re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower()
    if name.startswith("flashinfer-"):
        keep = os.environ["FLASHINFER_INSTALL_MODE"] == "reuse" and (
            (name == "flashinfer-cubin" and dist.version == os.environ["FLASHINFER_RELEASE_VERSION"])
            or (name in {"flashinfer-jit-cache", "flashinfer-jit-cache-sm100a"}
                and dist.version == os.environ["FLASHINFER_CACHE_VERSION"])
        )
        if not keep:
            remove.append(dist.metadata["Name"])
        if name == "flashinfer-python":
            # Earlier image builds may have added AOT overrides outside the
            # wheel RECORD; uninstall alone would leave those stale binaries.
            shutil.rmtree(dist.locate_file("flashinfer/data/aot"), ignore_errors=True)
    elif name in preserve or name.startswith("nvidia-"):
        constraints.append(f"{dist.metadata['Name']}=={dist.version}")
(build_dir / "constraints.txt").write_text("\n".join(constraints) + "\n")
(build_dir / "uninstall.txt").write_text("".join(f"{name}\n" for name in remove))
requirements = {"wheel"}
for project in map(Path, (".", "flashinfer-cubin", "flashinfer-jit-cache-provider", "flashinfer-jit-cache")):
    requirements.update(tomllib.loads((project / "pyproject.toml").read_text())["build-system"]["requires"])
requirements.update(get_build_dependency_requirements(torch.version.cuda.split(".")[0]))
(build_dir / "build-requirements.txt").write_text("\n".join(sorted(requirements)) + "\n")
PY

    # Keep matching official binary packages in reuse mode. uv also reuses
    # installed wheels, so a matching base image needs no cubin/provider download.
    if [[ -s "${BUILD_DIR}/uninstall.txt" ]]; then
        uv pip uninstall --system -r "${BUILD_DIR}/uninstall.txt"
    fi
    uv pip install --system --constraint "${BUILD_DIR}/constraints.txt" \
        -r "${BUILD_DIR}/build-requirements.txt" -r requirements.txt

    # Official cubins require the public release version. The actual fork SHA
    # remains in _build_meta and our image manifest; version checks stay enabled.
    if [[ "${FLASHINFER_INSTALL_MODE}" == reuse ]]; then
        unset FLASHINFER_LOCAL_VERSION
    else
        export FLASHINFER_LOCAL_VERSION="g${FLASHINFER_HEAD_SHA}"
    fi
    unset FLASHINFER_DEV_RELEASE_SUFFIX FLASHINFER_DISABLE_JIT FLASHINFER_DISABLE_VERSION_CHECK
    unset FLASHINFER_CUBIN_DIR
    # Keep build hooks from silently replacing runtime dependencies outside uv's
    # constrained resolution above (supported by newer FlashInfer branches).
    export FLASHINFER_BUILD_NO_PIP=1
    uv pip install --system --no-deps --no-build-isolation .
    export FLASHINFER_JIT_CACHE_PROVIDER_ARCH="${FLASHINFER_CUDA_ARCH_LIST}"
    export FLASHINFER_JIT_CACHE_PROVIDER_ARCHS="${FLASHINFER_CUDA_ARCH_LIST}"
    export MAX_JOBS="${MAX_JOBS:-4}"
    export FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-1}"
    if [[ "${FLASHINFER_INSTALL_MODE}" == reuse ]]; then
        echo ">>> Reuse official cubins and SM100a JIT-cache wheels"
        uv pip install --system --no-deps --only-binary=:all: \
            "flashinfer-cubin==${FLASHINFER_RELEASE_VERSION}" --index-url https://flashinfer.ai/whl
        uv pip install --system --no-deps --only-binary=:all: \
            "flashinfer-jit-cache==${FLASHINFER_CACHE_VERSION}" \
            "flashinfer-jit-cache-sm100a==${FLASHINFER_CACHE_VERSION}" \
            --index-url "https://flashinfer.ai/whl/${FLASHINFER_CACHE_VERSION##*+}"
    else
        # Cubins are downloaded/packaged; the provider compiles the broad AOT set.
        uv pip install --system --no-deps --no-build-isolation ./flashinfer-cubin
        echo ">>> Build FlashInfer JIT-cache provider for ${FLASHINFER_JIT_CACHE_PROVIDER_ARCH}"
        uv pip install --system --no-deps --no-build-isolation --verbose ./flashinfer-jit-cache-provider
        uv pip install --system --no-deps --no-build-isolation ./flashinfer-jit-cache
    fi

    cd /
    if [[ "${FLASHINFER_INSTALL_MODE}" == reuse ]]; then
        # Invalidate every affected cached specialization, including shapes we
        # do not explicitly warm. Those shapes must JIT from the patched source.
        python3 - <<'PY'
import json
import shutil
from pathlib import Path

from flashinfer_jit_cache.providers import sm100a

manifest_path = Path(sm100a.__file__).with_name("manifest.json")
manifest = json.loads(manifest_path.read_text())
affected = [name for name in manifest["modules"]
            if name.startswith(("checkpointing_ssu_", "replayssm_materialize_"))]
for name in affected:
    shutil.rmtree(manifest_path.parent / "jit_cache" / name)
manifest["modules"] = [name for name in manifest["modules"] if name not in affected]
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Invalidated {len(affected)} official ReplaySSM specializations")
PY
    fi

    # Verify discovery, provenance, and a real GPU operation from the installed
    # provider outside the checkout, with an empty workspace and JIT disabled.
    FLASHINFER_WORKSPACE_BASE="${BUILD_DIR}/smoke-cache" FLASHINFER_DISABLE_JIT=1 \
        python3 - "${BUILD_DIR}/jit-cache.env" <<'PY'
import importlib
import os
import sys
from pathlib import Path

import flashinfer
import flashinfer_cubin
import flashinfer_jit_cache
import torch
import vllm
from flashinfer._build_meta import __git_commit__
from flashinfer.jit import env as jit_env
from flashinfer.jit.norm import gen_norm_module

assert __git_commit__ == os.environ["FLASHINFER_HEAD_SHA"], __git_commit__
assert flashinfer.__version__ == flashinfer_cubin.__version__
reuse = os.environ["FLASHINFER_INSTALL_MODE"] == "reuse"
binary_sha = os.environ["FLASHINFER_RELEASE_SHA"] if reuse else __git_commit__
cache_version = os.environ["FLASHINFER_CACHE_VERSION"] if reuse else flashinfer.__version__
assert flashinfer_cubin.__git_version__ == binary_sha
assert flashinfer_jit_cache.__version__ == cache_version
assert flashinfer_jit_cache.__git_version__ == binary_sha
provider_id = "sm" + os.environ["FLASHINFER_JIT_CACHE_PROVIDER_ARCH"].replace(".", "")
providers = jit_env.FLASHINFER_AOT_PROVIDERS
assert len(providers) == 1 and providers[0].provider_id == provider_id, providers
provider = providers[0]
package = importlib.import_module(f"flashinfer_jit_cache.providers.{provider_id}")
assert package.__git_version__ == binary_sha, package.__git_version__
assert provider.version == package.__version__ == cache_version
assert {"norm", "fmha_gen", "fused_moe_trtllm_sm100"} <= provider.modules, provider.modules
for name in provider.modules:
    library = provider.jit_cache_dir / name / f"{name}.so"
    assert library.is_file() and library.stat().st_size > 0, f"Missing provider module: {library}"
assert gen_norm_module().aot_path == provider.jit_cache_dir / "norm" / "norm.so"
assert torch.cuda.is_available(), "The build job needs a GPU for its JIT smoke check"
x = torch.randn(16, 1024, device="cuda", dtype=torch.bfloat16)
weight = torch.ones(1024, device="cuda", dtype=torch.bfloat16)
# Exercise the provider's CUDA module directly; the public RMSNorm API can
# choose a separate CuTe DSL implementation on newer FlashInfer branches.
actual = torch.empty_like(x)
gen_norm_module().build_and_load().rmsnorm(actual, x, weight, 1e-6, False)
expected = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + 1e-6)
torch.testing.assert_close(actual, expected.to(x.dtype), rtol=1e-2, atol=1e-2)
torch.cuda.synchronize()
print(f"FlashInfer {flashinfer.__version__} from {flashinfer.__file__}")
print(f"vLLM {vllm.__version__}; torch {torch.__version__}")
print(f"Installed {provider.distribution} with {len(provider.modules)} precompiled modules")
print("Installed FlashInfer GPU smoke check passed with JIT disabled")
Path(sys.argv[1]).write_text(
    f"FLASHINFER_JIT_CACHE_PROVIDER={provider.distribution}\n"
    f"FLASHINFER_JIT_CACHE_VERSION={provider.version}\n"
    f"FLASHINFER_JIT_CACHE_SHA={package.__git_version__}\n"
    f"FLASHINFER_JIT_CACHE_ARCH={os.environ['FLASHINFER_JIT_CACHE_PROVIDER_ARCH']}\n"
    f"FLASHINFER_JIT_CACHE_MODULE_COUNT={len(provider.modules)}\n"
)
PY

    # Ported from build-super-vl-rl-v0251-thin.sh at 36e6e73cdc: compile the
    # TRTLLM-GEN host dispatchers against this branch's cubins, plus the Mamba
    # variants observed in the FP16-cache ReplaySSM run. Store them in the
    # installed package's AOT directory so they survive cache cleanup and work
    # even when runtime jobs mount a different HOME or FlashInfer workspace.
    # Keep one module list for compilation and verification in a fresh process.
    cat > "${BUILD_DIR}/warmup.py" <<'PY'
import hashlib
import os
import shutil
import sys
from functools import partial
from pathlib import Path

import torch

from flashinfer.artifacts import ArtifactPath, CheckSumHash
from flashinfer.jit import env as jit_env
from flashinfer.jit.attention.modules import gen_trtllm_gen_fmha_module
from flashinfer.jit.core import build_jit_specs
from flashinfer.jit.fused_moe import gen_trtllm_gen_fused_moe_sm100_module
from flashinfer.jit.mamba.checkpointing_ssu import gen_checkpointing_ssu_module
from flashinfer.jit.mamba.replayssm_materialize import gen_replayssm_materialize_module

modules = (
    ("FMHA", gen_trtllm_gen_fmha_module, ArtifactPath.TRTLLM_GEN_FMHA, CheckSumHash.TRTLLM_GEN_FMHA),
    (
        "FUSED_MOE_SM100",
        gen_trtllm_gen_fused_moe_sm100_module,
        ArtifactPath.TRTLLM_GEN_BMM,
        CheckSumHash.TRTLLM_GEN_BMM,
    ),
    # Exact specializations from super_3.5_GA-BF16-p2d2-rssm-rustfrontend-mcfp16-bxyu.log.
    (
        "CHECKPOINTING_SSU",
        partial(
            gen_checkpointing_ssu_module,
            state_dtype=torch.float16,
            input_dtype=torch.bfloat16,
            dt_dtype=torch.bfloat16,
            weight_dtype=torch.bfloat16,
            matrixA_dtype=torch.float32,
            stateIndex_dtype=torch.int32,
            state_scale_dtype=None,
            dim=64,
            dstate=128,
            npredicted=6,
            max_window=16,
            heads_per_group=16,
            num_groups=2,
            philox_rounds=5,
            enable_pdl=False,
        ),
        None,
        None,
    ),
    (
        "REPLAYSSM_MATERIALIZE",
        partial(
            gen_replayssm_materialize_module,
            state_dtype=torch.float16,
            input_dtype=torch.bfloat16,
            matrixA_dtype=torch.float32,
            dim=64,
            dstate=128,
            heads_per_group=16,
            max_window=16,
            philox_rounds=5,
        ),
        None,
        None,
    ),
)
verify_only = sys.argv[1] == "--verify"
records = []
for label, generate_spec, artifact, expected_sha in modules:
    if artifact is not None and not verify_only:
        manifest = jit_env.FLASHINFER_CUBIN_DIR / artifact / "checksums.txt"
        manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
        if manifest_sha != expected_sha:
            raise RuntimeError(f"{label} cubin manifest does not match this FlashInfer branch: {manifest}")
        records.append(f"{label}_ARTIFACT_PATH={artifact}\n{label}_MANIFEST_SHA256={manifest_sha}\n")

    spec = generate_spec()
    destination = jit_env.FLASHINFER_AOT_DIR / spec.name / f"{spec.name}.so"
    reuse_binary = os.environ["FLASHINFER_INSTALL_MODE"] == "reuse" and artifact is not None
    if reuse_binary:
        # FMHA and fused MoE are unchanged in the checked diff. Load the official
        # provider instead of recompiling or copying its dispatchers.
        destination = spec.aot_path
        assert destination.is_file(), f"Missing official module: {destination}"
        if not verify_only:
            assert artifact.encode() in destination.read_bytes(), f"Wrong artifact in {destination}"
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            records.append(
                f"{label}_MODULE_NAME={spec.name}\n{label}_DISPATCHER_PATH={destination}\n"
                f"{label}_DISPATCHER_SHA256={digest}\n"
            )
            print(f"Reusing official module: {destination}")
            continue
    if verify_only:
        assert spec.aot_path == destination and destination.is_file(), f"Missing warmed module: {destination}"
        spec.build_and_load()
        print(f"Loaded warmed module with JIT disabled: {destination}")
        continue

    print(f"Warming JIT module: {spec.name}", flush=True)
    build_jit_specs([spec], verbose=True, skip_prebuilt=False)
    compiled = jit_env.FLASHINFER_JIT_DIR / spec.name / f"{spec.name}.so"
    if artifact is not None and artifact.encode() not in compiled.read_bytes():
        raise RuntimeError(f"Compiled {label} dispatcher does not reference the expected cubins: {artifact}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(compiled, destination)
    dispatcher_sha = hashlib.sha256(destination.read_bytes()).hexdigest()
    records.append(
        f"{label}_MODULE_NAME={spec.name}\n"
        f"{label}_DISPATCHER_PATH={destination}\n"
        f"{label}_DISPATCHER_SHA256={dispatcher_sha}\n"
    )
    print(f"Saved warmed module: {destination}")
if not verify_only:
    Path(sys.argv[1]).write_text("".join(records))
PY

    echo ">>> Warm up TRTLLM-GEN and ReplaySSM JIT modules"
    FLASHINFER_WORKSPACE_BASE="${BUILD_DIR}/warmup-cache" \
        python3 "${BUILD_DIR}/warmup.py" "${BUILD_DIR}/warmup.env"

    # Prove a fresh process can load all saved modules without their build
    # workspace or permission to JIT compile a replacement.
    rm -rf "${BUILD_DIR}/warmup-cache"
    FLASHINFER_WORKSPACE_BASE="${BUILD_DIR}/warmup-verify-cache" FLASHINFER_DISABLE_JIT=1 \
        python3 "${BUILD_DIR}/warmup.py" --verify

    # Leave the vLLM build manifest intact when layering onto a custom image.
    {
        printf 'BASE_IMAGE=%s\nFLASHINFER_REPO=%s\nFLASHINFER_BRANCH=%s\nFLASHINFER_SHA=%s\n' \
            "${BASE_IMAGE}" "${FLASHINFER_REPO}" "${FLASHINFER_BRANCH}" "${FLASHINFER_HEAD_SHA}"
        printf 'FLASHINFER_INSTALL_MODE=%s\nFLASHINFER_RELEASE_SHA=%s\n' \
            "${FLASHINFER_INSTALL_MODE}" "${FLASHINFER_RELEASE_SHA}"
        python3 -c 'import flashinfer; print(f"FLASHINFER_VERSION={flashinfer.__version__}")'
        cat "${BUILD_DIR}/jit-cache.env"
        cat "${BUILD_DIR}/warmup.env"
    } > "${BUILD_ROOT}/flashinfer-build.env"
    cat "${BUILD_ROOT}/flashinfer-build.env"
    uv cache clean
    python3 -m pip cache purge || true
    apt-get clean
    rm -rf /var/lib/apt/lists/*
    exit 0
fi

###############################################################################
# Login node.
###############################################################################
if [[ $# -ne 1 || -z "${SLURM_ACCOUNT:-}" || -z "${BASE_IMAGE:-}" || -z "${FLASHINFER_BRANCH:-}" ]]; then
    echo "Usage: SLURM_ACCOUNT=<account> BASE_IMAGE=<vllm image or .sqsh> FLASHINFER_BRANCH=<branch> $0 <OUTPUT.sqsh>" >&2
    exit 2
fi
export BASE_IMAGE FLASHINFER_BRANCH
OUT_SQSH="$1"
mkdir -p "$(dirname "${OUT_SQSH}")"
OUT_SQSH="$(cd "$(dirname "${OUT_SQSH}")" && pwd)/$(basename "${OUT_SQSH}")"
if [[ -e "${OUT_SQSH}" || -L "${OUT_SQSH}" ]]; then
    echo "ERROR: ${OUT_SQSH} already exists." >&2
    exit 1
fi

FLASHINFER_HEAD_SHA=$(git ls-remote --exit-code "${FLASHINFER_REPO}" "refs/heads/${FLASHINFER_BRANCH}" | cut -f1) || {
    echo "ERROR: could not resolve branch ${FLASHINFER_BRANCH} in ${FLASHINFER_REPO}." >&2
    exit 1
}
if [[ ! "${FLASHINFER_HEAD_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: expected one branch commit, got: ${FLASHINFER_HEAD_SHA}" >&2
    exit 1
fi
if [[ -n "${EXPECTED_FLASHINFER_SHA:-}" && "${FLASHINFER_HEAD_SHA}" != "${EXPECTED_FLASHINFER_SHA}" ]]; then
    echo "ERROR: branch moved: expected ${EXPECTED_FLASHINFER_SHA}, got ${FLASHINFER_HEAD_SHA}." >&2
    exit 1
fi
export FLASHINFER_HEAD_SHA

# Execute a snapshot so edits while this job queues cannot change its build.
SNAP=$(mktemp "$(dirname "${OUT_SQSH}")/.flashinfer-build.XXXXXX")
trap 'rm -f "${SNAP}"' EXIT
cp "$0" "${SNAP}"
MOUNTS="$(dirname "${SNAP}"):$(dirname "${SNAP}")"
[[ ! -d /lustre ]] || MOUNTS="${MOUNTS},/lustre:/lustre"

echo "Base   ${BASE_IMAGE}"
echo "Source ${FLASHINFER_REPO} ${FLASHINFER_BRANCH} @ ${FLASHINFER_HEAD_SHA}"
echo "Output ${OUT_SQSH}"
if ! srun \
    --account="${SLURM_ACCOUNT}" \
    --partition="${SLURM_PARTITION:-batch}" \
    --job-name=super-vl-evals-flashinfer \
    --nodes=1 --ntasks=1 --segment=1 \
    --gpus-per-node=4 --mem=0 \
    --time="${SLURM_TIME:-04:00:00}" \
    --container-image="${BASE_IMAGE}" \
    --container-mounts="${MOUNTS}" \
    --no-container-mount-home \
    --qos="${SLURM_QOS:-interactive}" \
    --container-save="${OUT_SQSH}" \
    --export=ALL \
    bash "${SNAP}" __inside_build
then
    echo "ERROR: build failed; removing ${OUT_SQSH}." >&2
    rm -f "${OUT_SQSH}"
    exit 1
fi

ls -lh "${OUT_SQSH}"
sha256sum "${OUT_SQSH}" | tee "${OUT_SQSH}.sha256"
echo "Manifest: ${BUILD_ROOT}/flashinfer-build.env inside the saved image"
