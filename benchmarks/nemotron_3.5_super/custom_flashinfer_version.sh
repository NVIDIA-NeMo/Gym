#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Layer a FlashInfer source build onto a vLLM image using Slurm/Pyxis.
# Example (run on a login node; the output directory must be shared):
#   SLURM_ACCOUNT=my_account BASE_IMAGE=/lustre/images/custom-vllm.sqsh \
#     FLASHINFER_BRANCH=my-branch bash custom_flashinfer_version.sh /lustre/images/custom-flashinfer.sqsh
# Optional: FLASHINFER_REPO, EXPECTED_FLASHINFER_SHA, SLURM_PARTITION,
# SLURM_QOS, SLURM_TIME (default 04:00:00), SLURM_CPUS_PER_TASK (default 64), MAX_JOBS (default 32),
# FLASHINFER_NVCC_THREADS (default 1), FLASHINFER_CUDA_ARCH_LIST (10.0a or 10.0f).
# FLASHINFER_BUILD_MODE=auto (default), reuse (require compatible official
# binaries), source (rebuild everything), or incremental (reuse a verified custom image
# and rebuild FLASHINFER_REBUILD_MODULES, a comma-separated list of module names/globs); FLASHINFER_CUBIN_DOWNLOAD_THREADS=32.
# Reuse official binaries for checked ReplaySSM-only changes; otherwise build
# the fork's full SM100 provider. Incremental builds retain unselected modules.
# Incremental example: FLASHINFER_REBUILD_MODULES=fused_moe_trtllm_sm100,fmha_gen
# Existing warmup generators also support checkpointing_ssu_* and replayssm_materialize_*.
# Optional Prims BF16 warmup: FLASHINFER_PRIMS_WARMUP=1 and
# FLASHINFER_PRIMS_WARMUP_DUMP=/lustre/.../call-01 (trusted standalone MoE capture).
# FLASHINFER_PRIMS_WARMUP_ITERATIONS=2; FLASHINFER_PRIMS_WEIGHT_LAYOUT=BlockMajorK.
# Keep the CUDA toolkit in the image for other kernels' runtime JIT.
set -Eeuo pipefail

export FLASHINFER_REPO="${FLASHINFER_REPO:-https://github.com/bxyu-nvidia/flashinfer.git}"
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-10.0a}"
export FLASHINFER_BUILD_MODE="${FLASHINFER_BUILD_MODE:-auto}"
export FLASHINFER_REBUILD_MODULES="${FLASHINFER_REBUILD_MODULES:-fused_moe_trtllm_sm100}"
export FLASHINFER_PRIMS_WARMUP="${FLASHINFER_PRIMS_WARMUP:-0}"
export FLASHINFER_PRIMS_WARMUP_DUMP="${FLASHINFER_PRIMS_WARMUP_DUMP:-}"
export FLASHINFER_PRIMS_WARMUP_ITERATIONS="${FLASHINFER_PRIMS_WARMUP_ITERATIONS:-2}"
export FLASHINFER_PRIMS_WEIGHT_LAYOUT="${FLASHINFER_PRIMS_WEIGHT_LAYOUT:-BlockMajorK}"
export FLASHINFER_CUBIN_DOWNLOAD_THREADS="${FLASHINFER_CUBIN_DOWNLOAD_THREADS:-32}"
export SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-128}"
export MAX_JOBS="${MAX_JOBS:-64}"
export FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-1}"
case "${FLASHINFER_BUILD_MODE}" in
    auto|reuse|source|incremental) ;;
    *) echo "ERROR: FLASHINFER_BUILD_MODE must be auto, reuse, source, or incremental." >&2; exit 2 ;;
esac
case "${FLASHINFER_CUDA_ARCH_LIST}" in
    10.0a|10.0f) ;;
    *) echo "ERROR: FLASHINFER_CUDA_ARCH_LIST must be 10.0a or 10.0f for this SM100 image." >&2; exit 2 ;;
esac
BUILD_ROOT=/opt/super-vl-evals
case "${FLASHINFER_PRIMS_WARMUP}" in
    0|1) ;;
    *) echo "ERROR: FLASHINFER_PRIMS_WARMUP must be 0 or 1." >&2; exit 2 ;;
esac
if [[ "${FLASHINFER_PRIMS_WARMUP}" == 1 ]]; then
    if [[ ! "${FLASHINFER_PRIMS_WARMUP_ITERATIONS}" =~ ^[1-9][0-9]*$ || ! -d "${FLASHINFER_PRIMS_WARMUP_DUMP}" ]]; then
        echo "ERROR: Prims warmup needs a positive iteration count and an existing FLASHINFER_PRIMS_WARMUP_DUMP directory." >&2
        exit 2
    fi
    case "${FLASHINFER_PRIMS_WEIGHT_LAYOUT}" in
        BlockMajorK|MajorK) ;;
        *) echo "ERROR: FLASHINFER_PRIMS_WEIGHT_LAYOUT must be BlockMajorK or MajorK." >&2; exit 2 ;;
    esac
    for dump_file in metadata.json inputs.pt outputs.pt; do
        if [[ ! -f "${FLASHINFER_PRIMS_WARMUP_DUMP}/${dump_file}" ]]; then
            echo "ERROR: Prims warmup dump is missing ${dump_file}." >&2
            exit 2
        fi
    done
fi

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
    # Isolate provider builds as well as warmup from the base image's JIT cache.
    export FLASHINFER_WORKSPACE_BASE="${BUILD_DIR}/build-cache"
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
    if [[ "${FLASHINFER_BUILD_MODE}" == incremental ]]; then
        # Read the installed package outside the new checkout. Reuse only an
        # image whose native binaries and ReplaySSM overrides were verified.
        (
            cd /
            FLASHINFER_DISABLE_JIT=1 python3 "${BUILD_ROOT}/verify-flashinfer.py" \
                --verify "${BUILD_ROOT}/flashinfer-warmup.env"
            python3 - "${BUILD_DIR}" "${BUILD_ROOT}" <<'PY'
import importlib
import os
import shlex
import shutil
import sys
from pathlib import Path

import flashinfer
import flashinfer_cubin
import flashinfer_jit_cache
from flashinfer._build_meta import __git_commit__
from flashinfer.jit import env as jit_env

build_dir, build_root = map(Path, sys.argv[1:])
manifest = dict(line.split("=", 1) for line in (build_root / "flashinfer-build.env").read_text().splitlines())
assert manifest.get("FLASHINFER_REPLAY_VERIFIED") == "1", "Base image needs verified ReplaySSM binaries"
assert manifest["FLASHINFER_SHA"] == __git_commit__, "Base image source provenance mismatch"
assert manifest["FLASHINFER_JIT_CACHE_ARCH"] == os.environ["FLASHINFER_CUDA_ARCH_LIST"], "Base image architecture mismatch"
assert flashinfer.__version__ == flashinfer_cubin.__version__, "Base image cubin version mismatch"
assert len(jit_env.FLASHINFER_AOT_PROVIDERS) == 1, "Expected one base image provider"
provider = jit_env.FLASHINFER_AOT_PROVIDERS[0]
package = importlib.import_module(f"flashinfer_jit_cache.providers.{provider.provider_id}")
assert package.__git_version__ == flashinfer_jit_cache.__git_version__ == flashinfer_cubin.__git_version__
version = flashinfer.__version__
release = os.environ["FLASHINFER_RELEASE_VERSION"]
assert version == release or version.startswith(release + "+"), "Target must use the same FlashInfer release"
values = {
    "FLASHINFER_BASE_SHA": __git_commit__,
    "FLASHINFER_BASE_BINARY_SHA": package.__git_version__,
    "FLASHINFER_BASE_CACHE_VERSION": provider.version,
    "FLASHINFER_BASE_LOCAL_VERSION": version.partition("+")[2],
    "FLASHINFER_RELEASE_SHA": manifest.get("FLASHINFER_RELEASE_SHA", ""),
}
(build_dir / "base.env").write_text("".join(f"export {key}={shlex.quote(value)}\n" for key, value in values.items()))
shutil.copytree(jit_env.FLASHINFER_AOT_DIR, build_dir / "base-aot")
shutil.copy2(build_root / "flashinfer-warmup.env", build_dir / "base-warmup.env")
PY
        )
        # Generated locally from the verified installed package, not the branch.
        source "${BUILD_DIR}/base.env"
        git fetch --depth=1 origin "${FLASHINFER_BASE_SHA}"
        git diff --name-only --no-renames -z "${FLASHINFER_BASE_SHA}" HEAD -- > "${BUILD_DIR}/changed-files"
        python3 - "${BUILD_DIR}/changed-files" <<'PY'
import fnmatch
import os
import sys
from pathlib import Path

patterns = [item.strip() for item in os.environ["FLASHINFER_REBUILD_MODULES"].split(",")]
families = ("fused_moe_trtllm_sm100", "fmha_gen", "checkpointing_ssu_*", "replayssm_materialize_*")
if any(not item or item not in families for item in patterns):
    raise SystemExit(f"Incremental rebuild supports comma-separated selections from {families}; got {patterns}")
for changed_file in Path(sys.argv[1]).read_bytes().decode().split("\0"):
    if not changed_file:
        continue
    required = ()
    if changed_file == "csrc/trtllm_fused_moe_kernel_launcher.cu":
        required = ("fused_moe_trtllm_sm100",)
    elif changed_file == "csrc/trtllm_fmha_kernel_launcher.cu":
        required = ("fmha_gen",)
    elif changed_file == "csrc/replayssm_materialize.cu":
        required = ("replayssm_materialize_*",)
    elif changed_file in (
        "include/flashinfer/mamba/kernel_checkpointing_ssu.cuh",
        "include/flashinfer/mamba/kernel_checkpointing_ssu_common.cuh",
        "include/flashinfer/mamba/kernel_checkpointing_ssu_main.cuh",
    ):
        required = ("checkpointing_ssu_*", "replayssm_materialize_*")
    elif any(fnmatch.fnmatchcase(changed_file, pattern) for pattern in (
        "flashinfer/autotuner/*.py", "flashinfer/fused_moe/backends/prims_ts/*.py",
        "flashinfer/prims_ts/*.py", "flashinfer/jit/cute_dsl_core.py",
        "tests/*", "benchmarks/*", "docs/*", "*.md", "prims_ts_batched_gemm_kernel_notes.py",
    )):
        continue
    else:
        raise SystemExit(f"Cannot safely reuse binaries after change: {changed_file}. Use FLASHINFER_BUILD_MODE=source.")
    missing = set(required) - set(patterns)
    if missing:
        raise SystemExit(f"Change to {changed_file} requires FLASHINFER_REBUILD_MODULES to include {sorted(missing)}")
PY
        # Changes to a module family invalidate even its unwarmed specializations.
        export FLASHINFER_INSTALL_MODE=incremental
    elif [[ "${FLASHINFER_BUILD_MODE}" != source && "${FLASHINFER_CUDA_ARCH_LIST}" == 10.0a ]]; then
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
        keep = (os.environ["FLASHINFER_INSTALL_MODE"] == "incremental" and name != "flashinfer-python") or (
            os.environ["FLASHINFER_INSTALL_MODE"] == "reuse" and (
            (name == "flashinfer-cubin" and dist.version == os.environ["FLASHINFER_RELEASE_VERSION"])
            or (name in {"flashinfer-jit-cache", "flashinfer-jit-cache-sm100a"}
                and dist.version == os.environ["FLASHINFER_CACHE_VERSION"])
            )
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
    elif [[ "${FLASHINFER_INSTALL_MODE}" == incremental ]]; then
        # Binary packages retain their version; the new Python SHA is recorded
        # independently in _build_meta and the image manifests.
        if [[ -n "${FLASHINFER_BASE_LOCAL_VERSION}" ]]; then
            export FLASHINFER_LOCAL_VERSION="${FLASHINFER_BASE_LOCAL_VERSION}"
        else
            unset FLASHINFER_LOCAL_VERSION
        fi
    else
        export FLASHINFER_LOCAL_VERSION="g${FLASHINFER_HEAD_SHA}"
    fi
    unset FLASHINFER_DEV_RELEASE_SUFFIX FLASHINFER_DISABLE_JIT FLASHINFER_DISABLE_VERSION_CHECK
    unset FLASHINFER_CUBIN_DIR
    # Keep build hooks from silently replacing runtime dependencies outside uv's
    # constrained resolution above (supported by newer FlashInfer branches).
    export FLASHINFER_BUILD_NO_PIP=1
    uv pip install --no-cache --system --no-deps --no-build-isolation .
    export FLASHINFER_JIT_CACHE_PROVIDER_ARCH="${FLASHINFER_CUDA_ARCH_LIST}"
    export FLASHINFER_JIT_CACHE_PROVIDER_ARCHS="${FLASHINFER_CUDA_ARCH_LIST}"
    if [[ "${FLASHINFER_INSTALL_MODE}" == incremental ]]; then
        (
            cd /
            python3 - "${BUILD_DIR}/base-aot" <<'PY'
import fnmatch
import json
import os
import shutil
import sys

from flashinfer.jit import env as jit_env

shutil.copytree(sys.argv[1], jit_env.FLASHINFER_AOT_DIR, dirs_exist_ok=True)
# A family can have other cached shapes besides the explicit warmup cases.
# Remove all selected family variants so those shapes compile patched sources.
patterns = [p.strip() for p in os.environ["FLASHINFER_REBUILD_MODULES"].split(",") if "*" in p]
for provider in jit_env.FLASHINFER_AOT_PROVIDERS:
    manifest_path = provider.jit_cache_dir.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    affected = [name for name in manifest["modules"] if any(fnmatch.fnmatchcase(name, p) for p in patterns)]
    for name in affected:
        shutil.rmtree(provider.jit_cache_dir / name)
    manifest["modules"] = [name for name in manifest["modules"] if name not in affected]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
for directory in jit_env.FLASHINFER_AOT_DIR.iterdir():
    if directory.is_dir() and any(fnmatch.fnmatchcase(directory.name, p) for p in patterns):
        shutil.rmtree(directory)
PY
        )
        echo ">>> Reuse base image cubins, JIT-cache provider, and unchanged AOT modules"
    elif [[ "${FLASHINFER_INSTALL_MODE}" == reuse ]]; then
        echo ">>> Reuse official cubins and SM100a JIT-cache wheels"
        uv pip install --system --no-deps --only-binary=:all: \
            "flashinfer-cubin==${FLASHINFER_RELEASE_VERSION}" --index-url https://flashinfer.ai/whl
        uv pip install --system --no-deps --only-binary=:all: \
            "flashinfer-jit-cache==${FLASHINFER_CACHE_VERSION}" \
            "flashinfer-jit-cache-sm100a==${FLASHINFER_CACHE_VERSION}" \
            --index-url "https://flashinfer.ai/whl/${FLASHINFER_CACHE_VERSION##*+}"
    else
        # Cubins are downloaded/packaged; the provider compiles the broad AOT set.
        uv pip install --no-cache --system --no-deps --no-build-isolation ./flashinfer-cubin
        echo ">>> Build FlashInfer JIT-cache provider for ${FLASHINFER_JIT_CACHE_PROVIDER_ARCH}"
        uv pip install --no-cache --system --no-deps --no-build-isolation --verbose ./flashinfer-jit-cache-provider
        uv pip install --no-cache --system --no-deps --no-build-isolation ./flashinfer-jit-cache
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
if os.environ["FLASHINFER_INSTALL_MODE"] == "incremental":
    binary_sha = os.environ["FLASHINFER_BASE_BINARY_SHA"]
    cache_version = os.environ["FLASHINFER_BASE_CACHE_VERSION"]
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
import fnmatch
import hashlib
import os
import shutil
import sys
from functools import partial
from pathlib import Path

import torch

from flashinfer._build_meta import __git_commit__
from flashinfer.artifacts import ArtifactPath, CheckSumHash
from flashinfer.jit import env as jit_env
from flashinfer.jit.attention.modules import gen_trtllm_gen_fmha_module
from flashinfer.jit.core import build_jit_specs
from flashinfer.jit.fused_moe import gen_trtllm_gen_fused_moe_sm100_module
from flashinfer.jit.mamba.checkpointing_ssu import gen_checkpointing_ssu_module
from flashinfer.jit.mamba.replayssm_materialize import gen_replayssm_materialize_module

modules = [
    ("FMHA", gen_trtllm_gen_fmha_module, ArtifactPath.TRTLLM_GEN_FMHA, CheckSumHash.TRTLLM_GEN_FMHA),
    (
        "FUSED_MOE_SM100",
        gen_trtllm_gen_fused_moe_sm100_module,
        ArtifactPath.TRTLLM_GEN_BMM,
        CheckSumHash.TRTLLM_GEN_BMM,
    ),
]
# TP4 Nemotron: both PD prefill (one token) and speculative decode (six
# tokens), with RN and SR. Other shapes JIT in a build-specific workspace.
for rounds in (0, 5):
    for tokens in (1, 6):
        modules.append(
            (
                f"CHECKPOINTING_SSU_NP{tokens}_PR{rounds}",
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
                    npredicted=tokens,
                    max_window=16,
                    heads_per_group=16,
                    num_groups=2,
                    philox_rounds=rounds,
                    enable_pdl=False,
                ),
                None,
                None,
            )
        )
    modules.append(
        (
            f"REPLAYSSM_MATERIALIZE_PR{rounds}",
            partial(
                gen_replayssm_materialize_module,
                state_dtype=torch.float16,
                input_dtype=torch.bfloat16,
                matrixA_dtype=torch.float32,
                dim=64,
                dstate=128,
                heads_per_group=16,
                max_window=16,
                philox_rounds=rounds,
            ),
            None,
            None,
        )
    )
verify_only = sys.argv[1] == "--verify"
records = [f"FLASHINFER_SHA={__git_commit__}\n"]
expected = dict(line.split("=", 1) for line in Path(sys.argv[2]).read_text().splitlines()) if verify_only else {}
install_mode = expected["FLASHINFER_INSTALL_MODE"] if verify_only else os.environ["FLASHINFER_INSTALL_MODE"]
records.append(f"FLASHINFER_INSTALL_MODE={install_mode}\n")
base_records = {}
rebuild_modules = set()
if install_mode == "incremental" and not verify_only:
    base_records = dict(line.split("=", 1) for line in Path(os.environ["BUILD_DIR"], "base-warmup.env").read_text().splitlines())
    records.append(f"FLASHINFER_BASE_SHA={os.environ['FLASHINFER_BASE_SHA']}\n")
    patterns = [item.strip() for item in os.environ["FLASHINFER_REBUILD_MODULES"].split(",")]
    module_names = {generate_spec().name for _, generate_spec, _, _ in modules}
    for pattern in patterns:
        matches = {name for name in module_names if fnmatch.fnmatchcase(name, pattern)}
        if not matches:
            raise RuntimeError(f"No warmup generator matches {pattern}; available modules: {sorted(module_names)}")
        rebuild_modules.update(matches)
    records.append(f"FLASHINFER_REBUILD_MODULES={','.join(sorted(rebuild_modules))}\n")
if verify_only:
    assert __git_commit__ == expected["FLASHINFER_SHA"], "Installed FlashInfer commit changed"
for index, name in enumerate(
    (
        "kernel_checkpointing_ssu.cuh",
        "kernel_checkpointing_ssu_main.cuh",
        "kernel_checkpointing_ssu_common.cuh",
    )
):
    header = jit_env.FLASHINFER_INCLUDE_DIR / "flashinfer" / "mamba" / name
    digest = hashlib.sha256(header.read_bytes()).hexdigest()
    key = f"REPLAY_HEADER_{index}_SHA256"
    if verify_only:
        assert digest == expected[key], f"Installed header changed: {header}"
    else:
        records.append(f"{key}={digest}\n")
for label, generate_spec, artifact, expected_sha in modules:
    if artifact is not None and not verify_only:
        manifest = jit_env.FLASHINFER_CUBIN_DIR / artifact / "checksums.txt"
        manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
        if manifest_sha != expected_sha:
            raise RuntimeError(f"{label} cubin manifest does not match this FlashInfer branch: {manifest}")
        records.append(f"{label}_ARTIFACT_PATH={artifact}\n{label}_MANIFEST_SHA256={manifest_sha}\n")

    spec = generate_spec()
    destination = jit_env.FLASHINFER_AOT_DIR / spec.name / f"{spec.name}.so"
    if install_mode == "incremental" and not verify_only and spec.name not in rebuild_modules:
        destination = spec.aot_path
        assert str(destination) == base_records[f"{label}_DISPATCHER_PATH"], f"Base module resolution changed: {label}"
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        assert digest == base_records[f"{label}_DISPATCHER_SHA256"], f"Base module changed: {label}"
        records.append(
            f"{label}_MODULE_NAME={spec.name}\n{label}_DISPATCHER_PATH={destination}\n"
            f"{label}_DISPATCHER_SHA256={digest}\n"
        )
        print(f"Reusing verified base module: {destination}")
        continue
    reuse_binary = install_mode == "reuse" and artifact is not None
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
        destination = Path(expected[f"{label}_DISPATCHER_PATH"])
        assert spec.aot_path == destination and destination.is_file(), f"Missing warmed module: {destination}"
        assert hashlib.sha256(destination.read_bytes()).hexdigest() == expected[f"{label}_DISPATCHER_SHA256"], (
            f"Installed binary changed: {destination}"
        )
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

    export BUILD_DIR
    echo ">>> Warm up TRTLLM-GEN and ReplaySSM JIT modules (mode: ${FLASHINFER_INSTALL_MODE})"
    FLASHINFER_WORKSPACE_BASE="${BUILD_DIR}/warmup-cache" \
        python3 "${BUILD_DIR}/warmup.py" "${BUILD_DIR}/warmup.env"

    # Prove a fresh process can load all saved modules without their build
    # workspace or permission to JIT compile a replacement.
    rm -rf "${BUILD_DIR}/warmup-cache"
    FLASHINFER_WORKSPACE_BASE="${BUILD_DIR}/warmup-verify-cache" FLASHINFER_DISABLE_JIT=1 \
        python3 "${BUILD_DIR}/warmup.py" --verify "${BUILD_DIR}/warmup.env"
    install -m 644 "${BUILD_DIR}/warmup.py" "${BUILD_ROOT}/verify-flashinfer.py"
    install -m 644 "${BUILD_DIR}/warmup.env" "${BUILD_ROOT}/flashinfer-warmup.env"

    # Prims calls cute.compile directly. Preserve its disk cache in the image,
    # separately from the native .so overrides and disposable build workspace.
    if [[ "${FLASHINFER_PRIMS_WARMUP}" == 1 ]]; then
        export CUTE_DSL_CACHE_DIR="${BUILD_ROOT}/prims-cute-cache"
        export CUTE_DSL_DISABLE_FILE_CACHING=False
        mkdir -p "${CUTE_DSL_CACHE_DIR}"
        echo ">>> Warm Prims BF16 FC1/FC2 from ${FLASHINFER_PRIMS_WARMUP_DUMP}"
        python3 "${BUILD_DIR}/source/benchmarks/replay_flashinfer_dump.py" \
            --module flashinfer.fused_moe.backends.prims_ts.bf16_op \
            --function prims_ts_bf16_moe \
            --weight-layout "${FLASHINFER_PRIMS_WEIGHT_LAYOUT}" \
            --warmup "${FLASHINFER_PRIMS_WARMUP_ITERATIONS}" --iterations 1 \
            "${FLASHINFER_PRIMS_WARMUP_DUMP}"
        # A second process exercises disk-cache reuse and checks outputs again.
        python3 "${BUILD_DIR}/source/benchmarks/replay_flashinfer_dump.py" \
            --module flashinfer.fused_moe.backends.prims_ts.bf16_op \
            --function prims_ts_bf16_moe \
            --weight-layout "${FLASHINFER_PRIMS_WEIGHT_LAYOUT}" \
            --warmup 0 --iterations 1 "${FLASHINFER_PRIMS_WARMUP_DUMP}"
        if [[ -z "$(find "${CUTE_DSL_CACHE_DIR}" -type f ! -name .warmed -print -quit)" ]]; then
            echo "ERROR: Prims warmup produced no CuTe-DSL disk-cache files." >&2
            exit 1
        fi
        printf '%s\n' "${FLASHINFER_HEAD_SHA}" > "${CUTE_DSL_CACHE_DIR}/.warmed"
    fi

    # Leave the vLLM build manifest intact when layering onto a custom image.
    {
        printf 'BASE_IMAGE=%s\nFLASHINFER_REPO=%s\nFLASHINFER_BRANCH=%s\n' \
            "${BASE_IMAGE}" "${FLASHINFER_REPO}" "${FLASHINFER_BRANCH}"
        printf 'FLASHINFER_RELEASE_SHA=%s\n' "${FLASHINFER_RELEASE_SHA}"
        printf 'FLASHINFER_PRIMS_WARMUP=%s\n' "${FLASHINFER_PRIMS_WARMUP}"
        if [[ "${FLASHINFER_PRIMS_WARMUP}" == 1 ]]; then
            printf 'FLASHINFER_PRIMS_WARMUP_DUMP=%s\nFLASHINFER_PRIMS_WEIGHT_LAYOUT=%s\n' \
                "${FLASHINFER_PRIMS_WARMUP_DUMP}" "${FLASHINFER_PRIMS_WEIGHT_LAYOUT}"
            printf 'FLASHINFER_PRIMS_CUTE_CACHE=%s\n' "${CUTE_DSL_CACHE_DIR}"
        fi
        python3 -c 'import flashinfer; print(f"FLASHINFER_VERSION={flashinfer.__version__}")'
        cat "${BUILD_DIR}/jit-cache.env"
        cat "${BUILD_DIR}/warmup.env"
        printf 'FLASHINFER_REPLAY_VERIFIED=1\n'
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
if [[ "${FLASHINFER_PRIMS_WARMUP}" == 1 ]]; then
    FLASHINFER_PRIMS_WARMUP_DUMP="$(cd "${FLASHINFER_PRIMS_WARMUP_DUMP}" && pwd)"
    MOUNTS="${MOUNTS},${FLASHINFER_PRIMS_WARMUP_DUMP}:${FLASHINFER_PRIMS_WARMUP_DUMP}"
fi

echo "Base   ${BASE_IMAGE}"
echo "Source ${FLASHINFER_REPO} ${FLASHINFER_BRANCH} @ ${FLASHINFER_HEAD_SHA}"
echo "Output ${OUT_SQSH}"
if ! srun \
    --account="${SLURM_ACCOUNT}" \
    --partition="${SLURM_PARTITION:-batch}" \
    --job-name=super-vl-evals-flashinfer \
    --nodes=1 --ntasks=1 --segment=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
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
