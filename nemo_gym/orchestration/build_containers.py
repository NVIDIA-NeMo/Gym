# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import re
import shutil
from asyncio import create_subprocess_shell
from asyncio.subprocess import PIPE, STDOUT
from collections.abc import Iterator
from contextlib import contextmanager
from os import killpg
from pathlib import Path
from shlex import join, quote, split
from signal import SIGKILL, SIGTERM
from tempfile import TemporaryDirectory
from textwrap import dedent
from time import perf_counter
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, PrivateAttr, field_validator

from nemo_gym.config.config_types import BaseNeMoGymCLIConfig
from nemo_gym.config.global_config import get_global_config_dict


class FlashInferConfig(BaseNeMoGymCLIConfig):
    repo: str
    branch: str
    invalidate_modules: List[str]

    cpus_per_task: int


class VLLMConfig(BaseNeMoGymCLIConfig):
    """Install vLLM Python and Rust sources while reusing a compatible CUDA/C++ wheel."""

    repo: str
    branch: str
    version: str
    precompiled_wheel_commit: str


class MooncakeConfig(BaseNeMoGymCLIConfig):
    """Build Mooncake's CUDA 13 Transfer Engine and Store from source."""

    repo: str
    branch: str
    cpus_per_task: int = Field(gt=0)
    build_parallel_level: int = Field(gt=0)
    extra_apt_packages: Optional[List[str]] = None


class VLLMRouterConfig(BaseNeMoGymCLIConfig):
    """Build and install the router with the worker load-accounting fix."""

    repo: str
    branch: str
    cpus_per_task: int = Field(gt=0)


class WarmupConfig(BaseNeMoGymCLIConfig):
    """Run the vLLM startup-only launcher after installing all build stages.

    Environment values are literal (no shell expansion). CONTAINER,
    SAVE_WARMED_CONTAINER, and SBATCH_WAIT are managed by the builder.
    Account, partition, QoS, and GPU resources come from slurm; the time limit is 30 minutes.
    Run from the Gym checkout, as when invoking the launcher directly.
    """

    env: dict[str, str | int | float | bool]

    @field_validator("env")
    @classmethod
    def validate_env(cls, values: dict[str, str | int | float | bool]) -> dict[str, str | int | float | bool]:
        """Reject malformed names and overrides of stage-owned settings."""
        for key, value in values.items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError(f"Invalid warmup environment variable: {key!r}")
            if key in {
                "CONTAINER",
                "SAVE_WARMED_CONTAINER",
                "SBATCH_WAIT",
                "SBATCH_ACCOUNT",
                "SBATCH_PARTITION",
                "SBATCH_QOS",
                "SBATCH_GPUS_PER_NODE",
                "SBATCH_GRES",
            }:
                raise ValueError(f"{key} is managed by the warmup stage; omit it from warmup.env")
            if isinstance(value, str) and "\0" in value:
                raise ValueError(f"Warmup environment variable {key} contains a NUL byte")
        return values


class RunConfig(BaseNeMoGymCLIConfig):
    """Submit evaluation or serving with the built image; empty args launch only vLLM."""

    env: dict[str, str | int | float | bool]
    args: list[str] = Field(default_factory=list)

    @field_validator("args", mode="before")
    @classmethod
    def parse_args(cls, value: object) -> object:
        """Accept an argv list or shell-style quoting and line continuations, without shell expansion."""
        if isinstance(value, str):
            return split(value.replace("\\\n", ""))
        return value

    @field_validator("env")
    @classmethod
    def validate_env(cls, values: dict[str, str | int | float | bool]) -> dict[str, str | int | float | bool]:
        """Keep submission settings valid and prevent accidental warmup exports."""
        for key, value in values.items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError(f"Invalid run environment variable: {key!r}")
            if key in {"CONTAINER", "SAVE_WARMED_CONTAINER", "SBATCH_WAIT"}:
                raise ValueError(f"{key} is managed by the run stage; omit it from run.env")
            if isinstance(value, str) and "\0" in value:
                raise ValueError(f"Run environment variable {key} contains a NUL byte")
        return values


class BuildContainerConfig(BaseNeMoGymCLIConfig):
    input_container_fpath: Path
    output_container_fpath: Path
    log_file: Optional[Path] = None
    python_executable: Optional[str] = Field(
        default=None,
        description="Absolute Python path inside the container for package build/install stages; omit for system Python.",
    )

    @field_validator("python_executable")
    @classmethod
    def validate_python_executable(cls, value: Optional[str]) -> Optional[str]:
        """Validate container paths without resolving them on the submission host."""
        if value is not None and (not value.startswith("/") or "\0" in value):
            raise ValueError("python_executable must be an absolute path inside the container")
        return value

    slurm: Dict[str, Any] = Field(default_factory=dict)
    vllm: Optional[VLLMConfig] = None
    flashinfer: Optional[FlashInferConfig] = None
    mooncake: Optional[MooncakeConfig] = None
    vllm_router: Optional[VLLMRouterConfig] = None
    warmup: Optional[WarmupConfig] = None
    run: Optional[RunConfig] = None
    run_after_build: bool = False
    run_only: bool = False

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)

        if self.run_after_build and self.run_only:
            raise ValueError("--run and --run-only are mutually exclusive")
        if (self.run_after_build or self.run_only) and self.run is None:
            raise ValueError("Define run.env in the container config before using --run or --run-only")

        self.input_container_fpath = self.input_container_fpath.resolve()
        self.output_container_fpath = self.output_container_fpath.resolve()

        if not self.log_file:
            self.log_file = self.output_container_fpath.with_suffix(".log")
        self.log_file = self.log_file.resolve()


class CommandOutput(BaseModel):
    returncode: int
    output_lines: List[str]


async def _run_command(command: str, log_fpath: Path) -> CommandOutput:
    with log_fpath.open("a") as log_file:
        log_file.write(f"Command:\n{command}\n")

        process = await create_subprocess_shell(command, stdout=PIPE, stderr=STDOUT, start_new_session=True)
        output_lines = []

        try:
            assert process.stdout is not None

            async for line in process.stdout:
                line = line.decode(errors="replace").rstrip()

                print(line, flush=True)
                output_lines.append(line)
                log_file.write(line + "\n")

            returncode = await process.wait()
        except BaseException:
            # Let srun forward termination to its remote tasks and release its
            # allocation. SIGKILL alone strands those tasks on the cluster.
            try:
                killpg(process.pid, SIGTERM)
            except ProcessLookupError:
                pass  # Already exited.
            try:
                await asyncio.wait_for(process.communicate(), timeout=10)
            except TimeoutError:
                try:
                    killpg(process.pid, SIGKILL)
                except ProcessLookupError:
                    pass
                await process.communicate()
            raise

    return CommandOutput(
        returncode=returncode,
        output_lines=output_lines,
    )


class BaseBuilder(BaseModel):
    config: BaseNeMoGymCLIConfig

    # Log file is separate from the config so that we dont have the duplicate this log file property on every config.
    log_file: Path

    def _python_env_vars(self, container_config: BuildContainerConfig) -> dict[str, str]:
        """Select an existing container interpreter without forwarding host venv settings."""
        if container_config.python_executable is None:
            return {}
        return {"BUILD_PYTHON": container_config.python_executable}

    def _build_command(self, script: str, name: str, *args: str) -> str:
        """Use the selected interpreter for installs and subprocess build tools."""
        setup = dedent(
            r"""
            set -Eeuo pipefail
            BUILD_PIP_ARGS=(--system)
            if [[ -n "${BUILD_PYTHON:-}" ]]; then
                [[ -x "$BUILD_PYTHON" ]] || {
                    echo "Missing container Python executable: $BUILD_PYTHON" >&2; exit 1;
                }
                "$BUILD_PYTHON" -c 'import sys; print("Build Python:", sys.executable, "prefix:", sys.prefix)'
                export PATH="$(dirname "$BUILD_PYTHON"):$PATH"
                export PYO3_PYTHON="$BUILD_PYTHON"
                BUILD_PIP_ARGS=(--python "$BUILD_PYTHON")
            else
                BUILD_PYTHON=python3
            fi
            """
        ).lstrip()
        return join(["bash", "-c", setup + script, name, *args])

    async def perform_preflight_checks(self) -> None:
        raise NotImplementedError

    def get_srun_command_str(self) -> str:
        raise NotImplementedError

    def get_srun_env_vars(self, container_config: BuildContainerConfig) -> Dict[str, Any]:
        raise NotImplementedError

    def get_srun_slurm_vars(self, container_config: BuildContainerConfig) -> Dict[str, Any]:
        raise NotImplementedError

    async def _preflight_check_git_repo(self, repo: str, branch: str) -> str:
        """
        Returns the head sha
        """

        ref = f"refs/heads/{branch}"
        command_output = await _run_command(f"git ls-remote --exit-code {quote(repo)} {quote(ref)}", self.log_file)

        error_message = f"Preflight git repo check failed for {repo} @ {branch}. See {self.log_file}"
        if command_output.returncode != 0 or not command_output.output_lines:
            raise RuntimeError(error_message)

        matches = [line.split() for line in command_output.output_lines if line.split()[-1:] == [ref]]
        if len(matches) != 1 or len(matches[0]) != 2 or not re.fullmatch(r"[0-9a-f]{40}", matches[0][0]):
            raise RuntimeError(error_message)

        head_sha = matches[0][0]
        print(f"Preflight git repo for {repo} @ {branch} found commit SHA {head_sha}")
        return head_sha


class FlashInferBuilder(BaseBuilder):
    config: FlashInferConfig

    _head_sha: Optional[str] = PrivateAttr(default=None)

    async def perform_preflight_checks(self) -> None:
        """Resolve the branch once so a queued build installs the checked commit."""
        self._head_sha = None
        self._head_sha = await self._preflight_check_git_repo(self.config.repo, self.config.branch)

    def get_srun_command_str(self) -> str:
        """Reuse matching binaries, align cubins if needed, and invalidate modules without JIT."""
        if self._head_sha is None:
            raise RuntimeError("Run FlashInfer preflight checks before generating the build command")

        for pattern in self.config.invalidate_modules:
            if not re.fullmatch(r"[A-Za-z0-9_*?-]+", pattern):
                raise ValueError(f"Invalid FlashInfer module name/glob: {pattern!r}")

        script = dedent(
            r"""
            set -Eeuo pipefail
            command -v git >/dev/null || { echo "Missing build tool: git" >&2; exit 1; }
            BUILD_DIR=$(mktemp -d /tmp/flashinfer-slim.XXXXXX)
            export BUILD_DIR
            trap 'cd /; rm -rf "$BUILD_DIR"' EXIT
            unset FLASHINFER_DISABLE_JIT FLASHINFER_DISABLE_VERSION_CHECK FLASHINFER_CUBIN_DIR
            git init "$BUILD_DIR/source"
            git -C "$BUILD_DIR/source" remote add origin "$FLASHINFER_REPO"
            git -C "$BUILD_DIR/source" fetch --depth=1 origin "$FLASHINFER_HEAD_SHA"
            git -C "$BUILD_DIR/source" checkout --detach FETCH_HEAD
            git -C "$BUILD_DIR/source" submodule update --init --recursive --depth=1

            command -v uv >/dev/null || "$BUILD_PYTHON" -m pip install uv
            # Python 1: check the Python release through metadata before importing
            # FlashInfer, which rejects mismatched cubins at import time. Align only
            # the cubin wheel when needed, then preserve the remaining binary packages.
            # Save the version suffix and back up existing AOT overrides.
            # Run outside the checkout so imports refer to the base image's package.
            cd /
            "$BUILD_PYTHON" - "${BUILD_PIP_ARGS[@]}" <<'PY'
            import importlib.metadata as metadata
            import json
            import os
            import shutil
            import subprocess
            import sys
            from pathlib import Path

            root = Path(os.environ["BUILD_DIR"])
            release = (root / "source/version.txt").read_text().strip()
            version = metadata.version("flashinfer-python")
            if version.partition("+")[0] != release:
                raise RuntimeError(f"Slim builds require matching releases: branch={release}, Python={version}")
            try:
                cubin_version = metadata.version("flashinfer-cubin")
            except metadata.PackageNotFoundError:
                cubin_version = None
            if cubin_version != version:
                print(f"Aligning flashinfer-cubin: {cubin_version or 'not installed'} -> {version}", flush=True)
                subprocess.run(
                    ["uv", "pip", "install", *sys.argv[1:], "--no-deps", "--only-binary=:all:",
                     "--index-url", "https://flashinfer.ai/whl", f"flashinfer-cubin==={version}"],
                    check=True,
                )
                if metadata.version("flashinfer-cubin") != version:
                    raise RuntimeError(f"Failed to install flashinfer-cubin=={version} in {sys.executable}")

            import flashinfer
            import flashinfer_cubin
            from flashinfer.jit import env

            if flashinfer.__version__ != version or flashinfer_cubin.__version__ != version:
                raise RuntimeError("FlashInfer imports do not match the installed package metadata")
            (root / "local-version").write_text(version.partition("+")[2])
            if env.FLASHINFER_AOT_DIR.exists():
                shutil.copytree(env.FLASHINFER_AOT_DIR, root / "aot")
            packages = {dist.metadata["Name"]: dist.version for dist in metadata.distributions()
                        if dist.metadata["Name"].lower().replace("_", "-").startswith("flashinfer-")
                        and dist.metadata["Name"].lower().replace("_", "-") != "flashinfer-python"}
            (root / "binary-packages.json").write_text(json.dumps(packages))
            (root / "old-cache-root").write_text(str(env.FLASHINFER_CACHE_DIR))
            PY
            export FLASHINFER_LOCAL_VERSION="$(cat "$BUILD_DIR/local-version")"
            unset FLASHINFER_DEV_RELEASE_SUFFIX
            export FLASHINFER_BUILD_NO_PIP=1
            export BUILD_NVEP=0 BUILD_NIXL_EP=0 BUILD_NCCL_EP=0
            # Python 2 (pip): replace only flashinfer-python with this branch. Use the
            # image's build dependencies (--no-build-isolation) and leave runtime
            # dependencies, including the aligned cubins/providers, installed as-is (--no-deps).
            uv pip install "${BUILD_PIP_ARGS[@]}" --no-deps --no-build-isolation --force-reinstall "$BUILD_DIR/source"

            # Python 3: restore unrelated AOT overrides, remove matching module families
            # from package/provider/JIT caches, and update provider manifests. No spec
            # generators, shape lists, CUDA compilation, or kernel launches are needed.
            "$BUILD_PYTHON" - <<'PY'
            import fnmatch
            import importlib.metadata as metadata
            import json
            import os
            import shutil
            from pathlib import Path

            from flashinfer._build_meta import __git_commit__
            from flashinfer.jit import env

            root = Path(os.environ["BUILD_DIR"])
            if __git_commit__ != os.environ["FLASHINFER_HEAD_SHA"]:
                raise RuntimeError("Installed FlashInfer source commit differs from requested commit")
            packages = json.loads((root / "binary-packages.json").read_text())
            for package, version in packages.items():
                if metadata.version(package) != version:
                    raise RuntimeError(f"Unexpected binary package change: {package}")
            if (root / "aot").exists():
                shutil.copytree(root / "aot", env.FLASHINFER_AOT_DIR, dirs_exist_ok=True)
            patterns = os.environ["FLASHINFER_INVALIDATE_MODULES"].split(",")


            def selected(name: str) -> bool:
                return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


            removed = []
            # Scan all architectures, including cached specializations not used at startup.
            module_roots = set(env.FLASHINFER_AOT_DIRS)
            cache_roots = {Path((root / "old-cache-root").read_text()), env.FLASHINFER_CACHE_DIR,
                           Path("/opt/flashinfer-warmup-workspace")}
            for cache_root in cache_roots:
                module_roots.update(cache_root.glob("**/cached_ops"))
            for module_root in sorted(module_roots):
                for directory in sorted(module_root.glob("*")):
                    if selected(directory.name):
                        if directory.is_symlink():
                            directory.unlink()
                        elif directory.is_dir():
                            shutil.rmtree(directory)
                        else:
                            continue
                        removed.append(str(directory))
                        print(f"Invalidated {directory}", flush=True)
            # Remove provider registry entries as well as binaries so missing entries fall
            # through to JIT rather than advertising a prebuilt specialization.
            for provider in env.FLASHINFER_AOT_PROVIDERS:
                manifest_path = provider.jit_cache_dir.parent / "manifest.json"
                manifest = json.loads(manifest_path.read_text())
                remaining = [name for name in manifest["modules"] if not selected(name)]
                if remaining != manifest["modules"]:
                    manifest["modules"] = remaining
                    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            PY

            # These derived caches/plans describe the previous installation. Recreate
            # them during warmup instead of validating stale manifests or reusing graphs.
            rm -rf /opt/flashinfer-autotune-cache \
                /opt/vllm-cache /opt/prims-cute-cache

            exit 0
            """
        ).lstrip()

        return self._build_command(script, "flashinfer-build")

    def get_srun_env_vars(self, container_config: BuildContainerConfig) -> Dict[str, Any]:
        return {
            **self._python_env_vars(container_config),
            "FLASHINFER_REPO": self.config.repo,
            "FLASHINFER_HEAD_SHA": self._head_sha,
            "FLASHINFER_INVALIDATE_MODULES": ",".join(self.config.invalidate_modules),
            "FLASHINFER_NVCC_THREADS": "1",
            "MAX_JOBS": "64",
        }

    def get_srun_slurm_vars(self, container_config: BuildContainerConfig) -> Dict[str, Any]:
        return {
            "job-name": "build-flashinfer",
            "nodes": "1",
            "ntasks": "1",
            "segment": "1",
            "cpus-per-task": f"{self.config.cpus_per_task}",
            "time": "01:00:00",
            "container-image": str(container_config.output_container_fpath),
            "no-container-mount-home": True,
            "container-save": str(container_config.output_container_fpath),
        }


class VLLMBuilder(BaseBuilder):
    """Build a thin vLLM installation with Rust artifacts from the pinned source."""

    config: VLLMConfig
    _head_sha: Optional[str] = PrivateAttr(default=None)

    async def perform_preflight_checks(self) -> None:
        """Resolve the branch once so a queued build installs the checked commit."""
        self._head_sha = None
        self._head_sha = await self._preflight_check_git_repo(self.config.repo, self.config.branch)

    def get_srun_command_str(self) -> str:
        """Reuse wheel CUDA/C++ binaries, rebuild Rust, and verify the installed package."""
        if self._head_sha is None:
            raise RuntimeError("Run vLLM preflight checks before generating the build command")

        script = dedent(
            r"""
            set -Eeuo pipefail
            BUILD_DIR=$(mktemp -d /tmp/vllm-build.XXXXXX)
            export BUILD_DIR
            trap 'cd /; rm -rf "$BUILD_DIR"' EXIT
            command -v python >/dev/null || ln -sf "$(command -v python3)" /usr/local/bin/python
            export DEBIAN_FRONTEND=noninteractive
            apt-get update -y
            apt-get install -y --no-install-recommends git ca-certificates curl build-essential pkg-config perl \
                2>&1 | tail -5
            command -v uv >/dev/null || "$BUILD_PYTHON" -m pip install uv 2>&1 | tail -3

            git init "$BUILD_DIR/source"
            git -C "$BUILD_DIR/source" remote add origin "$VLLM_REPO"
            git -C "$BUILD_DIR/source" fetch --depth=1 origin "$VLLM_HEAD_SHA"
            git -C "$BUILD_DIR/source" checkout --detach FETCH_HEAD
            cd "$BUILD_DIR/source"
            vllm_sha=$(git rev-parse HEAD)
            test "$vllm_sha" = "$VLLM_HEAD_SHA"

            # The wheel supplies CUDA/C++ code; only Python, Rust and the script's
            # allowed metadata/documentation changes can differ from that wheel.
            git fetch --depth=1 https://github.com/vllm-project/vllm.git "$VLLM_PRECOMPILED_WHEEL_COMMIT"
            precompiled_base_sha=$(git rev-parse 'FETCH_HEAD^{commit}')
            changed_files=$(git diff --name-only "$precompiled_base_sha" HEAD --)
            unsupported_changes=$(
                printf '%s\n' "$changed_files" | grep -Ev \
                    '(^$|\.(py|txt|yaml)$|^rust/|^rust-toolchain\.toml$|^tools/build_rust\.sh$|^docs/.*\.md$)' || true
            )
            if [[ -n "$unsupported_changes" ]]; then
                echo "ERROR: precompiled CUDA/C++ build cannot consume these changes (Rust is rebuilt separately):" >&2
                echo "$unsupported_changes" >&2
                exit 1
            fi

            export VLLM_USE_PRECOMPILED=1
            export VLLM_PRECOMPILED_WHEEL_COMMIT
            export SETUPTOOLS_SCM_PRETEND_VERSION="$VLLM_VERSION"
            uv pip install "${BUILD_PIP_ARGS[@]}" . --prerelease=allow --torch-backend=auto --index-strategy unsafe-best-match

            # Installation copies wheel Rust artifacts too. Rebuild AFTER installing
            # and unset both reuse switches so the source and launcher stay in sync.
            uv pip install "${BUILD_PIP_ARGS[@]}" 'setuptools>=77.0.3,<81' 'setuptools-scm>=9.2.0' 'setuptools-rust>=1.9.0' wheel
            env -u VLLM_USE_PRECOMPILED -u VLLM_USE_PRECOMPILED_RUST bash tools/build_rust.sh
            installed_vllm_dir=$(cd / && "$BUILD_PYTHON" -c \
                'import vllm; from pathlib import Path; print(Path(vllm.__file__).parent)')
            install -m 755 vllm/vllm-rs "$installed_vllm_dir/vllm-rs"
            cmp vllm/vllm-rs "$installed_vllm_dir/vllm-rs"
            for rust_extension in vllm/_rust_*.so; do
                [[ -f "$rust_extension" ]] || continue
                install -m 755 "$rust_extension" "$installed_vllm_dir/"
            done

            rm -rf /opt/uv/cache /root/.cache/uv /root/.cache/pip /var/lib/apt/lists/* 2>/dev/null || true
            apt-get clean || true
            rm -rf "$BUILD_DIR/source"
            find /usr/lib/aarch64-linux-gnu /usr/local/cuda-13.0 -name '*_static*.a' -delete 2>/dev/null || true
            """
        ).lstrip()
        return self._build_command(script, "vllm-build")

    def get_srun_env_vars(self, container_config: BuildContainerConfig) -> Dict[str, Any]:
        """Pass build configuration as literal environment values."""
        if self._head_sha is None:
            raise RuntimeError("Run vLLM preflight checks before generating the build environment")
        return {
            **self._python_env_vars(container_config),
            "BASE_IMAGE": str(container_config.input_container_fpath),
            "VLLM_REPO": self.config.repo,
            "VLLM_BRANCH": self.config.branch,
            "VLLM_HEAD_SHA": self._head_sha,
            "VLLM_VERSION": self.config.version,
            "VLLM_PRECOMPILED_WHEEL_COMMIT": self.config.precompiled_wheel_commit,
        }

    def get_srun_slurm_vars(self, container_config: BuildContainerConfig) -> Dict[str, Any]:
        """Build into the working image used by all container stages."""
        return {
            "job-name": "build-vllm",
            "nodes": "1",
            "ntasks": "1",
            "segment": "1",
            "gpus-per-node": "4",
            "mem": "0",
            "time": "01:00:00",
            "container-image": str(container_config.output_container_fpath),
            "no-container-mount-home": True,
            "container-save": str(container_config.output_container_fpath),
        }


class VLLMRouterBuilder(BaseBuilder):
    """Install a source-built router wheel into the working container."""

    config: VLLMRouterConfig
    _head_sha: Optional[str] = PrivateAttr(default=None)

    async def perform_preflight_checks(self) -> None:
        """Resolve the branch once so queued builds use the checked commit."""
        self._head_sha = None
        self._head_sha = await self._preflight_check_git_repo(self.config.repo, self.config.branch)

    def get_srun_command_str(self) -> str:
        """Build with the container's Python and verify the load-accounting fix."""
        if self._head_sha is None:
            raise RuntimeError("Run router preflight checks before generating the build command")
        script = dedent(
            r"""
            set -Eeuo pipefail
            export DEBIAN_FRONTEND=noninteractive
            command -v uv >/dev/null
            # Some aarch64 images fail the libc-bin trigger on 64K-page kernels.
            # Check the installed tools and headers even if dpkg reports failure.
            apt-get update
            apt-get install -y --no-install-recommends \
                build-essential pkg-config libssl-dev protobuf-compiler curl git ca-certificates \
                || { dpkg --configure -a || true; }
            for tool in cc c++ pkg-config protoc curl git; do
                command -v "$tool" >/dev/null || { echo "Missing build tool: $tool" >&2; exit 1; }
            done
            test -f /usr/include/openssl/ssl.h

            BUILD_DIR=$(mktemp -d /tmp/vllm-router-build.XXXXXX)
            export BUILD_DIR
            trap 'cd /; rm -rf "$BUILD_DIR"' EXIT
            export CARGO_HOME="$BUILD_DIR/cargo"
            export RUSTUP_HOME="$BUILD_DIR/rustup"
            export CARGO_TARGET_DIR="$BUILD_DIR/target"
            export PATH="$CARGO_HOME/bin:$PATH"
            export CARGO_INCREMENTAL=0
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
                | sh -s -- -y --no-modify-path --profile minimal --default-toolchain 1.95.0
            rustc --version
            cargo --version

            git init "$BUILD_DIR/source"
            cd "$BUILD_DIR/source"
            git remote add origin "$VLLM_ROUTER_REPO"
            git fetch --depth=1 origin "$VLLM_ROUTER_COMMIT"
            git checkout --detach FETCH_HEAD
            test "$(git rev-parse HEAD)" = "$VLLM_ROUTER_COMMIT"
            # grep errors must not be mistaken for absence of the bad code.
            test -d src
            if grep -rn 'LOAD_RESET_INTERVAL' src/; then
                echo 'ERROR: periodic load reset still present; expected router#216' >&2
                exit 1
            else
                test "$?" -eq 1
            fi
            cargo test --lib -- \
                core::worker::tests::test_worker_load_guard \
                core::worker::tests::test_decrement_at_zero_clamps \
                core::worker::tests::test_load_guard \
                core::worker_registry::tests::test_health_checker_preserves

            # Match the interpreter that will run the router, without installing
            # the wheel build dependencies into system site-packages.
            buildenv="$BUILD_DIR/buildenv"
            uv venv --python "$BUILD_PYTHON" "$buildenv"
            uv pip install --python "$buildenv/bin/python" setuptools wheel setuptools-rust build
            "$buildenv/bin/python" -m build --wheel --no-isolation --outdir dist
            wheels=(dist/*.whl)
            test "${#wheels[@]}" -eq 1 && test -f "${wheels[0]}"
            uv pip install "${BUILD_PIP_ARGS[@]}" --force-reinstall "${wheels[0]}"
            # Import outside the source checkout to test the installed package.
            cd /
            "$BUILD_PYTHON" -c "import vllm_router_rs; print('vllm_router_rs import OK')"
            vllm-router --help >/dev/null
            router_so=$("$BUILD_PYTHON" -c 'import vllm_router_rs; print(vllm_router_rs.__file__)')
            if grep -aqF 'Resetting worker loads (cycle' "$router_so"; then
                echo 'ERROR: installed router still resets worker loads periodically' >&2
                exit 1
            else
                test "$?" -eq 1
            fi
            if ! grep -aqF 'Attempted to decrement load counter that is already at 0' "$router_so"; then
                echo 'ERROR: installed router is missing the #216 load-guard changes' >&2
                exit 1
            fi
            uv pip show "${BUILD_PIP_ARGS[@]}" vllm-router
            rm -rf "$BUILD_DIR"
            uv cache clean
            apt-get clean
            rm -rf /var/lib/apt/lists/*
            """
        ).lstrip()
        return self._build_command(script, "vllm-router-build")

    def get_srun_env_vars(self, container_config: BuildContainerConfig) -> Dict[str, Any]:
        """Pass pinned source and compilation concurrency literally."""
        if self._head_sha is None:
            raise RuntimeError("Run router preflight checks before generating the build environment")
        return {
            **self._python_env_vars(container_config),
            "VLLM_ROUTER_REPO": self.config.repo,
            "VLLM_ROUTER_COMMIT": self._head_sha,
            "CARGO_BUILD_JOBS": str(self.config.cpus_per_task),
        }

    def get_srun_slurm_vars(self, container_config: BuildContainerConfig) -> Dict[str, Any]:
        """Save the installed wheel in the image used by subsequent stages."""
        return {
            "job-name": "build-vllm-router",
            "nodes": "1",
            "ntasks": "1",
            "segment": "1",
            "cpus-per-task": str(self.config.cpus_per_task),
            "mem": "0",
            "time": "00:15:00",
            "container-image": str(container_config.output_container_fpath),
            "no-container-mount-home": True,
            "container-save": str(container_config.output_container_fpath),
        }


class MooncakeBuilder(BaseBuilder):
    """Build CUDA 13 Mooncake and smoke-test the installed Store over TCP."""

    config: MooncakeConfig
    _head_sha: Optional[str] = PrivateAttr(default=None)

    async def perform_preflight_checks(self) -> None:
        """Pin the configured branch before submitting the build."""
        self._head_sha = None
        self._head_sha = await self._preflight_check_git_repo(self.config.repo, self.config.branch)

    def get_srun_command_str(self) -> str:
        """Build the native Store and Transfer Engine, then verify installed artifacts."""
        if self._head_sha is None:
            raise RuntimeError("Run Mooncake preflight checks before generating the build command")
        script = dedent(
            r"""
            set -Eeuo pipefail
            export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
            : "${BASE_IMAGE:?}" "${MOONCAKE_HEAD_SHA:?}"
            echo ">>> Mooncake ${MOONCAKE_BRANCH} @ ${MOONCAKE_HEAD_SHA} on $(hostname)"
            if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]] || ! "${CUDA_HOME}/bin/nvcc" --version | grep -q 'release 13\.'; then
                echo "ERROR: BASE_IMAGE needs a CUDA 13 development toolkit at CUDA_HOME=${CUDA_HOME}." >&2
                exit 1
            fi
            export PATH="${CUDA_HOME}/bin:${PATH}"
            export DEBIAN_FRONTEND=noninteractive
            apt-get update -y
            # Install the C++ Store/TE dependencies directly; dependencies.sh also
            # installs Go and optional backends that this serving configuration omits.
            apt-get install -y --no-install-recommends \
                git ca-certificates build-essential cmake ninja-build pkg-config python3-dev \
                libibverbs-dev libgoogle-glog-dev libgflags-dev libjsoncpp-dev libunwind-dev \
                libnuma-dev libboost-dev libssl-dev libyaml-cpp-dev libcurl4-openssl-dev \
                liburing-dev libmsgpack-dev libzstd-dev libasio-dev libxxhash-dev
            # Newer Ubuntu releases split the C++ headers out of libmsgpack-dev.
            if apt-cache show libmsgpack-cxx-dev >/dev/null 2>&1; then
                apt-get install -y --no-install-recommends libmsgpack-cxx-dev
            fi
            if (( $# )); then
                # Restore files that earlier image stages may have stripped.
                apt-get install -y --no-install-recommends --reinstall -- "$@"
            fi
            command -v uv >/dev/null || "$BUILD_PYTHON" -m pip install uv

            BUILD_DIR=$(mktemp -d /tmp/mooncake-build.XXXXXX)
            export BUILD_DIR
            trap 'cd /; rm -rf "${BUILD_DIR}"' EXIT
            git init "${BUILD_DIR}/source"
            cd "${BUILD_DIR}/source"
            git remote add origin "${MOONCAKE_REPO}"
            # Use the commit resolved on the login node even if main moves while queued.
            git fetch --depth=1 origin "${MOONCAKE_HEAD_SHA}"
            git checkout --detach FETCH_HEAD
            test "$(git rev-parse HEAD)" = "${MOONCAKE_HEAD_SHA}"
            git submodule update --init --recursive --depth=1

            # Both distributions own the same mooncake package and CLI entry points.
            rm -f /opt/mooncake-build.env
            uv pip uninstall "${BUILD_PIP_ARGS[@]}" mooncake-transfer-engine-cuda13 mooncake-transfer-engine
            uv pip install "${BUILD_PIP_ARGS[@]}" . \
                --config-settings=cmake.build-type=Release \
                --config-settings=cmake.define.USE_CUDA=ON \
                --config-settings="cmake.define.CUDAToolkit_ROOT=${CUDA_HOME}" \
                --config-settings=cmake.define.WITH_TE=ON \
                --config-settings=cmake.define.WITH_STORE=ON \
                --config-settings=cmake.define.BUILD_EXAMPLES=OFF \
                --config-settings=cmake.define.BUILD_UNIT_TESTS=OFF \
                --config-settings=cmake.define.BUILD_BENCHMARK=OFF \
                --config-settings=cmake.define.WITH_EP=OFF \
                --config-settings=cmake.define.WITH_STORE_GO=OFF \
                --config-settings=cmake.define.WITH_STORE_RUST=OFF \
                --config-settings=cmake.define.WITH_STORE_RS=OFF \
                --config-settings=cmake.define.WITH_P2P_STORE=OFF

            # Publish provenance only after installation succeeds. Quote values so
            # the manifest can also be sourced safely as shell assignments.
            printf 'MOONCAKE_REPO=%q\nMOONCAKE_BRANCH=%q\nMOONCAKE_HEAD_SHA=%q\nBASE_IMAGE=%q\n' \
                "$MOONCAKE_REPO" "$MOONCAKE_BRANCH" "$MOONCAKE_HEAD_SHA" "$BASE_IMAGE" \
                > /opt/mooncake-build.env

            cd /
            rm -rf "${BUILD_DIR}"
            uv cache clean
            apt-get clean
            rm -rf /var/lib/apt/lists/*
            """
        ).lstrip()
        return self._build_command(script, "mooncake-build", *(self.config.extra_apt_packages or []))

    def get_srun_env_vars(self, container_config: BuildContainerConfig) -> Dict[str, Any]:
        """Pass source and build settings as literal environment values."""
        if self._head_sha is None:
            raise RuntimeError("Run Mooncake preflight checks before generating the build environment")
        return {
            **self._python_env_vars(container_config),
            "BASE_IMAGE": str(container_config.input_container_fpath),
            "MOONCAKE_REPO": self.config.repo,
            "MOONCAKE_BRANCH": self.config.branch,
            "MOONCAKE_HEAD_SHA": self._head_sha,
            "CMAKE_BUILD_PARALLEL_LEVEL": str(self.config.build_parallel_level),
        }

    def get_srun_slurm_vars(self, container_config: BuildContainerConfig) -> Dict[str, Any]:
        """Build into the working image shared by all stages."""
        return {
            "job-name": "build-mooncake",
            "nodes": "1",
            "ntasks": "1",
            "segment": "1",
            "cpus-per-task": str(self.config.cpus_per_task),
            "gpus-per-node": "4",
            "mem": "0",
            "time": "00:30:00",
            "container-image": str(container_config.output_container_fpath),
            "no-container-mount-home": True,
            "container-save": str(container_config.output_container_fpath),
        }


@contextmanager
def _timed_stage(name: str) -> Iterator[None]:
    """Report wall time, including preflight checks and time waiting for Slurm."""
    start = perf_counter()
    print(f"[{name}] starting", flush=True)
    status = "failed"
    try:
        yield
        status = "completed"
    finally:
        print(f"[{name}] {status} in {perf_counter() - start:.1f}s", flush=True)


class BuildContainerHelper(BaseModel):
    config: BuildContainerConfig

    def _slurm_launcher_env(self) -> dict[str, str]:
        mapping = {
            "account": "SBATCH_ACCOUNT",
            "partition": "SBATCH_PARTITION",
            "qos": "SBATCH_QOS",
            "gpus-per-node": "SBATCH_GPUS_PER_NODE",
            "gres": "SBATCH_GRES",
        }
        return {env_var: str(self.config.slurm[key]) for key, env_var in mapping.items() if key in self.config.slurm}

    async def run(self) -> None:
        try:
            await self._run()
        except:
            await self._clean_on_failure()
            raise

    async def _run(self) -> None:
        print(f"""Resolved fpaths:
- Input container: {self.config.input_container_fpath}
- Output container: {self.config.output_container_fpath}
- Log file: {self.config.log_file}
""")
        self.config.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.config.log_file.unlink(missing_ok=True)
        self.config.output_container_fpath.parent.mkdir(parents=True, exist_ok=True)

        with _timed_stage("Copy container"):
            shutil.copy(self.config.input_container_fpath, self.config.output_container_fpath)
        print("Copied input container fpath to output (not finished, just starting)")

        builders: list[BaseBuilder] = []
        if self.config.vllm is not None:
            builders.append(VLLMBuilder(config=self.config.vllm, log_file=self.config.log_file))
        if self.config.flashinfer is not None:
            builders.append(FlashInferBuilder(config=self.config.flashinfer, log_file=self.config.log_file))
        if self.config.vllm_router is not None:
            builders.append(VLLMRouterBuilder(config=self.config.vllm_router, log_file=self.config.log_file))

        # Install Mooncake last so earlier dependency resolution cannot replace it.
        if self.config.mooncake is not None:
            builders.append(MooncakeBuilder(config=self.config.mooncake, log_file=self.config.log_file))

        for builder in builders:
            with _timed_stage(type(builder).__name__.removesuffix("Builder")):
                await builder.perform_preflight_checks()
                command_output = await self._run_single_srun(
                    builder_command_str=builder.get_srun_command_str(),
                    builder_env_vars=builder.get_srun_env_vars(self.config),
                    builder_srun_slurm_vars=builder.get_srun_slurm_vars(self.config),
                )
                if command_output.returncode != 0:
                    raise RuntimeError(
                        f"Command failed with returncode {command_output.returncode}. See {self.config.log_file}"
                    )

        if self.config.warmup is not None:
            print("Running warmup. Currently, the warmup logs aren't printed and need to be checked manually.")
            with _timed_stage("Warmup"):
                await self._warmup_container()

    async def _warmup_container(self) -> None:
        # TODO @bxyu-nvidia: Eventually this sbatch_external_vllm will migrate to Gym-native orchestration and we can just import the relevant thing
        launcher = Path("benchmarks/nemotron_3.5_super/sbatch_external_vllm.sh").resolve()
        if not launcher.is_file():
            raise FileNotFoundError(f"Warmup launcher not found: {launcher}. Run from the Gym checkout.")
        assert self.config.warmup is not None
        output = self.config.output_container_fpath
        # The launcher publishes via a hard link and requires a nonexistent target.
        # Keep the input image intact until the completed job has exported its caches.
        # The launcher mounts a directory beneath this path. Image names can
        # contain colons, which Pyxis interprets as mount-field separators.
        with TemporaryDirectory(prefix=".vllm-warmup-", dir=output.parent) as directory:
            warmed = Path(directory) / "warmed.sqsh"
            job_file = Path(directory) / "job-id"
            env = {
                key: str(int(value)) if isinstance(value, bool) else str(value)
                for key, value in self.config.warmup.env.items()
            }
            env.update(self._slurm_launcher_env())
            env.update(
                CONTAINER=str(output),
                SAVE_WARMED_CONTAINER=str(warmed),
                SBATCH_WAIT="1",
            )
            # sbatch prints its ID before --wait completes. Record it without
            # changing the launcher's output, so interruption can cancel that job.
            script = (
                "set -euo pipefail\n"
                f'sbatch() {{ command sbatch "$@" | tee {quote(str(job_file))}; }}\n'
                "export -f sbatch\n"
                f"exec bash {quote(str(launcher))}\n"
            )
            command = join(["env", *(f"{key}={value}" for key, value in env.items()), "bash", "-c", script])
            try:
                result = await _run_command(command, self.config.log_file)
            except BaseException:
                # Killing the local sbatch waiter does not cancel the allocation.
                job = job_file.read_text().strip() if job_file.exists() else ""
                match = re.fullmatch(r"([0-9]+)(?:;([A-Za-z0-9_.-]+))?", job)
                if match:
                    cancel = ["scancel"]
                    if match[2]:
                        cancel.extend(["--clusters", match[2]])
                    cancel.append(match[1])
                    cancelled = await _run_command(join(cancel), self.config.log_file)
                    if cancelled.returncode:
                        print(f"Could not cancel warmup job {job}; see {self.config.log_file}")
                raise
            if result.returncode != 0:
                raise RuntimeError(f"Warmup failed with returncode {result.returncode}. See {self.config.log_file}")
            if not warmed.is_file() or warmed.stat().st_size == 0:
                raise RuntimeError(
                    f"Warmup completed without a nonempty image at {warmed}. See {self.config.log_file}"
                )
            warmed.replace(output)
        print(f"Saved warmed container: {output}")

    async def _run_container(self) -> None:
        """Submit the YAML's run configuration without rebuilding or modifying the image."""
        config = self.config.run
        if config is None:
            raise ValueError("Define run.env in the container config before using --run or --run-only")
        launcher = Path("benchmarks/nemotron_3.5_super/sbatch_external_vllm.sh").resolve()
        if not launcher.is_file():
            raise FileNotFoundError(f"Launcher not found: {launcher}. Run from the Gym checkout.")
        if not self.config.output_container_fpath.is_file():
            raise FileNotFoundError(f"Built image not found: {self.config.output_container_fpath}")
        env = self._slurm_launcher_env()
        env.update(
            {key: str(int(value)) if isinstance(value, bool) else str(value) for key, value in config.env.items()}
        )
        env["CONTAINER"] = str(self.config.output_container_fpath)
        if config.args and not env.get("EXPERIMENT_NAME"):
            raise ValueError("Set run.env.EXPERIMENT_NAME for the evaluation run")
        self.config.log_file.parent.mkdir(parents=True, exist_ok=True)
        Path("slurm-logs").mkdir(exist_ok=True)
        command = join(
            [
                "env",
                "-u",
                "SAVE_WARMED_CONTAINER",
                "-u",
                "SBATCH_WAIT",
                *(f"{key}={value}" for key, value in env.items()),
                "bash",
                str(launcher),
                *config.args,
            ]
        )
        result = await _run_command(command, self.config.log_file)
        if result.returncode:
            raise RuntimeError(
                f"Run submission failed with returncode {result.returncode}. See {self.config.log_file}"
            )

    async def _clean_on_failure(self) -> None:
        self.config.output_container_fpath.unlink(missing_ok=True)
        print(f"Removed incomplete output container: {self.config.output_container_fpath}", flush=True)

    async def _run_single_srun(
        self,
        builder_command_str: str,
        builder_env_vars: Dict[str, Any],
        builder_srun_slurm_vars: Dict[str, Any],
    ) -> CommandOutput:
        env_vars = builder_env_vars.copy()
        srun_slurm_vars = self.config.slurm | builder_srun_slurm_vars
        # Export names only: values stay shell-quoted in srun's environment,
        # including values containing commas. Do not inherit the host virtualenv.
        srun_slurm_vars["export"] = ",".join(env_vars) or "NONE"
        # A unique name identifies even a pending allocation, before a job ID
        # or remote task is available. Never cancel unrelated builds by name.
        job_name = f"{srun_slurm_vars.get('job-name', 'build-container')}-{uuid4().hex}"
        srun_slurm_vars["job-name"] = job_name

        env_vars_str = " \\\n".join(f"{k}={quote(v)}" for k, v in env_vars.items())
        srun_slurm_vars_strs = []
        for k, v in srun_slurm_vars.items():
            if isinstance(v, str):
                srun_slurm_vars_strs.append(f"--{k}={quote(v)}")
            elif isinstance(v, bool) and v:
                srun_slurm_vars_strs.append(f"--{k}")
            else:
                raise NotImplementedError(f"Unknown srun slurm variable type: {k=} {v=} {type(v)=}")
        srun_slurm_vars_str = " \\\n    ".join(srun_slurm_vars_strs)

        srun_command_str = f"""{env_vars_str} \\
srun \\
    {srun_slurm_vars_str} \\
    {builder_command_str}"""

        try:
            return await _run_command(srun_command_str, self.config.log_file)
        except BaseException:
            cancel = ["scancel", "--name", job_name]
            if clusters := srun_slurm_vars.get("clusters"):
                cancel.extend(["--clusters", str(clusters)])
            try:
                result = await _run_command(join(cancel), self.config.log_file)
                if result.returncode:
                    print(f"Could not cancel build job {job_name}; see {self.config.log_file}", flush=True)
            except Exception as exc:
                print(f"Could not cancel build job {job_name}: {exc}", flush=True)
            raise


def main() -> None:
    """Run container build stages using Gym's resolved configuration."""
    global_config_dict = get_global_config_dict()
    bch = BuildContainerHelper(config=BuildContainerConfig.model_validate(global_config_dict))

    async def execute() -> None:
        if not bch.config.run_only:
            await bch.run()
        if bch.config.run_only or bch.config.run_after_build:
            await bch._run_container()

    asyncio.run(execute())


if __name__ == "__main__":
    import sys

    from nemo_gym.cli.main import main as cli_main

    # Share --config translation and override handling with `gym build-containers`.
    sys.argv.insert(1, "build-containers")
    cli_main()
