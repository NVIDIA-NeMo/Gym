# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import re
import shutil
from asyncio import create_subprocess_shell
from asyncio.subprocess import PIPE, STDOUT
from os import killpg
from pathlib import Path
from shlex import join, quote
from signal import SIGKILL
from textwrap import dedent
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, PrivateAttr

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


class BuildContainerConfig(BaseNeMoGymCLIConfig):
    input_container_fpath: Path
    output_container_fpath: Path
    log_file: Optional[Path] = None

    slurm: Dict[str, Any] = Field(default_factory=dict)
    vllm: Optional[VLLMConfig] = None
    flashinfer: Optional[FlashInferConfig] = None

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)

        self.input_container_fpath = self.input_container_fpath.resolve()
        self.output_container_fpath = self.output_container_fpath.resolve()

        if not self.log_file:
            self.log_file = self.output_container_fpath.with_suffix(".log")
        self.log_file = self.log_file.resolve()

        print(f"""Resolved fpaths:
- Input container: {self.input_container_fpath}
- Output container: {self.output_container_fpath}
- Log file: {self.log_file}
""")


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
        except:
            try:
                killpg(process.pid, SIGKILL)
            except ProcessLookupError:
                pass  # Already exited.

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
        """Installation preserves binary packages and invalidates selected modules without JIT."""
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

            # Python 1: check that the branch and installed Python/cubin packages share
            # a release. Save the version suffix, back up existing AOT overrides, and
            # record binary-package versions for comparison after installation.
            # Run outside the checkout so imports refer to the base image's package.
            cd /
            python3 - <<'PY'
            import importlib.metadata as metadata
            import json
            import os
            import shutil
            from pathlib import Path

            import flashinfer
            import flashinfer_cubin
            from flashinfer.jit import env

            root = Path(os.environ["BUILD_DIR"])
            release = (root / "source/version.txt").read_text().strip()
            version = flashinfer.__version__
            if version != flashinfer_cubin.__version__ or version.partition("+")[0] != release:
                raise RuntimeError(f"Slim builds require matching releases: branch={release}, Python={version}, "
                                   f"cubins={flashinfer_cubin.__version__}")
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
            # dependencies, including cubins/providers, installed as-is (--no-deps).
            python3 -m pip install --no-deps --no-build-isolation --force-reinstall "$BUILD_DIR/source"

            # Python 3: restore unrelated AOT overrides, remove matching module families
            # from package/provider/JIT caches, and update provider manifests. No spec
            # generators, shape lists, CUDA compilation, or kernel launches are needed.
            python3 - <<'PY'
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
            rm -rf /opt/flashinfer-autotune-cache /opt/vllm-cache /opt/prims-cute-cache

            exit 0
            """
        ).lstrip()

        return join(["bash", "-c", script, "flashinfer-build"])

    def get_srun_env_vars(self, container_config: BuildContainerConfig) -> Dict[str, Any]:
        return {
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
            "export": "ALL",
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
            python3 -m pip install uv 2>&1 | tail -3

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
            uv pip install --system . --prerelease=allow --torch-backend=auto --index-strategy unsafe-best-match

            # Installation copies wheel Rust artifacts too. Rebuild AFTER installing
            # and unset both reuse switches so the source and launcher stay in sync.
            uv pip install --system 'setuptools>=77.0.3,<81' 'setuptools-scm>=9.2.0' 'setuptools-rust>=1.9.0' wheel
            env -u VLLM_USE_PRECOMPILED -u VLLM_USE_PRECOMPILED_RUST bash tools/build_rust.sh
            installed_vllm_dir=$(cd / && python3 -c \
                'import vllm; from pathlib import Path; print(Path(vllm.__file__).parent)')
            install -m 755 vllm/vllm-rs "$installed_vllm_dir/vllm-rs"
            cmp vllm/vllm-rs "$installed_vllm_dir/vllm-rs"
            for rust_extension in vllm/_rust_*.so; do
                [[ -f "$rust_extension" ]] || continue
                install -m 755 "$rust_extension" "$installed_vllm_dir/"
            done

            # Validate installed files, not imports shadowed by the source checkout.
            cd /
            python3 - "$installed_vllm_dir" <<'PY'
            import ast
            import subprocess
            import sys
            from pathlib import Path

            package = Path(sys.argv[1])
            tree = ast.parse((package / "v1/utils.py").read_text())
            manager = next(node for node in tree.body
                           if isinstance(node, ast.ClassDef) and node.name == "RustFrontendProcessManager")
            flags = {node.value for node in ast.walk(manager)
                     if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("--")}
            help_text = subprocess.check_output([str(package / "vllm-rs"), "frontend", "--help"], text=True)
            missing = flags - set(help_text.split())
            if missing:
                raise SystemExit(f"Installed Rust frontend does not support Python launcher flags: {sorted(missing)}")
            print("Rust frontend accepts the installed Python launcher's flags")
            PY
            python3 - <<'PY'
            import torch
            import vllm
            from vllm.vllm_flash_attn import flash_attn_varlen_func
            from vllm.model_executor.models.registry import ModelRegistry

            assert "NemotronH_Omni_Reasoning_V3" in ModelRegistry.get_supported_archs()
            print(f"vLLM {vllm.__version__}; torch {torch.__version__}")
            PY

            rm -rf /opt/uv/cache /root/.cache/uv /root/.cache/pip /var/lib/apt/lists/* 2>/dev/null || true
            apt-get clean || true
            rm -rf "$BUILD_DIR/source"
            find /usr/lib/aarch64-linux-gnu /usr/local/cuda-13.0 -name '*_static*.a' -delete 2>/dev/null || true
            """
        ).lstrip()
        return join(["bash", "-c", script, "vllm-build"])

    def get_srun_env_vars(self, container_config: BuildContainerConfig) -> Dict[str, Any]:
        """Pass build configuration as literal environment values."""
        if self._head_sha is None:
            raise RuntimeError("Run vLLM preflight checks before generating the build environment")
        return {
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
            "export": "ALL",
        }


class BuildContainerHelper(BaseModel):
    config: BuildContainerConfig

    async def run(self) -> None:
        try:
            await self._run()
        except:
            await self._clean_on_failure()
            raise

    async def _run(self) -> None:
        self.config.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.config.log_file.unlink(missing_ok=True)
        self.config.output_container_fpath.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy(self.config.input_container_fpath, self.config.output_container_fpath)
        print("Copied input container fpath to output (not finished, just starting)")

        builders: list[BaseBuilder] = []
        if self.config.vllm is not None:
            builders.append(VLLMBuilder(config=self.config.vllm, log_file=self.config.log_file))
        if self.config.flashinfer is not None:
            builders.append(FlashInferBuilder(config=self.config.flashinfer, log_file=self.config.log_file))

        for builder in builders:
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

    async def _clean_on_failure(self) -> None:
        command_output = await _run_command(
            command=join(["rm", "-f", "--", str(self.config.output_container_fpath)]),
            log_fpath=self.config.log_file,
        )
        assert command_output.returncode == 0

    async def _run_single_srun(
        self,
        builder_command_str: str,
        builder_env_vars: Dict[str, Any],
        builder_srun_slurm_vars: Dict[str, Any],
    ) -> CommandOutput:
        env_vars = builder_env_vars.copy()
        srun_slurm_vars = self.config.slurm | builder_srun_slurm_vars

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

        return await _run_command(srun_command_str, self.config.log_file)


def main() -> None:
    """Run container build stages using Gym's resolved configuration."""
    global_config_dict = get_global_config_dict()
    bch = BuildContainerHelper(config=BuildContainerConfig.model_validate(global_config_dict))
    asyncio.run(bch.run())


if __name__ == "__main__":
    import sys

    from nemo_gym.cli.main import main as cli_main

    # Share --config translation and override handling with `gym build-containers`.
    sys.argv.insert(1, "build-containers")
    cli_main()
