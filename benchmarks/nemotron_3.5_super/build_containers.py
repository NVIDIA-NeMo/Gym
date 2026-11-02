# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from asyncio import create_subprocess_shell
from asyncio.subprocess import PIPE, STDOUT
from os import killpg
from pathlib import Path
from shlex import join, quote
from signal import SIGKILL
from textwrap import dedent
from typing import Any, Dict, List, Optional

import aiofiles
from pydantic import BaseModel, Field, PrivateAttr

from nemo_gym.config_types import BaseNeMoGymCLIConfig


class FlashInferConfig(BaseNeMoGymCLIConfig):
    repo: str
    branch: str
    invalidate_modules: List[str]


class BuildContainerConfig(BaseNeMoGymCLIConfig):
    input_container_fpath: str
    output_container_fpath: str
    log_file: Path

    slurm: Dict[str, Any] = Field(default_factory=dict)
    flashinfer: Optional[FlashInferConfig] = None


class CommandOutput(BaseModel):
    returncode: int
    output_lines: List[str]


class BaseBuilder(BaseModel):
    config: BaseNeMoGymCLIConfig

    # Log file is separate from the config so that we dont have the duplicate this log file property on every config.
    log_file: Path

    async def perform_preflight_checks(self) -> None:
        raise NotImplementedError

    def get_command_str(self) -> str:
        raise NotImplementedError

    async def _run_command(self, command: str) -> CommandOutput:
        async with aiofiles.open(self.log_file, "a") as log_file:
            process = await create_subprocess_shell(command, stdout=PIPE, stderr=STDOUT, start_new_session=True)
            output_lines = []

            try:
                assert process.stdout is not None

                async for line in process.stdout:
                    line = line.decode(errors="replace").rstrip()

                    print(line, flush=True)
                    output_lines.append(line)
                    await log_file.write(line + "\n")

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

    async def _preflight_check_git_repo(self, repo: str, branch: str) -> str:
        """
        Returns the head sha
        """

        ref = f"refs/heads/{branch}"
        command_output = await self._run_command(command=f"git ls-remote --exit-code {quote(repo)} {quote(ref)}")

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

    def get_command_str(self) -> str:
        """Return a Bash command to run inside the target container.

        Call perform_preflight_checks first. The container must supply BASE_IMAGE
        for provenance; FLASHINFER_CUDA_ARCH_LIST defaults to 10.0a. Installation
        preserves binary packages and invalidates selected modules without JIT.
        Slurm submission and saving the container belong to the caller.
        """
        if self._head_sha is None:
            raise RuntimeError("Run FlashInfer preflight checks before generating the build command")
        for pattern in self.config.invalidate_modules:
            if not re.fullmatch(r"[A-Za-z0-9_*?-]+", pattern):
                raise ValueError(f"Invalid FlashInfer module name/glob: {pattern!r}")
        # Pass configuration as positional arguments so shell metacharacters in
        # repo URLs and module patterns are never evaluated as command text.
        script = dedent(
            r"""
            set -Eeuo pipefail
            export FLASHINFER_REPO="$1"
            export FLASHINFER_HEAD_SHA="$2"
            export FLASHINFER_INVALIDATE_MODULES="$3"
            export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-10.0a}"
            export BASE_IMAGE
            : "${FLASHINFER_HEAD_SHA:?}" "${BASE_IMAGE:?}"
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
            record = {key: os.environ[key] for key in
                      ("BASE_IMAGE", "FLASHINFER_HEAD_SHA", "FLASHINFER_CUDA_ARCH_LIST", "FLASHINFER_INVALIDATE_MODULES")}
            record["invalidated_paths"] = removed
            record["retained_packages"] = packages
            (root / "flashinfer-slim.json").write_text(json.dumps(record, indent=2) + "\n")
            print(f"Invalidated {len(removed)} native cache entries; selected modules will JIT on first use", flush=True)
            PY
            mkdir -p /opt
            cp "$BUILD_DIR/flashinfer-slim.json" /opt/flashinfer-slim.json
            # These derived caches/plans describe the previous installation. Recreate
            # them during warmup instead of validating stale manifests or reusing graphs.
            rm -rf /opt/flashinfer-autotune-cache \
                /opt/vllm-cache /opt/prims-cute-cache
            # A previous full-build verifier describes the old Python sources/modules.
            # Keep it for reference, but don't let the launcher apply it to this build.
            if [[ -f /opt/flashinfer-build.env ]]; then
                mv /opt/flashinfer-build.env /opt/flashinfer-build.before-slim.env
            fi
            exit 0
            """
        ).lstrip()
        return join(
            [
                "bash",
                "-c",
                script,
                "flashinfer-build",
                self.config.repo,
                self._head_sha,
                ",".join(self.config.invalidate_modules),
            ]
        )


class BuildContainerHelper:
    config: BuildContainerConfig

    def run(self) -> None:
        pass
