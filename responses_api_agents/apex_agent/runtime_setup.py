# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trusted host setup for pinned Apex and Archipelago runtime sources."""

from __future__ import annotations

import base64
import hashlib
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


HARNESS_REVISION = "1fd94befbb570eb6effe76b1895e5d599e820227"


class ApexImageBuildConfig(BaseModel):
    """Pinned upstream source used only when Gym must build a missing SIF."""

    enabled: bool
    source_repo: str
    source_revision: str
    source_root: Optional[str]
    source_github_token: Optional[str]
    dockerfile: str
    docker_tag: str
    timeout: int = Field(gt=0)


def _git_environment(token: str | None) -> dict[str, str]:
    env = os.environ.copy()
    for variable in ("GIT_ASKPASS", "SSH_ASKPASS"):
        helper = env.get(variable)
        if helper and not Path(helper).is_file():
            env.pop(variable, None)
    env["GIT_TERMINAL_PROMPT"] = "0"
    if token:
        encoded = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        env["GIT_CONFIG_COUNT"] = "1"
        env["GIT_CONFIG_KEY_0"] = "http.https://github.com/.extraheader"
        env["GIT_CONFIG_VALUE_0"] = f"Authorization: Basic {encoded}"
    return env


def _resolve_pinned_source(
    *,
    cache_dir: Path,
    cache_name: str,
    repo: str,
    revision: str,
    source_root: str | None,
    github_token: str | None,
) -> Path:
    """Resolve a pinned commit from a local checkout or a cached bare repository."""
    if source_root:
        root = Path(source_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"source checkout does not exist: {root}")
    else:
        root = cache_dir / f"{cache_name}.git"
        if not root.exists():
            subprocess.run(
                ["git", "init", "--bare", "--initial-branch=main", str(root)],
                check=True,
                timeout=30,
            )
        try:
            subprocess.run(
                ["git", "-C", str(root), "fetch", "--depth=1", repo, revision],
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
                env=_git_environment(github_token),
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            raise RuntimeError(
                f"could not fetch pinned source {repo}@{revision}: {detail}. Start Gym from a Git-authenticated "
                "shell, configure a local source root, use an SSH repository URL, or provide a GitHub token."
            ) from exc

    try:
        resolved = subprocess.run(
            ["git", "-C", str(root), "rev-parse", f"{revision}^{{commit}}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"source checkout does not contain {revision}: {detail}") from exc
    if resolved != revision:
        raise RuntimeError(f"source revision resolved to {resolved}, expected {revision}")
    return root


def prepare_harness_source_archive(
    *,
    agent_dir: Path,
    repo: str,
    source_root: str | None,
    github_token: str | None,
) -> Path:
    """Fetch and export the clean, pinned Apex harness commit."""
    cache_dir = agent_dir / "deps"
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / f"apex-harness-source-{HARNESS_REVISION[:20]}.tar.gz"
    if archive.exists():
        return archive

    root = _resolve_pinned_source(
        cache_dir=cache_dir,
        cache_name="apex-harness-source",
        repo=repo,
        revision=HARNESS_REVISION,
        source_root=source_root,
        github_token=github_token,
    )
    temporary = archive.with_suffix(".tmp")
    temporary.unlink(missing_ok=True)
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "archive",
                "--format=tar.gz",
                f"--output={temporary}",
                HARNESS_REVISION,
            ],
            check=True,
            timeout=60,
        )
        temporary.replace(archive)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"could not export pinned Apex harness revision {HARNESS_REVISION}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return archive


def _prepare_archipelago_context(agent_dir: Path, build: ApexImageBuildConfig) -> Path:
    cache_dir = agent_dir / "deps"
    cache_dir.mkdir(parents=True, exist_ok=True)
    context = cache_dir / f"archipelago-context-{build.source_revision[:20]}"
    if context.is_dir():
        return context

    root = _resolve_pinned_source(
        cache_dir=cache_dir,
        cache_name="archipelago-source",
        repo=build.source_repo,
        revision=build.source_revision,
        source_root=build.source_root,
        github_token=build.source_github_token,
    )
    archive = cache_dir / f".archipelago-{build.source_revision[:20]}.tar"
    temporary = cache_dir / f".{context.name}.building"
    if temporary.exists():
        shutil.rmtree(temporary)
    archive.unlink(missing_ok=True)
    try:
        subprocess.run(
            ["git", "-C", str(root), "archive", "--format=tar", f"--output={archive}", build.source_revision],
            check=True,
            timeout=120,
        )
        temporary.mkdir()
        with tarfile.open(archive) as source:
            source.extractall(temporary, filter="data")
        temporary.replace(context)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"could not export pinned Archipelago revision {build.source_revision}") from exc
    finally:
        archive.unlink(missing_ok=True)
        if temporary.exists():
            shutil.rmtree(temporary)
    return context


def _build_archipelago_sif(agent_dir: Path, target: Path, build: ApexImageBuildConfig) -> Path:
    context_dir = _prepare_archipelago_context(agent_dir, build)
    dockerfile = Path(build.dockerfile).expanduser()
    if not dockerfile.is_absolute():
        dockerfile = context_dir / dockerfile
    dockerfile = dockerfile.resolve()
    if not dockerfile.is_file():
        raise FileNotFoundError(f"Archipelago Dockerfile does not exist: {dockerfile}")
    if shutil.which("docker") is None:
        raise RuntimeError("docker is required to auto-build the Archipelago image")
    if shutil.which("apptainer") is None:
        raise RuntimeError("apptainer is required to convert the Archipelago Docker image to SIF")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.building")
    temporary.unlink(missing_ok=True)
    docker_command = ["docker", "build"]
    docker_env = os.environ.copy()
    if build.source_github_token:
        secret_name = "NEMO_GYM_APEX_GITHUB_TOKEN"
        docker_env[secret_name] = build.source_github_token
        docker_command.extend(["--secret", f"id=github_token,env={secret_name}"])
    docker_command.extend(["--file", str(dockerfile), "--tag", build.docker_tag, str(context_dir)])
    try:
        subprocess.run(docker_command, check=True, timeout=build.timeout, env=docker_env)
        subprocess.run(
            ["apptainer", "build", "--force", str(temporary), f"docker-daemon://{build.docker_tag}"],
            check=True,
            timeout=build.timeout,
        )
        temporary.replace(target)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Archipelago image build failed with exit code {exc.returncode}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return target


def resolve_image(
    *,
    agent_dir: Path,
    parent_dir: Path,
    image: str,
    image_build: ApexImageBuildConfig,
    sandbox_provider: str | Dict[str, Any],
) -> str:
    """Resolve an existing SIF/OCI image or build the configured pinned source once."""
    image = image.strip()
    if not image:
        raise ValueError("apex_agents_image must resolve to a non-empty image reference")
    is_apptainer = isinstance(sandbox_provider, dict) and "apptainer" in sandbox_provider
    if image.endswith(".sif") or image.startswith(("/", ".")):
        if not is_apptainer:
            raise ValueError("local or .sif images require the Apptainer sandbox provider")
        path = Path(image).expanduser()
        if not path.is_absolute():
            path = parent_dir / path
        path = path.resolve()
        if path.is_file():
            return str(path)
        if not image_build.enabled:
            raise FileNotFoundError(
                f"configured Apex image does not exist: {path}; supply a prebuilt SIF/OCI image or enable image_build"
            )
        return str(_build_archipelago_sif(agent_dir, path, image_build))
    if "://" in image and not image.startswith("docker://"):
        if not is_apptainer:
            raise ValueError(f"image URI {image!r} is not portable; use a bare OCI image reference")
        return image
    return image.removeprefix("docker://")


def harness_cache_path(
    *,
    agent_dir: Path,
    setup_path: Path,
    requirements_path: Path,
    image: str,
    source_archive: Path,
) -> Path:
    with source_archive.open("rb") as stream:
        source_digest = hashlib.file_digest(stream, "sha256").digest()
    recipe = hashlib.sha256(
        setup_path.read_bytes() + requirements_path.read_bytes() + image.encode("utf-8") + source_digest
    ).hexdigest()
    cache_dir = agent_dir / "deps"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"apex-harness-{recipe[:20]}.tar.gz"
