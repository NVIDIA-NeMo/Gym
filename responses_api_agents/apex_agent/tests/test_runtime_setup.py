# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from responses_api_agents.apex_agent.runtime_setup import (
    ApexImageBuildConfig,
    _prepare_archipelago_context,
    resolve_image,
)


def _build(enabled: bool = True) -> ApexImageBuildConfig:
    return ApexImageBuildConfig(
        enabled=enabled,
        source_repo="https://github.com/Mercor-Intelligence/archipelago.git",
        source_revision="0cb5c476c219a9df637e0bd37fb86b2361f4ab89",
        source_root=None,
        source_github_token=None,
        dockerfile="environment/Dockerfile",
        docker_tag="nemo-gym-archipelago:test",
        timeout=60,
    )


def test_resolve_image_reuses_existing_sif(tmp_path: Path) -> None:
    image = tmp_path / "archipelago.sif"
    image.touch()

    assert resolve_image(
        agent_dir=tmp_path,
        parent_dir=tmp_path,
        image=str(image),
        image_build=_build(),
        sandbox_provider={"apptainer": {}},
    ) == str(image)


def test_resolve_image_uses_prebuilt_oci_reference(tmp_path: Path) -> None:
    image = "registry.example/archipelago@sha256:1234"

    assert (
        resolve_image(
            agent_dir=tmp_path,
            parent_dir=tmp_path,
            image=image,
            image_build=_build(),
            sandbox_provider={"apptainer": {}},
        )
        == image
    )


def test_resolve_image_builds_missing_sif_once(monkeypatch, tmp_path: Path) -> None:
    image = tmp_path / "archipelago.sif"
    build_sif = MagicMock(return_value=image)
    monkeypatch.setattr("responses_api_agents.apex_agent.runtime_setup._build_archipelago_sif", build_sif)

    assert resolve_image(
        agent_dir=tmp_path,
        parent_dir=tmp_path,
        image=str(image),
        image_build=_build(),
        sandbox_provider={"apptainer": {}},
    ) == str(image)
    build_sif.assert_called_once_with(tmp_path, image, _build())


@pytest.mark.skipif(shutil.which("git") is None, reason="git is required")
def test_archipelago_context_exports_clean_pinned_revision(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "--initial-branch=main", str(source)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(source), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.name", "Test"], check=True)
    (source / "environment").mkdir()
    (source / "environment" / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    (source / "tracked.txt").write_text("pinned\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "-m", "fixture"], check=True, capture_output=True)
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (source / "tracked.txt").write_text("dirty working tree\n", encoding="utf-8")
    build = _build().model_copy(update={"source_root": str(source), "source_revision": revision})

    context = _prepare_archipelago_context(tmp_path / "agent", build)

    assert (context / "tracked.txt").read_text(encoding="utf-8") == "pinned\n"
    assert not (context / ".git").exists()
