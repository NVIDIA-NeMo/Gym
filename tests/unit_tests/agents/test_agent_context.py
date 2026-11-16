# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import errno
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from nemo_gym.agents.base_responses_api_agent import AgentSeedRejectedError
from nemo_gym.agents.utils.agent_context import resolve_agent_context_artifact
from nemo_gym.server_runtime.agent_context import AgentContextArtifact


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "root" / "documents" / "x").mkdir(parents=True)
    (tmp_path / "root" / "documents" / "x" / "manifest.json").write_text("{}")
    (tmp_path / "secret.txt").write_text("not the agent's")
    return tmp_path / "root"


def test_an_artifact_resolves_under_the_agents_root(root: Path) -> None:
    manifest = AgentContextArtifact(path="documents/x/manifest.json")

    assert (
        resolve_agent_context_artifact(manifest, root, kind="file") == (root / "documents/x/manifest.json").resolve()
    )
    assert resolve_agent_context_artifact(AgentContextArtifact(path="documents/x"), root, kind="directory").is_dir()


def test_an_artifact_cannot_reach_outside_the_root_through_a_symlink(root: Path) -> None:
    (root / "documents" / "leak.txt").symlink_to(root.parent / "secret.txt")

    with pytest.raises(AgentSeedRejectedError, match="resolves outside"):
        resolve_agent_context_artifact(AgentContextArtifact(path="documents/leak.txt"), root, kind="file")


@pytest.mark.parametrize(
    ("path", "kind"),
    [
        ("documents/x/missing.json", "file"),
        ("documents/x", "file"),
        ("documents/x/manifest.json", "directory"),
        ("documents/x/manifest.json/child", "file"),
    ],
)
def test_a_missing_or_wrong_kind_of_artifact_refuses_the_seed(root: Path, path: str, kind: str) -> None:
    with pytest.raises(AgentSeedRejectedError, match=f"is not a {kind} under") as raised:
        resolve_agent_context_artifact(AgentContextArtifact(path=path), root, kind=kind)

    assert raised.value.status_code == 422


def test_a_missing_root_is_reported_as_the_agents_configuration(root: Path) -> None:
    with pytest.raises(AgentSeedRejectedError, match="agent_context root .* does not exist") as raised:
        resolve_agent_context_artifact(
            AgentContextArtifact(path="documents/x/manifest.json"), root / "missing", kind="file"
        )

    assert raised.value.status_code == 422


@pytest.mark.parametrize(
    ("path", "reason"),
    [("documents/x/manifest\x00.json", "embedded null"), ("documents/" + "a" * 5000, "File name too long")],
    ids=["nul-byte", "long-name"],
)
def test_a_path_the_filesystem_cannot_look_up_refuses_the_seed(root: Path, path: str, reason: str) -> None:
    """A retry would meet the same error, so the seed answers a terminal 422, not a 500 that is retried."""
    # A NUL byte cannot pass validation; an artifact built without it reaches the filesystem.
    artifact = AgentContextArtifact.model_construct(path=path)

    with pytest.raises(AgentSeedRejectedError, match=reason) as raised:
        resolve_agent_context_artifact(artifact, root, kind="file")

    assert raised.value.status_code == 422


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any directory")
def test_an_unreadable_artifact_refuses_the_seed(root: Path) -> None:
    (root / "documents" / "x").chmod(0)
    try:
        with pytest.raises(AgentSeedRejectedError, match="Permission denied"):
            resolve_agent_context_artifact(AgentContextArtifact(path="documents/x/manifest.json"), root, kind="file")
    finally:
        (root / "documents" / "x").chmod(0o755)


def test_an_io_error_on_the_artifacts_storage_stays_retryable(root: Path) -> None:
    with (
        patch.object(Path, "is_file", side_effect=OSError(errno.EIO, "Input/output error")),
        pytest.raises(OSError) as raised,
    ):
        resolve_agent_context_artifact(AgentContextArtifact(path="documents/x/manifest.json"), root, kind="file")

    assert not isinstance(raised.value, AgentSeedRejectedError)


def test_a_transient_symlink_lookup_error_cannot_return_a_path_outside_the_root(root: Path) -> None:
    link = root / "link"
    link.symlink_to(root.parent, target_is_directory=True)
    real_lstat = os.lstat
    failed = False

    def fail_once_on_link(path: str, *args, **kwargs) -> os.stat_result:
        nonlocal failed
        if not failed and Path(path) == link:
            failed = True
            raise OSError(errno.EIO, "Input/output error")
        return real_lstat(path, *args, **kwargs)

    with patch("os.lstat", side_effect=fail_once_on_link), pytest.raises(OSError) as raised:
        resolve_agent_context_artifact(AgentContextArtifact(path="link/secret.txt"), root, kind="file")

    assert failed
    assert raised.value.errno == errno.EIO
    assert not isinstance(raised.value, AgentSeedRejectedError)
    with pytest.raises(AgentSeedRejectedError, match="resolves outside"):
        resolve_agent_context_artifact(AgentContextArtifact(path="link/secret.txt"), root, kind="file")


def test_a_symlink_loop_refuses_the_seed(root: Path) -> None:
    (root / "loop").symlink_to(root / "loop")

    with pytest.raises(AgentSeedRejectedError, match="cannot be resolved") as raised:
        resolve_agent_context_artifact(AgentContextArtifact(path="loop"), root, kind="file")

    assert raised.value.status_code == 422
