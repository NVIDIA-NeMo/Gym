# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

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
    [("documents/x/missing.json", "file"), ("documents/x", "file"), ("documents/x/manifest.json", "directory")],
)
def test_a_missing_or_wrong_kind_of_artifact_refuses_the_seed(root: Path, path: str, kind: str) -> None:
    with pytest.raises(AgentSeedRejectedError, match=f"is not a {kind} under") as raised:
        resolve_agent_context_artifact(AgentContextArtifact(path=path), root, kind=kind)

    assert raised.value.status_code == 422
