# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from nemo_gym.environment.authoring import (
    EnvironmentDefinitionError,
    load_environment,
    load_environment_callable,
    materialize_single_task,
    materialize_single_task_jsonl,
)
from nemo_gym.single_agent_episode_types import SINGLE_AGENT_TASK_INPUT_CONTRACT


ROOT = Path(__file__).parents[2]
HELLO_WORLD = ROOT / "environments" / "hello_world"


def test_load_and_materialize_hello_world() -> None:
    loaded = load_environment(HELLO_WORLD)

    assert loaded.definition.name == "hello-world"
    assert loaded.definition.episode_protocol == SINGLE_AGENT_TASK_INPUT_CONTRACT

    task = materialize_single_task(loaded)

    assert task.task_id.taskset == "hello-world"
    assert task.task_id.task_id == "hello-world-001"
    assert task.task_id.revision == "1.0.0"
    assert task.task_input.task_data == {}
    assert "/workspace/hello-gym.txt" in task.task_input.responses_create_params.input[0].content
    assert materialize_single_task_jsonl(loaded).endswith("\n")


def test_load_environment_local_verifier() -> None:
    loaded = load_environment(HELLO_WORLD)

    verifier = load_environment_callable(
        loaded,
        loaded.definition.task.verifier.implementation,
        description="task verifier",
    )

    assert verifier.__name__ == "verify"


def test_rejects_reference_outside_environment(tmp_path: Path) -> None:
    (tmp_path / "runtime").mkdir()
    (tmp_path / "runtime" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (tmp_path / "environment.yaml").write_text(
        """
name: unsafe
version: 1.0.0
description: Unsafe fixture
license: Apache-2.0
episode_protocol: nemo_gym.single_agent.v1
task:
  id: unsafe-001
  instruction: ../instruction.md
  verifier:
    implementation: verifier.py:verify
runtime:
  dockerfile: runtime/Dockerfile
  workdir: /workspace
"""
    )
    (tmp_path.parent / "instruction.md").write_text("outside")
    (tmp_path / "verifier.py").write_text("async def verify(attempt, verifier_input): return 1.0\n")

    with pytest.raises(EnvironmentDefinitionError, match="escapes the environment root"):
        load_environment(tmp_path)


@pytest.mark.parametrize(
    "environment_name",
    ["hello_world", "hello_taskset", "hello_verifier_reuse", "hello_mcp_tool"],
)
def test_all_hello_definitions_parse(environment_name: str) -> None:
    loaded = load_environment(ROOT / "environments" / environment_name)

    assert loaded.definition.name.startswith("hello-")
