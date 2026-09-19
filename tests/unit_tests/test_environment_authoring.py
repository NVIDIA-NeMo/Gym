# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from shutil import copytree

import pytest

from nemo_gym.environment.authoring import (
    EnvironmentDefinitionError,
    load_environment,
    load_environment_callable,
    materialize_single_task,
    materialize_tasks,
    materialize_tasks_jsonl,
)
from nemo_gym.single_agent_episode_types import SINGLE_AGENT_TASK_INPUT_CONTRACT


ROOT = Path(__file__).parents[2]
HELLO_WORLD = ROOT / "environments" / "hello_world"
HELLO_TASKSET = ROOT / "environments" / "hello_taskset"
HELLO_VERIFIER_REUSE = ROOT / "environments" / "hello_verifier_reuse"


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
    assert len(materialize_tasks_jsonl(loaded).splitlines()) == 1


def test_load_environment_local_verifier() -> None:
    loaded = load_environment(HELLO_WORLD)

    verifier = load_environment_callable(
        loaded,
        loaded.definition.task.verifier.implementation,
        description="task verifier",
    )

    assert verifier.__name__ == "verify"


def test_materialize_file_taskset() -> None:
    loaded = load_environment(HELLO_TASKSET)

    assert [(taskset.name, taskset.path) for taskset in loaded.definition.tasksets] == [
        ("example", "tasksets/example.jsonl")
    ]
    tasks = materialize_tasks(loaded, taskset="example")

    assert [task.materialized.task_id.task_id for task in tasks] == [
        "hello-taskset-001",
        "hello-taskset-002",
    ]
    hello, goodbye = tasks
    assert "/workspace/hello-gym.txt" in hello.materialized.task_input.responses_create_params.input[0].content
    assert "Hello from NeMo Gym!" in hello.materialized.task_input.responses_create_params.input[0].content
    assert hello.verifier.verifier_input == {
        "path": "/workspace/hello-gym.txt",
        "content": "Hello from NeMo Gym!",
    }
    assert "/workspace/goodbye-gym.txt" in goodbye.materialized.task_input.responses_create_params.input[0].content
    assert len(materialize_tasks_jsonl(loaded, taskset="example").splitlines()) == 2


def test_materialize_directory_tasks_with_selected_verifiers() -> None:
    loaded = load_environment(HELLO_VERIFIER_REUSE)

    assert [(taskset.name, taskset.path) for taskset in loaded.definition.tasksets] == [("default", "tasks/")]
    tasks = materialize_tasks(loaded)

    assert [task.materialized.task_id.task_id for task in tasks] == [
        "exact-greeting",
        "uppercase-greeting",
    ]
    exact, uppercase = tasks
    assert exact.materialized.task_id.taskset == "default"
    assert "/workspace/hello-gym.txt" in exact.materialized.task_input.responses_create_params.input[0].content
    assert exact.verifier.implementation == "nemo_gym.verifiers.files:text_file_equals"
    assert exact.verifier.verifier_input["path"] == "/workspace/hello-gym.txt"
    assert "/workspace/shout.txt" in uppercase.materialized.task_input.responses_create_params.input[0].content
    assert uppercase.verifier.implementation == "tasks/uppercase-greeting/verifier.py:verify"


def test_directory_taskset_requires_at_least_one_task_manifest(tmp_path: Path) -> None:
    environment_root = tmp_path / "hello_verifier_reuse"
    copytree(HELLO_VERIFIER_REUSE, environment_root)
    (environment_root / "tasks/exact-greeting/task.yaml").unlink()
    (environment_root / "tasks/uppercase-greeting/task.yaml").unlink()

    with pytest.raises(EnvironmentDefinitionError, match="contains no task.yaml files"):
        load_environment(environment_root)


def test_directory_task_rejects_reference_outside_environment(tmp_path: Path) -> None:
    environment_root = tmp_path / "hello_verifier_reuse"
    copytree(HELLO_VERIFIER_REUSE, environment_root)
    task_definition_path = environment_root / "tasks/uppercase-greeting/task.yaml"
    task_definition_path.write_text(
        task_definition_path.read_text().replace("verifier.py:verify", "../../../outside.py:verify")
    )

    with pytest.raises(EnvironmentDefinitionError, match="escapes the environment root"):
        load_environment(environment_root)


def test_taskset_names_must_be_unique(tmp_path: Path) -> None:
    environment_root = tmp_path / "hello_taskset"
    copytree(HELLO_TASKSET, environment_root)
    definition_path = environment_root / "environment.yaml"
    definition_path.write_text(
        definition_path.read_text().replace(
            "  - name: example\n    path: tasksets/example.jsonl\n",
            "  - name: example\n    path: tasksets/example.jsonl\n"
            "  - name: example\n    path: tasksets/example.jsonl\n",
        )
    )

    with pytest.raises(EnvironmentDefinitionError, match="taskset names must be unique"):
        load_environment(environment_root)


def test_file_taskset_validates_task_data_model(tmp_path: Path) -> None:
    environment_root = tmp_path / "hello_taskset"
    copytree(HELLO_TASKSET, environment_root)
    (environment_root / "tasksets/example.jsonl").write_text(
        '{"task_id":"invalid","task_data":{"expected_path":"relative.txt","expected_content":"Hello"}}\n'
    )

    with pytest.raises(EnvironmentDefinitionError, match="Invalid taskset row"):
        materialize_tasks(load_environment(environment_root), taskset="example")


def test_instruction_rejects_unknown_task_data_reference(tmp_path: Path) -> None:
    environment_root = tmp_path / "hello_taskset"
    copytree(HELLO_TASKSET, environment_root)
    (environment_root / "instruction.md").write_text("Create {{ task_data.missing }}")

    with pytest.raises(EnvironmentDefinitionError, match="Unknown task_data reference"):
        materialize_tasks(load_environment(environment_root), taskset="example")


def test_loading_is_protocol_neutral_but_materialization_dispatches_by_protocol(tmp_path: Path) -> None:
    environment_root = tmp_path / "hello_world"
    copytree(HELLO_WORLD, environment_root)
    definition_path = environment_root / "environment.yaml"
    definition_path.write_text(definition_path.read_text().replace("nemo_gym.single_agent.v1", "example.other.v1"))

    loaded = load_environment(environment_root)

    assert loaded.definition.episode_protocol == "example.other.v1"
    with pytest.raises(EnvironmentDefinitionError, match="Unsupported episode protocol"):
        materialize_tasks(loaded)


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
