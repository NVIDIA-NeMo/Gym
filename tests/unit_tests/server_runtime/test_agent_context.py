# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nemo_gym.server_runtime.agent_context import AGENT_CONTEXT_MAX_VALUES_BYTES, AgentContext, AgentContextArtifact
from nemo_gym.server_runtime.single_agent_turn_types import SingleAgentTurnTaskInput


@pytest.mark.parametrize("path", ["manifest.json", "test_media/documents/x/manifest.json", "media/./figure.png"])
def test_an_artifact_names_a_path_relative_to_the_agents_root(path: str) -> None:
    assert AgentContextArtifact(path=path).path == path


@pytest.mark.parametrize("path", ["", "/etc/passwd", "../outside", "media/../../outside", "media\\figure.png"])
def test_an_artifact_path_cannot_leave_the_agents_root(path: str) -> None:
    with pytest.raises(ValidationError):
        AgentContextArtifact(path=path)


def test_a_key_is_either_a_value_or_an_artifact() -> None:
    with pytest.raises(ValidationError, match=r"not both: \['media'\]"):
        AgentContext(values={"media": "inline"}, artifacts={"media": {"path": "media"}})


def test_inline_values_stay_small_and_large_content_goes_by_reference() -> None:
    AgentContext(values={"prompt": "x" * (AGENT_CONTEXT_MAX_VALUES_BYTES - 20)})
    with pytest.raises(ValidationError, match="pass large content as an artifact"):
        AgentContext(values={"image": "x" * AGENT_CONTEXT_MAX_VALUES_BYTES})


def test_fields_put_values_and_artifacts_in_one_namespace() -> None:
    context = AgentContext(values={"tag": "figqa2"}, artifacts={"media": {"path": "test_media/figs/1"}})

    assert context.fields() == {"tag": "figqa2", "media": AgentContextArtifact(path="test_media/figs/1")}


def test_a_rows_agent_context_moves_out_of_the_task_data_the_resources_server_gets() -> None:
    row = {
        "responses_create_params": {"input": "Describe the figure."},
        "verifier_metadata": {"answer": "a cat"},
        "agent_context": {"artifacts": {"media": {"path": "test_media/figs/1"}}},
    }

    task_input = SingleAgentTurnTaskInput.model_validate(row)

    assert task_input.agent_context == AgentContext(artifacts={"media": {"path": "test_media/figs/1"}})
    assert task_input.task_data == {"verifier_metadata": {"answer": "a cat"}}
    # The canonical form round-trips with the context in its own field.
    assert SingleAgentTurnTaskInput.model_validate(task_input.model_dump(mode="json")) == task_input


def test_a_row_without_agent_context_dumps_as_before() -> None:
    """Checkpoints digest the task input, so a task without context must dump without the new key."""
    task_input = SingleAgentTurnTaskInput.model_validate({"responses_create_params": {"input": "Hi"}, "id": "1"})

    assert task_input.agent_context is None
    assert "agent_context" not in task_input.model_dump(mode="json")


def test_agent_context_in_the_canonical_task_data_container_is_lifted_too() -> None:
    task_input = SingleAgentTurnTaskInput.model_validate(
        {
            "responses_create_params": {"input": "Hi"},
            "task_data": {"id": "1", "agent_context": {"values": {"tag": "figqa2"}}},
        }
    )

    assert task_input.agent_context == AgentContext(values={"tag": "figqa2"})
    assert task_input.task_data == {"id": "1"}
