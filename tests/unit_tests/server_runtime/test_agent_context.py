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


def test_an_artifact_path_cannot_carry_a_nul_byte() -> None:
    """The filesystem rejects it with a ValueError, so it is refused as data before any lookup."""
    with pytest.raises(ValidationError, match="NUL byte"):
        AgentContextArtifact(path="media/figure\x00.png")


def test_a_key_is_either_a_value_or_an_artifact() -> None:
    with pytest.raises(ValidationError, match=r"not both: \['media'\]"):
        AgentContext(values={"media": "inline"}, artifacts={"media": {"path": "media"}})


def test_inline_values_stay_small_and_large_content_goes_by_reference() -> None:
    AgentContext(values={"prompt": "x" * (AGENT_CONTEXT_MAX_VALUES_BYTES - 20)})
    with pytest.raises(ValidationError, match="pass large content as an artifact"):
        AgentContext(values={"image": "x" * AGENT_CONTEXT_MAX_VALUES_BYTES})


def test_the_values_limit_counts_utf8_bytes_not_escaped_json() -> None:
    # '{"prompt":"..."}' is 13 bytes around the text, and each CJK character is 3 bytes as UTF-8 (6 when escaped).
    fits = (AGENT_CONTEXT_MAX_VALUES_BYTES - 13) // 3
    AgentContext(values={"prompt": "\u4e2d" * fits})
    with pytest.raises(ValidationError, match=f"take {13 + 3 * (fits + 1)} bytes as UTF-8 JSON"):
        AgentContext(values={"prompt": "\u4e2d" * (fits + 1)})


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


@pytest.mark.parametrize("agent_context", [{}, {"values": {}, "artifacts": {}}, None])
def test_an_empty_agent_context_is_no_context(agent_context: dict | None) -> None:
    """A data-preparation script that always writes the key leaves the task input as a row without it."""
    row = {"responses_create_params": {"input": "Hi"}, "id": "1"}

    task_input = SingleAgentTurnTaskInput.model_validate({**row, "agent_context": agent_context})

    assert task_input.agent_context is None
    assert task_input.model_dump(mode="json") == SingleAgentTurnTaskInput.model_validate(row).model_dump(mode="json")


def test_agent_context_in_the_canonical_task_data_container_is_lifted_too() -> None:
    task_input = SingleAgentTurnTaskInput.model_validate(
        {
            "responses_create_params": {"input": "Hi"},
            "task_data": {"id": "1", "agent_context": {"values": {"tag": "figqa2"}}},
        }
    )

    assert task_input.agent_context == AgentContext(values={"tag": "figqa2"})
    assert task_input.task_data == {"id": "1"}
