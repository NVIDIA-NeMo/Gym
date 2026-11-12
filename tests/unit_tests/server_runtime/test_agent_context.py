# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from environment_servers.nemo_user_sim.episode_contracts import UserSimEpisodeRequest
from nemo_gym.core.responses_api_types import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.data.task_materialization import materialize_task
from nemo_gym.server_runtime._checkpoint.environment import task_digest
from nemo_gym.server_runtime.agent_context import AGENT_CONTEXT_MAX_BYTES, AgentContext, AgentContextArtifact
from nemo_gym.server_runtime.single_agent_turn_types import SingleAgentTurnAgentContext, SingleAgentTurnTaskInput


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
    # The complete context JSON contributes 39 bytes around this string.
    AgentContext(values={"prompt": "x" * (AGENT_CONTEXT_MAX_BYTES - 39)})
    with pytest.raises(ValidationError, match="pass large content as an artifact"):
        AgentContext(values={"prompt": "x" * (AGENT_CONTEXT_MAX_BYTES - 38)})


def test_the_context_limit_counts_utf8_bytes_not_escaped_json() -> None:
    # The complete context adds 39 bytes, and each CJK character is 3 bytes as UTF-8 (6 when escaped).
    fits = (AGENT_CONTEXT_MAX_BYTES - 39) // 3
    AgentContext(values={"prompt": "\u4e2d" * fits})
    with pytest.raises(ValidationError, match=f"takes {39 + 3 * (fits + 1)} bytes as UTF-8 JSON"):
        AgentContext(values={"prompt": "\u4e2d" * (fits + 1)})


def test_the_context_limit_includes_artifact_references() -> None:
    AgentContext(artifacts={"media": {"path": "x" * (AGENT_CONTEXT_MAX_BYTES - 47)}})
    with pytest.raises(ValidationError, match="over the 65536-byte limit"):
        AgentContext(artifacts={"media": {"path": "x" * (AGENT_CONTEXT_MAX_BYTES - 46)}})
    with pytest.raises(ValidationError, match="over the 65536-byte limit"):
        AgentContext(artifacts={str(index): {"path": "x" * 1000} for index in range(1000)})


def test_the_context_limit_counts_values_and_artifacts_together() -> None:
    with pytest.raises(ValidationError, match="over the 65536-byte limit"):
        AgentContext(values={"prompt": "x" * 32768}, artifacts={"media": {"path": "x" * 32768}})


def test_an_agents_parsed_artifact_cannot_rewrite_the_stored_context() -> None:
    class MediaContext(BaseModel):
        media: AgentContextArtifact

    context = AgentContext(artifacts={"media": {"path": "figs/1"}})
    parsed = MediaContext.model_validate(context.fields())

    with pytest.raises(ValidationError, match="frozen"):
        parsed.media.path = "figs/2"

    assert context.artifacts["media"].path == "figs/1"
    assert parsed.media.path == "figs/1"


def test_an_agent_cannot_reassign_the_stored_context() -> None:
    context = AgentContext(values={"tag": "figqa2"}, artifacts={"media": {"path": "figs/1"}})

    with pytest.raises(ValidationError, match="frozen"):
        context.values = {"tag": "other"}
    with pytest.raises(ValidationError, match="frozen"):
        context.artifacts = {}

    assert context.fields() == {"tag": "figqa2", "media": AgentContextArtifact(path="figs/1")}


def test_fields_put_values_and_artifacts_in_one_namespace() -> None:
    context = AgentContext(values={"tag": "figqa2"}, artifacts={"media": {"path": "test_media/figs/1"}})

    assert context.fields() == {"tag": "figqa2", "media": AgentContextArtifact(path="test_media/figs/1")}


def test_a_rows_agent_context_moves_out_of_the_task_data_the_resources_server_gets() -> None:
    row = {
        "responses_create_params": {"input": "Describe the figure."},
        "verifier_metadata": {"answer": "a cat"},
        "agent_context": {"agent": {"artifacts": {"media": {"path": "test_media/figs/1"}}}},
    }

    task_input = SingleAgentTurnTaskInput.model_validate(row)

    context = AgentContext(artifacts={"media": {"path": "test_media/figs/1"}})
    assert task_input.agent_context == SingleAgentTurnAgentContext(agent=context)
    assert task_input.agent_seed_context == context
    assert task_input.task_data == {"verifier_metadata": {"answer": "a cat"}}
    # The canonical form round-trips with the context in its own field.
    assert SingleAgentTurnTaskInput.model_validate(task_input.model_dump(mode="json")) == task_input


def test_a_row_without_agent_context_dumps_as_before() -> None:
    """Checkpoints digest the task input, so a task without context must dump without the new key."""
    task_input = SingleAgentTurnTaskInput.model_validate({"responses_create_params": {"input": "Hi"}, "id": "1"})

    assert task_input.agent_context is None
    assert "agent_context" not in task_input.model_dump(mode="json")


@pytest.mark.parametrize(
    ("row", "task_data"),
    [
        (
            {"responses_create_params": {"input": "Hi"}, "verifier_metadata": {"answer": "a"}, "id": "1"},
            {"verifier_metadata": {"answer": "a"}, "id": "1"},
        ),
        (
            {"responses_create_params": {"input": "Hi"}, "task_data": {"id": "1", "answer": "a"}},
            {"id": "1", "answer": "a"},
        ),
    ],
)
@pytest.mark.parametrize(
    "agent_context", ["absent", None, {}, {"agent": None}, {"agent": {}}, {"agent": {"values": {}, "artifacts": {}}}]
)
def test_a_task_without_agent_context_keeps_its_checkpoint_digest(
    row: dict, task_data: dict, agent_context: str | dict | None
) -> None:
    """Preserve the pre-context canonical shape with the installed SDK's response defaults."""
    # Build the old shape independently of SingleAgentTurnTaskInput. SDK defaults may change its digest;
    # adding an absent or empty context must not change it under the same dependencies.
    expected = {
        "responses_create_params": NeMoGymResponseCreateParamsNonStreaming(input="Hi").model_dump(mode="json"),
        "task_data": task_data,
    }
    if agent_context != "absent":
        row = {**row, "agent_context": agent_context}

    actual = SingleAgentTurnTaskInput.model_validate(row).model_dump(mode="json")
    assert actual == expected
    assert task_digest(actual) == task_digest(expected)


@pytest.mark.parametrize("agent_context", [{}, {"agent": {}}, {"agent": {"values": {}, "artifacts": {}}}, None])
def test_an_empty_agent_context_is_no_context(agent_context: dict | None) -> None:
    """A data-preparation script that always writes the key leaves the task input as a row without it."""
    row = {"responses_create_params": {"input": "Hi"}, "id": "1"}

    task_input = SingleAgentTurnTaskInput.model_validate({**row, "agent_context": agent_context})

    assert task_input.agent_context is None and task_input.agent_seed_context is None
    assert task_input.model_dump(mode="json") == SingleAgentTurnTaskInput.model_validate(row).model_dump(mode="json")


def test_agent_context_in_the_canonical_task_data_container_is_lifted_too() -> None:
    task_input = SingleAgentTurnTaskInput.model_validate(
        {
            "responses_create_params": {"input": "Hi"},
            "task_data": {"id": "1", "agent_context": {"agent": {"values": {"tag": "figqa2"}}}},
        }
    )

    assert task_input.agent_seed_context == AgentContext(values={"tag": "figqa2"})
    assert task_input.task_data == {"id": "1"}


@pytest.mark.parametrize(
    ("agent_context", "others"),
    [
        # The context of one agent, not keyed by role.
        ({"values": {"tag": "figqa2"}}, "['values']"),
        ({"user": {"values": {"tag": "figqa2"}}}, "['user']"),
        ({"agent": {"values": {"tag": "figqa2"}}, "assistant": {}}, "['assistant']"),
    ],
)
@pytest.mark.parametrize("nested", [False, True], ids=["flat", "task_data"])
def test_single_agent_turn_refuses_a_role_it_does_not_have(agent_context: dict, others: str, nested: bool) -> None:
    row = {"responses_create_params": {"input": "Hi"}, "task_data": {"id": "1"}}
    (row["task_data"] if nested else row)["agent_context"] = agent_context

    with pytest.raises(ValidationError, match="single_agent_turn has one role, 'agent'") as error:
        SingleAgentTurnTaskInput.model_validate(row)

    assert f"got {others}" in str(error.value)
    assert error.value.errors()[0]["loc"] == ("agent_context",)


def test_the_size_limit_applies_to_each_roles_context() -> None:
    """The row's role map adds its own bytes; what one session keeps is the role's context."""
    prompt = "x" * (AGENT_CONTEXT_MAX_BYTES - 39)
    task_input = SingleAgentTurnTaskInput.model_validate(
        {"responses_create_params": {"input": "Hi"}, "agent_context": {"agent": {"values": {"prompt": prompt}}}}
    )

    assert task_input.agent_seed_context == AgentContext(values={"prompt": prompt})
    with pytest.raises(ValidationError, match="over the 65536-byte limit"):
        SingleAgentTurnTaskInput.model_validate(
            {
                "responses_create_params": {"input": "Hi"},
                "agent_context": {"agent": {"values": {"prompt": prompt + "x"}}},
            }
        )


@pytest.mark.parametrize("context", [None, {}, {"agent": {"values": {"tag": "figure"}}}])
def test_usersim_task_input_rejects_unsupported_agent_context_after_materialization(context: object) -> None:
    path = Path(__file__).resolve().parents[3] / "resources_servers/nemo_user_sim/data/example.jsonl"
    row = json.loads(path.read_text().splitlines()[0])
    row["agent_context"] = context
    task = materialize_task(row, taskset="nemo_user_sim:example", task_index=0)

    with pytest.raises(ValidationError) as error:
        UserSimEpisodeRequest.model_validate({"episode_id": {"rollout_id": "user-context"}, "task": task})

    assert any(
        item["loc"] == ("task", "task_input", "agent_context") and item["type"] == "extra_forbidden"
        for item in error.value.errors()
    )
