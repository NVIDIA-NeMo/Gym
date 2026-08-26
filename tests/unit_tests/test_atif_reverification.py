# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from nemo_gym.atif_reverification import (
    AtifProjectionError,
    AtifReverifyManifestEntry,
    atif_trajectory_to_response,
    build_atif_verify_payload,
    index_materialized_inputs,
    load_atif_manifest,
    load_atif_trajectory,
    project_atif_manifest_entries,
    project_atif_manifest_entry,
)
from nemo_gym.base_resources_server import ReverifyMode
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCallForTraining,
    NeMoGymResponseOutputMessageForTraining,
)
from nemo_gym.relay_atif import AtifTrajectoryV1_7
from nemo_gym.responses_converter import ResponsesConverter
from nemo_gym.server_utils import ServerClient
from resources_servers.mcqa.app import MCQAResourcesServer, MCQAResourcesServerConfig, MCQAVerifyRequest


_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "relay_atif_v1_7_tool_trajectory.json"
_FIXTURE_SHA256 = "431aae09e1a1a3cfd478c44f730d0432c0052eb06fe387d4d90aec4bacf4b660"  # pragma: allowlist secret
_RESPONSES_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "relay_atif_v1_7_responses_tool_trajectory.json"
_RESPONSES_FIXTURE_SHA256 = (
    "8d9c3c2d21c4ef0a8d9488eac560b1fdfaa532712b8a4fc628023a94d0bab825"  # pragma: allowlist secret
)


def _trajectory_data() -> dict[str, Any]:
    return {
        "schema_version": "ATIF-v1.7",
        "session_id": "run-1",
        "trajectory_id": "trajectory-1",
        "agent": {"name": "fixture-agent", "version": "1", "model_name": "fixture-model"},
        "steps": [
            {"step_id": 1, "source": "user", "message": "Use both tools."},
            {
                "step_id": 2,
                "source": "agent",
                "timestamp": "2026-08-24T12:00:00Z",
                "message": "",
                "reasoning_content": "I need both results.",
                "tool_calls": [
                    {"tool_call_id": "call-a", "function_name": "lookup", "arguments": {"q": "x"}},
                    {"tool_call_id": "call-b", "function_name": "calculate", "arguments": {"x": 2}},
                ],
                # Deliberately reverse the result order. Pairing must use source_call_id,
                # not the positional assumption used by the older Harbor-only mapper.
                "observation": {
                    "results": [
                        {"source_call_id": "call-b", "content": "4"},
                        {"source_call_id": "call-a", "content": "found"},
                    ]
                },
            },
            {"step_id": 3, "source": "agent", "message": "The answer is 4."},
        ],
    }


def _materialized_input() -> dict[str, Any]:
    return {
        "responses_create_params": {
            "input": [{"role": "user", "content": "What is the weather in Raleigh?"}],
            "instructions": "Answer using tools when useful.",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [
                {
                    "type": "function",
                    "name": "lookup_weather",
                    "description": "Return deterministic fixture weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                    "strict": False,
                }
            ],
        },
        TASK_INDEX_KEY_NAME: 7,
        ROLLOUT_INDEX_KEY_NAME: 2,
        "expected_answer": "72 and sunny",
    }


def test_relay_atif_parser_rejects_unknown_structural_fields() -> None:
    data = _trajectory_data()
    data["unknown_root_field"] = "would otherwise be silently ignored"

    with pytest.raises(ValidationError, match="extra_forbidden"):
        AtifTrajectoryV1_7.model_validate(data)


def test_atif_v1_7_parser_accepts_missing_optional_session_id() -> None:
    data = _trajectory_data()
    del data["session_id"]

    trajectory = AtifTrajectoryV1_7.model_validate(data)

    assert trajectory.session_id is None


def _weather_reward(payload: dict[str, Any]) -> float:
    """Small stateless verifier used to compare native and ATIF payloads."""

    NeMoGymResponseCreateParamsNonStreaming.model_validate(payload["responses_create_params"])
    response = NeMoGymResponse.model_validate(payload["response"])
    calls = {item.call_id: (item.name, item.arguments) for item in response.output if item.type == "function_call"}
    results = {item.call_id: item.output for item in response.output if item.type == "function_call_output"}
    answer = "".join(
        part.text
        for item in response.output
        if item.type == "message"
        for part in item.content
        if part.type == "output_text"
    )
    try:
        weather = json.loads(results.get("call-weather-1", ""))
    except (TypeError, json.JSONDecodeError):
        weather = None
    return float(
        calls.get("call-weather-1") == ("lookup_weather", '{"city":"Raleigh"}')
        and weather == {"condition": "sunny", "temperature_f": 72}
        and answer == "It is 72 degrees and sunny in Raleigh."
    )


def test_current_relay_fixture_builds_a_valid_gym_verify_request() -> None:
    loaded = load_atif_trajectory(_FIXTURE_PATH)
    materialized = _materialized_input()
    original = deepcopy(materialized)

    payload = build_atif_verify_payload(materialized, loaded.trajectory)
    NeMoGymResponseCreateParamsNonStreaming.model_validate(payload["responses_create_params"])
    response = NeMoGymResponse.model_validate(payload["response"])

    assert loaded.source_sha256 == _FIXTURE_SHA256
    assert loaded.trajectory.agent.version == "0.9.0"
    assert loaded.trajectory.agent.extra == {
        "fixture": "gym-400",
        "relay_revision": "2222222222222222222222222222222222222222",
    }
    assert materialized == original
    assert payload[TASK_INDEX_KEY_NAME] == 7
    assert payload["expected_answer"] == "72 and sunny"
    assert response.instructions == "Answer using tools when useful."
    assert response.parallel_tool_calls is False
    assert response.tools[0].name == "lookup_weather"
    assert [item.type for item in response.output] == [
        "function_call",
        "function_call_output",
        "message",
    ]
    assert response.output[1].call_id == "call-weather-1"
    assert response.output[1].output == '{"condition":"sunny","temperature_f":72}'


def test_atif_and_equivalent_native_response_receive_the_same_reward() -> None:
    loaded = load_atif_trajectory(_FIXTURE_PATH)
    materialized = _materialized_input()
    atif_payload = build_atif_verify_payload(materialized, loaded.trajectory)
    final_step = loaded.trajectory.steps[-1]
    assert final_step.extra is not None
    raw_request = final_step.extra["llm_request"]
    raw_response = final_step.extra["llm_response"]
    converter = ResponsesConverter(return_token_id_information=False, uses_reasoning_parser=False)
    native_items = converter.chat_completions_messages_to_responses_items(
        deepcopy(raw_request["messages"][2:]) + [deepcopy(raw_response["choices"][0]["message"])]
    )
    params = NeMoGymResponseCreateParamsNonStreaming.model_validate(materialized["responses_create_params"])
    native_response = NeMoGymResponse(
        id="native-response",
        created_at=raw_response["created"],
        model=raw_response["model"],
        object="response",
        output=native_items,
        parallel_tool_calls=params.parallel_tool_calls,
        tool_choice=params.tool_choice or "auto",
        tools=params.tools,
        instructions=params.instructions,
        status="completed",
    )
    native_payload = materialized | {"response": native_response.model_dump(mode="json")}

    assert [item.type for item in native_response.output] == ["function_call", "function_call_output", "message"]
    assert _weather_reward(native_payload) == 1.0
    assert _weather_reward(atif_payload) == _weather_reward(native_payload)


async def test_same_atif_trajectory_can_be_reverified_with_different_stateless_configs() -> None:
    """Rescore one projected ATIF response without rerunning the agent or tools."""

    loaded = load_atif_trajectory(_RESPONSES_FIXTURE_PATH)
    payload = build_atif_verify_payload(
        {
            "responses_create_params": {
                "input": [{"role": "user", "content": "Run the command, then answer B."}],
                "tools": [],
            },
            "options": [{"A": "plain answer"}, {"B": "boxed answer"}],
            "expected_answer": "B",
        },
        loaded.trajectory,
    )
    request = MCQAVerifyRequest.model_validate(payload)

    def verifier(grading_mode: str) -> MCQAResourcesServer:
        config = MCQAResourcesServerConfig(
            host="127.0.0.1",
            port=8080,
            entrypoint="",
            name="mcqa",
            grading_mode=grading_mode,
        )
        assert config.REVERIFY_MODE == ReverifyMode.STATELESS
        return MCQAResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    strict_result = await verifier("strict_single_letter_boxed").verify(request)
    answer_colon_result = await verifier("lenient_answer_colon").verify(request)

    assert request.response.output_text == r"\boxed{B}"
    assert strict_result.response == answer_colon_result.response == request.response
    assert (
        strict_result.responses_create_params
        == answer_colon_result.responses_create_params
        == request.responses_create_params
    )
    assert strict_result.reward == 1.0
    assert strict_result.extracted_answer == "B"
    assert answer_colon_result.reward == 0.0
    assert answer_colon_result.extracted_answer is None


def test_fixed_relay_responses_fixture_preserves_invocation_correlation() -> None:
    loaded = load_atif_trajectory(_RESPONSES_FIXTURE_PATH)

    response = atif_trajectory_to_response(loaded.trajectory)

    assert loaded.source_sha256 == _RESPONSES_FIXTURE_SHA256
    assert [item.type for item in response.output] == [
        "reasoning",
        "function_call",
        "function_call_output",
        "reasoning",
        "message",
    ]
    correlated_items = [item for item in response.output if hasattr(item, "call_id")]
    assert [item.call_id for item in correlated_items] == [
        "call-abab8ac6-3a43-46a2-9224-d14a2d380504",
        "call-abab8ac6-3a43-46a2-9224-d14a2d380504",
    ]
    assert all(item.call_id != "fc_56a9401eb39a449c982424abb3b0fdc2" for item in correlated_items)


def test_prefixed_responses_item_id_orphan_shape_is_rejected() -> None:
    data = json.loads(_RESPONSES_FIXTURE_PATH.read_text())
    tool_step = data["steps"][1]
    tool_step["tool_calls"][0]["tool_call_id"] = "fc_56a9401eb39a449c982424abb3b0fdc2"
    tool_step.pop("observation")
    final_step = data["steps"][2]
    final_step["step_id"] = 4
    data["steps"].insert(
        2,
        {
            "step_id": 3,
            "source": "system",
            "message": "",
            "observation": {"results": [{"source_call_id": None, "content": "RELAY_ATIF_GYM_OK"}]},
        },
    )

    with pytest.raises(AtifProjectionError, match="raw provider tool calls do not match"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


def test_manifest_explicitly_joins_trajectory_to_materialized_rollout() -> None:
    materialized = _materialized_input()
    indexed = index_materialized_inputs([materialized])
    entry = AtifReverifyManifestEntry.model_validate(
        {
            "trajectory_path": _FIXTURE_PATH.name,
            TASK_INDEX_KEY_NAME: 7,
            ROLLOUT_INDEX_KEY_NAME: 2,
            "expected_sha256": _FIXTURE_SHA256,
        }
    )

    projected = project_atif_manifest_entry(
        entry,
        indexed,
        manifest_directory=_FIXTURE_PATH.parent,
    )

    assert projected.task_index == 7
    assert projected.rollout_index == 2
    assert projected.trajectory_id == "gym-atif-spike-session"
    assert projected.session_id == "gym-atif-spike-session"
    assert projected.source_sha256 == _FIXTURE_SHA256
    assert projected.schema_version == "ATIF-v1.7"
    assert projected.projection_status == "complete"
    assert projected.payload["expected_answer"] == "72 and sunny"
    assert _weather_reward(projected.payload) == 1.0


@pytest.mark.parametrize(("field_name", "value"), [("task_index", 8), ("rollout_index", 3)])
def test_manifest_rejects_conflicting_gym_source_identity(
    tmp_path: Path,
    field_name: str,
    value: int,
) -> None:
    data = _trajectory_data()
    data["extra"] = {
        "nemo_gym": {
            "source": {"format": "ng_trajectory", "task_index": 7, "rollout_index": 2},
            "conversion": {"status": "complete"},
        }
    }
    data["extra"]["nemo_gym"]["source"][field_name] = value
    trajectory_path = tmp_path / "trajectory.json"
    trajectory_path.write_text(json.dumps(data))
    entry = AtifReverifyManifestEntry(
        trajectory_path=trajectory_path,
        task_index=7,
        rollout_index=2,
    )

    with pytest.raises(AtifProjectionError, match=f"source {field_name}.*conflicts with manifest"):
        project_atif_manifest_entry(
            entry,
            index_materialized_inputs([_materialized_input()]),
            manifest_directory=tmp_path,
        )


def test_manifest_rejects_wrong_source_hash() -> None:
    indexed = index_materialized_inputs([_materialized_input()])
    entry = AtifReverifyManifestEntry.model_validate(
        {
            "trajectory_path": _FIXTURE_PATH.name,
            TASK_INDEX_KEY_NAME: 7,
            ROLLOUT_INDEX_KEY_NAME: 2,
            "expected_sha256": "0" * 64,
        }
    )

    with pytest.raises(AtifProjectionError, match="source hash mismatch"):
        project_atif_manifest_entry(entry, indexed, manifest_directory=_FIXTURE_PATH.parent)


def test_manifest_loader_reports_the_invalid_jsonl_row(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "trajectory_path": "first.json",
                TASK_INDEX_KEY_NAME: 7,
                ROLLOUT_INDEX_KEY_NAME: 2,
            }
        )
        + "\nnot-json\n"
    )

    with pytest.raises(AtifProjectionError, match="invalid ATIF manifest row 2"):
        load_atif_manifest(manifest)


def test_manifest_loader_rejects_empty_input(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n")

    with pytest.raises(AtifProjectionError, match="contains no entries"):
        load_atif_manifest(manifest)


@pytest.mark.parametrize(
    ("loader", "filename", "match"),
    [
        (load_atif_trajectory, "missing-trajectory.json", "could not read ATIF trajectory"),
        (load_atif_manifest, "missing-manifest.jsonl", "could not read ATIF manifest"),
    ],
)
def test_missing_atif_inputs_raise_clean_config_errors(loader: Any, filename: str, match: str, tmp_path: Path) -> None:
    with pytest.raises(AtifProjectionError, match=match):
        loader(tmp_path / filename)


def test_manifest_batch_rejects_more_than_one_trajectory_for_a_rollout() -> None:
    indexed = index_materialized_inputs([_materialized_input()])
    entry = AtifReverifyManifestEntry.model_validate(
        {
            "trajectory_path": _FIXTURE_PATH.name,
            TASK_INDEX_KEY_NAME: 7,
            ROLLOUT_INDEX_KEY_NAME: 2,
        }
    )

    with pytest.raises(AtifProjectionError, match="maps materialized rollout.*more than once"):
        project_atif_manifest_entries(
            [entry, entry],
            indexed,
            manifest_directory=_FIXTURE_PATH.parent,
        )


def test_manifest_batch_accepts_distinct_trajectories_without_trajectory_ids(tmp_path: Path) -> None:
    first_data = _trajectory_data()
    first_data["trajectory_id"] = None
    first_data["session_id"] = "shared-run"
    second_data = deepcopy(first_data)
    second_data["steps"][-1]["message"] = "A distinct response."

    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_path.write_text(json.dumps(first_data))
    second_path.write_text(json.dumps(second_data))
    first_input = _materialized_input()
    second_input = _materialized_input() | {ROLLOUT_INDEX_KEY_NAME: 3}
    indexed = index_materialized_inputs([first_input, second_input])
    entries = [
        AtifReverifyManifestEntry(
            trajectory_path=first_path,
            task_index=7,
            rollout_index=2,
        ),
        AtifReverifyManifestEntry(
            trajectory_path=second_path,
            task_index=7,
            rollout_index=3,
        ),
    ]

    projected = project_atif_manifest_entries(entries, indexed, manifest_directory=tmp_path)

    assert len(projected) == 2
    assert projected[0].trajectory_content_sha256 != projected[1].trajectory_content_sha256


def test_manifest_batch_rejects_reformatted_copy_without_a_trajectory_id(tmp_path: Path) -> None:
    trajectory_data = _trajectory_data()
    trajectory_data["trajectory_id"] = None
    trajectory_data["session_id"] = "shared-run"

    compact_path = tmp_path / "compact.json"
    pretty_path = tmp_path / "pretty.json"
    compact_path.write_text(json.dumps(trajectory_data, separators=(",", ":")))
    pretty_path.write_text(json.dumps(trajectory_data, indent=2))

    first_input = _materialized_input()
    second_input = _materialized_input() | {ROLLOUT_INDEX_KEY_NAME: 3}
    indexed = index_materialized_inputs([first_input, second_input])
    entries = [
        AtifReverifyManifestEntry(
            trajectory_path=compact_path,
            task_index=7,
            rollout_index=2,
        ),
        AtifReverifyManifestEntry(
            trajectory_path=pretty_path,
            task_index=7,
            rollout_index=3,
        ),
    ]

    compact = project_atif_manifest_entry(entries[0], indexed, manifest_directory=tmp_path)
    pretty = project_atif_manifest_entry(entries[1], indexed, manifest_directory=tmp_path)

    assert compact.source_sha256 != pretty.source_sha256
    assert compact.trajectory_content_sha256 == pretty.trajectory_content_sha256
    assert [item["id"] for item in compact.payload["response"]["output"] if "id" in item] == [
        item["id"] for item in pretty.payload["response"]["output"] if "id" in item
    ]
    with pytest.raises(AtifProjectionError, match="repeats ATIF trajectory identity"):
        project_atif_manifest_entries(entries, indexed, manifest_directory=tmp_path)


def test_materialized_input_index_rejects_duplicate_rollout_keys() -> None:
    row = _materialized_input()

    with pytest.raises(AtifProjectionError, match="duplicate materialized input key"):
        index_materialized_inputs([row, row])


@pytest.mark.parametrize("value", ["7", 7.0, True])
def test_manifest_indices_do_not_coerce_non_integer_values(value: Any) -> None:
    with pytest.raises(ValueError):
        AtifReverifyManifestEntry.model_validate(
            {
                "trajectory_path": _FIXTURE_PATH.name,
                TASK_INDEX_KEY_NAME: value,
                ROLLOUT_INDEX_KEY_NAME: 2,
            }
        )


def test_observation_results_are_paired_by_source_call_id() -> None:
    response = atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(_trajectory_data()))
    outputs = {item.call_id: item.output for item in response.output if item.type == "function_call_output"}

    assert outputs == {"call-b": "4", "call-a": "found"}
    assert response.output[0].type == "reasoning"
    assert response.output[0].summary[0].text == "I need both results."
    assert response.output[0].content is None
    assert [item.type for item in response.output] == [
        "reasoning",
        "function_call",
        "function_call",
        "function_call_output",
        "function_call_output",
        "message",
    ]


def test_egress_shaped_training_metadata_is_restored_on_the_last_model_output() -> None:
    data = _trajectory_data()
    data["steps"][1]["metrics"] = {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "cached_tokens": 1,
        "extra": {"nemo_gym": {"reasoning_tokens": 2, "total_tokens": 15}},
    }
    data["steps"][-1]["metrics"] = {
        "prompt_tokens": 131,
        "completion_tokens": 13,
        "cached_tokens": 8,
        "prompt_token_ids": [10, 11],
        "completion_token_ids": [12, 13],
        "logprobs": [-0.25, -0.5],
        "extra": {
            "nemo_gym": {
                "reasoning_tokens": 3,
                "total_tokens": 144,
                "routed_experts": "nrlre1:uint16:2x1x1:AAAA",
            }
        },
    }
    data["final_metrics"] = {
        "total_prompt_tokens": 141,
        "total_completion_tokens": 18,
        "total_cached_tokens": 9,
        "total_steps": 3,
    }

    trajectory = AtifTrajectoryV1_7.model_validate(data)
    response = atif_trajectory_to_response(trajectory)

    final_item = response.output[-1]
    assert isinstance(final_item, NeMoGymResponseOutputMessageForTraining)
    assert final_item.prompt_token_ids == [10, 11]
    assert final_item.generation_token_ids == [12, 13]
    assert final_item.generation_log_probs == [-0.25, -0.5]
    assert final_item.routed_experts == "nrlre1:uint16:2x1x1:AAAA"
    persisted_item = build_atif_verify_payload(_materialized_input(), trajectory)["response"]["output"][-1]
    assert persisted_item["prompt_token_ids"] == [10, 11]
    assert persisted_item["generation_token_ids"] == [12, 13]
    assert persisted_item["generation_log_probs"] == [-0.25, -0.5]
    assert persisted_item["routed_experts"] == "nrlre1:uint16:2x1x1:AAAA"


@pytest.mark.parametrize(
    ("extension", "message"),
    [
        (
            {"source": {"invocation_status": "failed"}},
            "Gym source invocation is not completed",
        ),
        (
            {"conversion": {"status": "partial"}},
            "Gym ATIF conversion is not complete",
        ),
        (
            {"conversion": {"status": "unknown"}},
            "Gym ATIF conversion is not complete",
        ),
        ({"source": None}, "ATIF extra.nemo_gym.source must be an object"),
        ({"conversion": None}, "ATIF extra.nemo_gym.conversion must be an object"),
    ],
)
def test_declared_incomplete_gym_conversion_is_not_reported_complete(extension: dict[str, Any], message: str) -> None:
    data = _trajectory_data()
    data["extra"] = {"nemo_gym": extension}

    with pytest.raises(AtifProjectionError, match=message):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


def test_complete_gym_conversion_metadata_is_accepted() -> None:
    data = _trajectory_data()
    data["extra"] = {
        "nemo_gym": {
            "source": {"invocation_status": "completed"},
            "conversion": {"status": "complete"},
        }
    }

    assert atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data)).status == "completed"


def test_training_metadata_on_a_tool_step_is_attached_to_the_last_function_call() -> None:
    data = _trajectory_data()
    data["steps"][1]["metrics"] = {
        "prompt_token_ids": [10],
        "completion_token_ids": [11],
        "logprobs": [-0.25],
    }

    response = atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))
    calls = [item for item in response.output if item.type == "function_call"]

    assert not isinstance(calls[0], NeMoGymResponseFunctionToolCallForTraining)
    assert isinstance(calls[1], NeMoGymResponseFunctionToolCallForTraining)
    assert calls[1].prompt_token_ids == [10]
    assert calls[1].generation_token_ids == [11]
    assert calls[1].generation_log_probs == [-0.25]


@pytest.mark.parametrize("missing_field", ["prompt_token_ids", "completion_token_ids", "logprobs"])
def test_partial_training_metadata_is_rejected(missing_field: str) -> None:
    data = _trajectory_data()
    metrics = {
        "prompt_token_ids": [10],
        "completion_token_ids": [11],
        "logprobs": [-0.25],
    }
    del metrics[missing_field]
    data["steps"][-1]["metrics"] = metrics

    with pytest.raises(AtifProjectionError, match="training token metadata is incomplete"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


def test_misaligned_training_metadata_is_rejected() -> None:
    data = _trajectory_data()
    data["steps"][-1]["metrics"] = {
        "prompt_token_ids": [10],
        "completion_token_ids": [11, 12],
        "logprobs": [-0.25],
    }

    with pytest.raises(AtifProjectionError, match="token IDs and log probabilities must have the same length"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("prompt_token_ids", [True], "token IDs must be non-negative JSON integer arrays"),
        ("prompt_token_ids", ["10"], "token IDs must be non-negative JSON integer arrays"),
        ("completion_token_ids", [False], "token IDs must be non-negative JSON integer arrays"),
        ("completion_token_ids", ["11"], "token IDs must be non-negative JSON integer arrays"),
        ("completion_token_ids", [-1], "token IDs must be non-negative JSON integer arrays"),
        ("logprobs", [True], "log probabilities must be finite JSON number arrays"),
        ("logprobs", ["-0.25"], "log probabilities must be finite JSON number arrays"),
    ],
)
def test_training_metadata_rejects_coercible_non_json_number_types(field: str, value: list[Any], message: str) -> None:
    data = _trajectory_data()
    metrics = {
        "prompt_token_ids": [10],
        "completion_token_ids": [11],
        "logprobs": [-0.25],
    }
    metrics[field] = value
    data["steps"][-1]["metrics"] = metrics

    with pytest.raises(ValidationError, match=message):
        AtifTrajectoryV1_7.model_validate(data)


def test_training_metadata_rejects_boolean_routed_expert_indices() -> None:
    data = _trajectory_data()
    data["steps"][-1]["metrics"] = {
        "prompt_token_ids": [10],
        "completion_token_ids": [11],
        "logprobs": [-0.25],
        "extra": {"nemo_gym": {"routed_experts": [[[True]]]}},
    }

    with pytest.raises(AtifProjectionError, match="must contain only JSON integer indices"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


@pytest.mark.parametrize(
    ("target", "field"),
    [
        ("step", "prompt_tokens"),
        ("step", "completion_tokens"),
        ("step", "cached_tokens"),
        ("final", "total_prompt_tokens"),
        ("final", "total_completion_tokens"),
        ("final", "total_cached_tokens"),
        ("final", "total_steps"),
    ],
)
def test_atif_metric_counts_do_not_coerce_json_booleans(target: str, field: str) -> None:
    data = _trajectory_data()
    if target == "step":
        data["steps"][-1]["metrics"] = {field: True}
    else:
        data["final_metrics"] = {field: True}

    with pytest.raises(ValidationError):
        AtifTrajectoryV1_7.model_validate(data)


@pytest.mark.parametrize(
    ("target", "field"),
    [
        ("step", "prompt_tokens"),
        ("step", "completion_tokens"),
        ("step", "cached_tokens"),
        ("step", "cost_usd"),
        ("final", "total_prompt_tokens"),
        ("final", "total_completion_tokens"),
        ("final", "total_cached_tokens"),
        ("final", "total_cost_usd"),
        ("final", "total_steps"),
    ],
)
def test_atif_metrics_reject_negative_counts_and_costs(target: str, field: str) -> None:
    data = _trajectory_data()
    value = -1.0 if field.endswith("cost_usd") else -1
    if target == "step":
        data["steps"][-1]["metrics"] = {field: value}
    else:
        data["final_metrics"] = {field: value}

    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        AtifTrajectoryV1_7.model_validate(data)


@pytest.mark.parametrize("target", ("step", "final"))
def test_atif_usage_rejects_cached_tokens_greater_than_prompt_tokens(target: str) -> None:
    data = _trajectory_data()
    if target == "step":
        data["steps"][1]["metrics"] = {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "cached_tokens": 11,
        }
        data["final_metrics"] = {
            "total_prompt_tokens": 10,
            "total_completion_tokens": 2,
            "total_cached_tokens": 10,
        }
        message = "step 2 cached_tokens exceeds prompt_tokens"
    else:
        data["final_metrics"] = {
            "total_prompt_tokens": 10,
            "total_completion_tokens": 2,
            "total_cached_tokens": 11,
        }
        message = "total_cached_tokens exceeds total_prompt_tokens"

    with pytest.raises(AtifProjectionError, match=message):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


def test_gym_usage_extension_restores_reasoning_tokens() -> None:
    data = _trajectory_data()
    data["steps"][1]["metrics"] = {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "cached_tokens": 1,
        "extra": {"nemo_gym": {"reasoning_tokens": 2, "total_tokens": 15}},
    }
    data["steps"][2]["metrics"] = {
        "prompt_tokens": 20,
        "completion_tokens": 6,
        "cached_tokens": 2,
        "extra": {"nemo_gym": {"reasoning_tokens": 3, "total_tokens": 26}},
    }
    data["final_metrics"] = {
        "total_prompt_tokens": 30,
        "total_completion_tokens": 11,
        "total_cached_tokens": 3,
        "total_steps": 3,
    }

    response = atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))

    assert response.usage is not None
    assert response.usage.output_tokens_details.reasoning_tokens == 5
    assert response.usage.total_tokens == 41


def test_gym_usage_extension_keeps_partially_known_reasoning_tokens_unknown() -> None:
    data = _trajectory_data()
    data["steps"][1]["metrics"] = {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "extra": {"nemo_gym": {"reasoning_tokens": 2, "total_tokens": 15}},
    }
    data["steps"][2]["metrics"] = {
        "prompt_tokens": 20,
        "completion_tokens": 6,
        "extra": {"nemo_gym": {"total_tokens": 26}},
    }
    data["final_metrics"] = {"total_prompt_tokens": 30, "total_completion_tokens": 11, "total_steps": 3}

    response = atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))

    assert response.usage is not None
    assert response.usage.output_tokens_details.reasoning_tokens is None


@pytest.mark.parametrize("value", [True, -1, "2"])
def test_gym_usage_extension_rejects_invalid_reasoning_counts(value: Any) -> None:
    data = _trajectory_data()
    data["steps"][1]["metrics"] = {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "extra": {"nemo_gym": {"reasoning_tokens": value, "total_tokens": 15}},
    }
    data["steps"][2]["metrics"] = {
        "prompt_tokens": 20,
        "completion_tokens": 6,
        "extra": {"nemo_gym": {"reasoning_tokens": 3, "total_tokens": 26}},
    }
    data["final_metrics"] = {"total_prompt_tokens": 30, "total_completion_tokens": 11, "total_steps": 3}

    with pytest.raises(AtifProjectionError, match="reasoning_tokens metadata is not a non-negative integer"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


def test_gym_usage_extension_rejects_inconsistent_total_tokens() -> None:
    data = _trajectory_data()
    data["steps"][1]["metrics"] = {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "extra": {"nemo_gym": {"reasoning_tokens": 2, "total_tokens": 16}},
    }
    data["steps"][2]["metrics"] = {
        "prompt_tokens": 20,
        "completion_tokens": 6,
        "extra": {"nemo_gym": {"reasoning_tokens": 3, "total_tokens": 26}},
    }
    data["final_metrics"] = {"total_prompt_tokens": 30, "total_completion_tokens": 11, "total_steps": 3}

    with pytest.raises(AtifProjectionError, match="step 2 Gym total_tokens metadata does not match"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


def test_relay_usage_extension_restores_reasoning_tokens() -> None:
    data = _trajectory_data()
    data["steps"][1]["metrics"] = {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "extra": {"output_tokens_details": {"reasoning_tokens": 2}, "total_tokens": 15},
    }
    data["steps"][2]["metrics"] = {
        "prompt_tokens": 20,
        "completion_tokens": 6,
        "extra": {"output_tokens_details": {"reasoning_tokens": 3}, "total_tokens": 26},
    }
    data["final_metrics"] = {"total_prompt_tokens": 30, "total_completion_tokens": 11, "total_steps": 3}

    response = atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))

    assert response.usage is not None
    assert response.usage.output_tokens_details.reasoning_tokens == 5
    assert response.usage.total_tokens == 41


def test_partial_per_step_usage_is_not_reported_as_a_complete_aggregate() -> None:
    data = _trajectory_data()
    data["steps"][1]["metrics"] = {"prompt_tokens": 10, "completion_tokens": 5}
    data["final_metrics"] = {"total_prompt_tokens": 10, "total_completion_tokens": 5, "total_steps": 3}

    response = atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))

    assert response.usage is None


def test_gym_egress_shaped_partial_usage_does_not_block_reverification() -> None:
    data = _trajectory_data()
    data["steps"][1]["metrics"] = {
        "extra": {"nemo_gym": {"reasoning_tokens": 9, "total_tokens": 9}},
    }
    data["steps"][2]["metrics"] = {
        "prompt_tokens": 131,
        "completion_tokens": 13,
        "cached_tokens": 8,
        "extra": {"nemo_gym": {"reasoning_tokens": 3, "total_tokens": 144}},
    }
    data["final_metrics"] = {"total_steps": 3}
    data["extra"] = {
        "nemo_gym": {
            "source": {"invocation_status": "completed"},
            "conversion": {"profile": "ng-trajectory-to-atif-v1", "status": "complete"},
        }
    }

    response = atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))

    assert response.usage is None
    assert [item.type for item in response.output] == [
        "reasoning",
        "function_call",
        "function_call",
        "function_call_output",
        "function_call_output",
        "message",
    ]


def test_final_usage_must_match_complete_per_step_usage() -> None:
    data = _trajectory_data()
    data["steps"][1]["metrics"] = {"prompt_tokens": 10, "completion_tokens": 5}
    data["steps"][2]["metrics"] = {"prompt_tokens": 20, "completion_tokens": 6}
    data["final_metrics"] = {"total_prompt_tokens": 31, "total_completion_tokens": 11, "total_steps": 3}

    with pytest.raises(AtifProjectionError, match="do not match the complete per-model-step metrics"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


def test_partial_relay_cached_token_total_is_not_reported_as_complete() -> None:
    data = _trajectory_data()
    data["steps"][1]["metrics"] = {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "cached_tokens": 4,
    }
    data["steps"][2]["metrics"] = {
        "prompt_tokens": 20,
        "completion_tokens": 6,
    }
    # Relay sums each available metric independently, so this final cache value
    # covers only the first model step and is not a trajectory-wide total.
    data["final_metrics"] = {
        "total_prompt_tokens": 30,
        "total_completion_tokens": 11,
        "total_cached_tokens": 4,
        "total_steps": 3,
    }

    response = atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))

    assert response.usage is not None
    assert response.usage.input_tokens == 30
    assert response.usage.output_tokens == 11
    assert response.usage.input_tokens_details.cached_tokens is None


def test_routed_experts_is_restored_only_from_the_nemo_gym_extension() -> None:
    data = _trajectory_data()
    data["steps"][-1]["metrics"] = {
        "prompt_token_ids": [10],
        "completion_token_ids": [11],
        "logprobs": [-0.25],
        "extra": {"routed_experts": "not-the-nemo-gym-extension"},
    }

    response = atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))

    final_item = response.output[-1]
    assert isinstance(final_item, NeMoGymResponseOutputMessageForTraining)
    assert final_item.routed_experts is None


def test_routed_experts_without_required_training_metadata_is_rejected() -> None:
    data = _trajectory_data()
    data["steps"][-1]["metrics"] = {"extra": {"nemo_gym": {"routed_experts": [[[0, 1]]]}}}

    with pytest.raises(AtifProjectionError, match="training token metadata is incomplete"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


def test_tool_call_without_a_result_is_not_marked_completed() -> None:
    data = _trajectory_data()
    data["steps"][1]["observation"]["results"] = [
        result for result in data["steps"][1]["observation"]["results"] if result["source_call_id"] != "call-b"
    ]

    with pytest.raises(AtifProjectionError, match="no observation result.*call-b"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


@pytest.mark.parametrize("step_ids", [[1, 1, 3], [1, 3, 4], [2, 3, 4]])
def test_parser_requires_sequential_step_ids(step_ids: list[int]) -> None:
    data = _trajectory_data()
    for step, step_id in zip(data["steps"], step_ids, strict=True):
        step["step_id"] = step_id

    with pytest.raises(ValidationError, match="sequential from 1"):
        AtifTrajectoryV1_7.model_validate(data)


@pytest.mark.parametrize("field,value", [("reasoning_content", "hidden"), ("tool_calls", []), ("metrics", {})])
def test_parser_rejects_agent_only_fields_on_user_steps(field: str, value: Any) -> None:
    data = _trajectory_data()
    data["steps"][0][field] = value

    with pytest.raises(ValidationError, match="only valid for agent steps"):
        AtifTrajectoryV1_7.model_validate(data)


def test_parser_requires_object_tool_arguments() -> None:
    data = _trajectory_data()
    data["steps"][1]["tool_calls"][0]["arguments"] = "not-an-object"

    with pytest.raises(ValidationError, match="dictionary"):
        AtifTrajectoryV1_7.model_validate(data)


def test_raw_responses_reasoning_is_preserved_when_structured_reasoning_is_absent() -> None:
    data = _trajectory_data()
    data["steps"][1]["reasoning_content"] = None
    data["steps"][1]["extra"] = {
        "llm_response": {
            "status": "completed",
            "output": [
                {
                    "id": "rs-live-shape",
                    "type": "reasoning",
                    "summary": [],
                    # NVIDIA's OpenAI-compatible Responses endpoint omits the
                    # upstream reasoning_text discriminator on these parts.
                    "content": [{"text": "Use both tool results."}],
                },
                {
                    "id": "fc-a",
                    "call_id": "call-a",
                    "type": "function_call",
                    "name": "lookup",
                    "arguments": '{"q":"x"}',
                },
                {
                    "id": "fc-b",
                    "call_id": "call-b",
                    "type": "function_call",
                    "name": "calculate",
                    "arguments": '{"x":2}',
                },
            ],
        }
    }

    response = atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))

    assert response.output[0].type == "reasoning"
    assert response.output[0].id == "rs-live-shape"
    assert response.output[0].content[0].type == "reasoning_text"
    assert response.output[0].content[0].text == "Use both tool results."


def test_completed_empty_agent_answer_is_preserved_for_scoring() -> None:
    data = _trajectory_data()
    data["steps"] = [data["steps"][0], {"step_id": 2, "source": "agent", "message": ""}]

    response = atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))

    assert [item.type for item in response.output] == ["message"]
    assert response.output[0].content[0].text == ""
    assert response.status == "completed"


def test_aggregated_llm_step_is_rejected_instead_of_flattened() -> None:
    data = _trajectory_data()
    data["steps"][1]["llm_call_count"] = 2

    with pytest.raises(AtifProjectionError, match="aggregates 2 LLM calls"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


def test_unrecognized_raw_provider_shape_is_not_reported_as_complete() -> None:
    data = _trajectory_data()
    data["steps"][-1]["extra"] = {"llm_response": {"actions": [{"type": "future_action"}]}}

    with pytest.raises(AtifProjectionError, match="unrecognized raw provider response shape"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


@pytest.mark.parametrize("llm_response", [None, [], "completed", True])
def test_non_object_raw_provider_evidence_is_not_silently_ignored(llm_response: Any) -> None:
    data = _trajectory_data()
    data["steps"][-1]["extra"] = {"llm_response": llm_response}

    with pytest.raises(AtifProjectionError, match="llm_response is not a supported provider response object"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


@pytest.mark.parametrize(
    "outer_failure",
    [
        {"status": "failed"},
        {"error": {"message": "provider failed"}},
        {"incomplete_details": {"reason": "max_output_tokens"}},
    ],
)
def test_outer_relay_envelope_cannot_hide_failure_in_a_completed_raw_response(
    outer_failure: dict[str, Any],
) -> None:
    data = json.loads(_RESPONSES_FIXTURE_PATH.read_text())
    final_step = data["steps"][-1]
    completed_raw_response = final_step["extra"]["llm_response"]
    final_step["extra"]["llm_response"] = outer_failure | {"raw_response": completed_raw_response}

    with pytest.raises(AtifProjectionError, match="non-completed provider|failed or incomplete provider"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("actions", [{"type": "hidden_action"}]),
        ("content", "hidden output"),
    ],
)
def test_outer_relay_envelope_cannot_hide_output_in_a_completed_raw_response(
    field_name: str,
    value: Any,
) -> None:
    data = json.loads(_RESPONSES_FIXTURE_PATH.read_text())
    final_step = data["steps"][-1]
    completed_raw_response = final_step["extra"]["llm_response"]
    final_step["extra"]["llm_response"] = {
        "raw_response": completed_raw_response,
        field_name: value,
    }

    with pytest.raises(AtifProjectionError, match=rf"unsupported output field '{field_name}'"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


def test_raw_responses_tool_calls_must_match_canonical_atif() -> None:
    data = _trajectory_data()
    data["steps"][1]["extra"] = {
        "llm_response": {
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "id": "fc-a",
                    "call_id": "call-a",
                    "name": "lookup",
                    "arguments": '{"q":"x"}',
                }
            ],
        }
    }

    with pytest.raises(AtifProjectionError, match="raw provider tool calls do not match"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


@pytest.mark.parametrize(
    ("field", "match"), [("tool_call_id", "blank tool_call_id"), ("function_name", "blank function_name")]
)
def test_canonical_tool_identity_must_not_be_whitespace_only(field: str, match: str) -> None:
    data = _trajectory_data()
    data["steps"][1]["tool_calls"][0][field] = " \t "
    if field == "tool_call_id":
        data["steps"][1]["observation"]["results"][1]["source_call_id"] = " \t "

    with pytest.raises(AtifProjectionError, match=match):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


@pytest.mark.parametrize("provider", ["responses", "chat", "anthropic"])
@pytest.mark.parametrize(
    ("field", "match"), [("call_id", "non-blank invocation ID"), ("name", "non-blank function name")]
)
def test_raw_provider_tool_identity_must_not_be_whitespace_only(provider: str, field: str, match: str) -> None:
    data = _trajectory_data()
    if provider == "responses":
        tool_call = {
            "type": "function_call",
            "id": "fc-a",
            "call_id": "call-a",
            "name": "lookup",
            "arguments": '{"q":"x"}',
            "status": "completed",
        }
        tool_call[field] = " \t "
        raw_response = {"status": "completed", "output": [tool_call]}
    elif provider == "chat":
        tool_call = {
            "id": "call-a",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"q":"x"}'},
        }
        if field == "call_id":
            tool_call["id"] = " \t "
        else:
            tool_call["function"]["name"] = " \t "
        raw_response = {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {"role": "assistant", "content": None, "tool_calls": [tool_call]},
                }
            ]
        }
    else:
        tool_call = {"type": "tool_use", "id": "call-a", "name": "lookup", "input": {"q": "x"}}
        tool_call["id" if field == "call_id" else "name"] = " \t "
        raw_response = {
            "type": "message",
            "role": "assistant",
            "stop_reason": "tool_use",
            "content": [tool_call],
        }
    data["steps"][1]["extra"] = {"llm_response": raw_response}

    with pytest.raises(AtifProjectionError, match=match):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


@pytest.mark.parametrize("provider", ["responses", "chat", "anthropic"])
def test_raw_tool_argument_comparison_distinguishes_json_booleans_from_numbers(provider: str) -> None:
    data = _trajectory_data()
    tool_step = data["steps"][1]
    tool_step["tool_calls"] = [{"tool_call_id": "call-b", "function_name": "calculate", "arguments": {"x": True}}]
    tool_step["observation"] = {"results": [{"source_call_id": "call-b", "content": "4"}]}

    if provider == "responses":
        raw_response = {
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "id": "fc-b",
                    "call_id": "call-b",
                    "name": "calculate",
                    "arguments": '{"x":1}',
                    "status": "completed",
                }
            ],
        }
    elif provider == "chat":
        raw_response = {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-b",
                                "type": "function",
                                "function": {"name": "calculate", "arguments": '{"x":1}'},
                            }
                        ],
                    },
                }
            ]
        }
    else:
        raw_response = {
            "type": "message",
            "role": "assistant",
            "stop_reason": "tool_use",
            "content": [{"type": "tool_use", "id": "call-b", "name": "calculate", "input": {"x": 1}}],
        }
    tool_step["extra"] = {"llm_response": raw_response}

    with pytest.raises(AtifProjectionError, match="raw provider tool calls do not match"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_atif_loader_rejects_non_json_numeric_constants(constant: str, tmp_path: Path) -> None:
    path = tmp_path / "trajectory.json"
    payload = json.dumps(_trajectory_data()).replace('{"x": 2}', f'{{"x": {constant}}}')
    path.write_text(payload)

    with pytest.raises(AtifProjectionError, match="invalid ATIF trajectory"):
        load_atif_trajectory(path)


def test_atif_loader_rejects_float_overflow(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.json"
    payload = json.dumps(_trajectory_data()).replace('{"x": 2}', '{"x": 1e400}')
    path.write_text(payload)

    with pytest.raises(AtifProjectionError, match="exceeds the finite float range"):
        load_atif_trajectory(path)


def test_atif_loader_rejects_nonzero_float_underflow(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.json"
    payload = json.dumps(_trajectory_data()).replace('{"x": 2}', '{"x": 1e-999}')
    path.write_text(payload)

    with pytest.raises(AtifProjectionError, match="underflows the finite float range"):
        load_atif_trajectory(path)


def test_atif_loader_accepts_true_zero_with_an_extreme_exponent(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.json"
    payload = json.dumps(_trajectory_data()).replace('{"x": 2}', '{"x": 0e-99999999999999999999}')
    path.write_text(payload)

    loaded = load_atif_trajectory(path)

    assert loaded.trajectory.steps[1].tool_calls is not None
    assert loaded.trajectory.steps[1].tool_calls[1].arguments["x"] == 0.0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_in_memory_projection_rejects_nonfinite_tool_arguments(value: float) -> None:
    data = _trajectory_data()
    data["steps"][1]["tool_calls"][1]["arguments"]["x"] = value

    with pytest.raises(AtifProjectionError, match="non-finite JSON number"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


def test_atif_loader_preserves_arbitrary_size_json_integers(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.json"
    data = _trajectory_data()
    large_integer = 2**100
    data["steps"][1]["tool_calls"][1]["arguments"]["x"] = large_integer
    path.write_text(json.dumps(data))

    loaded = load_atif_trajectory(path)

    assert loaded.trajectory.steps[1].tool_calls is not None
    assert loaded.trajectory.steps[1].tool_calls[1].arguments["x"] == large_integer


def test_atif_loader_rejects_duplicate_json_object_keys(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.json"
    payload = json.dumps(_trajectory_data()).replace(
        '"arguments": {"x": 2}',
        '"arguments": {"x": 2, "x": 3}',
    )
    path.write_text(payload)

    with pytest.raises(AtifProjectionError, match="duplicate object key 'x'"):
        load_atif_trajectory(path)


def test_atif_manifest_loader_rejects_duplicate_json_object_keys(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        '{"trajectory_path":"trajectory.json","_ng_task_index":7,"_ng_task_index":8,"_ng_rollout_index":2}\n'
    )

    with pytest.raises(AtifProjectionError, match="duplicate object key '_ng_task_index'"):
        load_atif_manifest(path)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_raw_tool_arguments_reject_non_json_numeric_constants(constant: str) -> None:
    data = _trajectory_data()
    data["steps"][1]["extra"] = {
        "llm_response": {
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "id": "fc-a",
                    "call_id": "call-a",
                    "name": "lookup",
                    "arguments": f'{{"q":{constant}}}',
                    "status": "completed",
                },
                {
                    "type": "function_call",
                    "id": "fc-b",
                    "call_id": "call-b",
                    "name": "calculate",
                    "arguments": '{"x":2}',
                    "status": "completed",
                },
            ],
        }
    }

    with pytest.raises(AtifProjectionError, match="invalid JSON arguments"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


def test_raw_tool_arguments_reject_float_overflow() -> None:
    data = _trajectory_data()
    data["steps"][1]["extra"] = {
        "llm_response": {
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "id": "fc-a",
                    "call_id": "call-a",
                    "name": "lookup",
                    "arguments": '{"q":1e400}',
                    "status": "completed",
                },
                {
                    "type": "function_call",
                    "id": "fc-b",
                    "call_id": "call-b",
                    "name": "calculate",
                    "arguments": '{"x":2}',
                    "status": "completed",
                },
            ],
        }
    }

    with pytest.raises(AtifProjectionError, match="exceeds the finite float range"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


def test_raw_tool_arguments_reject_nonzero_float_underflow() -> None:
    data = _trajectory_data()
    data["steps"][1]["extra"] = {
        "llm_response": {
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "id": "fc-a",
                    "call_id": "call-a",
                    "name": "lookup",
                    "arguments": '{"q":1e-999}',
                    "status": "completed",
                },
                {
                    "type": "function_call",
                    "id": "fc-b",
                    "call_id": "call-b",
                    "name": "calculate",
                    "arguments": '{"x":2}',
                    "status": "completed",
                },
            ],
        }
    }

    with pytest.raises(AtifProjectionError, match="underflows the finite float range"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


def test_raw_tool_arguments_reject_duplicate_json_object_keys() -> None:
    data = _trajectory_data()
    data["steps"][1]["extra"] = {
        "llm_response": {
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "id": "fc-a",
                    "call_id": "call-a",
                    "name": "lookup",
                    "arguments": '{"q":"x","q":"x"}',
                    "status": "completed",
                },
                {
                    "type": "function_call",
                    "id": "fc-b",
                    "call_id": "call-b",
                    "name": "calculate",
                    "arguments": '{"x":2}',
                    "status": "completed",
                },
            ],
        }
    }

    with pytest.raises(AtifProjectionError, match="duplicate object key 'q'"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


def test_raw_tool_arguments_preserve_arbitrary_size_json_integers() -> None:
    data = _trajectory_data()
    large_integer = 2**100
    data["steps"][1]["tool_calls"][1]["arguments"]["x"] = large_integer
    data["steps"][1]["extra"] = {
        "llm_response": {
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "id": "fc-a",
                    "call_id": "call-a",
                    "name": "lookup",
                    "arguments": '{"q":"x"}',
                    "status": "completed",
                },
                {
                    "type": "function_call",
                    "id": "fc-b",
                    "call_id": "call-b",
                    "name": "calculate",
                    "arguments": f'{{"x":{large_integer}}}',
                    "status": "completed",
                },
            ],
        }
    }

    response = atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))

    calls = [item for item in response.output if item.type == "function_call"]
    assert json.loads(calls[1].arguments)["x"] == large_integer


def test_raw_provider_message_must_match_canonical_atif() -> None:
    data = _trajectory_data()
    data["steps"][-1]["extra"] = {
        "llm_response": {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "A different answer."}],
                }
            ],
        }
    }

    with pytest.raises(AtifProjectionError, match="raw provider message does not match"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


@pytest.mark.parametrize("item_type", ["computer_call", "mcp_call", "custom_tool_call", "local_shell_call"])
def test_known_unprojected_responses_actions_are_rejected(item_type: str) -> None:
    data = _trajectory_data()
    data["steps"][-1]["extra"] = {"llm_response": {"output": [{"type": item_type}]}}

    with pytest.raises(AtifProjectionError, match="unsupported Responses output item"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda data: data.update(schema_version="ATIF-v1.6"), "unsupported ATIF schema version"),
        (lambda data: data.update(continued_trajectory_ref="next.json"), "continued ATIF trajectories"),
        (
            lambda data: data.update(
                subagent_trajectories=[
                    {
                        "schema_version": "ATIF-v1.7",
                        "trajectory_id": "child-1",
                        "agent": {"name": "child", "version": "1"},
                        "steps": [],
                    }
                ]
            ),
            "embedded subagent trajectories",
        ),
        (lambda data: data["steps"][1].update(is_copied_context=True), "copied continuation context"),
        (
            lambda data: data["steps"][1].update(
                message=[
                    {
                        "type": "image",
                        "source": {"media_type": "image/png", "path": "fixture.png"},
                    }
                ]
            ),
            "multimodal ATIF content",
        ),
        (lambda data: data["steps"][1].update(timestamp="2026-08-24T12:00:00"), "timestamp has no timezone"),
    ],
)
def test_strict_initial_scope_rejects_unsupported_trajectories(mutate: Any, match: str) -> None:
    data = _trajectory_data()
    mutate(data)
    trajectory = AtifTrajectoryV1_7.model_validate(data)

    with pytest.raises(AtifProjectionError, match=match):
        atif_trajectory_to_response(trajectory)


def test_tool_result_without_content_or_relay_extension_is_rejected() -> None:
    data = _trajectory_data()
    data["steps"][1]["observation"]["results"][0] = {"source_call_id": "call-b"}
    trajectory = AtifTrajectoryV1_7.model_validate(data)

    with pytest.raises(AtifProjectionError, match="neither content nor Relay extra.tool_result"):
        atif_trajectory_to_response(trajectory)


def test_empty_standard_tool_result_content_is_rejected_as_ambiguous() -> None:
    data = _trajectory_data()
    data["steps"][1]["observation"]["results"][0] = {
        "source_call_id": "call-b",
        "content": [],
    }

    with pytest.raises(AtifProjectionError, match="empty content-part list"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


def test_relay_empty_array_tool_result_is_preserved_from_the_structured_extension() -> None:
    data = _trajectory_data()
    data["steps"][1]["observation"]["results"][0] = {
        "source_call_id": "call-b",
        "extra": {"tool_result": []},
    }

    response = atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))
    outputs = {item.call_id: item.output for item in response.output if item.type == "function_call_output"}

    assert outputs["call-b"] == "[]"


@pytest.mark.parametrize(("trajectory_id", "session_id"), [(None, "run-1"), ("", ""), ("  ", "\t")])
def test_missing_or_empty_atif_ids_use_a_deterministic_content_identity(
    trajectory_id: str | None,
    session_id: str | None,
) -> None:
    data = _trajectory_data()
    data["trajectory_id"] = trajectory_id
    data["session_id"] = session_id
    trajectory = AtifTrajectoryV1_7.model_validate(data)

    first = atif_trajectory_to_response(trajectory)
    second = atif_trajectory_to_response(trajectory)

    assert first.id == second.id
    assert [item.id for item in first.output] == [item.id for item in second.output]


def test_interactive_input_after_agent_output_is_rejected_instead_of_dropped() -> None:
    data = _trajectory_data()
    data["steps"].append({"step_id": 4, "source": "user", "message": "Try a different answer."})
    data["steps"].append({"step_id": 5, "source": "agent", "message": "The answer is 5."})
    trajectory = AtifTrajectoryV1_7.model_validate(data)

    with pytest.raises(AtifProjectionError, match="non-agent step 4 appears after agent output"):
        atif_trajectory_to_response(trajectory)


def test_non_agent_observation_is_rejected_instead_of_dropped() -> None:
    data = _trajectory_data()
    data["steps"][0]["observation"] = {"results": [{"content": "external event"}]}
    trajectory = AtifTrajectoryV1_7.model_validate(data)

    with pytest.raises(AtifProjectionError, match="non-agent step 1 contains an observation"):
        atif_trajectory_to_response(trajectory)


def test_external_subagent_reference_is_rejected_without_being_dropped() -> None:
    data = _trajectory_data()
    data["steps"][1]["observation"]["results"][0]["subagent_trajectory_ref"] = [
        {"trajectory_path": "child-trajectory.json"}
    ]
    trajectory = AtifTrajectoryV1_7.model_validate(data)

    with pytest.raises(AtifProjectionError, match="references a subagent trajectory"):
        atif_trajectory_to_response(trajectory)


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ({"invocation": {"status": "failed"}}, "non-completed Relay invocation status"),
        ({"invocation": {"status": "in_progress"}}, "non-completed Relay invocation status"),
        ({"tool_invocations": [{"status": "timeout"}]}, "tool invocation 0 has non-completed status"),
        ({"llm_response": {"status": "incomplete"}}, "provider response status"),
        ({"llm_response": {"status": "queued"}}, "provider response status"),
        ({"llm_response": {"choices": [{"finish_reason": "length"}]}}, "non-terminal chat finish_reason"),
        ({"llm_response": {"stop_reason": "max_tokens"}}, "non-terminal Anthropic stop_reason"),
    ],
)
def test_known_failed_or_incomplete_outcomes_are_not_relabelled_completed(
    extra: dict[str, Any],
    match: str,
) -> None:
    data = _trajectory_data()
    data["steps"][-1]["extra"] = extra
    trajectory = AtifTrajectoryV1_7.model_validate(data)

    with pytest.raises(AtifProjectionError, match=match):
        atif_trajectory_to_response(trajectory)


def test_unknown_invocation_status_is_not_relabelled_completed() -> None:
    data = _trajectory_data()
    data["steps"][-1]["extra"] = {"invocation": {"status": "success"}}

    with pytest.raises(AtifProjectionError, match="non-completed Relay invocation status"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ({"invocation": None}, "Relay invocation metadata is not an object"),
        ({"invocation": "failed"}, "Relay invocation metadata is not an object"),
        ({"tool_invocations": None}, "Relay tool_invocations metadata is not a list"),
        ({"tool_invocations": {"status": "failed"}}, "Relay tool_invocations metadata is not a list"),
        ({"tool_invocations": [None]}, "Relay tool invocation 0 metadata is not an object"),
    ],
)
def test_malformed_relay_status_containers_are_not_silently_ignored(
    extra: dict[str, Any],
    match: str,
) -> None:
    data = _trajectory_data()
    data["steps"][-1]["extra"] = extra

    with pytest.raises(AtifProjectionError, match=match):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


def test_absent_optional_relay_status_metadata_remains_supported() -> None:
    data = _trajectory_data()
    data["steps"][-1]["extra"] = {}

    response = atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))

    assert response.status == "completed"


@pytest.mark.parametrize("status", ["failed", "timeout", "cancelled", "incomplete", True])
def test_tool_call_extra_status_is_not_relabelled_completed(status: Any) -> None:
    data = _trajectory_data()
    data["steps"][1]["tool_calls"][0]["extra"] = {"status": status}

    with pytest.raises(AtifProjectionError, match="tool call 0 has non-completed status"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


def test_parser_rejects_negative_total_steps() -> None:
    data = _trajectory_data()
    data["final_metrics"] = {"total_steps": -1}

    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        AtifTrajectoryV1_7.model_validate(data)


def test_message_and_tool_calls_in_one_step_are_rejected_without_an_ordering_claim() -> None:
    data = _trajectory_data()
    data["steps"][1]["message"] = "I will call both tools."

    with pytest.raises(AtifProjectionError, match="both message text and tool calls"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


@pytest.mark.parametrize(
    ("step_index", "item_index", "status"),
    [(1, 0, "incomplete"), (1, 1, "in_progress"), (2, 1, "incomplete")],
)
def test_noncompleted_responses_output_items_are_not_relabelled_completed(
    step_index: int,
    item_index: int,
    status: str,
) -> None:
    data = json.loads(_RESPONSES_FIXTURE_PATH.read_text())
    data["steps"][step_index]["extra"]["llm_response"]["output"][item_index]["status"] = status

    with pytest.raises(AtifProjectionError, match="has non-completed status"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


def test_canonical_and_raw_responses_reasoning_are_not_silently_collapsed() -> None:
    data = json.loads(_RESPONSES_FIXTURE_PATH.read_text())
    data["steps"][1]["reasoning_content"] = "A different canonical explanation."

    with pytest.raises(AtifProjectionError, match="both canonical reasoning_content and raw Responses reasoning"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


def test_responses_output_requires_a_completed_provider_status() -> None:
    data = json.loads(_RESPONSES_FIXTURE_PATH.read_text())
    del data["steps"][-1]["extra"]["llm_response"]["status"]

    with pytest.raises(AtifProjectionError, match="provider status is not completed"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


def test_responses_reasoning_after_output_is_rejected_instead_of_reordered() -> None:
    data = json.loads(_RESPONSES_FIXTURE_PATH.read_text())
    data["steps"][1]["extra"]["llm_response"]["output"].reverse()

    with pytest.raises(AtifProjectionError, match="reasoning item.*appears after output"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


@pytest.mark.parametrize("provider", ["responses", "chat", "anthropic"])
def test_raw_provider_messages_must_have_the_assistant_role(provider: str) -> None:
    if provider == "responses":
        data = json.loads(_RESPONSES_FIXTURE_PATH.read_text())
        data["steps"][-1]["extra"]["llm_response"]["output"][1]["role"] = "user"
    elif provider == "chat":
        data = json.loads(_FIXTURE_PATH.read_text())
        data["steps"][-1]["extra"]["llm_response"]["choices"][0]["message"]["role"] = "user"
    else:
        data = _trajectory_data()
        data["steps"][-1]["extra"] = {
            "llm_response": {
                "type": "message",
                "role": "user",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": data["steps"][-1]["message"]}],
            }
        }

    with pytest.raises(AtifProjectionError, match="non-assistant role"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


def test_chat_response_requires_a_terminal_finish_reason() -> None:
    data = json.loads(_FIXTURE_PATH.read_text())
    data["steps"][-1]["extra"]["llm_response"]["choices"][0]["finish_reason"] = None

    with pytest.raises(AtifProjectionError, match="non-terminal chat finish_reason"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


@pytest.mark.parametrize("stop_reason", [None, "max_tokens", "pause_turn"])
def test_anthropic_response_requires_a_terminal_stop_reason(stop_reason: str | None) -> None:
    data = _trajectory_data()
    data["steps"][-1]["extra"] = {
        "llm_response": {
            "type": "message",
            "role": "assistant",
            "stop_reason": stop_reason,
            "content": [{"type": "text", "text": data["steps"][-1]["message"]}],
        }
    }

    with pytest.raises(AtifProjectionError, match="non-terminal Anthropic stop_reason"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


def test_raw_text_part_boundaries_must_match_canonical_atif() -> None:
    data = _trajectory_data()
    data["steps"][-1]["message"] = [
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b\nc"},
    ]
    data["steps"][-1]["extra"] = {
        "llm_response": {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "a\nb"},
                        {"type": "output_text", "text": "c"},
                    ],
                }
            ],
        }
    }

    with pytest.raises(AtifProjectionError, match="raw provider message does not match"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (
            "annotations",
            [
                {
                    "type": "url_citation",
                    "url": "https://example.com",
                    "title": "source",
                    "start_index": 0,
                    "end_index": 1,
                }
            ],
        ),
        ("logprobs", [{"token": "B", "logprob": -0.1, "bytes": [66]}]),
    ],
)
def test_unprojected_responses_text_metadata_is_rejected(field: str, value: list[dict[str, Any]]) -> None:
    data = json.loads(_RESPONSES_FIXTURE_PATH.read_text())
    data["steps"][-1]["extra"]["llm_response"]["output"][1]["content"][0][field] = value

    with pytest.raises(AtifProjectionError, match=f"non-empty {field}"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


def test_unprojected_chat_message_annotations_are_rejected() -> None:
    data = json.loads(_FIXTURE_PATH.read_text())
    data["steps"][-1]["extra"]["llm_response"]["choices"][0]["message"]["annotations"] = [
        {"type": "url_citation", "url": "https://example.com"}
    ]

    with pytest.raises(AtifProjectionError, match="chat message contains non-empty annotations"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


def test_unprojected_chat_choice_logprobs_are_rejected() -> None:
    data = json.loads(_FIXTURE_PATH.read_text())
    data["steps"][-1]["extra"]["llm_response"]["choices"][0]["logprobs"] = {
        "content": [{"token": "It", "logprob": -0.1}]
    }

    with pytest.raises(AtifProjectionError, match="chat choice contains non-empty logprobs"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


def test_unprojected_anthropic_text_citations_are_rejected() -> None:
    data = _trajectory_data()
    data["steps"][-1]["extra"] = {
        "llm_response": {
            "type": "message",
            "role": "assistant",
            "stop_reason": "end_turn",
            "content": [
                {
                    "type": "text",
                    "text": data["steps"][-1]["message"],
                    "citations": [{"type": "web_search_result_location", "url": "https://example.com"}],
                }
            ],
        }
    }

    with pytest.raises(AtifProjectionError, match="Anthropic text block 0 contains non-empty citations"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


@pytest.mark.parametrize("item_index", [0, 1])
def test_unknown_responses_output_fields_are_rejected_even_when_null(item_index: int) -> None:
    data = json.loads(_RESPONSES_FIXTURE_PATH.read_text())
    data["steps"][1]["extra"]["llm_response"]["output"][item_index]["future_semantics"] = None

    with pytest.raises(AtifProjectionError, match="contains unsupported fields.*future_semantics"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


def test_unknown_responses_content_part_fields_are_rejected() -> None:
    data = json.loads(_RESPONSES_FIXTURE_PATH.read_text())
    data["steps"][-1]["extra"]["llm_response"]["output"][1]["content"][0]["future_semantics"] = "x"

    with pytest.raises(AtifProjectionError, match="contains unsupported fields.*future_semantics"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


def test_unknown_chat_message_fields_are_rejected_even_when_null() -> None:
    data = json.loads(_FIXTURE_PATH.read_text())
    data["steps"][-1]["extra"]["llm_response"]["choices"][0]["message"]["future_semantics"] = None

    with pytest.raises(AtifProjectionError, match="contains unsupported fields.*future_semantics"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


def test_unknown_chat_content_part_fields_are_rejected() -> None:
    data = json.loads(_FIXTURE_PATH.read_text())
    message = data["steps"][-1]["extra"]["llm_response"]["choices"][0]["message"]
    message["content"] = [{"type": "text", "text": message["content"], "future_semantics": "x"}]

    with pytest.raises(AtifProjectionError, match="contains unsupported fields.*future_semantics"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


@pytest.mark.parametrize("choice_index", [True, 0.0, "0", 1])
def test_chat_choice_index_must_be_integer_zero(choice_index: Any) -> None:
    data = json.loads(_FIXTURE_PATH.read_text())
    data["steps"][-1]["extra"]["llm_response"]["choices"][0]["index"] = choice_index

    with pytest.raises(AtifProjectionError, match="chat choice has inconsistent index"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


def test_matching_chat_reasoning_is_preserved() -> None:
    data = json.loads(_FIXTURE_PATH.read_text())
    data["steps"][-1]["reasoning_content"] = "Check the tool result."
    data["steps"][-1]["extra"]["llm_response"]["choices"][0]["message"]["reasoning_content"] = "Check the tool result."

    response = atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))

    assert response.output[-2].type == "reasoning"


def test_mismatched_chat_reasoning_is_rejected() -> None:
    data = json.loads(_FIXTURE_PATH.read_text())
    data["steps"][-1]["reasoning_content"] = "Canonical reasoning."
    data["steps"][-1]["extra"]["llm_response"]["choices"][0]["message"]["reasoning_content"] = "Raw reasoning."

    with pytest.raises(AtifProjectionError, match="raw chat reasoning does not match"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


def test_conflicting_chat_reasoning_aliases_are_rejected() -> None:
    data = json.loads(_FIXTURE_PATH.read_text())
    message = data["steps"][-1]["extra"]["llm_response"]["choices"][0]["message"]
    message["reasoning_content"] = "first"
    message["reasoning"] = "second"

    with pytest.raises(AtifProjectionError, match="chat reasoning aliases conflict"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


def test_unknown_anthropic_text_fields_are_rejected_even_when_null() -> None:
    data = _trajectory_data()
    data["steps"][-1]["extra"] = {
        "llm_response": {
            "type": "message",
            "role": "assistant",
            "stop_reason": "end_turn",
            "content": [
                {
                    "type": "text",
                    "text": data["steps"][-1]["message"],
                    "future_semantics": None,
                }
            ],
        }
    }

    with pytest.raises(AtifProjectionError, match="contains unsupported fields.*future_semantics"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


def test_unknown_anthropic_tool_use_fields_are_rejected() -> None:
    data = _trajectory_data()
    step = data["steps"][1]
    step["extra"] = {
        "llm_response": {
            "type": "message",
            "role": "assistant",
            "stop_reason": "tool_use",
            "content": [
                {"type": "tool_use", "id": "call-a", "name": "lookup", "input": {"q": "x"}},
                {
                    "type": "tool_use",
                    "id": "call-b",
                    "name": "calculate",
                    "input": {"x": 2},
                    "future_semantics": "x",
                },
            ],
        }
    }

    with pytest.raises(AtifProjectionError, match="contains unsupported fields.*future_semantics"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


def test_anthropic_direct_tool_callers_are_supported() -> None:
    data = _trajectory_data()
    step = data["steps"][1]
    step["extra"] = {
        "llm_response": {
            "type": "message",
            "role": "assistant",
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call-a",
                    "name": "lookup",
                    "input": {"q": "x"},
                    "caller": {"type": "direct"},
                },
                {
                    "type": "tool_use",
                    "id": "call-b",
                    "name": "calculate",
                    "input": {"x": 2},
                    "caller": {"type": "direct"},
                },
            ],
        }
    }

    response = atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))

    assert [item.call_id for item in response.output if item.type == "function_call"] == ["call-a", "call-b"]


@pytest.mark.parametrize("caller_type", ["code_execution_20250825", "code_execution_20260120"])
def test_anthropic_server_tool_callers_are_rejected(caller_type: str) -> None:
    data = _trajectory_data()
    step = data["steps"][1]
    step["extra"] = {
        "llm_response": {
            "type": "message",
            "role": "assistant",
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call-a",
                    "name": "lookup",
                    "input": {"q": "x"},
                    "caller": {"type": caller_type, "tool_id": "srvtoolu_1"},
                },
                {"type": "tool_use", "id": "call-b", "name": "calculate", "input": {"x": 2}},
            ],
        }
    }

    with pytest.raises(AtifProjectionError, match="unsupported server-tool caller"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


@pytest.mark.parametrize("raw_id", [" ", "duplicate"])
def test_raw_responses_reasoning_ids_must_be_nonblank_and_unique(raw_id: str) -> None:
    data = json.loads(_RESPONSES_FIXTURE_PATH.read_text())
    first_reasoning = data["steps"][1]["extra"]["llm_response"]["output"][0]
    first_reasoning["id"] = raw_id
    if raw_id == "duplicate":
        duplicate = deepcopy(first_reasoning)
        duplicate["content"] = [{"text": "A second reasoning item."}]
        data["steps"][1]["extra"]["llm_response"]["output"].insert(1, duplicate)

    match = "blank or invalid id" if not raw_id.strip() else "repeats item id"
    with pytest.raises(AtifProjectionError, match=match):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


@pytest.mark.parametrize(("has_tool_use", "stop_reason"), [(True, "end_turn"), (False, "tool_use")])
def test_anthropic_stop_reason_must_match_tool_use_content(has_tool_use: bool, stop_reason: str) -> None:
    data = _trajectory_data()
    step = data["steps"][1] if has_tool_use else data["steps"][-1]
    if has_tool_use:
        step["tool_calls"] = [{"tool_call_id": "call-b", "function_name": "calculate", "arguments": {"x": 2}}]
        step["observation"] = {"results": [{"source_call_id": "call-b", "content": "4"}]}
        content = [{"type": "tool_use", "id": "call-b", "name": "calculate", "input": {"x": 2}}]
    else:
        content = [{"type": "text", "text": step["message"]}]
    step["extra"] = {
        "llm_response": {
            "type": "message",
            "role": "assistant",
            "stop_reason": stop_reason,
            "content": content,
        }
    }

    with pytest.raises(AtifProjectionError, match="Anthropic stop_reason.*inconsistent"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


def test_observation_cannot_carry_both_standard_content_and_relay_tool_result() -> None:
    data = _trajectory_data()
    result = data["steps"][1]["observation"]["results"][0]
    result["extra"] = {"tool_result": {"value": "different"}}

    with pytest.raises(AtifProjectionError, match="contains both content and Relay extra.tool_result"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


def test_duplicate_tool_call_id_across_steps_is_rejected() -> None:
    data = _trajectory_data()
    data["steps"].append(
        {
            "step_id": 4,
            "source": "agent",
            "message": "",
            "tool_calls": [{"tool_call_id": "call-a", "function_name": "lookup", "arguments": {"q": "again"}}],
        }
    )
    trajectory = AtifTrajectoryV1_7.model_validate(data)

    with pytest.raises(AtifProjectionError, match="repeats tool_call_id.*across steps"):
        atif_trajectory_to_response(trajectory)


def test_multiple_agent_model_names_are_rejected_instead_of_collapsed() -> None:
    data = _trajectory_data()
    data["steps"][1]["model_name"] = "model-a"
    data["steps"][2]["model_name"] = "model-b"

    with pytest.raises(AtifProjectionError, match="multiple model names"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


def test_root_model_default_and_step_override_are_rejected_instead_of_collapsed() -> None:
    data = _trajectory_data()
    data["steps"][2]["model_name"] = "model-b"

    with pytest.raises(AtifProjectionError, match="multiple model names"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


def test_known_and_unknown_agent_model_names_are_rejected_instead_of_collapsed() -> None:
    data = _trajectory_data()
    data["agent"]["model_name"] = None
    data["steps"][1]["model_name"] = "model-a"

    with pytest.raises(AtifProjectionError, match="known and unknown model identity"):
        atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))


def test_uniform_step_model_names_are_preserved_without_a_root_default() -> None:
    data = _trajectory_data()
    data["agent"]["model_name"] = None
    data["steps"][1]["model_name"] = "model-a"
    data["steps"][2]["model_name"] = "model-a"

    response = atif_trajectory_to_response(AtifTrajectoryV1_7.model_validate(data))

    assert response.model == "model-a"


def test_multiple_outputs_for_one_tool_call_are_rejected() -> None:
    data = _trajectory_data()
    data["steps"][1]["observation"]["results"].append({"source_call_id": "call-b", "content": "second result"})
    trajectory = AtifTrajectoryV1_7.model_validate(data)

    with pytest.raises(AtifProjectionError, match="multiple outputs for tool call"):
        atif_trajectory_to_response(trajectory)


def test_materialized_task_must_supply_responses_create_params() -> None:
    trajectory = AtifTrajectoryV1_7.model_validate(_trajectory_data())

    with pytest.raises(AtifProjectionError, match="responses_create_params"):
        build_atif_verify_payload({"task_index": 1}, trajectory)
