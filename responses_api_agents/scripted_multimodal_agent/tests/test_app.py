# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Mapping
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request, Response

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from nemo_gym.visual_history import (
    CompactionScheduleConfig,
    ContextGuardConfig,
    HistoryPolicyConfig,
    RecencyHistoryPolicyConfig,
    VisualHistoryConfig,
    normalize_semantic_items,
)
from responses_api_agents.scripted_multimodal_agent.app import (
    ScriptedMultimodalAgent,
    ScriptedMultimodalAgentConfig,
    ScriptedMultimodalAgentRunRequest,
    _has_materializable_assistant_output,
    scripted_observations,
)


def _mock_http_response(payload: dict, cookies: dict | None = None) -> MagicMock:
    response = MagicMock()
    response.ok = True
    response.cookies = cookies or {}
    response.read = AsyncMock(return_value=json.dumps(payload).encode())
    return response


def _model_response(
    turn: int,
    *,
    prompt_token_ids: list[int] | None = None,
    generation_token_id: int | None = None,
) -> dict:
    if prompt_token_ids is None:
        prompt_token_ids = {
            0: [10],
            1: [10, 11, 12],
            2: [10, 11, 12, 13, 14],
            3: [20],
            4: [20, 21, 22],
        }[turn]
    if generation_token_id is None:
        generation_token_id = {0: 11, 1: 13, 2: 15, 3: 21, 4: 23}[turn]
    return {
        "id": f"resp-{turn}",
        "created_at": 1.0,
        "model": "dummy-model",
        "object": "response",
        "output": [
            {
                "id": f"msg-{turn}",
                "content": [
                    {
                        "annotations": [],
                        "text": f"assistant turn {turn}",
                        "type": "output_text",
                    }
                ],
                "role": "assistant",
                "status": "completed",
                "type": "message",
                "prompt_token_ids": prompt_token_ids,
                "generation_token_ids": [generation_token_id],
                "generation_log_probs": [-0.1 - turn],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def _config() -> ScriptedMultimodalAgentConfig:
    return ScriptedMultimodalAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="scripted_multimodal_agent",
        model_server=ModelServerRef(type="responses_api_models", name="model"),
        fixture="media_contract",
    )


def _active_recency_config() -> ScriptedMultimodalAgentConfig:
    config = _config()
    config.visual_history = VisualHistoryConfig(
        enabled=True,
        shadow_only=False,
        policy=HistoryPolicyConfig(
            type="recency",
            config=RecencyHistoryPolicyConfig(keep_last_image_groups=1),
        ),
    )
    return config


def _chunked_recency_config(actions_per_chunk: int) -> ScriptedMultimodalAgentConfig:
    config = _active_recency_config()
    config.visual_history.schedule = CompactionScheduleConfig(
        type="turn_chunked_recency",
        actions_per_chunk=actions_per_chunk,
    )
    return config


def _identity_shadow_config() -> ScriptedMultimodalAgentConfig:
    config = _config()
    config.visual_history = VisualHistoryConfig(
        enabled=True,
        shadow_only=True,
        policy=HistoryPolicyConfig(type="identity"),
        schedule=CompactionScheduleConfig(
            type="rolling_recency",
            actions_per_chunk=1,
        ),
    )
    return config


def _identity_authority_config() -> ScriptedMultimodalAgentConfig:
    config = _identity_shadow_config()
    config.visual_history.shadow_only = False
    return config


def _item_dict(item: Any) -> Mapping[str, Any]:
    if hasattr(item, "model_dump"):
        return item.model_dump()
    assert isinstance(item, Mapping)
    return item


def _request_image_urls(request_input: list[Any]) -> list[str]:
    return [
        part["image_url"]
        for item in request_input
        for part in _item_dict(item).get("content", [])
        if isinstance(part, Mapping) and part.get("type") == "input_image"
    ]


def _request_text_parts(request_input: list[Any]) -> list[str]:
    return [
        part["text"]
        for item in request_input
        for part in _item_dict(item).get("content", [])
        if isinstance(part, Mapping) and part.get("type") == "input_text"
    ]


def _image_urls(message) -> list[str]:
    return [part["image_url"] for part in message.model_dump()["content"] if part["type"] == "input_image"]


def _prefix_consistent_model_side_effect(num_turns: int):
    turn = 0

    def respond(*, url_path: str, json: Any, **_: Any) -> MagicMock:
        nonlocal turn
        assert url_path == "/v1/responses"
        assert turn < num_turns
        required_prefix = list(json.required_prefix_token_ids or [])
        response = _mock_http_response(
            _model_response(
                turn,
                prompt_token_ids=[*required_prefix, 1000 + turn],
                generation_token_id=2000 + turn,
            )
        )
        turn += 1
        return response

    return respond


class TestApp:
    def test_sanity(self) -> None:
        ScriptedMultimodalAgent(config=_config(), server_client=MagicMock(spec=ServerClient))

    def test_materializable_output_rejects_empty_assistant_shell(self) -> None:
        empty = _model_response(0)["output"]
        empty[0]["content"] = []

        assert not _has_materializable_assistant_output(empty)
        assert _has_materializable_assistant_output(_model_response(0)["output"])

    async def test_empty_assistant_shell_is_retried_without_entering_history(self) -> None:
        config = _config()
        config.num_turns = 1
        config.empty_response_retries = 1
        server = ScriptedMultimodalAgent(
            config=config,
            server_client=MagicMock(spec=ServerClient),
        )
        empty_response = _model_response(0)
        empty_response["output"][0]["content"] = []
        server.server_client.post.side_effect = [
            _mock_http_response(empty_response),
            _mock_http_response(_model_response(0)),
        ]
        request = MagicMock(spec=Request)
        request.cookies = {"session": "empty-retry"}

        result = await server.responses(
            request=request,
            response=Response(),
            body=NeMoGymResponseCreateParamsNonStreaming(input="initial text"),
        )

        assert server.server_client.post.call_count == 2
        assert len(result.output) == 1
        assert _item_dict(result.output[0])["content"][0]["text"] == "assistant turn 0"

    def test_computer_use_fixture_is_simple_and_deterministic(self) -> None:
        first = scripted_observations(fixture="computer_use", num_turns=8)
        second = scripted_observations(fixture="computer_use", num_turns=8)

        first_images = [_image_urls(message) for message in first]
        assert first_images == [_image_urls(message) for message in second]
        assert all(len(images) == 1 for images in first_images)
        assert len({images[0] for images in first_images}) == 8

    def test_text_padding_mode_preserves_media_length_without_images(self) -> None:
        observations = scripted_observations("text_padding")

        assert [_image_urls(message) for message in observations] == [[], [], [], [], []]
        text = [
            next(part["text"] for part in message.model_dump()["content"] if part["type"] == "input_text")
            for message in observations
        ]
        assert [value.count(" pad") for value in text] == [256, 256, 512, 0, 256]

    async def test_identity_shadow_matches_legacy_inputs_and_prompt_ids(self) -> None:
        responses_body = NeMoGymResponseCreateParamsNonStreaming(input="initial text")
        observed_inputs = []
        observed_prompt_ids = []
        for config in (
            _config(),
            _identity_shadow_config(),
            _identity_authority_config(),
        ):
            server = ScriptedMultimodalAgent(
                config=config,
                server_client=MagicMock(spec=ServerClient),
            )
            model_responses = []
            prior_context = []
            for turn in range(config.num_turns):
                prompt_token_ids = [*prior_context, 1000 + turn]
                generation_token_id = 2000 + turn
                model_responses.append(
                    _mock_http_response(
                        _model_response(
                            turn,
                            prompt_token_ids=prompt_token_ids,
                            generation_token_id=generation_token_id,
                        )
                    )
                )
                prior_context = [*prompt_token_ids, generation_token_id]
            server.server_client.post.side_effect = model_responses
            request = MagicMock(spec=Request)
            request.cookies = {"session": "identity-ab"}
            result = await server.responses(
                request=request,
                response=Response(),
                body=responses_body,
            )
            sent_inputs = [call.kwargs["json"].input for call in server.server_client.post.call_args_list]
            assert all(
                call.kwargs["json"]
                .model_dump(exclude_unset=True)
                .get(
                    "metadata",
                    {},
                )
                is not None
                for call in server.server_client.post.call_args_list
            )
            assert all(hasattr(item, "model_dump") for request_input in sent_inputs for item in request_input)
            observed_inputs.append([normalize_semantic_items(request_input) for request_input in sent_inputs])
            if result.completion_evidence:
                prompt_ids = [evidence.prompt_token_ids for evidence in result.completion_evidence]
            else:
                prompt_ids = [
                    tuple(_item_dict(item)["prompt_token_ids"])
                    for item in result.output
                    if "prompt_token_ids" in _item_dict(item)
                ]
            observed_prompt_ids.append(prompt_ids)

        assert observed_inputs[0] == observed_inputs[1]
        assert observed_inputs[0] == observed_inputs[2]
        assert observed_prompt_ids[0] == observed_prompt_ids[1]
        assert observed_prompt_ids[0] == observed_prompt_ids[2]

    async def test_responses_builds_exact_deterministic_visual_history(self) -> None:
        server = ScriptedMultimodalAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
        server.server_client.post.side_effect = [
            _mock_http_response(_model_response(turn), {"turn": str(turn)}) for turn in range(5)
        ]
        request = MagicMock(spec=Request)
        request.cookies = {"session": "test"}

        result = await server.responses(
            request=request,
            response=Response(),
            body=NeMoGymResponseCreateParamsNonStreaming(input="initial text"),
        )

        assert len(result.agent_input) == 1
        assert len(result.seed_obs) == 1
        assert len(result.output) == 9
        assert result.completion_evidence == []
        observations = [*result.seed_obs, *result.output[1::2]]
        image_urls = [_image_urls(message) for message in observations]
        assert [len(urls) for urls in image_urls] == [1, 1, 2, 0, 1]
        assert image_urls[0][0] == image_urls[4][0]
        assert image_urls[2][0] != image_urls[2][1]

        cumulative_input_lengths = [
            len(call.kwargs["json"].input) for call in server.server_client.post.call_args_list
        ]
        assert cumulative_input_lengths == [2, 4, 6, 8, 10]

    async def test_active_recency_rewrites_only_between_model_turns(self) -> None:
        server = ScriptedMultimodalAgent(
            config=_active_recency_config(),
            server_client=MagicMock(spec=ServerClient),
        )
        server.server_client.post.side_effect = [_mock_http_response(_model_response(turn)) for turn in range(5)]
        request = MagicMock(spec=Request)
        request.cookies = {"session": "recency-test"}

        result = await server.responses(
            request=request,
            response=Response(),
            body=NeMoGymResponseCreateParamsNonStreaming(input="initial text"),
        )

        request_inputs = [call.kwargs["json"].input for call in server.server_client.post.call_args_list]
        image_urls = [_request_image_urls(items) for items in request_inputs]
        original_images = [_image_urls(observation) for observation in scripted_observations()]
        image_a = original_images[0][0]
        image_b = original_images[1][0]
        image_c, image_d = original_images[2]

        assert image_urls == [
            [image_a],
            [image_a, image_b],
            [image_b, image_c, image_d],
            [image_c, image_d],
            [image_c, image_d, image_a],
        ]
        marker_counts = [_request_text_parts(items).count("[Earlier image omitted]") for items in request_inputs]
        assert marker_counts == [0, 0, 1, 1, 1]
        required_prefixes = [
            call.kwargs["json"].required_prefix_token_ids for call in server.server_client.post.call_args_list
        ]
        assert required_prefixes == [
            None,
            [10, 11],
            None,
            None,
            [20, 21],
        ]

        assert [list(evidence.prompt_token_ids) for evidence in result.completion_evidence] == [
            [10],
            [10, 11, 12],
            [10, 11, 12, 13, 14],
            [20],
            [20, 21, 22],
        ]
        assert [list(evidence.sampled_token_ids) for evidence in result.completion_evidence] == [
            [11],
            [13],
            [15],
            [21],
            [23],
        ]
        assert [len(evidence.media_ids) for evidence in result.completion_evidence] == [1, 2, 3, 2, 3]
        assert [evidence.context_epoch for evidence in result.completion_evidence] == [0, 0, 1, 2, 2]
        assert [evidence.expected_append_compatible for evidence in result.completion_evidence] == [
            False,
            True,
            False,
            False,
            True,
        ]
        assert [evidence.compaction_event_id for evidence in result.completion_evidence] == [
            None,
            None,
            result.boundary_events[0].event_id,
            result.boundary_events[1].event_id,
            None,
        ]
        assert all(len(evidence.policy_output_spans) == 1 for evidence in result.completion_evidence)
        assert len({evidence.completion_id for evidence in result.completion_evidence}) == 5
        assert all(evidence.evidence_source == "generation_response" for evidence in result.completion_evidence)
        referenced_media = {media_id for evidence in result.completion_evidence for media_id in evidence.media_ids}
        assert set(result.media_assets) == referenced_media
        assert len(result.media_assets) == 4
        assert all(
            asset["original_dimensions"] == (32, 32)
            and asset["color_mode"] == "RGB"
            and asset["source_format"] == "png"
            and asset["source_part"]["type"] == "input_image"
            for asset in result.media_assets.values()
        )
        assert [event.applies_to_step for event in result.boundary_events] == [
            3,
            4,
        ]

        # Compaction changes only the model-facing views. The verifier-facing
        # trajectory remains the complete deterministic interaction.
        observations = [*result.seed_obs, *result.output[1::2]]
        assert [_image_urls(message) for message in observations] == original_images

    async def test_k2_and_k4_freeze_history_inside_each_chunk(self) -> None:
        chunked_prompts = [
            [10],
            [10, 11, 12],
            [10, 11, 12, 13, 14],
            [10, 11, 12, 13, 14, 15, 16],
            [30],
        ]
        chunked_generations = [11, 13, 15, 17, 31]
        original_images = [_image_urls(observation) for observation in scripted_observations()]
        image_a = original_images[0][0]
        image_b = original_images[1][0]
        image_c, image_d = original_images[2]

        for actions_per_chunk, expected_chunk_sizes in (
            (2, [2, 2, 1]),
            (4, [4, 1]),
        ):
            server = ScriptedMultimodalAgent(
                config=_chunked_recency_config(actions_per_chunk),
                server_client=MagicMock(spec=ServerClient),
            )
            server.server_client.post.side_effect = [
                _mock_http_response(
                    _model_response(
                        turn,
                        prompt_token_ids=chunked_prompts[turn],
                        generation_token_id=chunked_generations[turn],
                    )
                )
                for turn in range(5)
            ]
            request = MagicMock(spec=Request)
            request.cookies = {"session": f"chunk-k{actions_per_chunk}"}

            result = await server.responses(
                request=request,
                response=Response(),
                body=NeMoGymResponseCreateParamsNonStreaming(input="initial text"),
            )

            request_inputs = [call.kwargs["json"].input for call in server.server_client.post.call_args_list]
            expected_images = (
                [
                    [image_a],
                    [image_a, image_b],
                    [image_b, image_c, image_d],
                    [image_b, image_c, image_d],
                    [image_c, image_d, image_a],
                ]
                if actions_per_chunk == 2
                else [
                    [image_a],
                    [image_a, image_b],
                    [image_a, image_b, image_c, image_d],
                    [image_a, image_b, image_c, image_d],
                    [image_c, image_d, image_a],
                ]
            )
            assert [_request_image_urls(items) for items in request_inputs] == expected_images
            required_prefixes = [
                call.kwargs["json"].required_prefix_token_ids for call in server.server_client.post.call_args_list
            ]
            assert required_prefixes == (
                [
                    None,
                    [10, 11],
                    None,
                    [10, 11, 12, 13, 14, 15],
                    None,
                ]
                if actions_per_chunk == 2
                else [
                    None,
                    [10, 11],
                    [10, 11, 12, 13],
                    [10, 11, 12, 13, 14, 15],
                    None,
                ]
            )
            assert [record.actual_action_count for record in result.chunk_records] == expected_chunk_sizes
            assert result.chunk_records[-1].early_close_reason == "terminal"
            assert [event.applies_to_step for event in result.boundary_events] == (
                [3, 5] if actions_per_chunk == 2 else [5]
            )
            assert all(event.schedule_name == "turn_chunked_recency" for event in result.boundary_events)

    async def test_short_and_long_trajectories_finalize_each_action_once(
        self,
    ) -> None:
        for num_turns, expected_chunk_sizes in (
            (1, [1]),
            (3, [2, 1]),
            (10, [2, 2, 2, 2, 2]),
        ):
            config = _chunked_recency_config(actions_per_chunk=2)
            config.num_turns = num_turns
            server = ScriptedMultimodalAgent(
                config=config,
                server_client=MagicMock(spec=ServerClient),
            )
            server.server_client.post.side_effect = _prefix_consistent_model_side_effect(num_turns)
            request = MagicMock(spec=Request)
            request.cookies = {"session": f"length-{num_turns}"}

            result = await server.responses(
                request=request,
                response=Response(),
                body=NeMoGymResponseCreateParamsNonStreaming(input="initial text"),
            )

            assert len(result.completion_evidence) == num_turns
            assert len(result.output) == 2 * num_turns - 1
            assert [record.actual_action_count for record in result.chunk_records] == expected_chunk_sizes
            action_ids = [action_id for record in result.chunk_records for action_id in record.eligible_action_ids]
            assert action_ids == [f"action-{turn:06d}" for turn in range(1, num_turns + 1)]
            if num_turns % 2:
                assert result.chunk_records[-1].early_close_reason == "terminal"
            else:
                assert result.chunk_records[-1].early_close_reason is None

    async def test_hundred_turn_recency_rollout_remains_exact_and_bounded(
        self,
    ) -> None:
        num_turns = 100
        actions_per_chunk = 5
        history_groups = 3
        config = _chunked_recency_config(actions_per_chunk)
        config.fixture = "computer_use"
        config.num_turns = num_turns
        assert isinstance(
            config.visual_history.policy.config,
            RecencyHistoryPolicyConfig,
        )
        config.visual_history.policy.config.keep_last_image_groups = history_groups

        observations = scripted_observations(
            fixture="computer_use",
            num_turns=num_turns,
        )
        image_turn_by_url = {
            _image_urls(observation)[0]: turn for turn, observation in enumerate(observations, start=1)
        }
        assert len(image_turn_by_url) == num_turns

        server = ScriptedMultimodalAgent(
            config=config,
            server_client=MagicMock(spec=ServerClient),
        )
        server.server_client.post.side_effect = _prefix_consistent_model_side_effect(num_turns)
        request = MagicMock(spec=Request)
        request.cookies = {"session": "hundred-turn-k5-recency"}

        result = await server.responses(
            request=request,
            response=Response(),
            body=NeMoGymResponseCreateParamsNonStreaming(input="initial text"),
        )

        model_calls = server.server_client.post.call_args_list
        assert len(model_calls) == num_turns
        visible_image_turns = [
            [image_turn_by_url[url] for url in _request_image_urls(call.kwargs["json"].input)] for call in model_calls
        ]
        expected_image_turns = []
        for turn in range(1, num_turns + 1):
            chunk_start = ((turn - 1) // actions_per_chunk) * actions_per_chunk + 1
            first_retained_turn = max(1, chunk_start - history_groups)
            expected_image_turns.append(list(range(first_retained_turn, turn + 1)))
        assert visible_image_turns == expected_image_turns
        assert max(map(len, visible_image_turns)) == (actions_per_chunk + history_groups)

        assert [record.actual_action_count for record in result.chunk_records] == [actions_per_chunk] * (
            num_turns // actions_per_chunk
        )
        assert [event.applies_to_step for event in result.boundary_events] == list(
            range(actions_per_chunk + 1, num_turns + 1, actions_per_chunk)
        )

        action_ids = [action_id for record in result.chunk_records for action_id in record.eligible_action_ids]
        completion_ids = [
            completion_id for record in result.chunk_records for completion_id in record.completion_evidence_ids
        ]
        assert action_ids == [f"action-{turn:06d}" for turn in range(1, num_turns + 1)]
        assert completion_ids == [completion.completion_id for completion in result.completion_evidence]
        assert [completion.segment_index for completion in result.completion_evidence] == [
            (turn - 1) // actions_per_chunk for turn in range(1, num_turns + 1)
        ]
        assert (
            max(len(completion.prompt_token_ids) for completion in result.completion_evidence)
            == 2 * actions_per_chunk - 1
        )
        assert (
            max(len(completion.media_ids) for completion in result.completion_evidence)
            == actions_per_chunk + history_groups
        )

        assert len(result.media_assets) == num_turns
        dumped = result.model_dump(mode="json")
        metadata = {
            key: dumped[key]
            for key in (
                "completion_evidence",
                "final_policy_decision",
                "lineage_deltas",
                "chunk_records",
                "boundary_events",
                "guard_records",
                "context_compaction_contract",
            )
        }
        serialized_metadata = json.dumps(metadata)
        assert "data:image/png;base64," not in serialized_metadata
        assert len(serialized_metadata) < num_turns * 25_000
        assert max(len(json.dumps(delta)) for delta in metadata["lineage_deltas"]) < 25_000

    async def test_text_only_integrated_path_never_invents_media_boundaries(
        self,
    ) -> None:
        config = _chunked_recency_config(actions_per_chunk=2)
        config.media_mode = "text_padding"
        server = ScriptedMultimodalAgent(
            config=config,
            server_client=MagicMock(spec=ServerClient),
        )
        server.server_client.post.side_effect = _prefix_consistent_model_side_effect(config.num_turns)
        request = MagicMock(spec=Request)
        request.cookies = {"session": "text-only"}

        result = await server.responses(
            request=request,
            response=Response(),
            body=NeMoGymResponseCreateParamsNonStreaming(input="initial text"),
        )

        assert all(
            not _request_image_urls(call.kwargs["json"].input) for call in server.server_client.post.call_args_list
        )
        assert all(not evidence.media_ids for evidence in result.completion_evidence)
        assert result.boundary_events == []
        assert [record.actual_action_count for record in result.chunk_records] == [
            2,
            2,
            1,
        ]

    async def test_reversed_same_shape_images_preserve_request_order(self) -> None:
        config = _active_recency_config()
        config.reverse_ordered_pair = True
        server = ScriptedMultimodalAgent(
            config=config,
            server_client=MagicMock(spec=ServerClient),
        )
        server.server_client.post.side_effect = _prefix_consistent_model_side_effect(config.num_turns)
        request = MagicMock(spec=Request)
        request.cookies = {"session": "reverse-order"}

        result = await server.responses(
            request=request,
            response=Response(),
            body=NeMoGymResponseCreateParamsNonStreaming(input="initial text"),
        )

        reversed_images = [
            _image_urls(observation) for observation in scripted_observations(reverse_ordered_pair=True)
        ]
        image_a = reversed_images[0][0]
        image_b = reversed_images[1][0]
        image_d, image_c = reversed_images[2]
        request_images = [
            _request_image_urls(call.kwargs["json"].input) for call in server.server_client.post.call_args_list
        ]
        assert request_images == [
            [image_a],
            [image_a, image_b],
            [image_b, image_d, image_c],
            [image_d, image_c],
            [image_d, image_c, image_a],
        ]
        assert len(result.completion_evidence[2].media_ids) == 3

    async def test_total_token_guard_rejects_before_model_generation(self) -> None:
        config = _chunked_recency_config(actions_per_chunk=4)
        config.visual_history.guards = ContextGuardConfig(
            max_total_tokens=1,
            reserved_generation_tokens=1,
        )
        server = ScriptedMultimodalAgent(
            config=config,
            server_client=MagicMock(spec=ServerClient),
        )
        server.server_client.post.return_value = _mock_http_response({"tokens": [10]})
        request = MagicMock(spec=Request)
        request.cookies = {"session": "token-reject"}

        with pytest.raises(RuntimeError, match="guards=total_tokens"):
            await server.responses(
                request=request,
                response=Response(),
                body=NeMoGymResponseCreateParamsNonStreaming(input="initial text"),
            )

        assert [call.kwargs["url_path"] for call in server.server_client.post.call_args_list] == ["/tokenize"]

    async def test_total_token_guard_closes_chunks_and_admits_rewritten_views(
        self,
    ) -> None:
        config = _chunked_recency_config(actions_per_chunk=4)
        config.visual_history.policy.config.keep_last_image_groups = 0
        config.visual_history.guards = ContextGuardConfig(
            max_total_tokens=5,
            reserved_generation_tokens=1,
        )
        server = ScriptedMultimodalAgent(
            config=config,
            server_client=MagicMock(spec=ServerClient),
        )
        tokenize_counts = iter([2, 4, 6, 2, 4, 6, 2])
        turn = 0

        def respond(*, url_path: str, json: Any, **_: Any) -> MagicMock:
            nonlocal turn
            if url_path == "/tokenize":
                return _mock_http_response({"tokens": list(range(next(tokenize_counts)))})
            assert url_path == "/v1/responses"
            required_prefix = list(json.required_prefix_token_ids or [])
            response = _mock_http_response(
                _model_response(
                    turn,
                    prompt_token_ids=[*required_prefix, 1000 + turn],
                    generation_token_id=2000 + turn,
                )
            )
            turn += 1
            return response

        server.server_client.post.side_effect = respond
        request = MagicMock(spec=Request)
        request.cookies = {"session": "token-admit-after-compaction"}

        result = await server.responses(
            request=request,
            response=Response(),
            body=NeMoGymResponseCreateParamsNonStreaming(input="initial text"),
        )

        assert [record.decision for record in result.guard_records] == [
            "admit",
            "admit",
            "admit_after_compaction",
            "admit",
            "admit_after_compaction",
        ]
        assert [record.post_compaction_value for record in result.guard_records] == [None, None, 3, None, 3]
        assert [record.actual_action_count for record in result.chunk_records] == [
            2,
            2,
            1,
        ]
        assert [record.early_close_reason for record in result.chunk_records] == [
            "guard:total_tokens",
            "guard:total_tokens",
            "terminal",
        ]
        assert [event.applies_to_step for event in result.boundary_events] == [
            3,
            5,
        ]
        assert turn == config.num_turns

    async def test_failed_boundary_model_call_is_stable_on_rollout_retry(
        self,
    ) -> None:
        config = _chunked_recency_config(actions_per_chunk=2)
        rollout_id = "boundary-model-call-retry"
        failed_server = ScriptedMultimodalAgent(
            config=config,
            server_client=MagicMock(spec=ServerClient),
        )
        turn = 0

        def fail_at_first_boundary(
            *,
            url_path: str,
            json: Any,
            **_: Any,
        ) -> MagicMock:
            nonlocal turn
            assert url_path == "/v1/responses"
            if turn == 2:
                failed = MagicMock()
                failed.ok = False
                failed.content.read = AsyncMock(return_value=b"model failure")
                failed.raise_for_status.side_effect = RuntimeError("model failure at boundary")
                return failed
            required_prefix = list(json.required_prefix_token_ids or [])
            response = _mock_http_response(
                _model_response(
                    turn,
                    prompt_token_ids=[*required_prefix, 1000 + turn],
                    generation_token_id=2000 + turn,
                )
            )
            turn += 1
            return response

        failed_server.server_client.post.side_effect = fail_at_first_boundary
        failed_request = MagicMock(spec=Request)
        failed_request.cookies = {"session": rollout_id}
        with pytest.raises(RuntimeError, match="model failure at boundary"):
            await failed_server.responses(
                request=failed_request,
                response=Response(),
                body=NeMoGymResponseCreateParamsNonStreaming(input="initial text"),
            )
        failed_boundary_request = (
            failed_server.server_client.post.call_args_list[2].kwargs["json"].model_dump(mode="json", warnings=False)
        )

        successful_results = []
        successful_boundary_requests = []
        for _ in range(2):
            retry_server = ScriptedMultimodalAgent(
                config=config,
                server_client=MagicMock(spec=ServerClient),
            )
            retry_server.server_client.post.side_effect = _prefix_consistent_model_side_effect(config.num_turns)
            retry_request = MagicMock(spec=Request)
            retry_request.cookies = {"session": rollout_id}
            result = await retry_server.responses(
                request=retry_request,
                response=Response(),
                body=NeMoGymResponseCreateParamsNonStreaming(input="initial text"),
            )
            successful_results.append(result.model_dump(mode="json", warnings=False))
            successful_boundary_requests.append(
                retry_server.server_client.post.call_args_list[2]
                .kwargs["json"]
                .model_dump(mode="json", warnings=False)
            )

        assert successful_boundary_requests == [
            failed_boundary_request,
            failed_boundary_request,
        ]
        assert successful_results[0]["boundary_events"] == (successful_results[1]["boundary_events"])
        assert successful_results[0]["completion_evidence"] == (successful_results[1]["completion_evidence"])
        assert successful_results[0]["lineage_deltas"] == (successful_results[1]["lineage_deltas"])

    @pytest.mark.parametrize(
        ("guards", "guard_name"),
        [
            (ContextGuardConfig(max_active_images=3), "active_images"),
            (
                ContextGuardConfig(
                    max_vision_tokens=300,
                    projected_vision_tokens_per_image=100,
                ),
                "vision_tokens",
            ),
        ],
    )
    async def test_media_guards_close_chunks_early_and_record_outcomes(
        self, guards: ContextGuardConfig, guard_name: str
    ) -> None:
        config = _chunked_recency_config(actions_per_chunk=4)
        config.visual_history.policy.config.keep_last_image_groups = 0
        config.visual_history.guards = guards
        server = ScriptedMultimodalAgent(
            config=config,
            server_client=MagicMock(spec=ServerClient),
        )
        prompts = [
            [10],
            [10, 11, 12],
            [20],
            [20, 21, 22],
            [20, 21, 22, 23, 30],
        ]
        generations = [11, 13, 21, 23, 31]
        server.server_client.post.side_effect = [
            _mock_http_response(
                _model_response(
                    turn,
                    prompt_token_ids=prompts[turn],
                    generation_token_id=generations[turn],
                )
            )
            for turn in range(5)
        ]
        request = MagicMock(spec=Request)
        request.cookies = {"session": "image-guard"}

        result = await server.responses(
            request=request,
            response=Response(),
            body=NeMoGymResponseCreateParamsNonStreaming(input="initial text"),
        )

        original_images = [_image_urls(observation) for observation in scripted_observations()]
        image_a = original_images[0][0]
        image_b = original_images[1][0]
        image_c, image_d = original_images[2]
        request_inputs = [call.kwargs["json"].input for call in server.server_client.post.call_args_list]
        assert [_request_image_urls(items) for items in request_inputs] == [
            [image_a],
            [image_a, image_b],
            [image_c, image_d],
            [image_c, image_d],
            [image_c, image_d, image_a],
        ]
        assert [record.decision for record in result.guard_records] == [
            "admit",
            "admit",
            "admit_after_compaction",
            "admit",
            "admit",
        ]
        assert [record.actual_action_count for record in result.chunk_records] == [2, 3]
        assert [record.early_close_reason for record in result.chunk_records] == [f"guard:{guard_name}", "terminal"]
        assert [event.applies_to_step for event in result.boundary_events] == [
            3,
        ]

    async def test_response_growth_is_linear_and_metadata_is_media_free(
        self,
    ) -> None:
        serialized_sizes = []
        for num_turns in (5, 10, 20):
            config = _chunked_recency_config(actions_per_chunk=2)
            config.num_turns = num_turns
            server = ScriptedMultimodalAgent(
                config=config,
                server_client=MagicMock(spec=ServerClient),
            )
            server.server_client.post.side_effect = _prefix_consistent_model_side_effect(num_turns)
            request = MagicMock(spec=Request)
            request.cookies = {"session": f"growth-{num_turns}"}

            result = await server.responses(
                request=request,
                response=Response(),
                body=NeMoGymResponseCreateParamsNonStreaming(input="initial text"),
            )
            serialized = result.model_dump_json()
            serialized_sizes.append(len(serialized))
            metadata = json.dumps(
                {
                    "completion_evidence": [value.model_dump(mode="json") for value in result.completion_evidence],
                    "final_policy_decision": result.model_dump(mode="json")["final_policy_decision"],
                    "lineage_deltas": result.model_dump(mode="json")["lineage_deltas"],
                    "chunk_records": [value.__dict__ for value in result.chunk_records],
                    "boundary_events": [value.__dict__ for value in result.boundary_events],
                    "guard_records": [value.__dict__ for value in result.guard_records],
                }
            )

            # Raw images remain only in the fixture's validation-compatible
            # transcript echo, never in evidence or bounded control records.
            assert "data:image/png;base64," in serialized
            assert "data:image/png;base64," not in metadata
            assert len(result.completion_evidence) == num_turns

        assert serialized_sizes[1] < serialized_sizes[0] * 2.5
        assert serialized_sizes[2] < serialized_sizes[1] * 2.5

    async def test_run_returns_fixed_reward_and_preserves_script_metadata(self) -> None:
        server = ScriptedMultimodalAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
        responses_body = NeMoGymResponseCreateParamsNonStreaming(input="initial text")

        inner_server = ScriptedMultimodalAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
        inner_server.server_client.post.side_effect = [_mock_http_response(_model_response(turn)) for turn in range(5)]
        request = MagicMock(spec=Request)
        request.cookies = {}
        scripted_response = await inner_server.responses(
            request=request,
            response=Response(),
            body=responses_body,
        )
        server.server_client.post.return_value = _mock_http_response(scripted_response.model_dump(mode="json"))

        result = await server.run(
            request=request,
            body=ScriptedMultimodalAgentRunRequest(responses_create_params=responses_body),
        )

        assert result.reward == 1.0
        assert len(result.response.seed_obs) == 1
        assert len(result.response.output) == 9

    async def test_run_selects_deterministic_reward_by_rollout_index(self) -> None:
        config = _config()
        config.reward_by_rollout_index = [0.0, 1.0]
        server = ScriptedMultimodalAgent(
            config=config,
            server_client=MagicMock(spec=ServerClient),
        )
        responses_body = NeMoGymResponseCreateParamsNonStreaming(input="initial text")

        inner_server = ScriptedMultimodalAgent(
            config=_config(),
            server_client=MagicMock(spec=ServerClient),
        )
        inner_server.server_client.post.side_effect = [_mock_http_response(_model_response(turn)) for turn in range(5)]
        request = MagicMock(spec=Request)
        request.cookies = {}
        scripted_response = await inner_server.responses(
            request=request,
            response=Response(),
            body=responses_body,
        )
        server.server_client.post.return_value = _mock_http_response(scripted_response.model_dump(mode="json"))

        result = await server.run(
            request=request,
            body=ScriptedMultimodalAgentRunRequest(
                responses_create_params=responses_body,
                context_compaction_rollout_index=1,
            ),
        )

        assert result.reward == 1.0

    async def test_rollout_index_reward_requires_in_range_identity(self) -> None:
        config = _config()
        config.reward_by_rollout_index = [0.0, 1.0]
        server = ScriptedMultimodalAgent(
            config=config,
            server_client=MagicMock(spec=ServerClient),
        )
        request = MagicMock(spec=Request)
        request.cookies = {}

        with pytest.raises(
            ValueError,
            match="outside reward_by_rollout_index",
        ):
            await server.run(
                request=request,
                body=ScriptedMultimodalAgentRunRequest(
                    responses_create_params=(NeMoGymResponseCreateParamsNonStreaming(input="initial text")),
                    context_compaction_rollout_index=2,
                ),
            )

        with pytest.raises(
            ValueError,
            match="requires context_compaction_rollout_index",
        ):
            await server.run(
                request=request,
                body=ScriptedMultimodalAgentRunRequest(
                    responses_create_params=(NeMoGymResponseCreateParamsNonStreaming(input="initial text")),
                ),
            )

    async def test_run_stamps_authoritative_rollout_and_group_identity(self) -> None:
        config = _active_recency_config()
        inner_server = ScriptedMultimodalAgent(
            config=config,
            server_client=MagicMock(spec=ServerClient),
        )
        inner_server.server_client.post.side_effect = _prefix_consistent_model_side_effect(config.num_turns)
        request = MagicMock(spec=Request)
        request.cookies = {}
        scripted_response = await inner_server.responses(
            request=type(
                "_Request",
                (),
                {"cookies": {"session": "group-1:batch-000000:row-000003"}},
            )(),
            response=Response(),
            body=NeMoGymResponseCreateParamsNonStreaming(input="initial text"),
        )

        server = ScriptedMultimodalAgent(
            config=config,
            server_client=MagicMock(spec=ServerClient),
        )
        server.server_client.post.return_value = _mock_http_response(scripted_response.model_dump(mode="json"))
        result = await server.run(
            request=request,
            body=ScriptedMultimodalAgentRunRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="initial text"),
                context_compaction_rollout_id=("group-1:batch-000000:row-000003"),
                context_compaction_group_id="group-1",
                context_compaction_task_id="task-1",
                context_compaction_rollout_index=3,
                context_compaction_attempt_index=0,
            ),
        )

        assert server.server_client.post.call_args.kwargs["cookies"]["session"] == ("group-1:batch-000000:row-000003")
        assert result.response.context_compaction_contract is not None
        assert result.response.context_compaction_contract.schema_version == 3
        assert result.response.context_compaction_contract.rollout_id == ("group-1:batch-000000:row-000003")
        assert result.response.context_compaction_contract.group_id == "group-1"
        assert result.response.context_compaction_contract.task_id == "task-1"
        assert result.response.context_compaction_contract.rollout_index == 3
        assert result.response.context_compaction_contract.attempt_index == 0
        assert result.response.context_compaction_contract.generation_contract.training_eligible is False
        assert len(result.response.model_call_metadata) == config.num_turns
        assert not hasattr(result.response, "completion_evidence")
        assert not hasattr(result.response, "agent_input")
        assert not hasattr(result.response, "seed_obs")
