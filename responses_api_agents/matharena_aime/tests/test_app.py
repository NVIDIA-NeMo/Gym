# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Response
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym.config_types import AggregateMetricsRequest
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.matharena_aime.app import (
    FORMAT_REPAIR_PROMPT,
    MathArenaAIMEAgent,
    MathArenaAIMEAgentConfig,
    MathArenaAIMERunRequest,
    _policy_params,
)


def agent(*, skip=False, offset=0):
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"observability_enabled": True}
    return MathArenaAIMEAgent(
        config=MathArenaAIMEAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="app.py",
            name="matharena_aime_agent",
            resources_server={"type": "resources_servers", "name": "matharena_aime"},
            model_server={"type": "responses_api_models", "name": "policy"},
            skip_verification=skip,
            skip_verification_reward=0.125,
            seed_offset=offset,
        ),
        server_client=client,
    ), client


def http(payload, *, cookies=None, error=False):
    result = MagicMock(status=500 if error else 200, ok=not error, cookies=cookies or {})
    result.read = AsyncMock(return_value=json.dumps(payload).encode())
    result.content.read = AsyncMock(return_value=json.dumps(payload).encode())
    result.raise_for_status.side_effect = RuntimeError("downstream HTTP failure")
    return result


def request(*, correlated=False):
    result = MagicMock(cookies={"incoming": "cookie"})
    result.path_params = {"rollout_id": "4-2"} if correlated else {}
    result.url.path = "/ng-rollout/4-2/v1/responses" if correlated else "/v1/responses"
    return result


def params(*, text_input=False, seed=2):
    return NeMoGymResponseCreateParamsNonStreaming.model_validate(
        {
            "input": "question" if text_input else [{"role": "user", "content": "question"}],
            "max_output_tokens": 50000,
            "temperature": 1.0,
            "top_p": 0.95,
            "metadata": {
                "extra_body": json.dumps(
                    {"seed": seed, "top_k": 64, "chat_template_kwargs": {"enable_thinking": True}}
                )
            },
        }
    )


def turn(name="first", text="Not boxed", *, incomplete=False):
    return {
        "id": name,
        "created_at": 1,
        "model": "test-policy",
        "object": "response",
        "status": "incomplete" if incomplete else "completed",
        "incomplete_details": {"reason": "max_output_tokens"} if incomplete else None,
        "output": [
            {
                "type": "reasoning",
                "id": name + "-reasoning",
                "summary": [{"type": "summary_text", "text": "unchanged reasoning"}],
                "encrypted_content": name + "-encrypted",
            },
            {
                "type": "message",
                "id": name + "-answer",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
                "prompt_token_ids": [1, 2],
                "generation_token_ids": [3, 4],
                "generation_log_probs": [-0.1, -0.2],
            },
        ],
        "usage": {
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 20,
            "output_tokens_details": {"reasoning_tokens": 15},
            "total_tokens": 30,
        },
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def decision(*, retry=True, valid=True, error=None):
    return {"needs_format_retry": retry, "parser_warning": 1 if retry else 0, "valid": valid, "verifier_error": error}


def combined(*, retry=True, valid=True, error=None):
    first = turn()
    turns = [first, turn("second", r"\boxed{2}")] if retry and valid else [first]
    result = dict(turns[-1])
    result["output"] = (
        first["output"] + [{"role": "user", "content": FORMAT_REPAIR_PROMPT, "type": "message"}] + turns[-1]["output"]
        if len(turns) == 2
        else first["output"]
    )
    result["_ng_matharena_turn_responses"] = turns
    result["_ng_matharena_format_check"] = decision(retry=retry, valid=valid, error=error)
    return result


def run_body():
    return MathArenaAIMERunRequest.model_validate(
        {
            "responses_create_params": params(),
            "expected_answer": 2,
            "language": "hi",
            "problem_idx": 1,
            "task_id": "hi-aime-1",
            "_ng_task_index": 4,
            "_ng_rollout_index": 2,
        }
    )


async def test_format_repair_replays_full_native_history_and_keeps_sampling_seed_and_cookies():
    srv, client = agent(offset=10)
    first, second = turn(incomplete=True), turn("second", r"\boxed{2}")
    client.post.side_effect = [
        http(first, cookies={"model_session": "first"}),
        http(decision(), cookies={"resource_session": "checked"}),
        http(second, cookies={"model_session": "second"}),
    ]
    output_http = Response()
    initial = params()
    before = initial.model_dump()
    result = await srv.responses(request(correlated=True), output_http, initial)
    assert initial.model_dump() == before
    assert client.post.await_count == 3
    first_call, check_call, second_call = client.post.await_args_list
    assert [call.kwargs["url_path"] for call in client.post.await_args_list] == [
        "/ng-rollout/4-2/v1/responses",
        "/needs_format_retry",
        "/ng-rollout/4-2/v1/responses",
    ]
    assert first_call.kwargs["cookies"] == {"incoming": "cookie"}
    assert check_call.kwargs["cookies"] == {"incoming": "cookie", "model_session": "first"}
    assert second_call.kwargs["cookies"] == {
        "incoming": "cookie",
        "model_session": "first",
        "resource_session": "checked",
    }
    assert set(check_call.kwargs["json"]) == {"response"}
    assert check_call.kwargs["json"]["response"]["id"] == "first"
    for call in (first_call, second_call):
        value = call.kwargs["json"]
        assert call.kwargs["server_name"] == "policy"
        assert value.max_output_tokens == 50000 and value.temperature == 1 and value.top_p == 0.95
        extra = json.loads(value.metadata["extra_body"])
        assert extra == {"seed": 12, "top_k": 64, "chat_template_kwargs": {"enable_thinking": True}}
        assert "expected_answer" not in value.model_dump()
    first_input = first_call.kwargs["json"].input
    second_input = second_call.kwargs["json"].input
    assert second_input[: len(first_input)] == first_input
    assert second_input[1].encrypted_content == "first-encrypted"
    assert second_input[2].generation_token_ids == [3, 4]
    assert second_input[2].prompt_token_ids == [1, 2]
    assert second_input[2].generation_log_probs == [-0.1, -0.2]
    assert second_input[-1].role == "user" and second_input[-1].content == FORMAT_REPAIR_PROMPT
    replay = NeMoGymResponseCreateParamsNonStreaming.model_validate(second_call.kwargs["json"].model_dump(mode="json"))
    assert replay.input[1].encrypted_content == "first-encrypted"
    assert replay.input[2].generation_token_ids == [3, 4]
    assert len(result.output) == 5
    assert result.output[2].role == "user"
    full_trajectory = [*first_input, *result.output]
    assert [(item.type, getattr(item, "role", None)) for item in full_trajectory] == [
        ("message", "user"),
        ("reasoning", None),
        ("message", "assistant"),
        ("message", "user"),
        ("reasoning", None),
        ("message", "assistant"),
    ]
    assert full_trajectory[-1].id == "second-answer"
    assert result.usage.input_tokens == 20 and result.usage.output_tokens == 40
    assert result.usage.output_tokens_details.reasoning_tokens == 30
    assert [item.id for item in result.model_extra["_ng_matharena_turn_responses"]] == ["first", "second"]
    assert any(b"model_session=second" in value for _, value in output_http.raw_headers)
    assert any(b"resource_session=checked" in value for _, value in output_http.raw_headers)


@pytest.mark.parametrize("text", [r"\boxed{999}", r"\boxed{None}", r"\boxed{2}"])
async def test_no_retry_for_any_parseable_answer_even_when_wrong(text):
    srv, client = agent()
    client.post.side_effect = [http(turn(text=text)), http(decision(retry=False))]
    result = await srv.responses(request(), Response(), params(text_input=True))
    assert client.post.await_count == 2
    assert len(result.model_extra["_ng_matharena_turn_responses"]) == 1
    assert client.post.await_args_list[0].kwargs["json"].input[0].content == "question"


async def test_never_retries_twice_even_when_repair_has_no_answer():
    srv, client = agent()
    client.post.side_effect = [http(turn()), http(decision()), http(turn("repair", "still not boxed"))]
    result = await srv.responses(request(), Response(), params())
    assert client.post.await_count == 3 and result.id == "repair"


async def test_invalid_format_check_does_not_trigger_model_retry():
    srv, client = agent()
    client.post.side_effect = [http(turn()), http(decision(valid=False, error="parser unavailable"))]
    result = await srv.responses(request(), Response(), params())
    assert client.post.await_count == 2
    assert not result.model_extra["_ng_matharena_format_check"].valid


def test_fastapi_keeps_raw_turns_repair_user_and_token_metadata():
    srv, client = agent()
    client.post.side_effect = [http(turn()), http(decision()), http(turn("repair", r"\boxed{2}"))]
    with TestClient(srv.setup_webserver()) as api:
        result = api.post("/v1/responses", json=params().model_dump(mode="json"))
        result.raise_for_status()
        payload = result.json()
    assert payload["_ng_matharena_format_check"]["needs_format_retry"]
    assert len(payload["_ng_matharena_turn_responses"]) == 2
    assert payload["output"][2]["role"] == "user"
    assert payload["output"][1]["generation_token_ids"] == [3, 4]
    NeMoGymResponse.model_validate(payload)


@pytest.mark.parametrize("phase", [0, 1, 2])
async def test_responses_propagates_each_downstream_http_error(phase):
    srv, client = agent()
    calls = [http(turn()), http(decision()), http(turn("repair"))]
    calls[phase] = http({}, error=True)
    client.post.side_effect = calls
    with pytest.raises(RuntimeError, match="HTTP failure"):
        await srv.responses(request(), Response(), params())
    assert client.post.await_count == phase + 1


async def test_invalid_model_and_parser_payloads_are_not_treated_as_wrong_answers():
    srv, client = agent()
    client.post.return_value = http({})
    with pytest.raises(RuntimeError, match="Invalid policy response"):
        await srv.responses(request(), Response(), params())
    client.post.side_effect = [http(turn()), http({"needs_format_retry": "yes"})]
    with pytest.raises(ValidationError):
        await srv.responses(request(), Response(), params())


@pytest.mark.parametrize("seed", range(4))
def test_native_repeat_seed_offset_once(seed):
    original = params(seed=seed)
    adjusted = _policy_params(original, seed_offset=10)
    assert json.loads(adjusted.metadata["extra_body"])["seed"] == 10 + seed
    assert json.loads(original.metadata["extra_body"])["seed"] == seed


@pytest.mark.parametrize("metadata", [None, {}, {"extra_body": "{}"}, {"extra_body": '{"top_k":64}'}])
def test_unseeded_standalone_request_unchanged(metadata):
    original = NeMoGymResponseCreateParamsNonStreaming(input="question", metadata=metadata)
    assert _policy_params(original, seed_offset=10) == original


@pytest.mark.parametrize("extra", ['{"seed":true}', '{"seed":"1"}', "[]", "null", "{invalid"])
def test_bad_seed_or_metadata_rejected(extra):
    with pytest.raises(ValueError):
        _policy_params(
            NeMoGymResponseCreateParamsNonStreaming(input="question", metadata={"extra_body": extra}), seed_offset=0
        )


async def test_run_native_seed_selfcall_verify_cookie_chain_and_metadata():
    srv, client = agent()
    client.post.side_effect = [
        http({}, cookies={"resource_session": "seeded"}),
        http(combined(), cookies={"model_session": "done"}),
        http({"reward": 1.0}),
    ]
    result = await srv.run(request(), run_body())
    calls = client.post.await_args_list
    assert [call.kwargs["url_path"] for call in calls] == ["/seed_session", "/ng-rollout/4-2/v1/responses", "/verify"]
    assert calls[1].kwargs["server_name"] == "matharena_aime_agent"
    assert "expected_answer" not in calls[1].kwargs["json"]
    verification = calls[2].kwargs["json"]
    assert verification["expected_answer"] == 2
    assert verification["language"] == "hi" and verification["task_id"] == "hi-aime-1"
    assert verification["format_retry_count"] == 1 and len(verification["turn_responses"]) == 2
    assert "_ng_matharena_turn_responses" not in verification["response"]
    assert calls[2].kwargs["cookies"] == {"incoming": "cookie", "resource_session": "seeded", "model_session": "done"}
    assert result.reward == 1 and result.format_retry_count == 1 and result.expected_answer == 2


@pytest.mark.parametrize("reason", [None, "parser timeout"])
async def test_invalid_retry_decision_masks_run_even_if_reverification_recovers(reason):
    srv, client = agent()
    client.post.side_effect = [http({}), http(combined(valid=False, error=reason)), http({"reward": 0.0})]
    result = await srv.run(request(), run_body())
    assert result.mask_sample and result.reward == 0 and result.format_retry_count == 0
    assert result.valid is False
    assert result.failure_kind == "matharena_aime:format_check_failed"
    assert result.failure_reason == (reason or "The format-retry decision could not be measured")


async def test_run_preserves_resource_failure_sidecar_and_skip_verification():
    srv, client = agent()
    client.post.side_effect = [
        http({}),
        http(combined(retry=False)),
        http(
            {
                "reward": -1,
                "mask_sample": True,
                "failure_kind": "matharena_aime:parser_timeout",
                "_ng_failure_class": "timeout",
            }
        ),
    ]
    result = await srv.run(request(), run_body())
    assert result.mask_sample and result.model_extra["_ng_failure_class"] == "timeout"
    srv, client = agent(skip=True)
    client.post.side_effect = [http({}), http(combined(retry=False))]
    result = await srv.run(request(), run_body())
    assert result.reward == 0.125 and result.verification_skipped and client.post.await_count == 2


@pytest.mark.parametrize("phase", [0, 1, 2])
async def test_run_propagates_each_downstream_http_error(phase):
    srv, client = agent()
    calls = [http({}), http(combined()), http({"reward": 1})]
    calls[phase] = http({}, error=True)
    client.post.side_effect = calls
    with pytest.raises(RuntimeError, match="HTTP failure"):
        await srv.run(request(), run_body())
    assert client.post.await_count == phase + 1


async def test_missing_turn_audit_rejected():
    srv, client = agent()
    payload = combined()
    payload.pop("_ng_matharena_turn_responses")
    client.post.side_effect = [http({}), http(payload)]
    with pytest.raises(RuntimeError, match="per-turn response audit"):
        await srv.run(request(), run_body())


async def test_aggregate_proxy_and_skip():
    srv, client = agent()
    client.post.return_value = http({"agent_metrics": {"accuracy": 0.5}, "key_metrics": {"accuracy": 0.5}})
    result = await srv.aggregate_metrics(AggregateMetricsRequest(verify_responses=[]))
    assert result.key_metrics == {"accuracy": 0.5}
    assert client.post.await_args.kwargs["url_path"] == "/aggregate_metrics"
    client.post.return_value = http({}, error=True)
    with pytest.raises(RuntimeError, match="HTTP failure"):
        await srv.aggregate_metrics(AggregateMetricsRequest(verify_responses=[]))
    srv, client = agent(skip=True)
    assert (await srv.aggregate_metrics(AggregateMetricsRequest(verify_responses=[]))).agent_metrics == {}
    client.post.assert_not_called()
