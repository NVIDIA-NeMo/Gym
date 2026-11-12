# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError
from starlette.requests import Request
from starlette.responses import Response
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.core import NeMoGymChatCompletion, NeMoGymResponse
from nemo_gym.models.chat_completions_converter import ChatCompletionsConverter
from nemo_gym.verification import JudgeError, judge_failsafe
from resources_servers.mcqa.app import MCQAResourcesServer, MCQAResourcesServerConfig, MCQAVerifyRequest
from resources_servers.video_caption.app import (
    VideoCaptionResourcesServer,
    VideoCaptionResourcesServerConfig,
    VideoCaptionVerifyRequest,
    VideoCaptionVerifyResponse,
)
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig, SimpleAgentRunRequest
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


def judge_response(text='{"visual_hits":[1,0]}', finish_reason="stop"):
    return NeMoGymChatCompletion.model_validate(
        {
            "id": "judge",
            "object": "chat.completion",
            "created": 0,
            "model": "judge",
            "choices": [
                {"index": 0, "finish_reason": finish_reason, "message": {"role": "assistant", "content": text}}
            ],
        }
    )


def request(caption="A person walks.", *, reference_caption=None):
    return VideoCaptionVerifyRequest(
        responses_create_params={"input": [{"role": "user", "content": "Describe the video."}]},
        verifier_metadata={
            "atomic_events": {"visual_events": ["Walking", "Sitting"], "pure_audio_events": 3},
            "reference_caption": reference_caption,
        },
        response=NeMoGymResponse(
            id="response",
            object="response",
            created_at=0,
            model="policy",
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
            output=[
                {
                    "id": "message",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": caption, "annotations": []}],
                }
            ],
        ),
    )


def server(**overrides):
    config = VideoCaptionResourcesServerConfig(
        name="video_caption",
        host="127.0.0.1",
        port=8080,
        entrypoint="app.py",
        judge_model_server={"type": "responses_api_models", "name": "video_caption_judge"},
        judge_chat_completion_create_params={"messages": [], "max_tokens": 20480, "reasoning_effort": "low"},
        caption_tokenizer_path="unused",
        judge_retry_base_delay_seconds=0,
        **overrides,
    )
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    with patch("resources_servers.video_caption.app.Tokenizer.from_file", return_value=tokenizer):
        return VideoCaptionResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


@pytest.fixture
def judge(monkeypatch):
    mock = AsyncMock(return_value=judge_response())
    monkeypatch.setattr("resources_servers.video_caption.app.call_judge", mock)
    return mock


@pytest.mark.asyncio
async def test_reward_and_judge_contract(judge):
    result = await server().verify(request("word " * 100))
    assert result.reward == 0.75
    assert result.caption_length_tokens == 100
    assert result.visual_recall_reward == 0.5
    assert result.length_reward == 1
    assert result.judge_attempts == 1
    assert result.degeneration_judge_attempts == 0
    assert result.degeneration_judge_response_text is None
    kwargs = judge.call_args.kwargs
    assert kwargs["server_name"] == "video_caption_judge"
    assert kwargs["url_path"] == "/v1/chat/completions"
    params = kwargs["json"]
    assert params.max_tokens == 20480
    assert params.reasoning_effort == "low"
    schema = params.response_format["json_schema"]["schema"]["properties"]["visual_hits"]
    assert schema["minItems"] == schema["maxItems"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("penalize_degeneration", [False, True])
@pytest.mark.parametrize(
    "reasoning_field,final_text,expected_reward",
    [
        ("reasoning", '{"visual_hits":[1,0]}', 0.75),
        ("reasoning_content", '{"visual_hits":[1,0]}', 0.75),
        (None, '{"visual_hits":[1,0]}', 0.75),
        ("reasoning", None, 0),
        ("reasoning", "", 0),
        ("reasoning", "invalid JSON", 0),
        ("reasoning", '{"visual_hits":[1]}', 0),
    ],
)
async def test_judge_through_gym_adapter(
    monkeypatch: pytest.MonkeyPatch,
    reasoning_field: str | None,
    final_text: str | None,
    expected_reward: float,
    penalize_degeneration: bool,
) -> None:
    # Exercise the real adapter, shared judge client, and verifier. Only the
    # upstream model and HTTP transport are mocked; reasoning conversion is not.
    monkeypatch.setattr("nemo_gym.server_utils.get_global_config_dict", lambda: {})
    adapter = VLLMModel(
        config=VLLMModelConfig(
            name="video_caption_judge",
            host="127.0.0.1",
            port=8081,
            entrypoint="app.py",
            base_url="http://127.0.0.1:8082/v1",
            api_key="unused",
            model="judge",
            return_token_id_information=False,
            uses_reasoning_parser=True,
            chat_template_kwargs={"enable_thinking": True, "reasoning_effort": "low"},
        ),
        server_client=MagicMock(spec=ServerClient, global_config_dict={}),
    )
    raw = judge_response(final_text).model_dump(mode="json", exclude_none=True)
    degeneration_text = '{"verdict":"pass"}'
    degeneration_raw = judge_response(degeneration_text).model_dump(mode="json", exclude_none=True)
    if reasoning_field:
        # Valid JSON inside reasoning must never substitute for a missing final answer.
        raw["choices"][0]["message"][reasoning_field] = 'Consider {"visual_hits":[1,1]} before checking the caption.'
        degeneration_raw["choices"][0]["message"][reasoning_field] = 'Check text quality. {"verdict":"fail"}'
    upstream = MagicMock()

    def upstream_response(**kwargs):
        schema_name = kwargs["response_format"]["json_schema"]["name"]
        return deepcopy(raw if schema_name == "visual_event_coverage" else degeneration_raw)

    upstream.create_chat_completion = AsyncMock(side_effect=upstream_response)
    adapter._clients = [upstream]
    http_request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [],
            "session": {SESSION_ID_KEY: "caption-judge-test"},
        }
    )
    wrapped = []

    async def route(*, server_name: str, url_path: str, **kwargs) -> MagicMock:
        assert (server_name, url_path) == ("video_caption_judge", "/v1/chat/completions")
        completion = await adapter.chat_completions(http_request, kwargs["json"])
        wrapped.append(completion.choices[0].message.content)
        response = MagicMock(ok=True)
        response.read = AsyncMock(return_value=completion.model_dump_json().encode())
        return response

    env = server(penalize_degeneration=penalize_degeneration)
    env.server_client.post = AsyncMock(side_effect=route)
    result = await judge_failsafe(env.verify)(request("word " * 100))
    if expected_reward:
        assert isinstance(result, VideoCaptionVerifyResponse), result.body.decode()
        assert result.reward == expected_reward
        assert result.visual_hits == [1, 0]
        assert result.judge_attempts == 1
        assert result.degeneration_judge_attempts == int(penalize_degeneration)
        assert upstream.create_chat_completion.await_count == 1 + int(penalize_degeneration)
        assert result.judge_response_text == final_text
        assert result.degeneration_judge_response_text == (degeneration_text if penalize_degeneration else None)
    else:
        body = json.loads(result.body)
        assert body["reward"] == body.get("length_reward", 0) == 0
        assert body["_ng_failure_class"] == "judge_failed"
        assert "3 attempts" in body["_ng_failure_judge_error"]
        assert upstream.create_chat_completion.await_count == 3
    if reasoning_field:
        assert all(text.startswith("<think>") for text in wrapped)
    else:
        assert wrapped == [final_text] + ([degeneration_text] if penalize_degeneration else [])
    sent = upstream.create_chat_completion.call_args_list[0].kwargs
    prompt_path = Path(__file__).parents[1] / "prompt_templates" / "visual_event_judge.txt"
    expected_caption = " ".join(["word"] * 100)
    assert sent["messages"] == [
        {
            "role": "user",
            "content": (
                f"{prompt_path.read_text().rstrip()}\n\nCANDIDATE CAPTION:\n{expected_caption}"
                '\n\nVISUAL EVENTS:\n["Walking", "Sitting"]'
            ),
        }
    ]
    assert sent["chat_template_kwargs"]["enable_thinking"] is True
    assert sent["reasoning_effort"] == "low"
    schema = sent["response_format"]["json_schema"]["schema"]["properties"]["visual_hits"]
    assert schema["minItems"] == schema["maxItems"] == 2
    properties = sent["response_format"]["json_schema"]["schema"]["properties"]
    assert set(properties) == {"visual_hits"}
    if expected_reward and penalize_degeneration:
        degeneration_sent = upstream.create_chat_completion.call_args_list[1].kwargs
        prompt = (prompt_path.parent / "degeneration_judge.txt").read_text().rstrip()
        assert degeneration_sent["messages"] == [
            {"role": "user", "content": f"{prompt}\n\nFINAL CAPTION:\n{expected_caption}"}
        ]
        assert degeneration_sent["response_format"]["json_schema"]["schema"]["required"] == ["verdict"]
        assert degeneration_sent["max_tokens"] == 20480
        assert degeneration_sent["reasoning_effort"] == "low"
        assert degeneration_sent["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "low"}
        assert degeneration_sent.get("temperature") is None
        assert degeneration_sent.get("top_p") is None
        assert degeneration_sent.get("top_k") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("penalize_degeneration", [False, True])
@pytest.mark.parametrize("caption", ["", " \n\t"])
@pytest.mark.parametrize("mode", ["recall", "precision", "f1"])
async def test_empty_caption_skips_judge_and_bonus(caption, judge, penalize_degeneration, mode):
    result = await server(penalize_degeneration=penalize_degeneration, visual_reward_type=mode).verify(
        request(caption)
    )
    assert result.reward == result.length_reward == result.caption_length_tokens == 0
    assert result.judge_attempts == 0
    assert result.degeneration_judge_attempts == 0
    assert result.event_extraction_attempts == result.precision_judge_attempts == 0
    assert not result.degeneration_checked
    judge.assert_not_called()


@pytest.mark.asyncio
async def test_third_attempt_success(judge):
    judge.side_effect = [JudgeError("unavailable"), judge_response("invalid JSON"), judge_response()]
    result = await server().verify(request("word " * 100))
    assert judge.await_count == result.judge_attempts == 3
    assert result.reward == 0.75


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        JudgeError("unavailable"),
        judge_response("invalid JSON"),
        judge_response('<think>unfinished reasoning {"visual_hits":[1,0]}'),
        judge_response('{"visual_hits":[1,0]}', "length"),
    ],
)
async def test_exhaustion_tagged_zero_without_length_bonus(failure, judge):
    judge.side_effect = [failure, failure, failure]
    response = await judge_failsafe(server().verify)(request("word " * 100))
    body = json.loads(response.body)
    assert judge.await_count == 3
    assert body["reward"] == 0
    assert body.get("length_reward", 0) == 0
    assert body["_ng_failure_class"] == "judge_failed"
    assert "3 attempts" in body["_ng_failure_judge_error"]


@pytest.mark.asyncio
async def test_timeout_bounds_each_of_three_attempts(judge):
    async def slow(*args, **kwargs):
        await asyncio.sleep(10)

    judge.side_effect = slow
    response = await judge_failsafe(server(judge_attempt_timeout_seconds=0.01).verify)(request())
    assert judge.await_count == 3
    assert json.loads(response.body)["reward"] == 0


@pytest.mark.asyncio
async def test_queue_wait_is_outside_timeout(judge):
    env = server(judge_endpoint_max_concurrency=1, judge_attempt_timeout_seconds=0.01)
    await env._judge_semaphore.acquire()
    task = asyncio.create_task(env.verify(request()))
    try:
        await asyncio.sleep(0.05)
        assert not task.done()
        judge.assert_not_called()
    finally:
        env._judge_semaphore.release()
    result = await task
    assert result.judge_attempts == 1


@pytest.mark.asyncio
async def test_cancellation_does_not_retry(judge):
    judge.side_effect = asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        await server().verify(request())
    assert judge.await_count == 1


@pytest.mark.asyncio
async def test_separate_judges_gate_reward_from_fail_verdict(judge):
    caption = "Wait, I should describe this again. " * 20
    verdict = {"verdict": "fail"}
    judge.side_effect = [
        judge_response('<think>{"visual_hits":[0,0]}</think>\n{"visual_hits":[1,1]}'),
        judge_response('<think>{"verdict":"pass"}</think>\n' + json.dumps(verdict)),
    ]
    result = await server(penalize_degeneration=True).verify(request(caption))
    assert judge.await_count == 2
    assert result.judge_attempts == result.degeneration_judge_attempts == 1
    assert result.reward == 0 and result.reward_before_degeneration == 1
    assert result.length_reward == result.visual_recall_reward == 1
    assert result.degeneration_checked and result.degeneration_detected
    assert json.loads(result.degeneration_judge_response_text) == verdict
    recall_params, degeneration_params = [call.kwargs["json"] for call in judge.call_args_list]
    for params in [recall_params, degeneration_params]:
        assert len(params.messages) == 1 and params.messages[0]["role"] == "user"
    assert recall_params.response_format["json_schema"]["schema"]["required"] == ["visual_hits"]
    assert degeneration_params.response_format["json_schema"]["schema"]["required"] == ["verdict"]
    assert 'VISUAL EVENTS:\n["Walking", "Sitting"]' in recall_params.messages[0]["content"]
    assert "VISUAL EVENTS:" not in degeneration_params.messages[0]["content"]


@pytest.mark.asyncio
async def test_invalid_degeneration_verdict_retries_until_third_attempt(judge):
    bad = judge_response('{"verdict":"unknown"}')
    good = judge_response('{"verdict":"pass"}')
    judge.side_effect = [judge_response(), bad, bad, good]
    result = await server(penalize_degeneration=True).verify(request("word " * 100))
    assert result.judge_attempts == 1 and result.degeneration_judge_attempts == 3
    assert judge.await_count == 4
    assert [call.kwargs["json"].response_format["json_schema"]["name"] for call in judge.call_args_list] == [
        "visual_event_coverage",
        "caption_degeneration",
        "caption_degeneration",
        "caption_degeneration",
    ]
    assert result.reward == 0.75 and not result.degeneration_detected


@pytest.mark.asyncio
async def test_missing_degeneration_verdict_exhausts_to_tagged_zero(judge):
    response = await judge_failsafe(server(penalize_degeneration=True).verify)(request("word " * 100))
    body = json.loads(response.body)
    assert judge.await_count == 4
    assert body["reward"] == 0
    assert body["_ng_failure_class"] == "judge_failed"
    assert "degeneration judge exhausted 3 attempts" in body["_ng_failure_judge_error"]


@pytest.mark.asyncio
async def test_separate_judges_have_independent_retry_budgets(judge):
    judge.side_effect = [
        JudgeError("unavailable"),
        judge_response("invalid JSON"),
        judge_response(),
        JudgeError("unavailable"),
        judge_response("invalid JSON"),
        judge_response('{"verdict":"pass"}'),
    ]
    result = await server(penalize_degeneration=True).verify(request("word " * 100))
    assert result.reward == 0.75
    assert result.judge_attempts == result.degeneration_judge_attempts == 3
    assert judge.await_count == 6


@pytest.mark.asyncio
async def test_recall_exhaustion_does_not_start_degeneration(judge):
    judge.side_effect = JudgeError("unavailable")
    response = await judge_failsafe(server(penalize_degeneration=True).verify)(request("word " * 100))
    body = json.loads(response.body)
    assert body["reward"] == 0 and body["_ng_failure_class"] == "judge_failed"
    assert "recall judge exhausted 3 attempts" in body["_ng_failure_judge_error"]
    assert judge.await_count == 3


@pytest.mark.asyncio
async def test_degeneration_timeout_does_not_repeat_successful_recall(judge):
    async def respond(*args, **kwargs):
        if kwargs["json"].response_format["json_schema"]["name"] == "visual_event_coverage":
            return judge_response()
        await asyncio.sleep(10)

    judge.side_effect = respond
    env = server(penalize_degeneration=True, judge_attempt_timeout_seconds=0.01)
    response = await judge_failsafe(env.verify)(request("word " * 100))
    body = json.loads(response.body)
    assert body["reward"] == 0 and body["_ng_failure_class"] == "judge_failed"
    assert "degeneration judge exhausted 3 attempts: TimeoutError" in body["_ng_failure_judge_error"]
    assert judge.await_count == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("has_final_answer", [False, True])
async def test_judgments_and_length_use_final_answer_without_policy_reasoning(judge, has_final_answer):
    caption = " ".join(["word"] * 100) if has_final_answer else ""
    reasoning = "HIDDEN_REASONING_SENTINEL Let me reconsider this. " * 1000
    converter = ChatCompletionsConverter(return_token_id_information=False, uses_reasoning_parser=True)
    output = converter.postprocess_chat_response(judge_response(f"<think>{reasoning}</think>{caption}").choices[0])
    body = request(caption)
    body.response = NeMoGymResponse.model_validate({**body.response.model_dump(), "output": output})
    assert body.response.output_text == caption
    judge.side_effect = [judge_response(), judge_response('{"verdict":"pass"}')]
    result = await server(penalize_degeneration=True).verify(body)
    if has_final_answer:
        assert result.reward == 0.75 and result.caption_length_tokens == 100
        assert judge.await_count == 2
        for call in judge.call_args_list:
            prompt = call.kwargs["json"].messages[0]["content"]
            assert caption in prompt
            assert "HIDDEN_REASONING_SENTINEL" not in prompt
            assert "<think>" not in prompt and "</think>" not in prompt
    else:
        assert result.reward == result.caption_length_tokens == 0
        judge.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,caption,expected",
    [
        ("caption", "word " * 100, 0.75),
        ("caption_f1", "word " * 100, 5 / 6),
        ("qa", r"\boxed{B}", 1.0),
        ("qa", r"\boxed{A}", 0.0),
    ],
)
async def test_caption_and_qa_agent_routing(kind, caption, expected, judge):
    gym_root = Path(__file__).resolve().parents[3]
    is_caption = kind.startswith("caption")
    name = "video_caption" if is_caption else "mcqa"
    config = OmegaConf.load(gym_root / f"resources_servers/{name}/configs/{name}.yaml")
    agent_name = f"{name}_simple_agent"
    config_values = OmegaConf.to_container(config[agent_name].responses_api_agents.simple_agent)
    client = MagicMock(spec=ServerClient)
    agent = SimpleAgent(
        config=SimpleAgentConfig(name=agent_name, host="127.0.0.1", port=8081, **config_values),
        server_client=client,
    )
    if is_caption:
        resource = server(visual_reward_type="f1" if kind == "caption_f1" else "recall")
        if kind == "caption_f1":
            judge.side_effect = [
                judge_response(),
                judge_response('{"visual_events":["Walking"]}'),
                judge_response('{"visual_hits":[1]}'),
            ]
    else:
        resource = MCQAResourcesServer(
            config=MCQAResourcesServerConfig(name="mcqa", host="127.0.0.1", port=8082, entrypoint="app.py"),
            server_client=client,
        )
    policy_response = request(caption).response
    http_request = Request({"type": "http", "method": "POST", "path": "/run", "headers": []})
    calls = []

    async def route(*, server_name, url_path, **kwargs):
        calls.append((server_name, url_path))
        payload = kwargs["json"]
        if (server_name, url_path) == (name, "/seed_session"):
            value = {}
        elif (server_name, url_path) == (agent_name, "/v1/responses"):
            value = (await agent.responses(http_request, Response(), payload)).model_dump(mode="json")
        elif (server_name, url_path) == ("policy_model", "/v1/responses"):
            assert "verifier_metadata" not in payload.model_dump()
            assert "HUMAN_REFERENCE" not in payload.model_dump_json()
            value = policy_response.model_dump(mode="json")
        elif (server_name, url_path) == (name, "/verify"):
            request_type = VideoCaptionVerifyRequest if is_caption else MCQAVerifyRequest
            value = (await resource.verify(request_type.model_validate(payload))).model_dump(mode="json")
        else:
            raise AssertionError((server_name, url_path))
        response = MagicMock(ok=True, cookies={})
        # Gym's shared decoding helper consumes the aiohttp response's bytes.
        response.read = AsyncMock(return_value=json.dumps(value).encode())
        return response

    client.post = AsyncMock(side_effect=route)
    body = request(caption, reference_caption="HUMAN_REFERENCE A person walks.").model_dump(exclude={"response"})
    if kind == "qa":
        body = {
            "responses_create_params": body["responses_create_params"],
            "expected_answer": "B",
            "options": [{"A": "red"}, {"B": "blue"}],
        }
    result = await agent.run(http_request, SimpleAgentRunRequest.model_validate(body))
    assert result.reward == pytest.approx(expected)
    assert calls == [
        (name, "/seed_session"),
        (agent_name, "/v1/responses"),
        ("policy_model", "/v1/responses"),
        (name, "/verify"),
    ]


def test_visual_reward_configuration_defaults_and_validation():
    assert server().config.visual_reward_type == "recall"
    with pytest.raises(ValidationError, match="visual_reward_type"):
        server(visual_reward_type="unsupported")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["precision", "f1"])
@pytest.mark.parametrize("reference", [None, "", " \n\t"])
async def test_precision_requires_human_reference_before_any_judging(mode, reference, judge):
    with pytest.raises(ValueError, match="nonempty verifier_metadata.reference_caption"):
        await server(visual_reward_type=mode).verify(request(reference_caption=reference))
    judge.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,expected", [("precision", 0.25), ("f1", 1 / 3)])
@pytest.mark.parametrize("degeneration", [False, True])
async def test_precision_uses_reference_caption_and_predicted_events_in_separate_requests(
    mode, expected, degeneration, judge
):
    caption = "PREDICTION " + "word " * 99
    reference = "HUMAN_REFERENCE A person walks past a red house."
    events = ["PREDICTED Walking", "PREDICTED Sitting", "PREDICTED Waving", "PREDICTED Leaving"]
    responses = [judge_response()] if mode == "f1" else []
    responses += [judge_response(json.dumps({"visual_events": events})), judge_response('{"visual_hits":[1,0,0,0]}')]
    if degeneration:
        responses.append(judge_response('{"verdict":"pass"}'))
    judge.side_effect = responses
    result = await server(visual_reward_type=mode, penalize_degeneration=degeneration).verify(
        request(caption, reference_caption=reference)
    )
    assert result.reward == pytest.approx((expected + 1) / 2)
    assert result.visual_reward == pytest.approx(expected)
    assert result.visual_reward_type == mode
    assert result.visual_recall_reward == (0.5 if mode == "f1" else None)
    assert result.visual_precision_reward == 0.25
    assert result.predicted_visual_events == events
    assert result.precision_visual_hits == [1, 0, 0, 0]
    assert result.event_extraction_attempts == result.precision_judge_attempts == 1
    assert result.judge_attempts == int(mode == "f1")
    assert result.degeneration_judge_attempts == int(degeneration)
    assert result.judge_response_text == ('{"visual_hits":[1,0]}' if mode == "f1" else None)
    assert json.loads(result.event_extraction_response_text) == {"visual_events": events}
    assert json.loads(result.precision_judge_response_text) == {"visual_hits": [1, 0, 0, 0]}
    params = [call.kwargs["json"] for call in judge.call_args_list]
    assert len(params) == 2 + int(mode == "f1") + int(degeneration)
    for param in params:
        assert len(param.messages) == 1 and param.messages[0]["role"] == "user"
        assert param.max_tokens == 20480 and param.reasoning_effort == "low"
        assert param.response_format["json_schema"]["strict"]
    offset = int(mode == "f1")
    prompt_path = Path(__file__).parents[1] / "prompt_templates"
    coverage_prompt = (prompt_path / "visual_event_judge.txt").read_text().rstrip()
    extraction_prompt = (prompt_path / "visual_event_extraction.txt").read_text().rstrip()
    assert params[offset].messages[0]["content"] == (
        f"{extraction_prompt}\n\n# Input Visual Caption\n"
        + json.dumps({"visual_caption": caption.strip()}, ensure_ascii=False)
    )
    assert params[offset + 1].messages[0]["content"] == (
        f"{coverage_prompt}\n\nCANDIDATE CAPTION:\n{reference}\n\nVISUAL EVENTS:\n" + json.dumps(events)
    )
    if mode == "f1":
        assert params[0].messages[0]["content"] == (
            f'{coverage_prompt}\n\nCANDIDATE CAPTION:\n{caption.strip()}\n\nVISUAL EVENTS:\n["Walking", "Sitting"]'
        )
    if degeneration:
        assert params[-1].messages[0]["content"].endswith(f"FINAL CAPTION:\n{caption.strip()}")
        assert reference not in params[-1].messages[0]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["precision", "f1"])
async def test_empty_extraction_is_zero_precision_and_skips_coverage_request(mode, judge):
    judge.side_effect = ([judge_response()] if mode == "f1" else []) + [
        judge_response('{"visual_events":[]}'),
        judge_response('{"verdict":"pass"}'),
    ]
    result = await server(visual_reward_type=mode, penalize_degeneration=True).verify(
        request("word " * 100, reference_caption="A person walks.")
    )
    assert result.predicted_visual_events == []
    assert result.visual_reward == result.visual_precision_reward == 0
    assert result.reward == 0.5 and result.length_reward == 1
    assert result.precision_judge_attempts == 0 and result.precision_judge_response_text is None
    assert result.degeneration_checked and not result.degeneration_detected
    assert judge.await_count == 2 + int(mode == "f1")


@pytest.mark.asyncio
async def test_each_f1_and_degeneration_stage_has_three_independent_attempts(judge):
    successes = [
        judge_response(),
        judge_response('{"visual_events":["Walking"]}'),
        judge_response('{"visual_hits":[1]}'),
        judge_response('{"verdict":"pass"}'),
    ]
    judge.side_effect = [
        item for good in successes for item in (JudgeError("unavailable"), judge_response("bad JSON"), good)
    ]
    result = await server(visual_reward_type="f1", penalize_degeneration=True).verify(
        request("word " * 100, reference_caption="A person walks.")
    )
    assert result.reward == pytest.approx((2 / 3 + 1) / 2)
    assert (
        result.judge_attempts
        == result.event_extraction_attempts
        == result.precision_judge_attempts
        == result.degeneration_judge_attempts
        == 3
    )
    assert judge.await_count == 12
    assert [call.kwargs["json"].response_format["json_schema"]["name"] for call in judge.call_args_list] == (
        ["visual_event_coverage"] * 3
        + ["visual_event_extraction"] * 3
        + ["visual_event_coverage"] * 3
        + ["caption_degeneration"] * 3
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stage,prefix", [("event extraction", 1), ("precision", 2)])
@pytest.mark.parametrize("failure", [JudgeError("unavailable"), judge_response("bad JSON")])
async def test_exhausted_precision_stages_return_tagged_zero_without_repeating_successes(
    stage, prefix, failure, judge
):
    successes = [judge_response(), judge_response('{"visual_events":["Walking"]}')]
    judge.side_effect = successes[:prefix] + [failure] * 3
    result = await judge_failsafe(server(visual_reward_type="f1", penalize_degeneration=True).verify)(
        request("word " * 100, reference_caption="A person walks.")
    )
    body = json.loads(result.body)
    assert body["reward"] == body.get("length_reward", 0) == 0
    assert body["_ng_failure_class"] == "judge_failed"
    assert f"{stage} judge exhausted 3 attempts" in body["_ng_failure_judge_error"]
    assert judge.await_count == prefix + 3


@pytest.mark.asyncio
async def test_precision_extraction_uses_only_final_policy_text(judge):
    caption = "word " * 100
    converter = ChatCompletionsConverter(return_token_id_information=False, uses_reasoning_parser=True)
    output = converter.postprocess_chat_response(
        judge_response(f"<think>PRIVATE_REASONING</think>{caption}").choices[0]
    )
    body = request(caption, reference_caption="A person walks.")
    body.response = NeMoGymResponse.model_validate({**body.response.model_dump(), "output": output})
    judge.side_effect = [judge_response('{"visual_events":["Walking"]}'), judge_response('{"visual_hits":[1]}')]
    result = await server(visual_reward_type="precision").verify(body)
    assert result.reward == 1 and result.caption_length_tokens == 100
    for call in judge.call_args_list:
        assert "PRIVATE_REASONING" not in call.kwargs["json"].messages[0]["content"]


@pytest.mark.asyncio
async def test_f1_through_real_gym_judge_adapter(monkeypatch):
    monkeypatch.setattr("nemo_gym.server_utils.get_global_config_dict", lambda: {})
    adapter = VLLMModel(
        config=VLLMModelConfig(
            name="video_caption_judge",
            host="127.0.0.1",
            port=8081,
            entrypoint="app.py",
            base_url="http://127.0.0.1:8082/v1",
            api_key="unused",
            model="judge",
            return_token_id_information=False,
            uses_reasoning_parser=True,
            chat_template_kwargs={"enable_thinking": True, "reasoning_effort": "low"},
        ),
        server_client=MagicMock(spec=ServerClient, global_config_dict={}),
    )
    finals = ['{"visual_hits":[1,0]}', '{"visual_events":["Walking"]}', '{"visual_hits":[1]}', '{"verdict":"pass"}']
    raw_responses = []
    for final in finals:
        raw = judge_response(final).model_dump(mode="json", exclude_none=True)
        raw["choices"][0]["message"]["reasoning"] = 'PRIVATE_JUDGE_REASONING {"visual_events":["Wrong"]}'
        raw_responses.append(raw)
    upstream = MagicMock()
    upstream.create_chat_completion = AsyncMock(side_effect=raw_responses)
    adapter._clients = [upstream]
    http_request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [],
            "session": {SESSION_ID_KEY: "f1-test"},
        }
    )

    async def route(*, server_name, url_path, **kwargs):
        assert (server_name, url_path) == ("video_caption_judge", "/v1/chat/completions")
        completion = await adapter.chat_completions(http_request, kwargs["json"])
        assert completion.choices[0].message.content.startswith("<think>")
        response = MagicMock(ok=True)
        response.read = AsyncMock(return_value=completion.model_dump_json().encode())
        return response

    env = server(visual_reward_type="f1", penalize_degeneration=True)
    env.server_client.post = AsyncMock(side_effect=route)
    result = await env.verify(request("word " * 100, reference_caption="A person walks."))
    assert result.visual_recall_reward == 0.5 and result.visual_precision_reward == 1
    assert result.visual_f1_reward == pytest.approx(2 / 3)
    assert result.reward == pytest.approx(5 / 6)
    assert result.predicted_visual_events == ["Walking"]
    assert [
        result.judge_response_text,
        result.event_extraction_response_text,
        result.precision_judge_response_text,
        result.degeneration_judge_response_text,
    ] == finals
    assert (
        result.judge_attempts
        == result.event_extraction_attempts
        == result.precision_judge_attempts
        == result.degeneration_judge_attempts
        == 1
    )
    assert upstream.create_chat_completion.await_count == 4
    for call in upstream.create_chat_completion.call_args_list:
        params = call.kwargs
        assert params["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "low"}
        assert params["max_tokens"] == 20480 and params["reasoning_effort"] == "low"
        assert len(params["messages"]) == 1 and params["messages"][0]["role"] == "user"
