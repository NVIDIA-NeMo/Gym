# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientConnectionError, ClientResponseError
from fastapi import HTTPException
from starlette.requests import Request

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.core import NeMoGymChatCompletionCreateParamsNonStreaming, NeMoGymResponseCreateParamsNonStreaming
from responses_api_models.sglang_model.app import SGLangModel, SGLangModelConfig


class FakeTokenizer:
    def __init__(self, full_prompt_ids: list[int], decoded: str = "answer") -> None:
        self.full_prompt_ids = full_prompt_ids
        self.decoded = decoded
        self.decode_calls: list[dict] = []

    def apply_chat_template(
        self,
        messages,
        *,
        tools=None,
        chat_template=None,
        add_generation_prompt,
        tokenize,
        **kwargs,
    ):
        if tokenize:
            return list(self.full_prompt_ids)
        assistant = next(msg for msg in messages if msg["role"] == "assistant" and msg["content"].startswith("gym_"))
        return assistant["content"] + "<|im_end|>\n" + ("FOLLOWUP" if add_generation_prompt else "")

    def __call__(self, text: str, *, add_special_tokens: bool):
        assert add_special_tokens is False
        if text == "<|im_end|>\n":
            return {"input_ids": [90, 91]}
        if text == "<|im_end|>\nFOLLOWUP":
            return {"input_ids": [90, 91, 30, 31]}
        if text == "FOLLOWUP":
            return {"input_ids": [30, 31]}
        raise AssertionError(f"unexpected tokenization input: {text!r}")

    def decode(self, token_ids, *, skip_special_tokens: bool, spaces_between_special_tokens: bool):
        self.decode_calls.append(
            {
                "token_ids": list(token_ids),
                "skip_special_tokens": skip_special_tokens,
                "spaces_between_special_tokens": spaces_between_special_tokens,
            }
        )
        return self.decoded


def test_native_prompt_renders_with_installed_chat_template_engine():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    model = make_model()
    model._sglang_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]")),
        chat_template="{{ messages[0]['content'] }}",
    )

    assert model._full_sglang_tokenize([{"role": "user", "content": "hello"}], None, {}) == [1]


class FakeSGLangClient:
    def __init__(self, result: dict) -> None:
        self.result = result
        self.calls: list[dict] = []

    async def create_generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def make_model(
    *,
    context_length: int = 64,
    tokenizer: FakeTokenizer | None = None,
    client: FakeSGLangClient | None = None,
) -> SGLangModel:
    config = SGLangModelConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="sglang_model",
        base_url="http://localhost:30000/v1",
        api_key="unused",  # pragma: allowlist secret
        model="local-tokenizer",
        context_length=context_length,
        return_token_id_information=True,
        uses_reasoning_parser=True,
    )
    model = SGLangModel(
        config=config,
        server_client=MagicMock(spec=ServerClient, global_config_dict={}),
    )
    if tokenizer is not None:
        model._sglang_tokenizer = tokenizer
    if client is not None:
        model._clients = [client]
    return model


async def test_generate_path_preserves_training_ids_reasoning_and_tools() -> None:
    tokenizer = FakeTokenizer(
        [1, 2, 3],
        decoded=(
            'private reasoning</think><tool_call>{"name":"shell","arguments":{"command":"ls"}}</tool_call><|im_end|>'
        ),
    )
    client = FakeSGLangClient(
        {
            "meta_info": {
                "output_ids": [11, 12],
                "output_token_logprobs": [
                    {"token_id": 11, "logprob": -0.1},
                    {"id": 12, "logprob": -0.2},
                ],
                "finish_reason": {"type": "stop"},
            }
        }
    )
    model = make_model(tokenizer=tokenizer, client=client)
    body = NeMoGymChatCompletionCreateParamsNonStreaming(
        messages=[{"role": "user", "content": "inspect"}],
        max_tokens=8,
    )
    request = SimpleNamespace(session={SESSION_ID_KEY: "session-1"})

    response = await model.chat_completions(request, body)

    assert client.calls == [
        {
            "input_ids": [1, 2, 3],
            "sampling_params": {
                "spaces_between_special_tokens": False,
                "max_new_tokens": 8,
            },
            "return_logprob": True,
        }
    ]
    choice = response.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.content == "<think>private reasoning</think>"
    assert choice.message.prompt_token_ids == [1, 2, 3]
    assert choice.message.generation_token_ids == [11, 12]
    assert choice.message.generation_log_probs == [-0.1, -0.2]
    assert choice.message.tool_calls[0].function.name == "shell"
    assert json_loads(choice.message.tool_calls[0].function.arguments) == {"command": "ls"}
    assert tokenizer.decode_calls == [
        {
            "token_ids": [11, 12],
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
        }
    ]


def json_loads(value: str) -> dict:
    import json

    return json.loads(value)


def sampled_client(ids=(20, 21)):
    return FakeSGLangClient(
        {
            "meta_info": {
                "output_ids": list(ids),
                "output_token_logprobs": [[-0.1, token] for token in ids],
                "finish_reason": {"type": "stop"},
            }
        }
    )


def session_request(session_id="rollout"):
    return Request({"type": "http", "session": {SESSION_ID_KEY: session_id}, "headers": []})


@pytest.mark.parametrize("ending", [[], [90], [90, 91]])
async def test_followup_splices_exact_sampled_ids_and_missing_boundary(ending):
    client = sampled_client([20, 21, *ending])
    model = make_model(tokenizer=FakeTokenizer([10, 11]), client=client)
    request = session_request()
    first = [{"role": "user", "content": "first"}]
    response = await model.chat_completions(request, NeMoGymChatCompletionCreateParamsNonStreaming(messages=first))
    assistant = response.choices[0].message.model_dump(exclude_none=True)
    await model.chat_completions(
        request,
        NeMoGymChatCompletionCreateParamsNonStreaming(
            messages=[*first, assistant, {"role": "user", "content": "continue"}],
        ),
    )
    assert client.calls[-1]["input_ids"] == [10, 11, 20, 21, 90, 91, 30, 31]


@pytest.mark.parametrize("max_tokens_field", ["max_completion_tokens", "max_tokens"])
async def test_explicit_max_tokens_is_clamped_to_remaining_context(
    max_tokens_field: str,
) -> None:
    tokenizer = FakeTokenizer([1, 2, 3])
    client = FakeSGLangClient(
        {
            "meta_info": {
                "output_ids": [11],
                "output_token_logprobs": [
                    {"token_id": 11, "logprob": -0.1},
                ],
            }
        }
    )
    model = make_model(
        context_length=10,
        tokenizer=tokenizer,
        client=client,
    )
    body = NeMoGymChatCompletionCreateParamsNonStreaming(
        messages=[{"role": "user", "content": "inspect"}],
        **{max_tokens_field: 10},
    )

    await model.chat_completions(
        SimpleNamespace(session={SESSION_ID_KEY: "session-clamp"}),
        body,
    )

    assert client.calls[0]["sampling_params"]["max_new_tokens"] == 7


async def test_over_context_terminates_without_calling_generate() -> None:
    tokenizer = FakeTokenizer([1, 2, 3, 4])
    client = FakeSGLangClient({"must": "not be used"})
    model = make_model(context_length=4, tokenizer=tokenizer, client=client)
    body = NeMoGymChatCompletionCreateParamsNonStreaming(
        messages=[{"role": "user", "content": "too long"}],
    )
    request = SimpleNamespace(session={SESSION_ID_KEY: "session-3"})

    response = await model.chat_completions(request, body)

    assert client.calls == []
    assert response.choices[0].finish_reason == "length"
    assert response.choices[0].message.prompt_token_ids == [1, 2, 3, 4]
    assert response.choices[0].message.generation_token_ids == []


@pytest.mark.parametrize("change", ["tools", "template", "assistant", "name", "tool_call_id"])
async def test_edited_continuation_fails_before_generate(change):
    client = sampled_client()
    model = make_model(tokenizer=FakeTokenizer([10]), client=client)
    request = session_request()
    history = [{"role": "tool", "name": "lookup", "content": "result", "tool_call_id": "one"}]
    controls = {"tools": [{"type": "function", "function": {"name": "read"}}]}
    response = await model.chat_completions(
        request,
        NeMoGymChatCompletionCreateParamsNonStreaming(
            messages=history,
            **controls,
        ),
    )
    assistant = response.choices[0].message.model_dump(exclude_none=True)
    if change == "tools":
        controls["tools"] = []
    elif change == "template":
        controls["metadata"] = {"chat_template_kwargs": '{"enable_thinking":false}'}
    elif change == "assistant":
        assistant["content"] = "edited"
    else:
        history[0][change] = "different"
    with pytest.raises(HTTPException) as exc:
        await model.chat_completions(
            request,
            NeMoGymChatCompletionCreateParamsNonStreaming(
                messages=[*history, assistant, {"role": "user", "content": "next"}],
                **controls,
            ),
        )
    assert exc.value.status_code == 409
    assert len(client.calls) == 1


def test_sglang_config_owns_context_and_tool_format() -> None:
    model = make_model(context_length=128)

    assert model.config.context_length == 128
    assert model.config.sglang_tool_format == "hermes"
    assert not hasattr(model.config, "engine")


def test_inline_chat_template_is_used_directly() -> None:
    model = make_model()
    model.config.sglang_chat_template = "inline-template"
    model.config.sglang_chat_template_path = "/must/not/be/read"

    assert model._get_sglang_chat_template() == "inline-template"


@pytest.mark.parametrize(
    "controls",
    [
        {"n": 2},
        {"tool_choice": "none"},
        {"parallel_tool_calls": False},
        {"response_format": {"type": "json_object"}},
        {"frequency_penalty": 1.0},
        {"presence_penalty": 1.0},
        {"logit_bias": {"1": 2}},
        {"seed": 4},
    ],
)
async def test_unsupported_chat_controls_fail_before_tokenization(controls):
    model = make_model(tokenizer=MagicMock(), client=sampled_client())
    body = NeMoGymChatCompletionCreateParamsNonStreaming(messages=[{"role": "user", "content": "first"}], **controls)
    with pytest.raises(HTTPException) as exc:
        await model.chat_completions(session_request(), body)
    assert exc.value.status_code == 422
    assert next(iter(controls)) in exc.value.detail
    model._sglang_tokenizer.apply_chat_template.assert_not_called()
    assert not model._clients[0].calls


@pytest.mark.parametrize(
    "controls",
    [
        {"tool_choice": "none"},
        {"parallel_tool_calls": False},
        {"text": {"format": {"type": "json_schema", "name": "result", "schema": {"type": "object"}}}},
    ],
)
@pytest.mark.parametrize("with_tools", [True, False])
async def test_unsupported_responses_controls_fail_before_tokenization(controls, with_tools):
    model = make_model(tokenizer=MagicMock(), client=sampled_client())
    body = NeMoGymResponseCreateParamsNonStreaming(
        input="first",
        tools=[{"type": "function", "name": "read", "parameters": {}, "strict": False}] if with_tools else [],
        **controls,
    )
    with pytest.raises(HTTPException) as exc:
        await model.responses(session_request(), body)
    assert exc.value.status_code == 422
    model._sglang_tokenizer.apply_chat_template.assert_not_called()
    assert not model._clients[0].calls


@pytest.mark.parametrize("opening", ["<think>", ""])
async def test_responses_reasoning_and_tool_replay(opening):
    client = sampled_client()
    tokenizer = FakeTokenizer(
        [1, 2],
        decoded=(
            opening + 'private</think>answer<tool_call>{"arguments":{"key":{"text":"}"}}, "name":"read"}</tool_call>'
        ),
    )
    model = make_model(tokenizer=tokenizer, client=client)
    request = session_request()
    first = [{"role": "user", "content": "first"}]
    tools = [{"type": "function", "name": "read", "parameters": {}, "strict": False}]
    result = await model.responses(request, NeMoGymResponseCreateParamsNonStreaming(input=first, tools=tools))
    reasoning, message, call = result.output
    assert reasoning.summary[0].text == "private"
    assert message.content[0].text == "answer"
    assert json_loads(call.arguments) == {"key": {"text": "}"}}
    replay = [item.model_dump(exclude_none=True) for item in result.output]
    next_input = [*first, *replay, {"type": "function_call_output", "call_id": call.call_id, "output": "ok"}]
    edited_message = {**replay[1], "content": [{**replay[1]["content"][0], "text": "edited answer"}]}
    with pytest.raises(HTTPException) as exc:
        await model.responses(
            request,
            NeMoGymResponseCreateParamsNonStreaming(
                input=[*first, replay[0], edited_message, replay[2], next_input[-1]],
                tools=tools,
            ),
        )
    assert exc.value.status_code == 409
    assert len(client.calls) == 1
    body = NeMoGymResponseCreateParamsNonStreaming(input=next_input, tools=tools)
    await model.responses(request, body)
    assert client.calls[-1]["input_ids"] == [1, 2, 20, 21, 90, 91, 30, 31]


@pytest.mark.parametrize("bad", ["{bad json}", "{}", "[]", '{"name":"", "arguments":{}}'])
def test_hermes_preserves_only_malformed_markup(bad):
    malformed = f"<tool_call>{bad}</tool_call>"
    valid = '<tool_call>{"arguments":{},"name":"read"}</tool_call>'
    _, content, calls = make_model()._parse_sglang_generation(f"before {malformed} then {valid} after")
    assert content == f"before {malformed} then  after"
    assert [call["function"]["name"] for call in calls] == ["read"]


@pytest.mark.parametrize(
    ("pins", "temperature", "limit"),
    [
        ({}, 0.8, 7),
        ({"temperature": 0.0, "max_tokens": 4}, 0.0, 4),
        ({"max_completion_tokens": 5}, 0.8, 5),
        ({"max_tokens": None, "temperature": None}, None, 1),
    ],
)
async def test_sampling_defaults_and_pins(pins, temperature, limit):
    client = sampled_client()
    model = make_model(tokenizer=FakeTokenizer([1, 2, 3]), client=client, context_length=10)
    model.config.extra_body = {"temperature": 0.1, "top_k": 5, "max_completion_tokens": 2}
    model.config.sampling_overrides = pins
    await model.chat_completions(
        session_request(),
        NeMoGymChatCompletionCreateParamsNonStreaming(
            messages=[{"role": "user", "content": "first"}],
            max_tokens=8,
            temperature=0.8,
            n=1,
            tool_choice="auto",
            parallel_tool_calls=True,
            response_format={"type": "text"},
        ),
    )
    sampling = client.calls[0]["sampling_params"]
    assert sampling["max_new_tokens"] == limit
    assert sampling["top_k"] == 5
    if temperature is None:
        assert "temperature" not in sampling
    else:
        assert sampling["temperature"] == temperature


@pytest.mark.parametrize("problem", ["evicted", "lost cookie with token metadata", "incompatible template"])
async def test_lost_splice_state_fails_before_generate(problem, monkeypatch):
    client = sampled_client()
    model = make_model(tokenizer=FakeTokenizer([1]), client=client)
    request = session_request()
    first = [{"role": "assistant", "content": "example"}, {"role": "user", "content": "first"}]
    body = NeMoGymChatCompletionCreateParamsNonStreaming(messages=first)
    result = await model.chat_completions(request, body)
    assistant = result.choices[0].message.model_dump(exclude_none=True)
    if problem == "incompatible template":
        monkeypatch.setattr(model._sglang_tokenizer, "apply_chat_template", lambda *args, **kwargs: "no anchor")
    else:
        monkeypatch.setattr(type(model), "_SESSION_CAPACITY", 1)
        await model.chat_completions(session_request("second"), body)
        if problem == "evicted":
            assistant = {"role": "assistant", "content": assistant["content"]}
        else:
            request = session_request()
    before = len(client.calls)
    with pytest.raises(HTTPException) as exc:
        await model.chat_completions(
            request,
            NeMoGymChatCompletionCreateParamsNonStreaming(
                messages=[*first, assistant, {"role": "user", "content": "next"}],
            ),
        )
    assert exc.value.status_code == 409
    assert len(client.calls) == before


@pytest.mark.parametrize("with_tools", [False, True])
async def test_template_boundary_and_conditional_header_with_real_renderer(with_tools):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from transformers import PreTrainedTokenizerFast

    # With no merges each character has exactly one token, so the two sampled IDs can be inspected directly.
    vocab = {char: i for i, char in enumerate("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ:<> ")}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(BPE(vocab, [])),
        chat_template=(
            "{% for m in messages %}{{ m.role }}:{{ m.content }}<EOT>{% endfor %}"
            "{% if add_generation_prompt %}assistant:{{ 'TOOL' if tools else 'TEXT' }}{% endif %}"
        ),
        eos_token="<EOT>",
    )
    sampled = [vocab["A"], tokenizer.eos_token_id]
    client = sampled_client(sampled)
    model = make_model(tokenizer=tokenizer, client=client, context_length=256)
    request = session_request()
    first = [{"role": "user", "content": "first"}]
    controls = {"tools": [{"type": "function", "function": {"name": "read"}}]} if with_tools else {}
    result = await model.chat_completions(
        request,
        NeMoGymChatCompletionCreateParamsNonStreaming(
            messages=first,
            **controls,
        ),
    )
    followup = [*first, result.choices[0].message.model_dump(exclude_none=True), {"role": "user", "content": "next"}]
    await model.chat_completions(
        request,
        NeMoGymChatCompletionCreateParamsNonStreaming(
            messages=followup,
            **controls,
        ),
    )
    mode = "TOOL" if with_tools else "TEXT"

    def ids(text):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    assert client.calls[0]["input_ids"] == ids("user:first<EOT>assistant:" + mode)
    assert client.calls[1]["input_ids"] == client.calls[0]["input_ids"] + sampled + ids(
        "user:next<EOT>assistant:" + mode
    )


@pytest.mark.parametrize("failure", ["connection", "server", "overflow", "bad request"])
async def test_generate_accounts_for_endpoint_health(failure):
    model = make_model(tokenizer=FakeTokenizer([1]))
    model.config.route_around_failing_endpoints = True
    model.config.endpoint_failure_threshold = 1
    clients = [sampled_client(), sampled_client()]
    for i, client in enumerate(clients):
        client.base_url = f"http://worker-{i}"
    model._clients = clients
    request = session_request()
    body = NeMoGymChatCompletionCreateParamsNonStreaming(messages=[{"role": "user", "content": "first"}])
    chosen = model._hashed_client("rollout")
    fallback = next(client for client in clients if client is not chosen)
    if failure == "connection":
        error = ClientConnectionError("unreachable")
    else:
        error = ClientResponseError(
            request_info=MagicMock(),
            history=(),
            status=400 if failure == "bad request" else 500,
            message="context length" if failure == "overflow" else "failed",
        )
    chosen.create_generate = AsyncMock(side_effect=error)
    if failure == "overflow":
        result = await model.chat_completions(request, body)
        assert result.choices[0].finish_reason == "length"
    else:
        with pytest.raises(type(error)):
            await model.chat_completions(request, body)
    if failure in ("overflow", "bad request"):
        assert not model._endpoint_failed(chosen)
        return
    assert model._endpoint_failed(chosen)
    await model.chat_completions(request, body)
    assert len(fallback.calls) == 1
    model._endpoint_health[chosen.base_url].failed_at -= model.config.endpoint_retry_after_s + 1
    chosen.create_generate = AsyncMock(return_value=chosen.result)
    # A new session with the same hash probes the recovered endpoint.
    session_id = next(str(i) for i in range(100) if model._hashed_client(str(i)) is chosen)
    await model.chat_completions(session_request(session_id), body)
    chosen.create_generate.assert_awaited_once()
    assert not model._endpoint_failed(chosen)


async def test_overlapping_calls_cannot_replace_an_accepted_sample():
    client = sampled_client()
    model = make_model(tokenizer=FakeTokenizer([1]), client=client)
    request = session_request()
    body = NeMoGymChatCompletionCreateParamsNonStreaming(messages=[{"role": "user", "content": "first"}])
    entered, release = asyncio.Event(), asyncio.Event()

    async def delayed_generate(**kwargs):
        entered.set()
        await release.wait()
        return sampled_client([99]).result

    generate = client.create_generate
    client.create_generate = delayed_generate
    delayed = asyncio.create_task(model.chat_completions(request, body))
    try:
        await entered.wait()
        client.create_generate = generate
        accepted = await model.chat_completions(request, body)
    finally:
        release.set()
    with pytest.raises(HTTPException) as exc:
        await delayed
    assert exc.value.status_code == 409
    await model.chat_completions(
        request,
        NeMoGymChatCompletionCreateParamsNonStreaming(
            messages=[
                *body.messages,
                accepted.choices[0].message.model_dump(exclude_none=True),
                {"role": "user", "content": "next"},
            ],
        ),
    )
    assert client.calls[-1]["input_ids"] == [1, 20, 21, 90, 91, 30, 31]
