# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import pytest

from responses_api_models.vllm_loglikelihood.app import (
    LogLikelihoodRequest,
    VLLMLogLikelihoodConfig,
    VLLMLogLikelihoodModel,
)


@pytest.fixture
def setup(monkeypatch):
    client = MagicMock()
    client.create_tokenize = AsyncMock(side_effect=lambda **kw: {"tokens": list(kw["prompt"].encode())})

    async def completion(**kw):
        n = len(kw["prompt"])
        return {"choices": [{"logprobs": {"token_logprobs": [None] + [-0.5] * (n - 1) + [-999]}}]}

    client.create_completion = AsyncMock(side_effect=completion)
    config = VLLMLogLikelihoodConfig(
        name="model",
        host="127.0.0.1",
        port=12345,
        entrypoint="app.py",
        base_url="http://unused/v1",
        api_key="dummy",
        model="test",
        return_token_id_information=False,
        uses_reasoning_parser=False,
    )
    model = VLLMLogLikelihoodModel.model_construct(config=config, server_client=MagicMock(global_config_dict={}))
    monkeypatch.setattr(VLLMLogLikelihoodModel, "_resolve_client", lambda self, request: client)
    return model, client


@pytest.mark.asyncio
async def test_echo_only_continuation_tokens(setup):
    model, client = setup
    result = await model.loglikelihood(
        MagicMock(), LogLikelihoodRequest(context="Answer:", continuations=[" A", " BB"])
    )
    assert [s.logprob for s in result.scores] == [-1, -1.5]
    assert result.scores[0].continuation_token_ids == [32, 65]
    assert result.scores[0].context_tokens == 7
    for call in client.create_completion.call_args_list:
        assert call.kwargs["echo"] and call.kwargs["logprobs"] == 1
        assert call.kwargs["max_tokens"] == 1 and call.kwargs["temperature"] == 0
        assert call.kwargs["seed"] == 42 and isinstance(call.kwargs["prompt"][0], int)
    assert all(not c.kwargs["add_special_tokens"] for c in client.create_tokenize.call_args_list)


@pytest.mark.asyncio
async def test_context_whitespace_moves_to_continuation(setup):
    model, _ = setup
    result = await model.loglikelihood(MagicMock(), LogLikelihoodRequest(context="Answer: ", continuations=["A", "B"]))
    assert result.scores[0].context_tokens == 7
    assert result.scores[0].continuation_token_ids == [32, 65]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["missing", "short", "nan", "overflow", "boundary", "empty", "whitespace"])
async def test_invalid_likelihood_fails_instead_of_scoring_zero(setup, failure):
    model, client = setup
    body = LogLikelihoodRequest(context="Answer:", continuations=[" A", " B"])
    if failure == "missing":
        client.create_completion.side_effect = None
        client.create_completion.return_value = {"choices": [{}]}
    elif failure in {"short", "nan"}:
        client.create_completion.side_effect = None
        client.create_completion.return_value = {
            "choices": [
                {
                    "logprobs": {
                        "token_logprobs": [None] * 5
                        if failure == "short"
                        else [None] + [-1.0] * 6 + [float("nan"), -1.0, -1.0]
                    }
                }
            ]
        }
    elif failure == "overflow":
        model.config.max_context_tokens = 9
    elif failure == "boundary":
        client.create_tokenize.side_effect = lambda **kw: {"tokens": [len(kw["prompt"]), 1]}
    elif failure == "empty":
        body.continuations[0] = ""
    else:
        body.context = "   "
    with pytest.raises(ValueError):
        await model.loglikelihood(MagicMock(), body)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["empty_context_tokens", "empty_choices"])
async def test_empty_engine_payload_rejected(setup, failure):
    model, client = setup
    if failure == "empty_context_tokens":
        client.create_tokenize.side_effect = None
        client.create_tokenize.return_value = {"tokens": []}
    else:
        client.create_completion.side_effect = None
        client.create_completion.return_value = {"choices": []}
    with pytest.raises(ValueError):
        await model.loglikelihood(MagicMock(), LogLikelihoodRequest(context="x", continuations=[" A", " B"]))


def test_likelihood_route_is_added_to_native_model(setup, monkeypatch):
    from fastapi import FastAPI

    from responses_api_models.vllm_model.app import VLLMModel

    app = FastAPI()
    monkeypatch.setattr(VLLMModel, "setup_webserver", lambda self: app)
    assert setup[0].setup_webserver() is app
    assert any(route.path == "/loglikelihood" for route in app.routes)


@pytest.mark.asyncio
async def test_explicit_special_tokens_keep_continuation_offsets(setup):
    model, client = setup
    model.config.likelihood_add_special_tokens = True
    client.create_tokenize.side_effect = lambda **kw: {"tokens": [999] + list(kw["prompt"].encode())}
    result = await model.loglikelihood(
        MagicMock(), LogLikelihoodRequest(context="Answer:", continuations=[" A", " BB"])
    )
    assert [score.logprob for score in result.scores] == [-1, -1.5]
    assert result.scores[0].context_tokens == 8
    assert result.scores[0].continuation_token_ids == [32, 65]
    assert all(call.kwargs["add_special_tokens"] for call in client.create_tokenize.call_args_list)


@pytest.mark.asyncio
@pytest.mark.parametrize("trailing_whitespace", ["", "\n\n"])
async def test_chat_template_scores_choices_after_assistant_prefix(setup, trailing_whitespace):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    model, client = setup
    model.config.render_chat_template = True
    model.config.use_completions_api = True
    model._chat_template_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        bos_token="<bos>",
        eos_token="<eos>",
        chat_template=(
            "{{ bos_token }}{% for message in messages %}"
            "{{ '<' + message['role'] + '>' + message['content'] + eos_token }}"
            "{% endfor %}{% if add_generation_prompt %}{{ '<assistant>' + suffix }}{% endif %}"
        ),
    )
    model.config.chat_template_kwargs = {"suffix": trailing_whitespace}
    prompt = "Question: demo\nAnswer: B\n\nQuestion: test\nAnswer:"
    rendered_context = "<bos><user>" + prompt + "<eos><assistant>"
    result = await model.loglikelihood(
        MagicMock(), LogLikelihoodRequest(context=prompt, continuations=[" A", " B", " C", " D"])
    )
    assert client.create_tokenize.call_args_list[0].kwargs["prompt"] == rendered_context
    for score, call in zip(result.scores, client.create_completion.call_args_list, strict=True):
        continuation = trailing_whitespace + score.continuation
        assert bytes(call.kwargs["prompt"]).decode() == rendered_context + continuation
        assert score.context_tokens == len(rendered_context.encode())
        assert bytes(score.continuation_token_ids).decode() == continuation
        assert score.logprob == -0.5 * len(continuation.encode())
    assert all(not call.kwargs["add_special_tokens"] for call in client.create_tokenize.call_args_list)
