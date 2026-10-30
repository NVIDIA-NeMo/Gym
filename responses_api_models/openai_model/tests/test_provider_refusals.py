# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientResponseError

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_models.openai_model import app
from responses_api_models.openai_model.provider_refusals import provider_refusal


BODY = {"error": {"code": "bio_policy", "message": "Provider declined this request."}}
ALLOWED = frozenset({"bio_policy"})


def server(**kwargs):
    config = app.SimpleModelServerConfig(
        host="127.0.0.1",
        port=8081,
        entrypoint="",
        name="policy_model",
        openai_base_url="https://example.invalid/v1",
        openai_api_key="unused",  # pragma: allowlist secret
        openai_model="dummy_model",
        **kwargs,
    )
    return app.SimpleModelServer(config=config, server_client=MagicMock(spec=ServerClient, global_config_dict={}))


def failure(status, body):
    error = ClientResponseError(MagicMock(), (), status=status, message="upstream")
    error.response_content = json.dumps(body).encode()
    return error


@pytest.mark.parametrize(
    "status,codes,expected",
    [(400, ALLOWED, True), (400, frozenset(), False), (401, ALLOWED, False), (503, ALLOWED, False)],
)
def test_only_explicit_policy_refusal_codes(status, codes, expected):
    assert (provider_refusal(status, BODY, codes) is not None) is expected
    assert provider_refusal(status, {"error": {"code": "bad_input", "message": "Not a refusal"}}, codes) is None


def test_unstructured_error_code_is_not_a_refusal():
    assert provider_refusal(400, {"code": [], "message": "Unstructured error"}, ALLOWED) is None


def test_nested_litellm_escaped_envelope():
    quoted = json.dumps(BODY).replace('"', '\\"')
    outer = json.dumps(
        {"error": {"code": "400", "message": "OpenAIException - " + quoted + "\nNo fallback attempted."}}
    )
    assert provider_refusal(400, outer, ALLOWED) == BODY["error"]


async def test_provider_refusal_is_typed_and_keeps_unknown_usage():
    model = server(provider_refusal_codes=ALLOWED)
    model._client = SimpleNamespace(create_response=AsyncMock(side_effect=failure(400, BODY)))
    response = await model.responses(NeMoGymResponseCreateParamsNonStreaming(input="A benign test"))
    assert response.output[0].content[0].type == "refusal"
    assert response.output[0].content[0].refusal == BODY["error"]["message"]
    assert response.usage is None
    assert response.metrics["provider_http_refusal"]["http_status"] == 400
    assert response.metrics["provider_http_refusal"]["error_code"] == "bio_policy"
    assert json.loads(response.metrics["provider_http_refusal"]["raw_error_body"]) == BODY
    assert model._client.create_response.await_count == 1


@pytest.mark.parametrize(
    "status,codes,body",
    [
        (400, frozenset(), BODY),
        (400, ALLOWED, {"error": {"code": "bad_input", "message": "Invalid request"}}),
        (401, ALLOWED, BODY),
        (503, ALLOWED, BODY),
    ],
)
async def test_default_and_unrelated_errors_are_not_converted(status, codes, body):
    model = server(provider_refusal_codes=codes)
    model._client = SimpleNamespace(create_response=AsyncMock(side_effect=failure(status, body)))
    with pytest.raises(ClientResponseError):
        await model.responses(NeMoGymResponseCreateParamsNonStreaming(input="A benign test"))


def test_configuration_cannot_label_capacity_error_as_refusal():
    with pytest.raises(ValueError):
        server(provider_refusal_codes=["service_unavailable"])


@pytest.mark.parametrize("message", ["First line\nSecond line", r"literal \n", 'embedded "quotes"', r"C:\path\file"])
@pytest.mark.parametrize("encoding", ["direct", "quote_only", "json_string"])
def test_provider_message_escapes_are_preserved(message, encoding):
    body = {"error": {"code": "bio_policy", "message": message}}
    encoded = json.dumps(body)
    if encoding == "quote_only":
        encoded = encoded.replace('"', '\\"')
    elif encoding == "json_string":
        encoded = json.dumps(encoded)
    wire = json.dumps(
        {"error": {"code": "400", "message": "OpenAIException - " + encoded + "\nNo fallback."}}
    ).encode()
    assert provider_refusal(400, wire, ALLOWED) == body["error"]
