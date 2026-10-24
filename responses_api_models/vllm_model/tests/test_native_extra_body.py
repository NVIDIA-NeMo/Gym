# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from contextlib import asynccontextmanager
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym.global_config import GlobalConfigDictParserConfig, get_global_config_dict
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import (
    ServerClient,
    get_global_aiohttp_client,
    global_aiohttp_client_exit,
    is_global_aiohttp_client_setup,
)
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


@pytest.fixture(autouse=True)
def reset_http_client():
    yield
    global_aiohttp_client_exit()


@pytest.fixture
def backend():
    captured = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            captured.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            payload = json.dumps(
                {
                    "id": "resp-test",
                    "created_at": 1,
                    "model": "model-test",
                    "object": "response",
                    "parallel_tool_calls": False,
                    "tool_choice": "auto",
                    "tools": [],
                    "status": "completed",
                    "output": [
                        {
                            "id": "msg-test",
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": "ok", "annotations": []}],
                        }
                    ],
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", captured
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def make_server(base_url, **kwargs):
    get_global_config_dict(
        global_config_dict_parser_config=GlobalConfigDictParserConfig(dotenv_path=None, skip_load_from_cli=True)
    )
    return VLLMModel(
        config=VLLMModelConfig(
            host="127.0.0.1",
            port=19000,
            name="model_test",
            entrypoint="app.py",
            base_url=base_url,
            api_key="EMPTY",
            model="model-test",
            is_responses_native=True,
            return_token_id_information=False,
            uses_reasoning_parser=False,
            **kwargs,
        ),
        server_client=ServerClient.model_construct(global_config_dict={}),
    )


def make_app(server):
    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            if is_global_aiohttp_client_setup():
                await get_global_aiohttp_client().close()

    app = FastAPI(lifespan=lifespan)
    server.setup_session_middleware(app)
    app.post("/v1/responses")(server.responses)
    return app


@pytest.mark.parametrize("metadata", [None, {}, {"extra_body": ""}, {"extra_body": "{}"}])
def test_native_default_keeps_request_and_has_no_budget(backend, metadata):
    base_url, captured = backend
    server = make_server(base_url)
    body = {"input": "hello", "max_output_tokens": 8192, "temperature": 0.3}
    if metadata is not None:
        body["metadata"] = metadata
    with TestClient(make_app(server)) as client:
        response = client.post("/v1/responses", json=body)
        assert response.status_code == 200, response.text
    assert captured == [("/v1/responses", {**body, "model": "model-test"})]
    assert "thinking_token_budget" not in captured[0][1]


@pytest.mark.parametrize("extension", [{"thinking_token_budget": 17}, {"generic_extension": {"enabled": True}}])
@pytest.mark.parametrize("config_extension", [None, {"thinking_token_budget": 29, "config_extension": True}])
def test_native_forwards_request_extensions_without_persistent_mutation(backend, extension, config_extension):
    base_url, captured = backend
    server = make_server(base_url, extra_body=deepcopy(config_extension))
    body = {"input": "hello", "max_output_tokens": 8192, "metadata": {"extra_body": json.dumps(extension)}}
    body_before = deepcopy(body)
    with TestClient(make_app(server)) as client:
        response = client.post("/v1/responses", json=body)
        assert response.status_code == 200, response.text
        assert captured[-1] == (
            "/v1/responses",
            {**(config_extension or {}), **extension, **body, "model": "model-test"},
        )
        response = client.post("/v1/responses", json={"input": "again", "max_output_tokens": 8192})
        assert response.status_code == 200, response.text
        assert captured[-1][1].get("thinking_token_budget") == (config_extension or {}).get("thinking_token_budget")
    assert body == body_before
    assert server.config.extra_body == config_extension


def test_native_explicit_fields_win_over_extensions_and_sampling_pins_win_last(backend):
    base_url, captured = backend
    server = make_server(
        base_url,
        extra_body={"max_output_tokens": 101, "temperature": 0.1, "top_p": 0.1},
        sampling_overrides={"top_p": 0.7},
    )
    body = {
        "input": "hello",
        "model": "caller-model",
        "max_output_tokens": 8192,
        "temperature": 0.3,
        "top_p": 0.3,
        "metadata": {"extra_body": json.dumps({"max_output_tokens": 202, "temperature": 0.2, "top_p": 0.2})},
    }
    with TestClient(make_app(server)) as client:
        response = client.post("/v1/responses", json=body)
        assert response.status_code == 200, response.text
    assert captured[0][1] == {**body, "model": "model-test", "top_p": 0.7}


@pytest.mark.parametrize("raw", ["{", "null", "3", '["invalid"]', "true", '"text"'])
def test_native_malformed_extensions_follow_existing_chat_failure(backend, raw):
    base_url, captured = backend
    server = make_server(base_url)
    metadata = {"extra_body": raw}
    with pytest.raises((ValueError, TypeError)):
        server._preprocess_chat_completion_create_params(
            None, {"messages": [{"role": "user", "content": "hello"}], "metadata": metadata}
        )
    with TestClient(make_app(server), raise_server_exceptions=False) as client:
        response = client.post("/v1/responses", json={"input": "hello", "metadata": metadata})
        assert response.status_code == 500
    assert captured == []


def test_request_schema_keeps_forbidding_unknown_fields():
    with pytest.raises(ValidationError):
        NeMoGymResponseCreateParamsNonStreaming(input="hello", thinking_token_budget=17)


@pytest.mark.parametrize("raw", ["[]", '[["thinking_token_budget", 17]]', "false", "1.5"])
def test_native_requires_extension_json_object_without_forwarding(backend, raw):
    base_url, captured = backend
    server = make_server(base_url)
    with TestClient(make_app(server), raise_server_exceptions=False) as client:
        response = client.post("/v1/responses", json={"input": "hello", "metadata": {"extra_body": raw}})
        assert response.status_code == 500
    assert captured == []


def test_native_projected_fields_win_and_nested_config_stays_request_local(backend):
    base_url, captured = backend
    extension = {"nested": {"values": [1, 2]}, "chat_template_kwargs": {"enable_thinking": False}}
    config = deepcopy(extension)
    server = make_server(base_url, extra_body=config, chat_template_kwargs={"enable_thinking": True})
    body = {
        "input": "hello",
        "max_output_tokens": 8192,
        "metadata": {
            "extra_body": json.dumps(
                {"model": "override", "input": "override", "max_output_tokens": 17, "nested": {"values": [3]}}
            )
        },
    }
    typed = NeMoGymResponseCreateParamsNonStreaming.model_validate(deepcopy(body))
    before = typed.model_dump()
    with TestClient(make_app(server)) as client:
        response = client.post("/v1/responses", json=body)
        assert response.status_code == 200, response.text
        wire = captured[-1][1]
        assert wire["model"] == "model-test" and wire["input"] == "hello"
        assert wire["max_output_tokens"] == 8192
        assert wire["chat_template_kwargs"] == {"enable_thinking": True}
        assert wire["nested"] == {"values": [3]}
        response = client.post("/v1/responses", json={"input": "again"})
        assert response.status_code == 200, response.text
        assert captured[-1][1]["nested"] == {"values": [1, 2]}
    assert config == extension and server.config.extra_body == extension
    assert typed.model_dump() == before


@pytest.mark.parametrize("value", [{"thinking_token_budget": 17}, 17, None])
def test_native_request_metadata_value_must_remain_a_string(backend, value):
    base_url, captured = backend
    server = make_server(base_url)
    with TestClient(make_app(server), raise_server_exceptions=False) as client:
        response = client.post("/v1/responses", json={"input": "hello", "metadata": {"extra_body": value}})
        assert response.status_code == 422
    assert captured == []
