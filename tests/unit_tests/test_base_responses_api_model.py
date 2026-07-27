# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pathlib import Path
from unittest.mock import MagicMock

from fastapi import Body, FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.testclient import TestClient
from omegaconf import OmegaConf

from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModel,
    BaseResponsesAPIModelConfig,
    ModelCallRecord,
    SimpleResponsesAPIModel,
)
from nemo_gym.capture_records import CaptureStore
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import (
    ServerClient,
)


TEST_ROLLOUT_ID = "my-test-rollout-id"


def _create_test_model_call_record() -> ModelCallRecord:
    return ModelCallRecord(
        status_code=200,
        route="my-test-route",
        timestamp_start=0.0,
        timestamp_end=0.0,
        model_ref=ModelServerRef(type="responses_api_models", name="my-server-name"),
        request=NeMoGymResponseCreateParamsNonStreaming(
            input=[
                NeMoGymEasyInputMessage(
                    role="user",
                    content=[{"type": "input_text", "text": "hello"}],
                    type="message",
                ),
            ]
        ),
        response=NeMoGymResponse(
            id="resp_123",
            created_at=1691418000,
            model="dummy_model",
            tools=[],
            parallel_tool_calls=True,
            tool_choice="auto",
            output=[
                NeMoGymResponseOutputMessage(
                    id="msg_123",
                    role="assistant",
                    status="completed",
                    type="message",
                    content=[
                        NeMoGymResponseOutputText(
                            text="hi :) how are you?",
                            type="output_text",
                            annotations=[],
                        )
                    ],
                )
            ],
            object="response",
        ),
        error_response=None,
        raw_request=dict(),
        raw_response=dict(),
    )


class _TestSimpleResponsesAPIModel(SimpleResponsesAPIModel):
    async def chat_completions(self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()):
        raise NotImplementedError

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()):
        return _create_test_model_call_record().response


class TestBaseResponsesAPIModel:
    def test_BaseResponsesAPIModel(self) -> None:
        config = BaseResponsesAPIModelConfig(host="", port=0, openai_api_key="123", entrypoint="", name="")
        BaseResponsesAPIModel(config=config)
        assert "should_capture_calls" not in BaseResponsesAPIModelConfig.model_fields
        assert "call_capture_dir" not in BaseResponsesAPIModelConfig.model_fields

    def test_SimpleResponsesAPIModel(self) -> None:
        config = BaseResponsesAPIModelConfig(host="", port=0, openai_api_key="123", entrypoint="", name="")

        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {}
        model = _TestSimpleResponsesAPIModel(config=config, server_client=server_client)
        model.setup_webserver()

    def _create_test_app_with_model_call_capture(
        self,
        tmp_path: Path,
        TestSimpleResponsesAPIModel_cls: type[SimpleResponsesAPIModel] = _TestSimpleResponsesAPIModel,
        should_capture_calls: bool = True,
    ) -> FastAPI:
        mock_server_client = MagicMock(spec=ServerClient)
        mock_server_client.global_config_dict = OmegaConf.create(
            {"should_capture_calls": should_capture_calls, "call_capture_dir": str(tmp_path)}
        )

        model_server = TestSimpleResponsesAPIModel_cls(
            config=BaseResponsesAPIModelConfig(
                host="",
                port=0,
                entrypoint="",
                name="my-test-server-name",
            ),
            server_client=mock_server_client,
        )

        return model_server.setup_webserver()

    def test_raised_call_is_captured_then_reraised(self, tmp_path: Path):
        """A model call that raises (not just a non-2xx) is captured with an exception category and the
        error is re-raised (response unchanged for the caller)."""

        class MyTestSimpleResponsesAPIModel(_TestSimpleResponsesAPIModel):
            async def responses(self, body: dict = Body()) -> dict:
                raise RuntimeError("kaboom")

        app = self._create_test_app_with_model_call_capture(tmp_path, MyTestSimpleResponsesAPIModel)

        client = TestClient(app, raise_server_exceptions=False)

        r = client.post("/v1/ng-rollout/r-raise/responses", json={"input": "x"})
        assert r.status_code == 500  # error propagated, response unchanged

        calls = CaptureStore(tmp_path).read("r-raise")
        assert len(calls) == 1
        assert calls[0].response is None
        assert calls[0].error_response is not None and "kaboom" in calls[0].error_response

    def test_per_rollout_url_prefix_correlates_and_is_openai_compatible(self, tmp_path: Path):
        """A caller attributes calls through the model base URL. The prefix is stripped before routing,
        while an ordinary unprefixed request remains unobserved."""
        app = self._create_test_app_with_model_call_capture(tmp_path)

        client = TestClient(app)

        # Prefixed base_url: routes to /v1/responses and correlates capture by the path id.
        r = client.post("/v1/ng-rollout/task7-roll2/responses", json={"input": "hi"})
        assert r.status_code == 200
        calls = CaptureStore(tmp_path).read("task7-roll2")
        assert len(calls) == 1

        # Prefixed base_url: routes to /v1/responses and correlates capture by the path id.
        r = client.post("/v1/responses/ng-rollout/task7-roll2/", json={"input": "hi"})
        assert r.status_code == 200
        calls = CaptureStore(tmp_path).read("task7-roll2")
        assert len(calls) == 2

        # Plain /v1 URL is routed normally but is not captured without an explicit rollout prefix.
        r2 = client.post("/v1/responses", json={"input": "hi"})
        assert r2.status_code == 200
        calls = CaptureStore(tmp_path).read("task7-roll2")
        assert len(calls) == 2

    def test_per_rollout_prefix_not_stripped_for_non_observed_paths_too(self, tmp_path: Path):
        """A prefixed but non-observed path (e.g. /v1/models) is not stripped and not found,
        and is not captured (composes with arbitrary endpoints, not just the observed dialects)."""
        app = self._create_test_app_with_model_call_capture(tmp_path)

        @app.get("/v1/models")
        async def _models() -> dict:
            return {"object": "list", "data": []}

        client = TestClient(app)

        r = client.get("/v1/ng-rollout/abc/models")
        assert r.status_code == 404
        assert CaptureStore(tmp_path).read("abc") == []

    def test_capture_records_non_json_response_as_error_doesnt_record(self, tmp_path: Path):
        class My_TestSimpleResponsesAPIModel(_TestSimpleResponsesAPIModel):
            async def responses(self, body: dict = Body()) -> PlainTextResponse:
                return PlainTextResponse("not json")

        app = self._create_test_app_with_model_call_capture(tmp_path, My_TestSimpleResponsesAPIModel)

        client = TestClient(app, raise_server_exceptions=False)

        r = client.post("/v1/ng-rollout/rnj/responses", json={"input": "x"})
        assert r.status_code == 500
        records = CaptureStore(tmp_path).read("rnj")
        assert len(records) == 0

    def test_rollout_prefix_not_stripped_when_capture_disabled(self, tmp_path: Path):
        app = self._create_test_app_with_model_call_capture(tmp_path, should_capture_calls=False)

        client = TestClient(app)
        # Dummy input to just test it doesn't 404
        assert client.post("/v1/responses", json={}).status_code == 422
        assert client.post("/v1/ng-rollout/3-0/responses", json={}).status_code == 404
