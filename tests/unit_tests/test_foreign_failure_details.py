# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle

import pytest
from aiohttp import ClientResponseError, ClientSession, web
from aiohttp.test_utils import TestServer
from pydantic import ValidationError

from nemo_gym.episode_types import ForeignErrorDetails
from nemo_gym.server_utils import raise_for_status
from nemo_gym.typed_exceptions import (
    FAILURE_EVENT_HEADER,
    ModelFailure,
    boundary_failure,
    encode_header,
    failure_events,
)


@pytest.mark.parametrize("status", [401, 500, None])
async def test_selected_foreign_details_survive_real_http_and_pickle_without_kind_inference(status):
    details = ForeignErrorDetails(exception_type="sdk.ApiError", message="authentication failed", http_status=status)
    original = ModelFailure(message="model request failed", operation="model.request", foreign_error=details)
    before = failure_events(original)

    async def endpoint(request):
        return web.json_response(
            {"error": "upstream unavailable"},
            status=503,
            headers={FAILURE_EVENT_HEADER: encode_header(boundary_failure(original, server="model"))},
        )

    app = web.Application()
    app.router.add_get("/failure", endpoint)
    async with TestServer(app) as server, ClientSession() as client:
        async with client.get(server.make_url("/failure")) as response:
            with pytest.raises(ClientResponseError) as caught:
                await raise_for_status(response)
    assert caught.value.status == 503
    rebuilt = caught.value.__cause__
    assert rebuilt.event_id == original.event_id
    assert rebuilt.foreign_error == details
    assert rebuilt.failure_kind is None
    assert failure_events(rebuilt) == before
    assert pickle.loads(pickle.dumps(rebuilt)).foreign_error == details


def test_foreign_details_are_optional_selected_and_bounded():
    original = ModelFailure(message="failed", operation="model.request")
    record = failure_events(original)[original.event_id]
    assert "foreign_error" not in record.model_dump(mode="json")
    with pytest.raises(ValidationError):
        ForeignErrorDetails(exception_type="sdk.ApiError", message="x" * 2001)
    with pytest.raises(ValidationError):
        ForeignErrorDetails(exception_type="sdk.ApiError", message="selected", request={"secret": "not accepted"})
    with pytest.raises(ValidationError):
        ModelFailure(message="failed", foreign_error=RuntimeError("not serializable data"))


@pytest.mark.parametrize("status", [401, 500, None])
async def test_judge_captures_selected_sdk_fields_before_translation_and_forwards_them(status):
    from types import SimpleNamespace

    from nemo_gym.judge import JudgeCallFailed, _judge_error

    sdk_error = ClientResponseError(
        SimpleNamespace(real_url="http://provider.invalid/judge"), (), status=status, message="authentication failed"
    )
    sdk_error.request_payload = {"secret": "must not be serialized"}
    original = _judge_error(sdk_error)
    original.__cause__ = sdk_error
    assert isinstance(original, JudgeCallFailed)
    assert original.foreign_error.exception_type == "aiohttp.client_exceptions.ClientResponseError"
    assert original.foreign_error.message == "authentication failed"
    assert original.foreign_error.http_status == status
    assert original.failure_kind == "judge_failed"
    event = failure_events(original)[original.event_id]
    assert event.foreign_error == original.foreign_error
    assert "must not be serialized" not in event.model_dump_json()

    async def endpoint(request):
        return web.json_response(
            {"error": "upstream unavailable"},
            status=503,
            headers={FAILURE_EVENT_HEADER: encode_header(boundary_failure(original, server="judge"))},
        )

    app = web.Application()
    app.router.add_get("/failure", endpoint)
    async with TestServer(app) as server, ClientSession() as client:
        async with client.get(server.make_url("/failure")) as response:
            with pytest.raises(ClientResponseError) as caught:
                await raise_for_status(response)
    assert caught.value.status == 503
    rebuilt = caught.value.__cause__
    assert rebuilt.event_id == original.event_id
    assert rebuilt.failure_kind == "judge_failed"
    assert rebuilt.foreign_error == original.foreign_error
    from nemo_gym.typed_exceptions import http_status

    assert http_status(rebuilt.__cause__) == status
    assert failure_events(rebuilt) == failure_events(original)
    assert pickle.loads(pickle.dumps(rebuilt)).foreign_error == original.foreign_error


def test_selected_sdk_fields_are_bounded_without_serializing_arbitrary_exception_text():
    from nemo_gym.typed_exceptions import foreign_error_details

    sdk_error = ClientResponseError(None, (), status=500, message="x" * 2001)
    selected = foreign_error_details(sdk_error)
    assert len(selected.message) == 2000
    assert selected.message.endswith(" [truncated]")
    assert foreign_error_details(RuntimeError("private request data")).message == ""


def test_selected_scalar_constructor_message_is_bounded_and_invalid_values_do_not_hide_failure():
    from nemo_gym.typed_exceptions import foreign_error_details

    error = RuntimeError("constructor message")
    assert foreign_error_details(error).message == ""
    selected = foreign_error_details(error, message=error.args[0])
    assert selected.message == "constructor message"
    error.message = {"request": "must not be serialized"}
    assert foreign_error_details(error).message == ""
    long_type = type("SdkError", (RuntimeError,), {"__qualname__": "SdkError" + "x" * 300})
    selected = foreign_error_details(long_type("failed"))
    assert len(selected.exception_type) == 256 and selected.exception_type.endswith(" [truncated]")
