# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from requests.exceptions import HTTPError, Timeout

from responses_api_models.local_vllm_model import app


@pytest.mark.parametrize("first_error", [None, HTTPError("503"), Timeout("warming up")])
def test_readiness_checks_unauthenticated_health_and_retries(monkeypatch, first_error):
    response = Mock()
    response.raise_for_status.side_effect = [first_error, None] if first_error else None
    get = Mock(return_value=response)
    sleep = Mock()
    monkeypatch.setattr(app.requests, "get", get)
    monkeypatch.setattr(app.ray, "get", lambda result: True)
    monkeypatch.setattr(app, "sleep", sleep)
    model = SimpleNamespace(
        config=SimpleNamespace(name="test", base_url=["http://127.0.0.1:1234/v1"], api_key="configured-test-key"),
        _local_vllm_model_actor=SimpleNamespace(is_alive=SimpleNamespace(remote=lambda: True)),
    )
    app.LocalVLLMModel.await_server_ready(model)
    for call in get.call_args_list:
        assert call.kwargs == {"url": "http://127.0.0.1:1234/health", "timeout": 5}
    assert get.call_count == (2 if first_error else 1)
    assert sleep.call_count == (1 if first_error else 0)


def test_readiness_rejects_dead_worker(monkeypatch):
    monkeypatch.setattr(app.ray, "get", lambda result: False)
    get = Mock()
    monkeypatch.setattr(app.requests, "get", get)
    model = SimpleNamespace(
        config=SimpleNamespace(name="test"),
        _local_vllm_model_actor=SimpleNamespace(is_alive=SimpleNamespace(remote=lambda: False)),
    )
    with pytest.raises(AssertionError, match="spinup failed"):
        app.LocalVLLMModel.await_server_ready(model)
    get.assert_not_called()
