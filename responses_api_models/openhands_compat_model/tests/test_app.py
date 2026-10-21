# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import runpy
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock

from nemo_gym.server_utils import (
    IS_NEMO_GYM_FASTAPI_ENTRYPOINT_KEY_NAME,
    IS_NEMO_GYM_FASTAPI_WORKER_KEY_NAME,
    SimpleServer,
)
from responses_api_models.openhands_compat_model.app import normalize_openhands_tool_messages


def test_legacy_tool_name_removed_without_mutating_history_or_call_identity():
    request = {
        "messages": [
            {"role": "tool", "name": "shell", "tool_call_id": "abc", "content": "result"},
            {"role": "assistant", "name": "assistant", "content": "done"},
        ]
    }
    original = deepcopy(request)
    normalized = normalize_openhands_tool_messages(request)
    assert normalized["messages"][0] == {"role": "tool", "tool_call_id": "abc", "content": "result"}
    assert normalized["messages"][1] == original["messages"][1]
    assert request == original


def test_uvicorn_workers_import_a_module_level_app(monkeypatch):
    # With num_workers > 1, each uvicorn worker imports `responses_api_models.openhands_compat_model.app:app`.
    path, name = str(Path(__file__).parents[1] / "app.py"), "responses_api_models.openhands_compat_model.app"
    started = MagicMock(return_value=object())
    monkeypatch.setattr(SimpleServer, "run_webserver", started)
    for key in (IS_NEMO_GYM_FASTAPI_WORKER_KEY_NAME, IS_NEMO_GYM_FASTAPI_ENTRYPOINT_KEY_NAME):
        monkeypatch.delenv(key, raising=False)
    assert "app" not in runpy.run_path(path, run_name=name)
    started.assert_not_called()
    monkeypatch.setenv(IS_NEMO_GYM_FASTAPI_WORKER_KEY_NAME, "1")
    monkeypatch.setenv(IS_NEMO_GYM_FASTAPI_ENTRYPOINT_KEY_NAME, "responses_api_models/openhands_compat_model/app.py")
    assert runpy.run_path(path, run_name=name)["app"] is started.return_value
    started.assert_called_once_with()
