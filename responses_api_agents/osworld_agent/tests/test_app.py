# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the OSWorld responses-api agent.

Heavy dependencies (``ray``, ``desktop_env``) are mocked at the module
boundary so the suite runs on a login node without OSWorld installed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.osworld_agent.app import (
    OSWorldAgent,
    OSWorldAgentConfig,
    OSWorldRunRequest,
    OSWorldVerifyResponse,
    _normalize_chat_message,
    _resolve_policy_model_name,
)


DEFAULT_OSWORLD_TASK: Dict[str, Any] = {
    "id": "test-task-001",
    "instruction": "Open Chrome and enable Do Not Track.",
    "snapshot": "chrome",
    "config": [],
    "evaluator": {"func": "exact_match"},
    "related_apps": ["chrome"],
}

DEFAULT_RUN_RESULT: Dict[str, Any] = {
    "reward": 1.0,
    "score": 1.0,
    "finished": True,
    "error": None,
    "artifact_dir": "/tmp/osworld-artifacts/chrome/test-task-001",
    "steps": [
        {
            "step": 0,
            "model_text": "```python\npyautogui.click(100, 200)\n```",
            "actions": ["pyautogui.click(100, 200)"],
            "reward": 0.0,
            "done": False,
            "info": {},
        },
        {
            "step": 1,
            "model_text": "```DONE```",
            "actions": ["DONE"],
            "reward": 1.0,
            "done": True,
            "info": {},
        },
    ],
}


def test_omni_runtime_model_overrides_stale_global_provenance(monkeypatch, caplog) -> None:
    monkeypatch.setenv("OMNI_MINI_VLLM_MODEL", "nvidia/nemotron-3-nano-omni")
    monkeypatch.delenv("OSWORLD_POLICY_MODEL_NAME", raising=False)

    with caplog.at_level("WARNING"):
        resolved = _resolve_policy_model_name(
            {"policy_model_name": "azure/anthropic/claude-opus-4-7"},
            "omni_mini_agent",
        )

    assert resolved == "nvidia/nemotron-3-nano-omni"
    assert "stale global policy_model_name" in caplog.text


def test_non_omni_runner_keeps_configured_policy_model(monkeypatch) -> None:
    monkeypatch.setenv("OMNI_MINI_VLLM_MODEL", "nvidia/nemotron-3-nano-omni")
    monkeypatch.delenv("OSWORLD_POLICY_MODEL_NAME", raising=False)

    assert (
        _resolve_policy_model_name(
            {"policy_model_name": "nvidia/minimaxai/minimax-m3"},
            "m3_agent",
        )
        == "nvidia/minimaxai/minimax-m3"
    )


def test_normalize_chat_message_preserves_reasoning_and_native_tool_calls() -> None:
    message = SimpleNamespace(
        content="Action: Click the target.",
        tool_calls=[
            SimpleNamespace(
                function=SimpleNamespace(
                    name="computer_use",
                    arguments='{"action":"left_click","coordinate":[500,250]}',
                )
            )
        ],
        model_extra={"reasoning_content": "Inspect the screenshot."},
    )

    normalized = _normalize_chat_message(message, structured=True)

    assert normalized["reasoning_content"] == "Inspect the screenshot."
    assert "<tool_call>" in normalized["content"]
    assert '"name": "computer_use"' in normalized["content"]
    assert '"coordinate": [500, 250]' in normalized["content"]


def test_normalize_chat_message_recovers_vllm_wrapped_reasoning() -> None:
    message = SimpleNamespace(
        content="<think>\nInspect the screenshot.\n</think>## Action:\nClick.\n## Code:\n```python\npass\n```",
        tool_calls=[],
        model_extra={},
    )

    normalized = _normalize_chat_message(message, structured=True)

    assert normalized["reasoning_content"] == "Inspect the screenshot."
    assert normalized["content"].startswith("## Action:")
    assert "<think>" not in normalized["content"]


def test_normalize_chat_message_extracts_one_text_part() -> None:
    message = SimpleNamespace(
        content=[
            {
                "type": "text",
                "text": "## Action:\nClick.\n## Code:\n```python\npyautogui.click(1, 2)\n```",
            }
        ],
        tool_calls=[],
        model_extra={},
    )

    normalized = _normalize_chat_message(message, structured=True)

    assert normalized["content"].startswith("## Action:")
    assert normalized["content"].endswith("pyautogui.click(1, 2)\n```")


def test_normalize_chat_message_selects_first_action_from_multiple_text_parts() -> None:
    message = SimpleNamespace(
        content=[
            {
                "type": "text",
                "text": "Click the first target.\n## Code:\n```python\npyautogui.click(1, 2)\n```",
            },
            {
                "type": "text",
                "text": "Finish.\n## Code:\n```python\ncomputer.terminate(status='success')\n```",
            },
        ],
        tool_calls=[],
        model_extra={},
    )

    normalized = _normalize_chat_message(message, structured=True)

    assert normalized["content"].startswith("## Action:\nClick the first target.")
    assert "pyautogui.click(1, 2)" in normalized["content"]
    assert "computer.terminate" not in normalized["content"]


def test_normalize_chat_message_recovers_serialized_text_parts() -> None:
    parts = [
        {
            "type": "text",
            "text": "Click the first target.\n## Code:\n```python\npyautogui.click(1, 2)\n```",
        },
        {
            "type": "text",
            "text": "Finish.\n## Code:\n```python\ncomputer.terminate(status='success')\n```",
        },
    ]
    message = SimpleNamespace(content=repr(parts), tool_calls=[], model_extra={})

    normalized = _normalize_chat_message(message, structured=True)

    assert normalized["content"].startswith("## Action:\nClick the first target.")
    assert "pyautogui.click(1, 2)" in normalized["content"]
    assert "computer.terminate" not in normalized["content"]


def test_normalize_chat_message_recovers_nested_serialized_text_parts() -> None:
    inner_parts = [
        {
            "type": "text",
            "text": "Click the first target.\n## Code:\n```python\npyautogui.click(1, 2)\n```",
        },
        {
            "type": "text",
            "text": "Finish.\n## Code:\n```python\ncomputer.terminate(status='success')\n```",
        },
    ]
    outer_parts = [{"type": "text", "text": repr(inner_parts)}]
    message = SimpleNamespace(content=outer_parts, tool_calls=[], model_extra={})

    normalized = _normalize_chat_message(message, structured=True)

    assert normalized["content"].startswith("## Action:\nClick the first target.")
    assert "pyautogui.click(1, 2)" in normalized["content"]
    assert "computer.terminate" not in normalized["content"]


def test_normalize_chat_message_recovers_serialized_parts_after_think_wrapper() -> None:
    parts = [
        {
            "type": "text",
            "text": "Click the first target.\n## Code:\n```python\npyautogui.click(1, 2)\n```",
        },
        {
            "type": "text",
            "text": "Finish.\n## Code:\n```python\ncomputer.terminate(status='success')\n```",
        },
    ]
    content = "<think>\nInspect the screenshot.\n</think>" + repr(parts)
    message = SimpleNamespace(content=content, tool_calls=[], model_extra={})

    normalized = _normalize_chat_message(message, structured=True)

    assert normalized["reasoning_content"] == "Inspect the screenshot."
    assert normalized["content"].startswith("## Action:\nClick the first target.")
    assert "pyautogui.click(1, 2)" in normalized["content"]
    assert "computer.terminate" not in normalized["content"]


def test_normalize_chat_message_recovers_malformed_serialized_parts() -> None:
    malformed = (
        "[{'type': 'text', 'text': 'Click user's target.\\n## Code:\\n"
        "```python\\npyautogui.click(1, 2)\\n```'},"
        " {'type': 'text', 'text': 'Finish.\\n## Code:\\n"
        '```python\\ncomputer.terminate(status=\\"success\\")\\n```\'}]'
    )
    content = "<think>\nInspect the screenshot.\n</think>" + malformed
    message = SimpleNamespace(content=content, tool_calls=[], model_extra={})

    normalized = _normalize_chat_message(message, structured=True)

    assert normalized["reasoning_content"] == "Inspect the screenshot."
    assert normalized["content"].startswith("## Action:\nExecute the first generated action.")
    assert "pyautogui.click(1, 2)" in normalized["content"]
    assert "computer.terminate" not in normalized["content"]


def test_normalize_chat_message_recovers_double_escaped_apostrophe() -> None:
    malformed = (
        "[{'type': 'text', 'text': 'GIMP\\\\'s theme is light.\\n## Action:\\n"
        "Finish.\\n## Code:\\n```code\\n"
        'computer.terminate(status=\\"success\\")\\n```\\n\'}]'
    )
    message = SimpleNamespace(content=malformed, tool_calls=[], model_extra={})

    normalized = _normalize_chat_message(message, structured=True)

    assert not normalized["content"].startswith("[")
    assert normalized["content"].startswith("## Action:\nExecute the first generated action.")
    assert "computer.terminate" in normalized["content"]


def test_normalize_chat_message_recovers_truncated_serialized_parts() -> None:
    malformed = (
        "[{'type': 'text', 'text': 'GIMP\\\\'s theme is light.\\n## Action:\\n"
        "Finish.\\n## Code:\\n```code\\n"
        'computer.terminate(status=\\"success\\")\\n```\\n\'}'
    )
    message = SimpleNamespace(content=malformed, tool_calls=[], model_extra={})

    normalized = _normalize_chat_message(message, structured=True)

    assert not normalized["content"].startswith("[")
    assert normalized["content"].startswith("## Action:\nExecute the first generated action.")
    assert "computer.terminate" in normalized["content"]


def test_normalize_chat_message_recovers_action_after_serialized_prefix() -> None:
    malformed = (
        "[{'type': 'text', 'text': \"The task is complete.\"}]\n"
        "## Action:\nMark the task as successfully completed.\n"
        "## Code:\n```code\ncomputer.terminate(status='success')\n```"
    )
    message = SimpleNamespace(content=malformed, tool_calls=[], model_extra={})

    normalized = _normalize_chat_message(message, structured=True)

    assert not normalized["content"].startswith("[")
    assert normalized["content"].startswith("## Action:\nExecute the first generated action.")
    assert "computer.terminate" in normalized["content"]


def make_config(**overrides: Any) -> OSWorldAgentConfig:
    base: Dict[str, Any] = dict(
        name="osworld_agent",
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        concurrency=1,
        provider_name="docker",
        headless=True,
        screen_width=1280,
        screen_height=800,
        require_a11y_tree=False,
        client_password="password",  # pragma: allowlist secret
        max_steps=3,
        max_trajectory_length=3,
        sleep_after_execution=0.0,
        cache_dir="cache",
        max_tokens=512,
        temperature=1.0,
        top_p=0.9,
    )
    base.update(overrides)
    return OSWorldAgentConfig(**base)


def make_run_request(
    osworld_task: Optional[Dict[str, Any]] = None,
    *,
    extra_metadata: Optional[Dict[str, Any]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> OSWorldRunRequest:
    metadata: Dict[str, Any] = {"task_id": "test-task-001", "domain": "chrome"}
    if osworld_task is not None:
        metadata["osworld_task"] = osworld_task
    if extra_metadata:
        metadata.update(extra_metadata)
    return OSWorldRunRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input=[], temperature=temperature, top_p=top_p
        ),
        verifier_metadata=metadata,
    )


def setup_server_client_mocks(mock_load_from_global_config, mock_get_first_server_config_dict):
    mock_client = MagicMock()
    mock_client.global_config_dict = {
        "policy_model_name": "test-policy",
        "policy_api_key": "test-key",  # pragma: allowlist secret
    }
    mock_load_from_global_config.return_value = mock_client
    mock_get_first_server_config_dict.return_value = {"host": "127.0.0.1", "port": 8000}


class TestApp:
    def test_sanity(self) -> None:
        OSWorldAgent(config=make_config(), server_client=MagicMock(spec=ServerClient))

    async def test_responses_not_implemented(self) -> None:
        agent = OSWorldAgent(config=make_config(), server_client=MagicMock(spec=ServerClient))
        with pytest.raises(NotImplementedError):
            await agent.responses(NeMoGymResponseCreateParamsNonStreaming(input=[], temperature=1.0, top_p=0.9))

    def test_endpoints_registration(self) -> None:
        agent = OSWorldAgent(config=make_config(), server_client=MagicMock(spec=ServerClient))
        app = agent.setup_webserver()
        client = TestClient(app, raise_server_exceptions=False)

        # /v1/responses raises NotImplementedError -> 500 (not 404).
        resp = client.post("/v1/responses", json={"input": [], "temperature": 1.0, "top_p": 0.9})
        assert resp.status_code == 500

        # /run is registered (anything other than 404 satisfies registration).
        run_resp = client.post("/run", json={})
        assert run_resp.status_code != 404

    async def test_run_missing_task_returns_empty_response(self) -> None:
        agent = OSWorldAgent(config=make_config(), server_client=MagicMock(spec=ServerClient))
        request = make_run_request(osworld_task=None)  # no osworld_task -> short-circuit
        response = await agent.run(request)
        assert isinstance(response, OSWorldVerifyResponse)
        assert response.reward == 0.0
        assert "osworld_error" in response.verifier_metadata
        assert "No 'osworld_task'" in response.verifier_metadata["osworld_error"]

    @patch("responses_api_agents.osworld_agent.app.ServerClient.load_from_global_config")
    @patch("responses_api_agents.osworld_agent.app.get_first_server_config_dict")
    @patch("responses_api_agents.osworld_agent.app._run_osworld_task_remote")
    @patch("asyncio.to_thread")
    async def test_run_successful_execution(
        self,
        mock_to_thread,
        mock_remote,
        mock_get_first_server_config_dict,
        mock_load_from_global_config,
    ) -> None:
        setup_server_client_mocks(mock_load_from_global_config, mock_get_first_server_config_dict)

        # Mock the Ray-remote ``.options(...).remote(...)`` call chain.
        future = MagicMock()
        mock_remote.options.return_value.remote.return_value = future
        mock_to_thread.return_value = DEFAULT_RUN_RESULT

        agent = OSWorldAgent(config=make_config(), server_client=MagicMock(spec=ServerClient))
        request = make_run_request(osworld_task=DEFAULT_OSWORLD_TASK, temperature=0.7, top_p=0.95)

        response = await agent.run(request)

        assert isinstance(response, OSWorldVerifyResponse)
        assert response.reward == 1.0
        assert response.verifier_metadata["osworld_score"] == 1.0
        assert response.verifier_metadata["osworld_finished"] is True
        assert response.verifier_metadata["osworld_error"] is None
        assert response.verifier_metadata["osworld_artifact_dir"] == DEFAULT_RUN_RESULT["artifact_dir"]
        assert response.verifier_metadata["osworld_model_name"] == "test-policy"
        assert response.response.model == "test-policy"
        # Two model steps -> two output messages. ``response.response`` is a
        # ``NeMoGymResponse`` Pydantic model (coerced from the dict in app.py),
        # so use attribute access.
        assert len(response.response.output) == 2
        # Per-request overrides win over the agent default.
        assert response.response.temperature == 0.7
        assert response.response.top_p == 0.95
        # Ray remote was dispatched exactly once with our task spec.
        mock_remote.options.assert_called_once()
        mock_remote.options.return_value.remote.assert_called_once()
        positional_args, _ = mock_remote.options.return_value.remote.call_args
        assert positional_args[0] == DEFAULT_OSWORLD_TASK
        assert positional_args[1]["evaluator_disable_gpu"] is True

    @patch("responses_api_agents.osworld_agent.app.ServerClient.load_from_global_config")
    @patch("responses_api_agents.osworld_agent.app.get_first_server_config_dict")
    @patch("responses_api_agents.osworld_agent.app._run_osworld_task_remote")
    @patch("asyncio.to_thread")
    async def test_run_ray_failure_returns_empty_response(
        self,
        mock_to_thread,
        mock_remote,
        mock_get_first_server_config_dict,
        mock_load_from_global_config,
    ) -> None:
        setup_server_client_mocks(mock_load_from_global_config, mock_get_first_server_config_dict)
        mock_remote.options.return_value.remote.return_value = MagicMock()
        mock_to_thread.side_effect = RuntimeError("docker daemon unreachable")

        agent = OSWorldAgent(config=make_config(), server_client=MagicMock(spec=ServerClient))
        request = make_run_request(osworld_task=DEFAULT_OSWORLD_TASK)

        response = await agent.run(request)

        assert response.reward == 0.0
        assert "RuntimeError" in response.verifier_metadata["osworld_error"]
        assert "docker daemon unreachable" in response.verifier_metadata["osworld_error"]

    @patch("responses_api_agents.osworld_agent.app.ServerClient.load_from_global_config")
    @patch("responses_api_agents.osworld_agent.app.get_first_server_config_dict")
    @patch("responses_api_agents.osworld_agent.app._run_osworld_task_remote")
    @patch("asyncio.to_thread")
    async def test_run_partial_score_thresholds_to_zero(
        self,
        mock_to_thread,
        mock_remote,
        mock_get_first_server_config_dict,
        mock_load_from_global_config,
    ) -> None:
        """Score < 1.0 -> reward 0.0 (matches gym's 0/1 reward convention)."""
        setup_server_client_mocks(mock_load_from_global_config, mock_get_first_server_config_dict)
        mock_remote.options.return_value.remote.return_value = MagicMock()
        mock_to_thread.return_value = {
            "reward": 0.0,
            "score": 0.4,
            "finished": False,
            "error": None,
            "steps": [],
        }

        agent = OSWorldAgent(config=make_config(), server_client=MagicMock(spec=ServerClient))
        response = await agent.run(make_run_request(osworld_task=DEFAULT_OSWORLD_TASK))

        assert response.reward == 0.0
        assert response.verifier_metadata["osworld_score"] == 0.4
        assert response.verifier_metadata["osworld_finished"] is False
