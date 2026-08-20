# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the OSWorld responses-api agent.

Heavy dependencies (``ray``, ``desktop_env``) are mocked at the module
boundary so the suite runs on a login node without OSWorld installed.
"""

from __future__ import annotations

import hashlib
import json
import os
from types import SimpleNamespace
from typing import Any, Dict, Literal, Optional
from unittest.mock import MagicMock, call, patch

import pytest
import ray
from fastapi.testclient import TestClient

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.osworld_agent.app import (
    OSWorldAgent,
    OSWorldAgentConfig,
    OSWorldRunRequest,
    OSWorldVerifyResponse,
    _append_model_io,
    _apply_sandbox_provider_overrides,
    _build_messages_model_fn,
    _build_response,
    _empty_response,
    _log_context_headers,
    _model_io_images,
    _normalize_chat_message,
    _resolve_policy_model_name,
    _resolve_run_rollout_purpose,
    _validate_runner_runtime,
)
from responses_api_agents.osworld_agent.trajectory import resolve_trajectory_identity


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


def test_full_model_io_writer_keeps_payload_and_indexes_images(monkeypatch, tmp_path) -> None:
    log_path = tmp_path / "model-io-agent.jsonl"
    monkeypatch.setenv("OSWORLD_MODEL_IO_LOG", str(log_path))
    data_url = "data:image/png;base64,YWJj"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": "inspect this image"},
            ],
        }
    ]
    image_index = _model_io_images(messages)

    _append_model_io(
        {
            "schema_version": 1,
            "event": "model_request",
            "openai_request": {"messages": messages},
            "embedded_images": image_index,
        }
    )

    row = json.loads(log_path.read_text(encoding="utf-8"))
    assert row["openai_request"]["messages"] == messages
    assert row["embedded_images"] == [
        {
            "message_index": 0,
            "part_index": 0,
            "data_url_chars": len(data_url),
            "encoded_sha256": hashlib.sha256(b"YWJj").hexdigest(),
            "decoded_bytes": 3,
            "decoded_sha256": hashlib.sha256(b"abc").hexdigest(),
        }
    ]


def test_log_context_headers_do_not_change_model_payload() -> None:
    context = {
        "run_id": "run-001",
        "adapter": "gym",
        "sampling_event_id": "sampling-training-001",
        "source_group_id": "dataset-group-001",
        "execution_id": "execution-001",
        "rollout_id": "rollout-001",
        "group_id": "group-001",
        "rollout_index": 4,
        "attempt_index": 2,
        "task_id": "task-001",
        "domain": "chrome",
        "task_attempt": 2,
        "step": 3,
        "parse_attempt": 1,
    }

    assert _log_context_headers(context) == {
        "x-nemo-gym-log-run-id": "run-001",
        "x-nemo-gym-log-adapter": "gym",
        "x-nemo-gym-log-sampling-event-id": "sampling-training-001",
        "x-nemo-gym-log-source-group-id": "dataset-group-001",
        "x-nemo-gym-log-execution-id": "execution-001",
        "x-nemo-gym-log-rollout-id": "rollout-001",
        "x-nemo-gym-log-group-id": "group-001",
        "x-nemo-gym-log-rollout-index": "4",
        "x-nemo-gym-log-attempt-index": "2",
        "x-nemo-gym-log-task-id": "task-001",
        "x-nemo-gym-log-domain": "chrome",
        "x-nemo-gym-log-task-attempt": "2",
        "x-nemo-gym-log-step": "3",
        "x-nemo-gym-log-parse-attempt": "1",
    }


@patch("openai.DefaultHttpxClient")
@patch("openai.OpenAI")
def test_messages_model_fn_propagates_task_context_in_headers_and_logs(
    mock_openai, mock_http_client, monkeypatch, tmp_path
) -> None:
    log_path = tmp_path / "model-io-agent.jsonl"
    monkeypatch.setenv("OSWORLD_MODEL_IO_LOG", str(log_path))
    message = SimpleNamespace(content="done", tool_calls=[], model_extra={})
    response = SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])
    client = mock_openai.return_value
    client.chat.completions.create.return_value = response
    call = _build_messages_model_fn(
        base_url="http://policy/v1",
        model_name="policy",
        api_key="test-key",  # pragma: allowlist secret
        log_context={
            "run_id": "run-001",
            "adapter": "gym",
            "rollout_id": "rollout-001",
            "group_id": "group-001",
            "rollout_index": 4,
            "attempt_index": 2,
            "task_id": "task-001",
        },
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "inspect"}]}]
    payload = {
        "model": "policy",
        "messages": messages,
        "max_tokens": 32,
        "temperature": 0.6,
        "_nemo_gym_return_message": True,
        "_osworld_log_context": {"step": 4, "parse_attempt": 2},
    }

    call(messages, payload)

    mock_http_client.assert_called_once_with(trust_env=False)
    mock_openai.assert_called_once_with(
        base_url="http://policy/v1",
        api_key="test-key",  # pragma: allowlist secret
        http_client=mock_http_client.return_value,
    )
    sent = client.chat.completions.create.call_args.kwargs
    assert sent["messages"] == messages
    assert "_osworld_log_context" not in sent
    assert sent["extra_headers"]["x-nemo-gym-log-task-id"] == "task-001"
    assert sent["extra_headers"]["x-nemo-gym-log-rollout-id"] == "rollout-001"
    assert sent["extra_headers"]["x-nemo-gym-log-group-id"] == "group-001"
    assert sent["extra_headers"]["x-nemo-gym-log-step"] == "4"
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [row["event"] for row in rows] == ["model_request", "model_response"]
    assert all(row["task_id"] == "task-001" for row in rows)
    assert all(row["rollout_id"] == "rollout-001" for row in rows)
    assert all(row["group_id"] == "group-001" for row in rows)
    assert all(row["step"] == 4 for row in rows)
    assert all(row["parse_attempt"] == 2 for row in rows)
    assert rows[0]["openai_request"] == {
        "model": "policy",
        "messages": messages,
        "max_tokens": 32,
        "temperature": 0.6,
    }


@patch("openai.DefaultHttpxClient")
@patch("openai.OpenAI")
def test_messages_model_fn_forwards_explicit_nemo_rl_rollout_purpose(mock_openai, mock_http_client) -> None:
    message = SimpleNamespace(content="done", tool_calls=[], model_extra={})
    client = mock_openai.return_value
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")]
    )
    call = _build_messages_model_fn(
        base_url="http://policy/v1",
        model_name="policy",
        api_key="test-key",  # pragma: allowlist secret
        rollout_purpose="evaluation",
    )
    messages = [{"role": "user", "content": "inspect"}]

    call(
        messages,
        {
            "model": "policy",
            "messages": messages,
            "max_tokens": 64,
            "temperature": 0.6,
        },
    )

    sent = client.chat.completions.create.call_args.kwargs
    assert json.loads(sent["metadata"]["extra_body"]) == {"nemo_rl_rollout_purpose": "evaluation"}


def test_omni_runtime_model_overrides_stale_global_provenance(monkeypatch, caplog) -> None:
    monkeypatch.setenv("NANO_OMNI_VLLM_MODEL", "nvidia/nemotron-3-nano-omni")
    monkeypatch.delenv("OSWORLD_POLICY_MODEL_NAME", raising=False)

    with caplog.at_level("WARNING"):
        resolved = _resolve_policy_model_name(
            {"policy_model_name": "azure/anthropic/claude-opus-4-7"},
            "nemotron_v3_nano_omni_agent",
        )

    assert resolved == "nvidia/nemotron-3-nano-omni"
    assert "stale global policy_model_name" in caplog.text


def test_non_omni_runner_keeps_configured_policy_model(monkeypatch) -> None:
    monkeypatch.setenv("NANO_OMNI_VLLM_MODEL", "nvidia/nemotron-3-nano-omni")
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


def test_normalize_chat_message_preserves_training_metadata_from_model_extra() -> None:
    raw_content = (
        "<think>Inspect the screenshot.</think>## Action:\nClick.\n## Code:\n```python\npyautogui.click(1, 2)\n```"
    )
    message = SimpleNamespace(
        content=raw_content,
        tool_calls=[],
        model_extra={
            "prompt_token_ids": [10, 11],
            "generation_token_ids": [20, 21],
            "generation_log_probs": [-0.1, -0.2],
        },
    )

    normalized = _normalize_chat_message(message, structured=True)

    assert normalized["raw_content"] == raw_content
    assert normalized["prompt_token_ids"] == [10, 11]
    assert normalized["generation_token_ids"] == [20, 21]
    assert normalized["generation_log_probs"] == [-0.1, -0.2]


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


def test_normalize_chat_message_recovers_serialized_click_part() -> None:
    parts = [
        {"type": "click", "x": 0.984, "y": 0.129},
        {
            "type": "text",
            "text": "## Action:\nClose the Chrome update notification.\n",
        },
    ]
    message = SimpleNamespace(content=repr(parts), tool_calls=[], model_extra={})

    normalized = _normalize_chat_message(message, structured=True)

    assert normalized["content"] == (
        "## Action:\nClose the Chrome update notification.\n## Code:\n```python\npyautogui.click(0.984, 0.129)\n```"
    )
    assert "normalization_error" not in normalized


def test_normalize_chat_message_recovers_action_click_after_think_wrapper() -> None:
    parts = [
        {
            "type": "action",
            "action": "click",
            "target": "close_tab",
            "input": {"x": 0.17, "y": 0.042},
        }
    ]
    raw_content = "<think>Close the unexpected tab.</think>\n" + repr(parts)
    message = SimpleNamespace(
        content=raw_content,
        tool_calls=[],
        model_extra={
            "prompt_token_ids": [10, 11],
            "generation_token_ids": [20, 21],
            "generation_log_probs": [-0.1, -0.2],
        },
    )

    normalized = _normalize_chat_message(message, structured=True)

    assert normalized["reasoning_content"] == "Close the unexpected tab."
    assert normalized["content"] == (
        "## Action:\nExecute the generated click action.\n## Code:\n```python\npyautogui.click(0.17, 0.042)\n```"
    )
    assert normalized["raw_content"] == raw_content
    assert normalized["generation_token_ids"] == [20, 21]
    assert "normalization_error" not in normalized


def test_structured_normalization_failure_preserves_exact_generation_evidence() -> None:
    raw_content = [{"type": "unsupported-native-action", "value": 7}]
    message = SimpleNamespace(
        content=raw_content,
        tool_calls=[],
        model_extra={
            "prompt_token_ids": [10, 11],
            "generation_token_ids": [20, 21],
            "generation_log_probs": [-0.1, -0.2],
            "routed_experts": "route-evidence",
        },
    )

    normalized = _normalize_chat_message(message, structured=True)

    assert normalized["content"] == repr(raw_content)
    assert normalized["raw_content"] == raw_content
    assert normalized["normalization_error"]["type"] == "ValueError"
    assert normalized["prompt_token_ids"] == [10, 11]
    assert normalized["generation_token_ids"] == [20, 21]
    assert normalized["generation_log_probs"] == [-0.1, -0.2]
    assert normalized["routed_experts"] == "route-evidence"


def test_post_think_normalization_failure_preserves_exact_generation_evidence() -> None:
    parts = [{"type": "unsupported-native-action", "value": 7}]
    raw_content = "<think>Inspect the screenshot.</think>\n" + repr(parts)
    message = SimpleNamespace(
        content=raw_content,
        tool_calls=[],
        model_extra={
            "prompt_token_ids": [10, 11],
            "generation_token_ids": [20, 21],
            "generation_log_probs": [-0.1, -0.2],
            "routed_experts": "route-evidence",
        },
    )

    normalized = _normalize_chat_message(message, structured=True)

    assert normalized["reasoning_content"] == "Inspect the screenshot."
    assert normalized["content"] == repr(parts)
    assert normalized["raw_content"] == raw_content
    assert normalized["normalization_error"]["type"] == "ValueError"
    assert normalized["prompt_token_ids"] == [10, 11]
    assert normalized["generation_token_ids"] == [20, 21]
    assert normalized["generation_log_probs"] == [-0.1, -0.2]
    assert normalized["routed_experts"] == "route-evidence"


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


def test_sandbox_provider_overrides_merge_only_selected_provider() -> None:
    provider = {
        "opensandbox": {
            "connection": {"domain": "sandbox.internal"},
            "create": {"timeout_s": 1500, "retries": 10},
        }
    }
    overrides = {
        "opensandbox": {"create": {"timeout_s": 180, "retries": 1}},
        "docker": {"create": {"start_timeout_s": 60}},
    }

    resolved = _apply_sandbox_provider_overrides(provider, overrides)

    assert resolved == {
        "opensandbox": {
            "connection": {"domain": "sandbox.internal"},
            "create": {"timeout_s": 180, "retries": 1},
        }
    }
    assert provider["opensandbox"]["create"] == {"timeout_s": 1500, "retries": 10}


def make_run_request(
    osworld_task: Optional[Dict[str, Any]] = None,
    *,
    extra_metadata: Optional[Dict[str, Any]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    rollout_purpose: Optional[Literal["training", "evaluation"]] = None,
) -> OSWorldRunRequest:
    metadata: Dict[str, Any] = {"task_id": "test-task-001", "domain": "chrome"}
    if osworld_task is not None:
        metadata["osworld_task"] = osworld_task
    if extra_metadata:
        metadata.update(extra_metadata)
    return OSWorldRunRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input=[],
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
        ),
        verifier_metadata=metadata,
        rollout_purpose=rollout_purpose,
    )


def test_resolve_run_rollout_purpose_accepts_metadata_carrier() -> None:
    request = make_run_request(rollout_purpose=None)
    request.responses_create_params.metadata = {"nemo_rl_rollout_purpose": "evaluation"}

    assert _resolve_run_rollout_purpose(request) == "evaluation"


def test_resolve_run_rollout_purpose_rejects_carrier_conflict() -> None:
    request = make_run_request(rollout_purpose="training")
    request.responses_create_params.metadata = {"nemo_rl_rollout_purpose": "evaluation"}

    with pytest.raises(ValueError, match="carriers disagree"):
        _resolve_run_rollout_purpose(request)


def test_build_response_always_emits_semantic_trajectory() -> None:
    request = make_run_request(osworld_task=DEFAULT_OSWORLD_TASK)
    response = _build_response(request, DEFAULT_RUN_RESULT, "test-policy", 1.0, 0.9)

    contract = response.response.trajectory_contract
    assert contract is not None
    assert contract["schema_version"] == 2
    assert contract["mode"] == "osworld_semantic_trajectory"
    assert contract["identity_source"] == "derived"
    assert contract["capabilities"]["semantic_trajectory"] is True
    assert contract["capabilities"]["exact_model_call_evidence"] is False
    assert response.response.context_compaction_contract is None
    assert len(response.response.trajectory_transitions or []) == 2


def test_build_exact_trace_response_preserves_noncontiguous_turns() -> None:
    request = OSWorldRunRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        verifier_metadata={"task_id": "task-001", "domain": "chrome"},
        context_compaction_contract_version=2,
        context_compaction_rollout_id="rollout-test-001",
        context_compaction_group_id="group-test-001",
        context_compaction_task_id="task-001",
        context_compaction_rollout_index=0,
        context_compaction_attempt_index=0,
    )
    result = {
        **DEFAULT_RUN_RESULT,
        "steps": [
            {
                "step": 0,
                "actions": ["pyautogui.click(10, 20)"],
                "reward": 0.0,
                "done": False,
                "info": {
                    "agent": {
                        "model_calls": [
                            {
                                "parse_attempt": 1,
                                "prompt_messages": [
                                    {"role": "system", "content": "system"},
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "image_url",
                                                "image_url": {"url": "data:image/png;base64,Zmlyc3Q="},
                                            }
                                        ],
                                    },
                                ],
                                "response": {
                                    "raw_content": "first action",
                                    "prompt_token_ids": [10, 11],
                                    "generation_token_ids": [20, 21],
                                    "generation_log_probs": [-0.1, -0.2],
                                },
                                "accepted": True,
                                "parse_error": None,
                                "parsed_actions": ["pyautogui.click(10, 20)"],
                            }
                        ]
                    }
                },
            },
            {
                "step": 1,
                "actions": ["DONE"],
                "reward": 1.0,
                "done": True,
                "info": {
                    "agent": {
                        "model_calls": [
                            {
                                "parse_attempt": 1,
                                "prompt_messages": [
                                    {"role": "system", "content": "rewritten system"},
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "image_url",
                                                "image_url": {"url": "data:image/png;base64,c2Vjb25k"},
                                            }
                                        ],
                                    },
                                ],
                                "response": {
                                    "raw_content": "finish",
                                    # Deliberately not prefixed by turn 1's prompt + completion.
                                    "prompt_token_ids": [99, 100],
                                    "generation_token_ids": [101],
                                    "generation_log_probs": [-0.3],
                                },
                                "accepted": True,
                                "parse_error": None,
                                "parsed_actions": ["DONE"],
                            }
                        ]
                    }
                },
            },
        ],
    }

    response = _build_response(
        request,
        result,
        "test-policy",
        1.0,
        0.9,
        max_trajectory_length=3,
        max_output_tokens=512,
    )

    trace_response = response.response
    output = [item.model_dump(exclude_none=True) for item in trace_response.output]
    assert [item["role"] for item in output] == ["assistant", "assistant"]
    assert output[0]["generation_token_ids"] == [20, 21]
    assert output[1]["prompt_token_ids"] == [99, 100]
    contract = trace_response.context_compaction_contract
    assert contract is not None
    assert contract["schema_version"] == 2
    assert contract["mode"] == "exact_trace_authority"
    assert contract["rollout_id"] == "rollout-test-001"
    assert contract["group_id"] == "group-test-001"
    assert contract["task_id"] == "task-001"
    assert contract["rollout_index"] == 0
    assert contract["attempt_index"] == 0
    assert contract["identity_source"] == "caller"
    evidence = trace_response.completion_evidence
    assert evidence is not None
    assert [item["segment_index"] for item in evidence] == [0, 1]
    assert [item["expected_append_compatible"] for item in evidence] == [False, False]
    assert evidence[1]["compaction_event_id"] is not None
    assert len(trace_response.boundary_events or []) == 1
    assert len(trace_response.media_assets or {}) == 2
    model_calls = trace_response.trajectory_model_calls or []
    assert len(model_calls) == 2
    assert model_calls[0]["state"]["prompt_messages"][1]["content"][0] == {
        "type": "input_image",
        "media_id": model_calls[0]["state"]["media_ids"][0],
        "detail": "high",
    }
    assert model_calls[0]["action"] == {
        "raw_completion": "first action",
        "parsed_actions": ["pyautogui.click(10, 20)"],
    }
    assert model_calls[0]["generation_evidence"]["generation_log_probs"] == [
        -0.1,
        -0.2,
    ]
    first_media_id = model_calls[0]["state"]["media_ids"][0]
    assert (trace_response.media_assets or {})[first_media_id]["source_part"][
        "image_url"
    ] == "data:image/png;base64,Zmlyc3Q="
    assert response.verifier_metadata["osworld_steps"][0]["info"]["agent"] == {"model_call_count": 1}
    assert [item["action"]["parsed_actions"] for item in trace_response.trajectory_transitions or []] == [
        ["pyautogui.click(10, 20)"],
        ["DONE"],
    ]


def test_build_exact_trace_response_derives_identity_for_benchmarking() -> None:
    request = make_run_request(osworld_task=DEFAULT_OSWORLD_TASK)
    result = {
        **DEFAULT_RUN_RESULT,
        "steps": [
            {
                "step": 0,
                "model_text": "action",
                "actions": ["DONE"],
                "reward": 1.0,
                "done": True,
                "info": {
                    "agent": {
                        "model_calls": [
                            {
                                "parse_attempt": 1,
                                "prompt_messages": [{"role": "user", "content": "inspect"}],
                                "response": {
                                    "raw_content": "action",
                                    "prompt_token_ids": [1],
                                    "generation_token_ids": [2],
                                    "generation_log_probs": [-0.1],
                                },
                                "accepted": True,
                                "parse_error": None,
                                "parsed_actions": ["DONE"],
                            }
                        ]
                    }
                },
            }
        ],
    }

    response = _build_response(request, result, "test-policy", 1.0, 0.9)

    trajectory_contract = response.response.trajectory_contract
    exact_contract = response.response.context_compaction_contract
    assert trajectory_contract is not None
    assert exact_contract is not None
    assert trajectory_contract["identity_source"] == "derived"
    assert trajectory_contract["training_eligibility"]["status"] == "ineligible"
    assert exact_contract["identity_source"] == "derived"
    assert exact_contract["rollout_id"] == trajectory_contract["rollout_id"]


def test_build_response_accepts_generic_caller_trajectory_identity() -> None:
    request = OSWorldRunRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        verifier_metadata={"task_id": "task-001", "domain": "chrome"},
        trajectory_identity={
            "schema_version": 1,
            "rollout_id": "rollout-generic-001",
            "group_id": "group-generic-001",
            "task_id": "task-001",
            "rollout_index": 2,
            "attempt_index": 0,
        },
    )

    response = _build_response(request, DEFAULT_RUN_RESULT, "test-policy", 1.0, 0.9)

    contract = response.response.trajectory_contract
    assert contract is not None
    assert contract["identity_source"] == "caller"
    assert contract["rollout_id"] == "rollout-generic-001"
    assert contract["rollout_index"] == 2


def test_execution_identity_is_correlated_but_excluded_from_semantic_digest() -> None:
    def build(execution_id: str):
        request = OSWorldRunRequest.model_validate(
            {
                "responses_create_params": {"input": []},
                "verifier_metadata": {
                    "task_id": "task-001",
                    "domain": "chrome",
                },
                "trajectory_identity": {
                    "schema_version": 1,
                    "sampling_event_id": "sampling-training-001",
                    "source_group_id": "dataset-group-001",
                    "rollout_id": "rollout-generic-001",
                    "group_id": "group-event-001",
                    "task_id": "task-001",
                    "rollout_index": 2,
                    "attempt_index": 0,
                },
                "_ng_execution_id": execution_id,
            }
        )
        assert "_ng_execution_id" not in request.model_dump()
        result = {
            **DEFAULT_RUN_RESULT,
            "execution_id": execution_id,
            "steps": [
                {
                    "step": 0,
                    "model_text": "```DONE```",
                    "actions": ["DONE"],
                    "reward": 1.0,
                    "done": True,
                    "info": {
                        "agent": {
                            "model_calls": [
                                {
                                    "parse_attempt": 1,
                                    "prompt_messages": [{"role": "user", "content": "inspect"}],
                                    "response": {
                                        "raw_content": "```DONE```",
                                        "prompt_token_ids": [1],
                                        "generation_token_ids": [2],
                                        "generation_log_probs": [-0.1],
                                    },
                                    "accepted": True,
                                    "parse_error": None,
                                    "parsed_actions": ["DONE"],
                                }
                            ]
                        }
                    },
                }
            ],
        }
        return _build_response(
            request,
            result,
            "test-policy",
            1.0,
            0.9,
        )

    first = build("execution-first")
    second = build("execution-second")

    first_contract = first.response.trajectory_contract
    second_contract = second.response.trajectory_contract
    assert first_contract is not None
    assert second_contract is not None
    assert first_contract["sampling_event_id"] == "sampling-training-001"
    assert first_contract["source_group_id"] == "dataset-group-001"
    assert first_contract["trajectory_contract_id"] == second_contract["trajectory_contract_id"]
    first_exact = first.response.context_compaction_contract
    second_exact = second.response.context_compaction_contract
    assert first_exact is not None
    assert first_exact == second_exact
    assert "execution_id" not in json.dumps(first_exact, sort_keys=True)
    assert first.response.execution_context == {
        "schema_version": 1,
        "execution_id": "execution-first",
        "sampling_event_id": "sampling-training-001",
        "source_group_id": "dataset-group-001",
        "rollout_id": "rollout-generic-001",
        "group_id": "group-event-001",
        "task_id": "task-001",
    }
    assert first.verifier_metadata["osworld_execution_id"] == "execution-first"


def test_build_response_rejects_partial_caller_identity() -> None:
    request = OSWorldRunRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        verifier_metadata={"task_id": "task-001"},
        context_compaction_contract_version=2,
    )

    with pytest.raises(ValueError, match="identity is incomplete"):
        _build_response(request, DEFAULT_RUN_RESULT, "test-policy", 1.0, 0.9)


def test_empty_response_preserves_explicit_semantic_execution_join() -> None:
    request = OSWorldRunRequest.model_validate(
        {
            "responses_create_params": {"input": []},
            "verifier_metadata": {"task_id": "task-001"},
            "trajectory_identity": {
                "schema_version": 1,
                "sampling_event_id": "sampling-evaluation-001",
                "source_group_id": "dataset-group-001",
                "rollout_id": "rollout-evaluation-001",
                "group_id": "group-evaluation-001",
                "task_id": "task-001",
                "rollout_index": 0,
                "attempt_index": 0,
            },
            "_ng_execution_id": "execution-empty-001",
        }
    )

    response = _empty_response(request, error="fixture unavailable")

    assert response.response.execution_context == {
        "schema_version": 1,
        "execution_id": "execution-empty-001",
        "sampling_event_id": "sampling-evaluation-001",
        "source_group_id": "dataset-group-001",
        "rollout_id": "rollout-evaluation-001",
        "group_id": "group-evaluation-001",
        "task_id": "task-001",
    }


def test_exact_trace_keeps_parser_retries_as_distinct_model_calls() -> None:
    request = make_run_request(osworld_task=DEFAULT_OSWORLD_TASK)
    calls = []
    for attempt, token_id, accepted in ((1, 20, False), (2, 21, True)):
        calls.append(
            {
                "parse_attempt": attempt,
                "prompt_messages": [{"role": "user", "content": f"attempt {attempt}"}],
                "response": {
                    "raw_content": f"sample {attempt}",
                    "prompt_token_ids": [10, attempt],
                    "generation_token_ids": [token_id],
                    "generation_log_probs": [-0.1 * attempt],
                },
                "accepted": accepted,
                "parse_error": None if accepted else "invalid Python",
                "parsed_actions": ["DONE"] if accepted else [],
            }
        )
    result = {
        **DEFAULT_RUN_RESULT,
        "steps": [
            {
                "step": 0,
                "model_text": "sample 2",
                "actions": ["DONE"],
                "reward": 1.0,
                "done": True,
                "info": {"agent": {"model_calls": calls}},
            }
        ],
    }

    response = _build_response(request, result, "test-policy", 1.0, 0.9)

    assert len(response.response.output) == 2
    assert [item["accepted"] for item in response.response.completion_evidence or []] == [
        False,
        True,
    ]
    [transition] = response.response.trajectory_transitions or []
    assert len(transition["state"]["model_call_ids"]) == 2
    assert transition["action"]["accepted_model_call_id"] == transition["state"]["model_call_ids"][1]


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

    def test_removed_training_switches_fail_loudly(self) -> None:
        with pytest.raises(ValueError, match="trajectory evidence is now automatic"):
            OSWorldAgent(
                config=make_config(training_mode=True),
                server_client=MagicMock(spec=ServerClient),
            )

    @patch("responses_api_agents.osworld_agent.app.load_attr")
    def test_startup_validates_runner_in_agent_runtime(self, mock_load_attr) -> None:
        agent = OSWorldAgent(
            config=make_config(
                runner_name="m3_agent",
                agent_class_path="custom_runtime.M3Agent",
            ),
            server_client=MagicMock(spec=ServerClient),
        )

        assert agent.sem is not None
        mock_load_attr.assert_called_once_with("custom_runtime.M3Agent")

    @patch("responses_api_agents.osworld_agent.app.load_attr")
    def test_pointer_startup_disables_unconfigured_parallel_tools(self, mock_load_attr, monkeypatch) -> None:
        monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        def load_pointer(_path: str) -> None:
            assert os.environ["ANTHROPIC_API_KEY"] == "__nemo_gym_anthropic_key_deferred__"  # pragma: allowlist secret

        mock_load_attr.side_effect = load_pointer

        class_path = _validate_runner_runtime(make_config(runner_name="pointer_agent"))

        assert class_path == "mm_agents.pointer.PointerAgent"
        assert os.environ["PARALLEL_API_KEY"] == "__nemo_gym_parallel_tools_disabled__"  # pragma: allowlist secret
        assert "ANTHROPIC_API_KEY" not in os.environ
        mock_load_attr.assert_called_once_with(class_path)

    def test_metrics_report_binary_and_raw_osworld_scores(self) -> None:
        agent = OSWorldAgent(config=make_config(), server_client=MagicMock(spec=ServerClient))
        tasks = [
            [
                {"reward": 1.0, "verifier_metadata": {"osworld_score": 1.0}},
                {"reward": 0.0, "verifier_metadata": {"osworld_score": 0.5}},
            ],
            [
                {
                    "reward": 0.0,
                    "mask_sample": True,
                    "verifier_metadata": {"osworld_score": 0.25},
                }
            ],
        ]

        metrics = agent.compute_metrics(tasks)

        assert metrics["osworld/scored_rollout_count"] == 3
        assert metrics["osworld/masked_rollout_count"] == 1
        assert metrics["osworld/binary_success_count"] == 1
        assert metrics["osworld/binary_success_rate"] == pytest.approx(100.0 / 3.0)
        assert metrics["osworld/raw_reward_sum"] == pytest.approx(1.75)
        assert metrics["osworld/raw_reward_rate"] == pytest.approx(175.0 / 3.0)

        key_metrics = agent.get_key_metrics({"mean/reward": 1.0, **metrics})
        assert key_metrics == {
            "mean/reward": 1.0,
            "osworld/binary_success_rate": pytest.approx(100.0 / 3.0),
            "osworld/raw_reward_rate": pytest.approx(175.0 / 3.0),
        }

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

    @patch("responses_api_agents.osworld_agent.app.ServerClient.load_from_global_config")
    @patch("responses_api_agents.osworld_agent.app.get_first_server_config_dict")
    @patch("responses_api_agents.osworld_agent.app._run_osworld_task_remote")
    @patch("asyncio.to_thread")
    def test_http_run_recovers_evaluation_rollout_purpose_from_metadata(
        self,
        mock_to_thread,
        mock_remote,
        mock_get_first_server_config_dict,
        mock_load_from_global_config,
    ) -> None:
        """Exercise the real FastAPI/Pydantic boundary used by NeMo-RL."""
        assert "rollout_purpose" in OSWorldRunRequest.__annotations__
        setup_server_client_mocks(mock_load_from_global_config, mock_get_first_server_config_dict)
        mock_remote.options.return_value.remote.return_value = MagicMock()
        mock_to_thread.return_value = {
            **DEFAULT_RUN_RESULT,
            "execution_id": "execution-http-test",
        }

        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {"observability_enabled": True}
        agent = OSWorldAgent(config=make_config(), server_client=server_client)
        request = make_run_request(
            osworld_task=DEFAULT_OSWORLD_TASK,
            temperature=0.6,
            top_p=0.95,
            max_output_tokens=768,
            rollout_purpose="evaluation",
        )
        payload = {
            **request.model_dump(mode="json"),
            "_ng_execution_id": "execution-http-test",
            "_ng_task_index": 4,
            "_ng_rollout_index": 0,
            "trajectory_identity": {
                "schema_version": 1,
                "sampling_event_id": "sampling-evaluation-http",
                "source_group_id": "dataset-group-http",
                "rollout_id": "rollout-evaluation-http",
                "group_id": "group-evaluation-http",
                "task_id": "test-task-001",
                "rollout_index": 0,
                "attempt_index": 0,
            },
        }
        # Reproduce a generic /run boundary that keeps the standard
        # responses_create_params model but discards a top-level extension.
        payload.pop("rollout_purpose")
        payload["responses_create_params"]["metadata"] = {"nemo_rl_rollout_purpose": "evaluation"}

        response = TestClient(agent.setup_webserver()).post("/run", json=payload)

        assert response.status_code == 200
        assert response.json()["rollout_purpose"] == "evaluation"
        assert response.json()["response"]["execution_context"] == {
            "schema_version": 1,
            "execution_id": "execution-http-test",
            "sampling_event_id": "sampling-evaluation-http",
            "source_group_id": "dataset-group-http",
            "rollout_id": "rollout-evaluation-http",
            "group_id": "group-evaluation-http",
            "task_id": "test-task-001",
        }
        positional_args, _ = mock_remote.options.return_value.remote.call_args
        assert positional_args[1]["rollout_purpose"] == "evaluation"
        assert positional_args[1]["execution_id"] == "execution-http-test"
        assert positional_args[1]["log_context"]["sampling_event_id"] == ("sampling-evaluation-http")
        assert positional_args[1]["log_context"]["rollout_id"] == ("rollout-evaluation-http")
        assert positional_args[1]["sandbox_spec"]["metadata"]["nemo-gym.execution-id"] == "execution-http-test"

    @patch("benchmarks.osworld.assets.ensure_osworld_assets")
    def test_setup_webserver_idempotently_prefetches_configured_assets(self, mock_ensure) -> None:
        mock_ensure.return_value = SimpleNamespace(
            task_count=5,
            asset_count=7,
            materialized_count=0,
            cache_dir="/cache",
        )
        agent = OSWorldAgent(
            config=make_config(asset_input_jsonl="/data/tasks.jsonl", setup_cache_dir="/cache"),
            server_client=MagicMock(spec=ServerClient),
        )

        agent.setup_webserver()

        mock_ensure.assert_called_once_with(
            "/data/tasks.jsonl",
            "/cache",
            token=os.environ.get("HF_TOKEN"),
            proxy_url=os.environ.get("OSWORLD_ASSET_PROXY_URL"),
        )

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
        monkeypatch,
    ) -> None:
        setup_server_client_mocks(mock_load_from_global_config, mock_get_first_server_config_dict)
        monkeypatch.setenv("RUN_TAG", "run-001")

        # Mock the Ray-remote ``.options(...).remote(...)`` call chain.
        future = MagicMock()
        mock_remote.options.return_value.remote.return_value = future
        mock_to_thread.return_value = DEFAULT_RUN_RESULT

        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {"observability_enabled": True}
        agent = OSWorldAgent(
            config=make_config(
                agent_kwargs={"parse_retries": 5},
                agent_kwargs_by_rollout_purpose={
                    "training": {"parse_retries": 1},
                    "evaluation": {"parse_retries": 5},
                },
            ),
            server_client=server_client,
        )
        request = make_run_request(
            osworld_task=DEFAULT_OSWORLD_TASK,
            temperature=0.7,
            top_p=0.95,
            max_output_tokens=4096,
            rollout_purpose="evaluation",
        )
        request = OSWorldRunRequest.model_validate(
            {
                **request.model_dump(),
                "_ng_task_index": 4,
                "_ng_rollout_index": 0,
                "trajectory_identity": {
                    "schema_version": 1,
                    "rollout_id": "rollout-eval-001",
                    "group_id": "group-eval-001",
                    "task_id": "test-task-001",
                    "rollout_index": 0,
                    "attempt_index": 0,
                },
            }
        )

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
        assert response.rollout_purpose == "evaluation"
        # Ray remote was dispatched exactly once with our task spec.
        mock_remote.options.assert_called_once()
        mock_remote.options.return_value.remote.assert_called_once()
        positional_args, _ = mock_remote.options.return_value.remote.call_args
        assert positional_args[0] == DEFAULT_OSWORLD_TASK
        assert positional_args[1]["base_url"] == "http://127.0.0.1:8000/ng-rollout/4-0/v1"
        assert positional_args[1]["evaluator_disable_gpu"] is True
        assert positional_args[1]["docker_port_lock_timeout"] == 300.0
        assert positional_args[1]["enable_proxy"] is False
        assert positional_args[1]["proxy_config_file"] is None
        assert positional_args[1]["max_tokens"] == 4096
        assert positional_args[1]["rollout_purpose"] == "evaluation"
        assert positional_args[1]["agent_kwargs"]["parse_retries"] == 5
        assert positional_args[1]["log_context"] == {
            "run_id": "run-001",
            "adapter": "gym",
            "rollout_purpose": "evaluation",
            "rollout_id": "rollout-eval-001",
            "group_id": "group-eval-001",
            "rollout_index": 0,
            "attempt_index": 0,
            "task_id": "test-task-001",
            "domain": "chrome",
            "task_attempt": 1,
        }
        runtime_env = mock_remote.options.call_args.kwargs["runtime_env"]
        assert runtime_env["py_executable"]
        assert runtime_env["env_vars"]["RUN_TAG"] == "run-001"

    def test_derived_run_identity_is_stable_for_standalone_benchmark(self) -> None:
        identity_a = resolve_trajectory_identity(
            request_extra={"_ng_rollout_index": 2, "_ng_attempt_index": 1},
            verifier_metadata={"task_id": "task-standalone"},
            model_name="policy-model",
        )
        identity_b = resolve_trajectory_identity(
            request_extra={"_ng_rollout_index": 2, "_ng_attempt_index": 1},
            verifier_metadata={"task_id": "task-standalone"},
            model_name="policy-model",
        )

        assert identity_a == identity_b
        assert identity_a["identity_source"] == "derived"
        assert identity_a["rollout_index"] == 2
        assert identity_a["attempt_index"] == 1

    def test_caller_run_identity_rejects_task_mismatch(self) -> None:
        with pytest.raises(
            ValueError,
            match="trajectory_identity.task_id must match verifier_metadata task_id",
        ):
            resolve_trajectory_identity(
                request_extra={
                    "trajectory_identity": {
                        "schema_version": 1,
                        "rollout_id": "rollout-001",
                        "group_id": "group-001",
                        "task_id": "wrong-task",
                        "rollout_index": 0,
                        "attempt_index": 0,
                    }
                },
                verifier_metadata={"task_id": "actual-task"},
                model_name="policy-model",
            )

    @patch("responses_api_agents.osworld_agent.app.ServerClient.load_from_global_config")
    @patch("responses_api_agents.osworld_agent.app.get_first_server_config_dict")
    @patch("responses_api_agents.osworld_agent.app._run_osworld_task_remote")
    @patch("asyncio.to_thread")
    async def test_run_forwards_rollout_diagnostic_paths_to_ray_child(
        self,
        mock_to_thread,
        mock_remote,
        mock_get_first_server_config_dict,
        mock_load_from_global_config,
        monkeypatch,
    ) -> None:
        setup_server_client_mocks(mock_load_from_global_config, mock_get_first_server_config_dict)
        monkeypatch.setenv("OSWORLD_MODEL_IO_LOG", "/workspace/output/r6f/osworld-model-io.jsonl")
        monkeypatch.setenv("OSWORLD_TASK_ARTIFACT_ROOT", "/workspace/output/r6f/osworld-tasks")
        monkeypatch.setenv("OSWORLD_VM_EXEC_LOG", "/workspace/output/r6f/osworld-vm-exec.jsonl")
        mock_remote.options.return_value.remote.return_value = MagicMock()
        mock_to_thread.return_value = DEFAULT_RUN_RESULT

        agent = OSWorldAgent(config=make_config(), server_client=MagicMock(spec=ServerClient))
        await agent.run(make_run_request(osworld_task=DEFAULT_OSWORLD_TASK))

        runtime_env = mock_remote.options.call_args.kwargs["runtime_env"]
        expected_env = {
            "OSWORLD_MODEL_IO_LOG": "/workspace/output/r6f/osworld-model-io.jsonl",
            "OSWORLD_TASK_ARTIFACT_ROOT": "/workspace/output/r6f/osworld-tasks",
            "OSWORLD_VM_EXEC_LOG": "/workspace/output/r6f/osworld-vm-exec.jsonl",
        }
        assert {name: runtime_env["env_vars"][name] for name in expected_env} == expected_env

    @patch("responses_api_agents.osworld_agent.app.ServerClient.load_from_global_config")
    @patch("responses_api_agents.osworld_agent.app.get_first_server_config_dict")
    @patch("responses_api_agents.osworld_agent.app._run_osworld_task_remote")
    @patch("asyncio.to_thread")
    async def test_run_resolves_named_gym_sandbox_config_to_plain_ray_payload(
        self,
        mock_to_thread,
        mock_remote,
        mock_get_first_server_config_dict,
        mock_load_from_global_config,
    ) -> None:
        setup_server_client_mocks(mock_load_from_global_config, mock_get_first_server_config_dict)
        global_config = mock_load_from_global_config.return_value.global_config_dict
        global_config["osworld_sandbox"] = {
            "default_metadata": {"sandbox-api": "docker-cli", "owner": "provider"},
            "docker": {"create": {"use_init": False}},
        }
        mock_remote.options.return_value.remote.return_value = MagicMock()
        mock_to_thread.return_value = DEFAULT_RUN_RESULT
        agent = OSWorldAgent(
            config=make_config(
                sandbox_provider="osworld_sandbox",
                sandbox_spec={
                    "image": "docker://osworld@sha256:fixed",
                    "metadata": {"owner": "agent"},
                },
                vm_path="/assets/Ubuntu.qcow2",
            ),
            server_client=MagicMock(spec=ServerClient),
        )

        response = await agent.run(make_run_request(osworld_task=DEFAULT_OSWORLD_TASK))

        assert response.reward == 1.0
        positional_args, _ = mock_remote.options.return_value.remote.call_args
        runner_kwargs = positional_args[1]
        assert runner_kwargs["base_url"] == "http://127.0.0.1:8000/v1"
        assert runner_kwargs["sandbox_provider_config"] == {"docker": {"create": {"use_init": False}}}
        assert runner_kwargs["sandbox_spec"] == {
            "image": "docker://osworld@sha256:fixed",
            "metadata": {"sandbox-api": "docker-cli", "owner": "agent"},
        }
        assert runner_kwargs["vm_path"] == "/assets/Ubuntu.qcow2"
        assert runner_kwargs["sandbox_vm_path"] is None

    @patch("responses_api_agents.osworld_agent.app.ServerClient.load_from_global_config")
    @patch("responses_api_agents.osworld_agent.app.get_first_server_config_dict")
    @patch("responses_api_agents.osworld_agent.app._run_osworld_task_remote")
    @patch("asyncio.to_thread")
    async def test_run_applies_osworld_scoped_opensandbox_create_budget(
        self,
        mock_to_thread,
        mock_remote,
        mock_get_first_server_config_dict,
        mock_load_from_global_config,
    ) -> None:
        setup_server_client_mocks(mock_load_from_global_config, mock_get_first_server_config_dict)
        global_config = mock_load_from_global_config.return_value.global_config_dict
        global_config["sandbox"] = {
            "opensandbox": {
                "connection": {"domain": "sandbox.internal"},
                "create": {"timeout_s": 1500, "retries": 10},
            }
        }
        mock_remote.options.return_value.remote.return_value = MagicMock()
        mock_to_thread.return_value = DEFAULT_RUN_RESULT
        agent = OSWorldAgent(
            config=make_config(
                sandbox_provider="sandbox",
                sandbox_provider_overrides={"opensandbox": {"create": {"timeout_s": 180, "retries": 1}}},
            ),
            server_client=MagicMock(spec=ServerClient),
        )

        await agent.run(make_run_request(osworld_task=DEFAULT_OSWORLD_TASK))

        runner_kwargs = mock_remote.options.return_value.remote.call_args.args[1]
        assert runner_kwargs["sandbox_provider_config"] == {
            "opensandbox": {
                "connection": {"domain": "sandbox.internal"},
                "create": {"timeout_s": 180, "retries": 1},
            }
        }

    @patch("responses_api_agents.osworld_agent.app.ServerClient.load_from_global_config")
    @patch("responses_api_agents.osworld_agent.app.get_first_server_config_dict")
    @patch("responses_api_agents.osworld_agent.app._run_osworld_task_remote")
    @patch("asyncio.to_thread")
    async def test_proxy_required_task_runs_directly_when_proxy_is_disabled(
        self,
        mock_to_thread,
        mock_remote,
        mock_get_first_server_config_dict,
        mock_load_from_global_config,
        monkeypatch,
    ) -> None:
        setup_server_client_mocks(mock_load_from_global_config, mock_get_first_server_config_dict)
        monkeypatch.setenv("PROXY_CONFIG_FILE", "/unused/proxy.json")
        mock_remote.options.return_value.remote.return_value = MagicMock()
        mock_to_thread.return_value = DEFAULT_RUN_RESULT
        task = {**DEFAULT_OSWORLD_TASK, "proxy": True}
        agent = OSWorldAgent(config=make_config(enable_proxy=False), server_client=MagicMock(spec=ServerClient))

        response = await agent.run(make_run_request(osworld_task=task))

        positional_args, _ = mock_remote.options.return_value.remote.call_args
        assert positional_args[1]["enable_proxy"] is False
        assert positional_args[1]["proxy_config_file"] is None
        assert response.mask_sample is False
        assert response.verifier_metadata["osworld_proxy_required"] is True
        assert response.verifier_metadata["osworld_proxy_enabled"] is False
        assert response.verifier_metadata["osworld_proxy_configured"] is False

    @patch("responses_api_agents.osworld_agent.app._run_osworld_task_remote")
    async def test_proxy_required_task_is_masked_in_explicit_strict_mode(self, mock_remote) -> None:
        task = {**DEFAULT_OSWORLD_TASK, "proxy": True}
        agent = OSWorldAgent(
            config=make_config(enable_proxy=False, allow_direct_proxy_tasks=False),
            server_client=MagicMock(spec=ServerClient),
        )

        response = await agent.run(make_run_request(osworld_task=task))

        assert response.mask_sample is True
        assert response.verifier_metadata["osworld_termination_reason"] == "proxy_required_but_disabled"
        assert response.verifier_metadata["osworld_proxy_required"] is True
        assert response.verifier_metadata["osworld_proxy_enabled"] is False
        mock_remote.options.assert_not_called()

    @patch("responses_api_agents.osworld_agent.app.ServerClient.load_from_global_config")
    @patch("responses_api_agents.osworld_agent.app.get_first_server_config_dict")
    @patch("responses_api_agents.osworld_agent.app._run_osworld_task_remote")
    @patch("asyncio.to_thread")
    async def test_proxy_required_task_can_run_directly_when_explicitly_allowed(
        self,
        mock_to_thread,
        mock_remote,
        mock_get_first_server_config_dict,
        mock_load_from_global_config,
        monkeypatch,
    ) -> None:
        setup_server_client_mocks(mock_load_from_global_config, mock_get_first_server_config_dict)
        monkeypatch.setenv("OSWORLD_ALLOW_DIRECT_PROXY_TASKS", "1")
        monkeypatch.setenv("PROXY_CONFIG_FILE", "/unused/proxy.json")
        mock_remote.options.return_value.remote.return_value = MagicMock()
        mock_to_thread.return_value = DEFAULT_RUN_RESULT
        task = {**DEFAULT_OSWORLD_TASK, "proxy": True}
        agent = OSWorldAgent(config=make_config(), server_client=MagicMock(spec=ServerClient))

        response = await agent.run(make_run_request(osworld_task=task))

        positional_args, _ = mock_remote.options.return_value.remote.call_args
        assert positional_args[1]["enable_proxy"] is False
        assert positional_args[1]["allow_direct_proxy_tasks"] is True
        assert positional_args[1]["proxy_config_file"] is None
        assert response.mask_sample is False
        assert response.verifier_metadata["osworld_proxy_required"] is True
        assert response.verifier_metadata["osworld_proxy_enabled"] is False

    @patch("responses_api_agents.osworld_agent.app._run_osworld_task_remote")
    async def test_direct_proxy_mode_rejects_remote_resources_before_ray(self, mock_remote) -> None:
        task = {**DEFAULT_OSWORLD_TASK, "proxy": True}
        agent = OSWorldAgent(
            config=make_config(
                allow_direct_proxy_tasks=True,
                resources_server={"type": "resources_servers", "name": "osworld_resources"},
            ),
            server_client=MagicMock(spec=ServerClient),
        )

        response = await agent.run(make_run_request(osworld_task=task))

        assert response.mask_sample is True
        assert response.verifier_metadata["osworld_termination_reason"] == "proxy_configuration_error"
        assert "remote Resources Server" in response.verifier_metadata["osworld_error"]
        mock_remote.options.assert_not_called()

    @patch("responses_api_agents.osworld_agent.app.ServerClient.load_from_global_config")
    @patch("responses_api_agents.osworld_agent.app.get_first_server_config_dict")
    @patch("responses_api_agents.osworld_agent.app._run_osworld_task_remote")
    @patch("asyncio.to_thread")
    async def test_proxy_enable_env_validates_and_reaches_ray(
        self,
        mock_to_thread,
        mock_remote,
        mock_get_first_server_config_dict,
        mock_load_from_global_config,
        monkeypatch,
        tmp_path,
    ) -> None:
        setup_server_client_mocks(mock_load_from_global_config, mock_get_first_server_config_dict)
        proxy_path = tmp_path / "proxy.json"
        proxy_path.write_text('[{"host":"proxy.example.com","port":3128}]\n', encoding="utf-8")
        monkeypatch.setenv("OSWORLD_ENABLE_PROXY", "1")
        monkeypatch.setenv("PROXY_CONFIG_FILE", str(proxy_path))
        mock_remote.options.return_value.remote.return_value = MagicMock()
        mock_to_thread.return_value = DEFAULT_RUN_RESULT
        task = {**DEFAULT_OSWORLD_TASK, "proxy": True}
        agent = OSWorldAgent(config=make_config(), server_client=MagicMock(spec=ServerClient))

        response = await agent.run(make_run_request(osworld_task=task))

        positional_args, _ = mock_remote.options.return_value.remote.call_args
        assert positional_args[1]["enable_proxy"] is True
        assert positional_args[1]["proxy_config_file"] == str(proxy_path)
        assert response.mask_sample is False
        assert response.verifier_metadata["osworld_proxy_required"] is True
        assert response.verifier_metadata["osworld_proxy_enabled"] is True
        assert response.verifier_metadata["osworld_proxy_configured"] is True

    async def test_invalid_proxy_env_value_is_masked(self, monkeypatch) -> None:
        monkeypatch.setenv("OSWORLD_ENABLE_PROXY", "sometimes")
        agent = OSWorldAgent(config=make_config(), server_client=MagicMock(spec=ServerClient))

        response = await agent.run(make_run_request(osworld_task=DEFAULT_OSWORLD_TASK))

        assert response.mask_sample is True
        assert response.verifier_metadata["osworld_termination_reason"] == "proxy_configuration_error"

    async def test_invalid_direct_proxy_env_value_is_masked(self, monkeypatch) -> None:
        monkeypatch.setenv("OSWORLD_ALLOW_DIRECT_PROXY_TASKS", "sometimes")
        agent = OSWorldAgent(config=make_config(), server_client=MagicMock(spec=ServerClient))

        response = await agent.run(make_run_request(osworld_task=DEFAULT_OSWORLD_TASK))

        assert response.mask_sample is True
        assert response.verifier_metadata["osworld_termination_reason"] == "proxy_configuration_error"

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

    @patch("responses_api_agents.osworld_agent.app.ray.cancel")
    @patch("responses_api_agents.osworld_agent.app.ServerClient.load_from_global_config")
    @patch("responses_api_agents.osworld_agent.app.get_first_server_config_dict")
    @patch("responses_api_agents.osworld_agent.app._run_osworld_task_remote")
    @patch("asyncio.to_thread")
    async def test_run_end_to_end_timeout_cancels_ray_task_and_masks_sample(
        self,
        mock_to_thread,
        mock_remote,
        mock_get_first_server_config_dict,
        mock_load_from_global_config,
        mock_ray_cancel,
    ) -> None:
        setup_server_client_mocks(mock_load_from_global_config, mock_get_first_server_config_dict)
        future = MagicMock()
        mock_remote.options.return_value.remote.return_value = future
        mock_to_thread.side_effect = [ray.exceptions.GetTimeoutError("deadline"), RuntimeError("cancelled")]

        agent = OSWorldAgent(
            config=make_config(task_timeout=12),
            server_client=MagicMock(spec=ServerClient),
        )
        response = await agent.run(make_run_request(osworld_task=DEFAULT_OSWORLD_TASK))

        mock_ray_cancel.assert_called_once_with(future, force=False)
        assert response.reward == 0.0
        assert response.mask_sample is True
        assert response.verifier_metadata["osworld_termination_reason"] == "task_timeout"
        assert response.verifier_metadata["osworld_error"] == ("task_timeout exceeded (12s) during end-to-end rollout")

    @patch("responses_api_agents.osworld_agent.app.ray.cancel")
    @patch("responses_api_agents.osworld_agent.app.ServerClient.load_from_global_config")
    @patch("responses_api_agents.osworld_agent.app.get_first_server_config_dict")
    @patch("responses_api_agents.osworld_agent.app._run_osworld_task_remote")
    @patch("asyncio.to_thread")
    async def test_run_end_to_end_timeout_force_cancels_after_cleanup_grace(
        self,
        mock_to_thread,
        mock_remote,
        mock_get_first_server_config_dict,
        mock_load_from_global_config,
        mock_ray_cancel,
    ) -> None:
        setup_server_client_mocks(mock_load_from_global_config, mock_get_first_server_config_dict)
        future = MagicMock()
        mock_remote.options.return_value.remote.return_value = future
        mock_to_thread.side_effect = [
            ray.exceptions.GetTimeoutError("deadline"),
            ray.exceptions.GetTimeoutError("cleanup grace"),
        ]

        agent = OSWorldAgent(
            config=make_config(task_timeout=12, task_cancel_grace_s=3),
            server_client=MagicMock(spec=ServerClient),
        )
        response = await agent.run(make_run_request(osworld_task=DEFAULT_OSWORLD_TASK))

        assert mock_ray_cancel.call_args_list == [call(future, force=False), call(future, force=True)]
        assert response.mask_sample is True
        assert response.verifier_metadata["osworld_termination_reason"] == "task_timeout"

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
