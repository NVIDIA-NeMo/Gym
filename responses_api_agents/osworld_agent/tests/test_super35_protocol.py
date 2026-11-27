# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Super 3.5 protocol behavior through the shared OSWorld agent loop."""

from __future__ import annotations

import base64
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml

from responses_api_agents.osworld_agent.adapter_agents import (
    MODEL_PROTOCOL_REGISTRY,
    OSWorldProtocolAgent,
    resolve_model_protocol,
)
from responses_api_agents.osworld_agent.agent_contract import assert_train_eval_parity, resolve_agent_contract
from responses_api_agents.osworld_agent.history_policy import HistoryPolicySpec
from responses_api_agents.osworld_agent.runner_registry import RunnerSpec, resolve_runner_spec
from responses_api_agents.osworld_agent.super35_protocol import (
    SUPER35_PROTOCOL,
    SUPER35_PROTOCOL_ID,
    SUPER35_TOOLS,
    parse_super35_response,
)


def xml(name: str, arguments: dict[str, str]) -> str:
    parameters = "".join(f"<parameter={key}>\n{value}\n</parameter>" for key, value in arguments.items())
    return f"<tool_call><function={name}>{parameters}</function></tool_call>"


def agent(**kwargs: Any) -> OSWorldProtocolAgent:
    return OSWorldProtocolAgent(
        model="test-super35",
        max_steps=100,
        model_protocol_id=SUPER35_PROTOCOL_ID,
        history_policy=HistoryPolicySpec.sink_window(sink=1, low_water=4, high_water=4),
        **kwargs,
    )


def parse(response: object) -> tuple[str, list[str], dict[str, Any]]:
    return parse_super35_response(response, screen_size=(1920, 1080), coordinate_type="relative", thinking=True)


def test_xml_projects_code_but_preserves_original_code_for_assistant_history() -> None:
    response = {
        "content": "Click the target.\n" + xml("exec_py", {"code_block": "pyautogui.click(0.5, 0.25)"}),
        "reasoning_content": "The target is visible.",
    }
    summary, actions, info = parse(response)
    assert summary == "Click the target."
    assert actions == ["pyautogui.click(960, 270)"]
    assert info["thought"] == "The target is visible."
    assert "pyautogui.click(0.5, 0.25)" in info["tool_call_xml"]
    assert "960" not in info["tool_call_xml"]


@pytest.mark.parametrize(
    ("name", "arguments", "expected"),
    [
        ("computer.wait", {}, "WAIT"),
        ("computer.terminate", {"status": "success"}, "DONE"),
        ("computer.terminate", {"status": "failure", "answer": "The required device is absent."}, "FAIL"),
    ],
)
def test_control_calls_require_explicit_tool_semantics(name: str, arguments: dict[str, str], expected: str) -> None:
    assert parse(xml(name, arguments))[1] == [expected]


def test_inline_thinking_and_python_comparisons_survive_xml_parsing() -> None:
    code = "if 1 < 2:\n    pyautogui.click(x=0.25, y=0.5)"
    _, actions, info = parse(
        "<think>Choose the visible target.</think>Click.\n" + xml("exec_py", {"code_block": code})
    )
    assert actions == ["if 1 < 2:\n    pyautogui.click(x=480, y=540)"]
    assert info["thought"] == "Choose the visible target."


def test_scroll_projects_positions_without_scaling_the_scroll_amount() -> None:
    assert parse(xml("exec_py", {"code_block": "pyautogui.scroll(-3, x=0.5, y=0.25)"}))[1] == [
        "pyautogui.scroll(-3, x=960, y=270)"
    ]


@pytest.mark.parametrize(
    "response",
    [
        xml("pyautogui.tripleClick", {"x": "0.5", "y": "0.5"}),
        xml("computer.terminate", {"answer": "success"}),
        xml("computer.terminate", {"status": "done"}),
        xml("computer.wait", {"seconds": "20"}),
        xml("exec_py", {"code_block": "pyautogui.click(960, 540)"}),
        xml("exec_py", {"code_block": "pyautogui.click(x, 0.5)"}),
        xml("exec_py", {"code_block": "pyautogui.click(True, 0.5)"}),
        xml("exec_py", {"code_block": "# no executable action"}),
        xml("exec_py", {"code_block": "pyautogui.click("}),
        xml("computer.wait", {}) + xml("computer.wait", {}),
        xml("computer.wait", {}) + "I am done.",
        "<tool_call><function=exec_py><parameter=code_block>pass</function></tool_call>",
        "<tool_call><function=exec_py><parameter=code_block>pass</parameter>"
        "<parameter=code_block>pass</parameter></function></tool_call>",
    ],
)
def test_invalid_samples_are_rejected_without_inventing_a_terminal_action(response: str) -> None:
    with pytest.raises((ValueError, SyntaxError)):
        parse(response)


def test_super35_rejects_server_extracted_calls_instead_of_trusting_repaired_xml() -> None:
    instance = agent(parse_retries=1)
    instance.call_llm = lambda *_: {
        "content": None,
        "reasoning_content": "Click the target.",
        "finish_reason": "tool_calls",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "exec_py",
                    "arguments": json.dumps({"code_block": "pyautogui.click(0.5, 0.25)"}),
                },
            }
        ],
    }
    _, actions, info = instance.predict("Click the target.", {"screenshot": b"screen"})
    assert actions == []
    assert info["agent_outcome"] == "model_response_unparseable"
    assert info["stop_rollout"] and info["model_call_completed"]


def test_missing_opening_tool_tag_is_rejected_then_retried_without_repair() -> None:
    # Shape observed in a real rollout: the server had extracted a valid click
    # despite the original generation omitting its opening <tool_call> tag.
    malformed = (
        "<function=exec_py>\n<parameter=code_block>\npyautogui.click(0.119, 0.069)\n"
        "</parameter>\n</function>\n</tool_call>"
    )
    replies = [
        {"content": malformed, "finish_reason": "stop"},
        {"content": xml("computer.wait", {}), "finish_reason": "stop"},
    ]
    instance = agent(parse_retries=2, parse_error_feedback=True)
    instance.call_llm = lambda *_: replies.pop(0)
    _, actions, info = instance.predict("Task", {"screenshot": b"screen"})
    assert actions == ["WAIT"]
    assert info["model_calls"][0]["response"]["content"] == malformed
    assert info["model_calls"][0]["accepted"] is False
    assert info["model_calls"][0]["parsed_actions"] == []
    assert info["model_calls"][1]["accepted"] is True


def test_history_retains_initial_and_recent_images_and_all_assistant_tool_turns() -> None:
    instance = agent()
    payloads = []

    def reply(payload, _model=None):
        payloads.append(payload)
        return {
            "content": xml("exec_py", {"code_block": "pyautogui.click(0.5, 0.25)"}),
            "reasoning_content": f"reasoning-{len(payloads)}",
        }

    instance.call_llm = reply
    for turn in range(6):
        instance.predict("Click the target.", {"screenshot": f"screen-{turn}".encode()})
    messages = payloads[-1]["messages"]
    assert messages[1]["content"][0]["text"] == "Task: Click the target.\n"
    images = [
        base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
        for message in messages
        if isinstance(message["content"], list)
        for part in message["content"]
        if part["type"] == "image_url"
    ]
    assert images == [b"screen-0", b"screen-3", b"screen-4", b"screen-5"]
    assistants = [message["content"] for message in messages if message["role"] == "assistant"]
    assert len(assistants) == 5
    for turn, content in enumerate(assistants, 1):
        assert f"reasoning-{turn}" in content
        assert "<function=exec_py>" in content
    assert not any(message["role"] == "tool" for message in messages)
    assert payloads[-1]["tools"] == list(SUPER35_TOOLS)
    assert payloads[-1]["tool_choice"] == "none"
    assert payloads[-1]["skip_special_tokens"] is False
    assert payloads[-1]["_nemo_gym_preserve_response"] is True
    assert "<tools>" not in messages[0]["content"]
    assert payloads[-1]["chat_template_kwargs"] == {"enable_thinking": True, "truncate_history_thinking": False}


def test_non_thinking_requests_and_history_agree() -> None:
    instance = agent(thinking=False)
    payloads = []
    instance.call_llm = lambda payload, *_: payloads.append(payload) or {"content": xml("computer.wait", {})}
    for _ in range(2):
        instance.predict("Wait.", {"screenshot": b"screen"})
    assert payloads[-1]["chat_template_kwargs"] == {"enable_thinking": False, "truncate_history_thinking": False}
    assert "<think>" not in payloads[-1]["messages"][2]["content"]


def test_optional_step_guidance_is_attached_only_to_the_current_observation() -> None:
    instance = agent(pre_done_checklist=True)
    payloads = []
    instance.call_llm = lambda payload, *_: payloads.append(payload) or {"content": xml("computer.wait", {})}
    for _ in range(2):
        instance.predict("Task", {"screenshot": b"screen"})
    messages = payloads[-1]["messages"]
    assert "Before returning computer.terminate" not in str(messages[1]["content"])
    assert "Before returning computer.terminate" in messages[-1]["content"][-1]["text"]


def test_parse_exhaustion_and_transport_errors_remain_distinct_from_explicit_failure() -> None:
    instance = agent(parse_retries=2)
    instance.call_llm = lambda *_: {"content": "not a tool call"}
    _, actions, info = instance.predict("Task", {"screenshot": b"screen"})
    assert actions == [] and info["stop_rollout"] and info["model_call_completed"]
    assert info["agent_outcome"] == "model_response_unparseable"
    instance.call_llm = lambda *_: (_ for _ in ()).throw(ConnectionError("endpoint unavailable"))
    _, actions, info = instance.predict("Task", {"screenshot": b"screen"})
    assert actions == [] and info["stop_rollout"] and not info["model_call_completed"]
    instance.call_llm = lambda *_: {"content": xml("computer.terminate", {"status": "failure"})}
    assert instance.predict("Task", {"screenshot": b"screen"})[1] == ["FAIL"]


def test_truncated_native_call_is_never_executed() -> None:
    instance = agent(parse_retries=1)
    instance.call_llm = lambda *_: {
        "content": xml("computer.terminate", {"status": "success"}),
        "finish_reason": "length",
    }
    _, actions, info = instance.predict("Task", {"screenshot": b"screen"})
    assert actions == []
    assert info["agent_outcome"] == "model_output_truncated"


def test_model_protocol_default_and_train_eval_parity_are_resolved_by_registry() -> None:
    spec = resolve_runner_spec("nemotron_super35_agent")
    assert spec.kind == "model_protocol_agent"
    assert spec.agent_class_path.endswith(".OSWorldProtocolAgent")
    kwargs = dict(
        runner_spec=spec,
        max_steps=100,
        max_trajectory_length=3,
        explicit_history_policy=HistoryPolicySpec.sink_window(sink=1, low_water=4, high_water=4).to_config(),
        model_protocol_id=None,
        parity_mode="strict",
    )
    training = resolve_agent_contract(**kwargs, rollout_purpose="training")
    evaluation = resolve_agent_contract(**kwargs, rollout_purpose="evaluation")
    assert training.model_protocol_id == SUPER35_PROTOCOL_ID
    assert_train_eval_parity(training, evaluation, parity_mode="strict")
    changed = resolve_agent_contract(**{**kwargs, "max_steps": 99}, rollout_purpose="evaluation")
    with pytest.raises(ValueError, match="contracts differ"):
        assert_train_eval_parity(training, changed, parity_mode="strict")


def test_another_model_can_register_its_protocol_without_a_new_agent_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    protocol = replace(SUPER35_PROTOCOL, protocol_id="test-other-model-v1", parser_id="test-other-parser-v1")
    monkeypatch.setitem(MODEL_PROTOCOL_REGISTRY, protocol.protocol_id, protocol)
    spec = RunnerSpec(name="other-model", kind="model_protocol_agent", model_protocol_id=protocol.protocol_id)
    resolved = resolve_agent_contract(
        runner_spec=spec,
        max_steps=100,
        max_trajectory_length=3,
        explicit_history_policy=None,
        model_protocol_id=None,
        rollout_purpose="evaluation",
        parity_mode="strict",
    )
    assert resolved.model_protocol_id == protocol.protocol_id
    assert resolve_model_protocol(resolved.model_protocol_id) is protocol
    assert (
        resolved.contract["model_protocol"]["wire_contract_id"]
        != SUPER35_PROTOCOL.to_contract(thinking=True)["wire_contract_id"]
    )


def test_renderer_and_request_changes_have_distinct_wire_contracts() -> None:
    baseline = SUPER35_PROTOCOL.to_contract(thinking=True)["wire_contract_id"]
    for changed in [
        replace(SUPER35_PROTOCOL, message_renderer_id="another-renderer-v1"),
        replace(SUPER35_PROTOCOL, chat_template_kwargs_thinking={"enable_thinking": False}),
        replace(SUPER35_PROTOCOL, tools=(), tool_choice="auto"),
        replace(SUPER35_PROTOCOL, tool_choice="auto"),
        replace(SUPER35_PROTOCOL, skip_special_tokens=True),
        replace(SUPER35_PROTOCOL, python_execution="direct"),
        replace(SUPER35_PROTOCOL, preserve_response=False),
    ]:
        assert changed.to_contract(thinking=True)["wire_contract_id"] != baseline


@pytest.mark.parametrize("log_model_io", [False, True])
@pytest.mark.parametrize("reasoning_field", ["reasoning_content", "reasoning"])
def test_gym_transport_preserves_native_calls_and_forwards_tools(
    monkeypatch, tmp_path, log_model_io, reasoning_field
) -> None:
    from responses_api_agents.osworld_agent import app

    client = MagicMock()
    message = SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                function=SimpleNamespace(
                    name="exec_py", arguments=json.dumps({"code_block": "pyautogui.click(0.5, 0.25)"})
                )
            )
        ],
        model_extra={},
    )
    setattr(message, reasoning_field, "\nClick the target.\n")
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")]
    )
    monkeypatch.setattr(app, "_build_policy_openai_client", lambda **_: client)
    if log_model_io:
        monkeypatch.setenv("OSWORLD_MODEL_IO_LOG", str(tmp_path / "model-io.jsonl"))
    caller = app._build_messages_model_fn(
        base_url="http://policy/v1", model_name="policy", api_key="test-key"
    )  # pragma: allowlist secret
    response = caller(
        [{"role": "user", "content": "Task"}],
        {
            "_nemo_gym_return_message": True,
            "_nemo_gym_require_stop": True,
            "_nemo_gym_preserve_response": True,
            "tools": list(SUPER35_TOOLS),
            "tool_choice": "auto",
            "chat_template_kwargs": {"enable_thinking": True, "truncate_history_thinking": False},
        },
    )
    assert client.chat.completions.create.call_args.kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": True, "truncate_history_thinking": False}
    }
    assert parse(response)[1] == ["pyautogui.click(960, 270)"]
    assert response[reasoning_field] == "\nClick the target.\n"
    assert parse(response)[2]["thought"] == "Click the target."
    assert response["content"] is None
    assert response["tool_calls"][0]["function"]["arguments"] == message.tool_calls[0].function.arguments
    assert client.chat.completions.create.call_args.kwargs["tools"] == list(SUPER35_TOOLS)
    assert client.chat.completions.create.call_args.kwargs["tool_choice"] == "auto"


def test_transport_does_not_repair_truncated_native_arguments_or_content() -> None:
    from responses_api_agents.osworld_agent.app import _normalize_chat_message

    arguments = '{"code_block":"pass"'
    message = SimpleNamespace(
        content=None,
        tool_calls=[SimpleNamespace(function=SimpleNamespace(name="exec_py", arguments=arguments))],
        reasoning_content="raw reasoning",
        model_extra={},
    )
    response = _normalize_chat_message(message, structured=True, preserve_response=True)
    assert response["tool_calls"][0]["function"]["arguments"] == arguments
    with pytest.raises(json.JSONDecodeError):
        parse(response)
    message.tool_calls = []
    message.content = "```python\nprint('unfinished')"
    response = _normalize_chat_message(message, structured=True, preserve_response=True)
    assert response["content"] == message.content
    with pytest.raises(ValueError):
        parse(response)


@pytest.mark.parametrize("log_model_io", [False, True])
def test_transport_keeps_raw_xml_and_special_tokens(monkeypatch, tmp_path, log_model_io) -> None:
    from responses_api_agents.osworld_agent import app

    raw = "\n<function=exec_py><parameter=code_block>pass</parameter></function></tool_call>\n"
    message = SimpleNamespace(content=raw, tool_calls=[], reasoning="\nOriginal reasoning.\n", model_extra={})
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")]
    )
    monkeypatch.setattr(app, "_build_policy_openai_client", lambda **_: client)
    if log_model_io:
        monkeypatch.setenv("OSWORLD_MODEL_IO_LOG", str(tmp_path / "model-io.jsonl"))
    caller = app._build_messages_model_fn(
        base_url="http://policy/v1", model_name="policy", api_key="test-key"
    )  # pragma: allowlist secret
    response = caller(
        [{"role": "user", "content": "Task"}],
        {
            "_nemo_gym_return_message": True,
            "_nemo_gym_preserve_response": True,
            "tools": list(SUPER35_TOOLS),
            "tool_choice": "none",
            "skip_special_tokens": False,
            "chat_template_kwargs": {"enable_thinking": True, "truncate_history_thinking": False},
        },
    )
    sent = client.chat.completions.create.call_args.kwargs
    assert sent["tool_choice"] == "none" and sent["tools"] == list(SUPER35_TOOLS)
    assert sent["extra_body"] == {
        "skip_special_tokens": False,
        "chat_template_kwargs": {"enable_thinking": True, "truncate_history_thinking": False},
    }
    assert response["content"] == raw and response["reasoning"] == message.reasoning
    with pytest.raises(ValueError, match="one complete"):
        parse(response)


@pytest.mark.parametrize("coordinate_type", ["absolute", "qwen25"])
def test_unsupported_coordinates_fail_at_both_configuration_entrypoints(coordinate_type: str) -> None:
    with pytest.raises(ValueError, match="does not support coordinate_type"):
        agent(coordinate_type=coordinate_type)
    spec = resolve_runner_spec("nemotron_super35_agent", agent_kwargs={"coordinate_type": coordinate_type})
    with pytest.raises(ValueError, match="does not support coordinate_type"):
        resolve_agent_contract(
            runner_spec=spec,
            max_steps=100,
            max_trajectory_length=3,
            explicit_history_policy=None,
            model_protocol_id=None,
            rollout_purpose="evaluation",
            parity_mode="strict",
        )


def test_super35_profile_wires_model_agent_history_and_request_forwarding() -> None:
    root = Path(__file__).resolve().parents[3]
    config = yaml.safe_load((root / "benchmarks/osworld/configs/osworld_agent_super35.yaml").read_text())
    model = config["policy_model"]["responses_api_models"]["vllm_model"]
    assert model["forward_request_chat_template_kwargs"] is True
    assert model["chat_template_kwargs"]["truncate_history_thinking"] is False
    assert model["extra_body"] == {"skip_special_tokens": False}
    settings = config["osworld_super35_agent"]["responses_api_agents"]["osworld_agent"]
    assert settings["runner_name"] == "nemotron_super35_agent"
    assert settings["model_protocol_id"] == SUPER35_PROTOCOL_ID
    assert HistoryPolicySpec.from_mapping(settings["history_policy"]).sink == 1
    assert settings["history_policy"]["params"]["low_water"] == 4
