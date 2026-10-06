# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import json
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import Request
from omegaconf import OmegaConf

from nemo_gym.context_management.client import PARENT_HEADER
from nemo_gym.server_utils import ServerClient
from responses_api_agents.nemotron_osworld import cc_app
from responses_api_agents.nemotron_osworld.cc_app import (
    NemotronOSWorldCCAgent,
    NemotronOSWorldCCAgentConfig,
    NemotronOSWorldCCRunRequest,
)
from responses_api_models.vllm_model.app import VLLMModelConfig


class _HTTPResponse:
    def __init__(self, payload: Any, *, cookies: dict[str, str] | None = None, status: int = 200):
        self._payload = json.dumps(payload).encode()
        self.cookies = cookies or {}
        self.status = status
        self.ok = status < 400

    async def read(self) -> bytes:
        return self._payload


def _model_response(index: int, text: str) -> dict[str, Any]:
    return {
        "id": f"response-{index}",
        "created_at": 1.0,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": None,
        "model": "vllm_local",
        "object": "response",
        "output": [
            {
                "id": f"message-{index}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "none",
        "tools": [],
    }


class _RunClient:
    global_config_dict = {"token_id_capture": {"enabled": True}}

    def __init__(
        self,
        *,
        screenshots: list[str] | None = None,
        responses: list[dict[str, Any]] | None = None,
        failure: str | None = None,
        model_delay_s: float = 0.0,
        verify_error: str | None = None,
    ):
        self.screenshots = list(screenshots or ["same", "second", "same"])
        self.responses = list(
            responses
            or [
                _model_response(1, "click 1"),
                _model_response(2, "click 2"),
                _model_response(3, "finish"),
            ]
        )
        self.failure = failure
        self.model_delay_s = model_delay_s
        self.verify_error = verify_error
        self.calls: list[dict[str, Any]] = []
        self.screenshot_index = 0

    async def post(self, **kwargs: Any) -> _HTTPResponse:
        self.calls.append(deepcopy(kwargs))
        path = kwargs["url_path"]
        if path == "/seed_session":
            return _HTTPResponse({"sandbox_id": "sandbox-1"}, cookies={"session": "resources-session"})
        if path == "/screenshot":
            if self.failure == "screenshot":
                raise ConnectionError("screenshot transport failed")
            image = self.screenshots[min(self.screenshot_index, len(self.screenshots) - 1)]
            self.screenshot_index += 1
            return _HTTPResponse({"image_base64": base64.b64encode(image.encode()).decode()})
        if path.endswith("/v1/responses"):
            if self.model_delay_s:
                await asyncio.sleep(self.model_delay_s)
            if self.failure == "model":
                raise ConnectionError("model transport failed")
            return _HTTPResponse(self.responses.pop(0), cookies={"model": "model-session"})
        if path == "/execute":
            if self.failure == "execute":
                raise ConnectionError("execute acknowledgement lost")
            return _HTTPResponse({"output": "ok", "returncode": 0})
        if path == "/verify":
            return _HTTPResponse(
                kwargs["json"]
                | {
                    "reward": 0.0 if self.verify_error else 1.0,
                    "official_evaluator": True,
                    "verify_error": self.verify_error,
                }
            )
        if path == "/release":
            return _HTTPResponse({"released": True})
        raise AssertionError(f"Unexpected path: {path}")


def _fake_contract():
    def parse(response, _screen_size, _coordinate_type, *, thinking):
        assert thinking is True
        text = response["content"]
        if text == "finish":
            return "terminate", ["DONE"], {"action": "terminate", "code": "DONE"}
        index = int(text.rsplit(" ", 1)[-1])
        code = f"pyautogui.click({index}, {index + 1})"
        return f"click-{index}", [code], {"action": f"click-{index}", "code": code}

    return (
        "thinking prompt {password}",
        "non-thinking prompt {password}",
        "# Task Instruction:\n{instruction}\n\n",
        parse,
    )


def test_normalize_vendor_parser_input_restores_implicit_action_header() -> None:
    raw = {
        "content": ("Click the File menu to open it.\n## Code:\n```python\npyautogui.click(0.1, 0.2)\n```"),
        "reasoning_content": "inspect the menu",
    }

    normalized = cc_app._normalize_vendor_parser_input(raw)

    assert normalized["content"] == (
        "## Action:\nClick the File menu to open it.\n\n## Code:\n```python\npyautogui.click(0.1, 0.2)\n```"
    )
    assert normalized["reasoning_content"] == raw["reasoning_content"]


def test_normalize_vendor_parser_input_preserves_explicit_action() -> None:
    response = {
        "content": ("## Action:\nClick the File menu.\n## Code:\n```python\npyautogui.click(0.1, 0.2)\n```"),
        "reasoning_content": "",
    }

    assert cc_app._normalize_vendor_parser_input(response) is response


def _config(**overrides: Any) -> NemotronOSWorldCCAgentConfig:
    values = {
        "host": "0.0.0.0",
        "port": 8080,
        "entrypoint": "",
        "name": "nemotron_osworld",
        "resources_server": {"type": "resources_servers", "name": "resources"},
        "model_server": {"type": "responses_api_models", "name": "model"},
        "max_steps": 3,
        "sleep_after_execution_s": 0.0,
        "llm_timeout_s": 1.0,
        "rollout_timeout_s": 10.0,
        "context_history": {
            "enabled": True,
            "policy": {
                "type": "recency",
                "config": {
                    "images": {
                        "enabled": True,
                        "protect_initial_context": False,
                        "keep_last_groups": 1,
                    }
                },
            },
            "schedule": {"type": "turn_chunked_recency", "actions_per_chunk": 2},
        },
    }
    values.update(overrides)
    return NemotronOSWorldCCAgentConfig.model_validate(values)


def _agent(client: _RunClient, **config: Any) -> NemotronOSWorldCCAgent:
    server = NemotronOSWorldCCAgent(
        config=_config(**config),
        server_client=MagicMock(spec=ServerClient),
    )
    server.server_client = client
    return server


def _body(owner: str | None = "dispatch_g0") -> NemotronOSWorldCCRunRequest:
    payload: dict[str, Any] = {
        "responses_create_params": {"input": "do the task"},
        "verifier_metadata": {"id": "task-1", "evaluator": {"func": "exact_match"}},
    }
    if owner is not None:
        payload["_ng_rollout_id"] = owner
    return NemotronOSWorldCCRunRequest.model_validate(payload)


def _request() -> Request:
    request = MagicMock(spec=Request)
    request.cookies = {"upstream": "kept"}
    return request


def test_cc_config_uses_nested_history_and_ordinary_token_free_vllm():
    root = Path(__file__).resolve().parents[3]
    config = OmegaConf.merge(
        {
            "policy_base_url": "http://worker.test/v1",
            "policy_api_key": "test-key",
            "policy_model_name": "test-model",
        },
        OmegaConf.load(root / "responses_api_models/vllm_model/configs/vllm_model.yaml"),
        OmegaConf.load(root / "responses_api_agents/nemotron_osworld/configs/nemotron_osworld_cc.yaml"),
    )
    agent_values = OmegaConf.to_container(
        config.nemotron_osworld.responses_api_agents.nemotron_osworld,
        resolve=True,
    )
    agent_config = NemotronOSWorldCCAgentConfig.model_validate(
        agent_values | {"host": "127.0.0.1", "port": 8080, "name": "nemotron_osworld"}
    )
    assert agent_config.context_history.policy.config.images.enabled is True
    assert agent_config.context_history.policy.config.images.keep_last_groups == 3

    model_types = config.policy_model.responses_api_models
    assert list(model_types) == ["vllm_model"]
    model_config = VLLMModelConfig.model_validate(
        OmegaConf.to_container(model_types.vllm_model, resolve=True)
        | {"host": "127.0.0.1", "port": 8081, "name": "policy_model"}
    )
    assert model_config.return_token_id_information is False


@pytest.mark.parametrize("suffix", ["", "_a" + "1" * 32])
async def test_run_emits_selected_parent_chain_new_root_and_token_free_receipt(monkeypatch, suffix):
    monkeypatch.setattr(cc_app, "_nemotron_contract", _fake_contract)
    owner = "dispatch_g0" + suffix
    transport = _RunClient()
    result = await _agent(transport).run(_request(), _body(owner))

    model_calls = [call for call in transport.calls if call["url_path"].endswith("/v1/responses")]
    assert [call["url_path"] for call in model_calls] == [
        f"/ng-rollout/{owner}_s0/training-token-capture/v1/responses",
        f"/ng-rollout/{owner}_s0/training-token-capture/v1/responses",
        f"/ng-rollout/{owner}_s1/training-token-capture/v1/responses",
    ]
    assert [json.loads(call["headers"][PARENT_HEADER]) for call in model_calls] == [
        None,
        "response-1",
        None,
    ]
    assert all(call["_retry"] is False for call in transport.calls)
    assert all(call["json"].previous_response_id is None for call in model_calls)
    assert all(call["json"].conversation is None for call in model_calls)
    assert all(call["json"].context_management is None for call in model_calls)
    assert all(call["json"].stream is None for call in model_calls)

    receipt = result.context_compaction_result
    assert receipt.logical_rollout_id == owner
    assert receipt.outcome == "completed"
    assert [[action.response_id for action in segment.selected_actions] for segment in receipt.segments] == [
        ["response-1", "response-2"],
        ["response-3"],
    ]
    assert result.response.id == receipt.segments[-1].selected_actions[-1].response_id == "response-3"
    wire_payload = result.model_dump(mode="json")
    assert wire_payload["response"]["id"] == "response-3"
    assert "context_compaction_result" in wire_payload
    assert "context_compaction_result" not in wire_payload["response"]
    assert [len(segment.media_occurrence_refs) for segment in receipt.segments] == [2, 1]
    first, second = receipt.segments
    assert first.media_occurrence_refs[0] == second.media_occurrence_refs[0]
    assert first.media_occurrence_refs[0] != first.media_occurrence_refs[1]
    assert [action.new_media_occurrence_refs for action in first.selected_actions] == [
        [first.media_occurrence_refs[0]],
        [first.media_occurrence_refs[1]],
    ]
    assert len(receipt.media_assets) == 2
    assert "prompt_token_ids" not in receipt.model_dump_json()
    assert "generation_token_ids" not in receipt.model_dump_json()
    assert "data:image" not in result.response.model_dump_json()

    verify = next(call for call in transport.calls if call["url_path"] == "/verify")
    assert verify["json"]["action_history"] == [
        "pyautogui.click(1, 2)",
        "pyautogui.click(2, 3)",
        "DONE",
    ]
    assert result.official_evaluator is True
    assert sum(call["url_path"] == "/release" for call in transport.calls) == 1


async def test_inference_only_run_generates_owner_without_capture_routes(monkeypatch):
    monkeypatch.setattr(cc_app, "_nemotron_contract", _fake_contract)
    transport = _RunClient()

    result = await _agent(transport, token_id_capture=False).run(_request(), _body(None))

    owner = result.context_compaction_result.logical_rollout_id
    assert owner.startswith("inference-")
    model_calls = [call for call in transport.calls if call["url_path"].endswith("/v1/responses")]
    assert [call["url_path"] for call in model_calls] == [
        f"/ng-rollout/{owner}_s0/v1/responses",
        f"/ng-rollout/{owner}_s0/v1/responses",
        f"/ng-rollout/{owner}_s1/v1/responses",
    ]
    assert all(call["headers"] == {} for call in model_calls)
    assert all("training-token-capture" not in call["url_path"] for call in model_calls)
    assert result.reward == 1.0


async def test_k2_n3_preserves_opening_screenshot_and_rewrites_on_fifth_action(monkeypatch):
    monkeypatch.setattr(cc_app, "_nemotron_contract", _fake_contract)
    transport = _RunClient(
        screenshots=["one", "two", "three", "four", "five"],
        responses=[
            _model_response(1, "click 1"),
            _model_response(2, "click 2"),
            _model_response(3, "click 3"),
            _model_response(4, "click 4"),
            _model_response(5, "finish"),
        ],
    )
    agent = _agent(
        transport,
        max_steps=5,
        context_history={
            "enabled": True,
            "policy": {
                "type": "recency",
                "config": {
                    "images": {
                        "enabled": True,
                        "protect_initial_context": True,
                        "keep_last_groups": 3,
                    }
                },
            },
            "schedule": {"type": "turn_chunked_recency", "actions_per_chunk": 2},
        },
    )

    result = await agent.run(_request(), _body())
    model_calls = [call for call in transport.calls if call["url_path"].endswith("/v1/responses")]
    image_counts = [
        sum(
            part.get("type") == "input_image"
            for item in call["json"].input
            for part in (
                item.model_dump().get("content", [])
                if hasattr(item, "model_dump") and isinstance(item.model_dump().get("content"), list)
                else []
            )
        )
        for call in model_calls
    ]

    assert image_counts == [1, 2, 3, 4, 4]
    assert [json.loads(call["headers"][PARENT_HEADER]) for call in model_calls] == [
        None,
        "response-1",
        "response-2",
        "response-3",
        None,
    ]
    assert [len(segment.selected_actions) for segment in result.context_compaction_result.segments] == [4, 1]


@pytest.mark.parametrize(
    "owner",
    [None, "unscoped", "dispatch_g0_a123", "dispatch_g0_a" + "z" * 32],
)
async def test_run_requires_exact_framework_owner_before_sandbox_mutation(owner):
    transport = _RunClient()
    with pytest.raises(ValueError, match="_ng_rollout_id|framework logical owner"):
        await _agent(transport).run(_request(), _body(owner))
    assert transport.calls == []


async def test_missing_capture_enablement_rejects_before_seed():
    transport = _RunClient()
    transport.global_config_dict = {}
    with pytest.raises(ValueError, match="requires token capture"):
        await _agent(transport).run(_request(), _body())
    assert transport.calls == []


@pytest.mark.parametrize("failure", ["screenshot", "model"])
async def test_pre_receipt_failure_is_not_retried_and_releases_sandbox(monkeypatch, failure):
    monkeypatch.setattr(cc_app, "_nemotron_contract", _fake_contract)
    transport = _RunClient(failure=failure)
    with pytest.raises(ConnectionError, match="transport failed"):
        await _agent(transport).run(_request(), _body())

    failed_path = "/screenshot" if failure == "screenshot" else "/v1/responses"
    assert (
        sum(
            call["url_path"] == failed_path
            or (failed_path == "/v1/responses" and call["url_path"].endswith(failed_path))
            for call in transport.calls
        )
        == 1
    )
    assert all(call["_retry"] is False for call in transport.calls)
    assert sum(call["url_path"] == "/verify" for call in transport.calls) == 0
    assert sum(call["url_path"] == "/release" for call in transport.calls) == 1


async def test_model_timeout_is_single_attempt_and_releases_sandbox(monkeypatch):
    monkeypatch.setattr(cc_app, "_nemotron_contract", _fake_contract)
    transport = _RunClient(model_delay_s=0.05)
    with pytest.raises(TimeoutError):
        await _agent(transport, llm_timeout_s=0.001).run(_request(), _body())
    assert sum(call["url_path"].endswith("/v1/responses") for call in transport.calls) == 1
    assert sum(call["url_path"] == "/release" for call in transport.calls) == 1


async def test_later_model_timeout_preserves_selected_trace_as_local_failure(
    monkeypatch,
):
    monkeypatch.setattr(cc_app, "_nemotron_contract", _fake_contract)
    transport = _RunClient()
    original_post = transport.post
    model_calls = 0

    async def delay_second_model_call(**kwargs):
        nonlocal model_calls
        if kwargs["url_path"].endswith("/v1/responses"):
            model_calls += 1
            if model_calls == 2:
                await asyncio.sleep(0.1)
        return await original_post(**kwargs)

    transport.post = delay_second_model_call
    result = await _agent(transport, llm_timeout_s=0.02).run(_request(), _body())

    assert model_calls == 2
    assert result.context_compaction_result.outcome == "execution_failure"
    assert result.instance_config["mask_sample"] is True
    verify = next(call for call in transport.calls if call["url_path"] == "/verify")
    assert verify["json"]["action_history"] == ["pyautogui.click(1, 2)", "FAIL"]


async def test_later_model_transport_failure_preserves_selected_trace_as_local_failure(
    monkeypatch,
):
    monkeypatch.setattr(cc_app, "_nemotron_contract", _fake_contract)
    transport = _RunClient()
    original_post = transport.post
    model_calls = 0

    async def fail_second_model_call(**kwargs):
        nonlocal model_calls
        if kwargs["url_path"].endswith("/v1/responses"):
            model_calls += 1
            if model_calls == 2:
                raise ConnectionError("model endpoint disappeared")
        return await original_post(**kwargs)

    transport.post = fail_second_model_call
    result = await _agent(transport).run(_request(), _body())

    assert model_calls == 2
    assert result.context_compaction_result.outcome == "execution_failure"
    assert result.instance_config["mask_sample"] is True
    verify = next(call for call in transport.calls if call["url_path"] == "/verify")
    assert verify["json"]["action_history"] == ["pyautogui.click(1, 2)", "FAIL"]
    assert sum(call["url_path"] == "/release" for call in transport.calls) == 1


async def test_action_transport_failure_is_rollout_local_and_officially_verified(monkeypatch):
    monkeypatch.setattr(cc_app, "_nemotron_contract", _fake_contract)
    transport = _RunClient(failure="execute")
    result = await _agent(transport).run(_request(), _body())

    assert result.response.id == "response-1"
    assert result.context_compaction_result.outcome == "execution_failure"
    assert result.instance_config["mask_sample"] is True
    assert sum(call["url_path"] == "/execute" for call in transport.calls) == 1
    verify = next(call for call in transport.calls if call["url_path"] == "/verify")
    assert verify["json"]["action_history"] == ["pyautogui.click(1, 2)", "FAIL"]
    assert sum(call["url_path"] == "/release" for call in transport.calls) == 1


async def test_step_cap_preserves_last_selected_response_and_marks_max_steps(monkeypatch):
    monkeypatch.setattr(cc_app, "_nemotron_contract", _fake_contract)
    transport = _RunClient(
        screenshots=["one", "two"],
        responses=[_model_response(1, "click 1"), _model_response(2, "click 2")],
    )
    result = await _agent(transport, max_steps=2).run(_request(), _body())

    assert result.response.id == "response-2"
    assert result.context_compaction_result.outcome == "max_steps"
    verify = next(call for call in transport.calls if call["url_path"] == "/verify")
    assert verify["json"]["action_history"] == ["pyautogui.click(1, 2)", "FAIL"]


async def test_official_evaluator_failure_marks_only_this_sample(monkeypatch):
    monkeypatch.setattr(cc_app, "_nemotron_contract", _fake_contract)
    result = await _agent(_RunClient(verify_error="evaluate_timeout")).run(_request(), _body())

    assert result.reward == 0.0
    assert result.verify_error == "evaluate_timeout"
    assert result.instance_config["mask_sample"] is True
    assert result.context_compaction_result.outcome == "completed"
