# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import json
from typing import Any
from unittest.mock import MagicMock

from fastapi import Request, Response

from nemo_gym.context_compaction import ContextCompactedResponse
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from nemo_gym.visual_history import (
    CompactionScheduleConfig,
    HistoryPolicyConfig,
    RecencyHistoryPolicyConfig,
    VisualHistoryConfig,
)
from responses_api_agents.nemotron_osworld import cc_app
from responses_api_agents.nemotron_osworld.cc_app import (
    NemotronOSWorldCCAgent,
    NemotronOSWorldCCAgentConfig,
    NemotronOSWorldCCRunRequest,
)


class _HTTPResponse:
    def __init__(self, payload: Any, cookies: dict[str, str] | None = None):
        self._payload = json.dumps(payload).encode()
        self.cookies = cookies or {}
        self.ok = True
        self.status = 200

    async def read(self) -> bytes:
        return self._payload


def _model_response(turn: int, required_prefix: list[int] | None) -> dict[str, Any]:
    terminal = turn == 4
    text = "finish" if terminal else f"click {turn}"
    return {
        "id": f"resp-{turn}",
        "created_at": 1.0,
        "model": "vllm_local",
        "object": "response",
        "output": [
            {
                "id": f"msg-{turn}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
                "prompt_token_ids": [*(required_prefix or []), 1000 + turn],
                "generation_token_ids": [2000 + turn],
                "generation_log_probs": [-0.1],
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "none",
        "tools": [],
    }


class _ResponsesLoopClient:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []
        self.turn = 0

    async def post(
        self,
        server_name: str,
        url_path: str,
        json: Any = None,
        cookies: Any = None,
        **_: Any,
    ) -> _HTTPResponse:
        self.calls.append(
            {
                "server_name": server_name,
                "url_path": url_path,
                "json": json,
                "cookies": dict(cookies or {}),
            }
        )
        if url_path == "/screenshot":
            image = base64.b64encode(f"png-{self.turn}".encode()).decode()
            return _HTTPResponse({"image_base64": image})
        if url_path == "/execute":
            return _HTTPResponse({"output": "ok", "returncode": 0})
        assert url_path == "/v1/responses"
        required_prefix = list(json.required_prefix_token_ids or [])
        result = _HTTPResponse(_model_response(self.turn, required_prefix))
        self.turn += 1
        return result


def _config() -> NemotronOSWorldCCAgentConfig:
    return NemotronOSWorldCCAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="nemotron_osworld_cc",
        resources_server={"type": "resources_servers", "name": "resources"},
        model_server={"type": "responses_api_models", "name": "model"},
        max_steps=5,
        sleep_after_execution_s=0.0,
        visual_history=VisualHistoryConfig(
            enabled=True,
            shadow_only=False,
            policy=HistoryPolicyConfig(
                type="recency",
                config=RecencyHistoryPolicyConfig(
                    keep_all_text=True,
                    keep_last_image_groups=3,
                ),
            ),
            schedule=CompactionScheduleConfig(
                type="turn_chunked_recency",
                actions_per_chunk=2,
            ),
        ),
    )


def _fake_contract():
    def parse(response, _screen_size, _coordinate_type, *, thinking):
        assert thinking is True
        turn = int(response["content"].rsplit(" ", 1)[-1]) if response["content"].startswith("click") else 4
        if turn == 4:
            return "terminate", ["DONE"], {"action": "terminate", "code": "DONE"}
        code = f"pyautogui.click({turn + 1}, {turn + 2})"
        return f"click-{turn}", [code], {"action": f"click-{turn}", "code": code}

    return (
        "thinking prompt {password}",
        "non-thinking prompt {password}",
        "# Task Instruction:\n{instruction}\n\n",
        parse,
    )


async def test_k2_n3_responses_path_records_every_action(monkeypatch) -> None:
    monkeypatch.setattr(cc_app, "_nemotron_contract", _fake_contract)
    client = _ResponsesLoopClient()
    agent = NemotronOSWorldCCAgent(
        config=_config(),
        server_client=MagicMock(spec=ServerClient),
    )
    agent.server_client = client
    request = MagicMock(spec=Request)
    request.cookies = {"_nemo_gym_osworld_cc_rollout_id": "rollout-7"}
    request.session = {"session_id": "web-session"}

    result = await agent.responses(
        request=request,
        response=Response(),
        body=NeMoGymResponseCreateParamsNonStreaming(input="do the task"),
    )

    assert isinstance(result, ContextCompactedResponse)
    assert result.context_compaction_contract.rollout_id == "rollout-7"
    assert len(result.completion_evidence) == 5
    assert [record.actual_action_count for record in result.chunk_records] == [2, 2, 1]
    # The K=2 boundary before step 3 is a no-op because N=3 still retains all
    # screenshots. The first materialized rewrite therefore occurs at step 5.
    assert [event.applies_to_step for event in result.boundary_events] == [5]

    model_calls = [call for call in client.calls if call["url_path"] == "/v1/responses"]
    assert len(model_calls) == 5
    assert all(call["server_name"] == "model" for call in model_calls)
    assert [call["json"].required_prefix_token_ids for call in model_calls] == [
        None,
        [1000, 2000],
        [1000, 2000, 1001, 2001],
        [1000, 2000, 1001, 2001, 1002, 2002],
        None,
    ]
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
    # The initial screenshot is protected, so the compacted view contains it
    # plus the three most recent non-initial image groups.
    assert image_counts == [1, 2, 3, 4, 4]
    assert all(call["url_path"] != "/v1/chat/completions" for call in client.calls)


class _RunClient:
    def __init__(self, compacted: ContextCompactedResponse):
        self.compacted = compacted
        self.calls: list[dict[str, Any]] = []

    async def post(
        self,
        server_name: str,
        url_path: str,
        json: Any = None,
        cookies: Any = None,
        **_: Any,
    ) -> _HTTPResponse:
        self.calls.append({"server_name": server_name, "url_path": url_path, "json": json})
        if url_path == "/seed_session":
            return _HTTPResponse({"sandbox_id": "sandbox"}, {"session": "resources-session"})
        if url_path == "/v1/responses":
            return _HTTPResponse(self.compacted.model_dump(mode="json"))
        if url_path == "/action_history":
            return _HTTPResponse({"action_history": ["DONE"]})
        assert url_path == "/verify"
        return _HTTPResponse({**json, "reward": 1.0})


async def test_run_stamps_schema3_identity_after_official_verify(monkeypatch) -> None:
    monkeypatch.setattr(cc_app, "_nemotron_contract", _fake_contract)
    loop_client = _ResponsesLoopClient()
    inner = NemotronOSWorldCCAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
    inner.server_client = loop_client
    inner_request = MagicMock(spec=Request)
    inner_request.cookies = {"_nemo_gym_osworld_cc_rollout_id": "rollout-7"}
    inner_request.session = {"session_id": "inner-session"}
    compacted = await inner.responses(
        request=inner_request,
        response=Response(),
        body=NeMoGymResponseCreateParamsNonStreaming(input="do the task"),
    )
    assert isinstance(compacted, ContextCompactedResponse)

    run_client = _RunClient(compacted)
    agent = NemotronOSWorldCCAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
    agent.server_client = run_client
    request = MagicMock(spec=Request)
    request.cookies = {}
    result = await agent.run(
        request=request,
        body=NemotronOSWorldCCRunRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="do the task"),
            verifier_metadata={"id": "task-7", "evaluator": {"func": "exact_match"}},
            context_compaction_rollout_id="rollout-7",
            context_compaction_group_id="group-2",
            context_compaction_task_id="task-7",
            context_compaction_rollout_index=3,
            context_compaction_attempt_index=1,
        ),
    )

    verify_index = next(i for i, call in enumerate(run_client.calls) if call["url_path"] == "/verify")
    assert verify_index < len(run_client.calls)
    verifier_contract = run_client.calls[verify_index]["json"]["response"]["context_compaction_contract"]
    assert verifier_contract["schema_version"] == 2
    assert verifier_contract["group_id"] is None
    contract = result.response.context_compaction_contract
    assert contract.schema_version == 3
    assert contract.rollout_id == "rollout-7"
    assert contract.group_id == "group-2"
    assert contract.task_id == "task-7"
    assert contract.rollout_index == 3
    assert contract.attempt_index == 1
    assert len(result.response.model_call_metadata) == 5
    assert not hasattr(result.response, "completion_evidence")
