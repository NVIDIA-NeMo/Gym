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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Response

import responses_api_agents.scicode_agent.app as app
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.scicode_agent.app import (
    ModelServerRef,
    ResourcesServerRef,
    ScicodeAgent,
    ScicodeAgentConfig,
    ScicodeAgentRunRequest,
)
from responses_api_agents.scicode_agent.step_utils import (
    PREFILLED_STEPS_CODE,
    extract_python_script,
    is_context_window_error,
    process_problem_steps,
)


_PROMPT_FPATH = "benchmarks/scicode/prompts/default.yaml"


def _config():
    return ScicodeAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="scicode_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="scicode"),
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        prompt_fpath=_PROMPT_FPATH,
    )


def _agent():
    return ScicodeAgent(config=_config(), server_client=MagicMock(spec=ServerClient))


def _model_json(code: str) -> dict:
    return {
        "id": "r",
        "created_at": 0.0,
        "model": "d",
        "object": "response",
        "output": [
            {
                "id": "m",
                "content": [{"annotations": [], "text": f"```python\n{code}\n```", "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }


class _Resp:
    def __init__(self, payload, cookies=None):
        self._payload = payload
        self.cookies = cookies or {}

    async def json(self):
        return self._payload


class _FakeRequest:
    cookies: dict = {}


def _run_request(problem_id="1", n_steps=2):
    sub_steps = [
        {
            "step_number": f"{problem_id}.{i + 1}",
            "step_description_prompt": f"desc {i}",
            "step_background": f"bg {i}",
            "function_header": f"def f{i}():",
            "return_line": "return None",
            "test_cases": ["assert True"],
        }
        for i in range(n_steps)
    ]
    return ScicodeAgentRunRequest(
        responses_create_params={"input": []},
        problem_id=problem_id,
        sub_steps=sub_steps,
        required_dependencies="import numpy as np",
        uuid=problem_id,
    )


# ----------------------------
# step_utils helpers
# ----------------------------
def test_extract_python_script_python_fence_strips_imports():
    assert extract_python_script("pre\n```python\nimport numpy as np\nx = 1\n```\npost") == "\nx = 1\n"


def test_extract_python_script_generic_fence():
    assert extract_python_script("```\ny = 2\n```") == "\ny = 2\n"


def test_extract_python_script_no_fence():
    assert extract_python_script("z = 3") == "z = 3"


def test_process_problem_steps_with_and_without_background():
    sub_steps = [
        {
            "step_description_prompt": "D0",
            "step_background": "B0",
            "function_header": "def f0():",
            "return_line": "r0",
        },
        {
            "step_description_prompt": "D1",
            "step_background": "B1",
            "function_header": "def f1():",
            "return_line": "r1",
        },
    ]
    prev = ["code0", None]
    ps_bg, ns_bg, prevcode = process_problem_steps(sub_steps, 1, prev, with_background=True)
    assert "B0" in ps_bg and "B1" in ns_bg and "def f1()" in ns_bg and prevcode == "code0"
    ps_no, ns_no, _ = process_problem_steps(sub_steps, 1, prev, with_background=False)
    assert "B0" not in ps_no and "B1" not in ns_no


def test_is_context_window_error():
    assert is_context_window_error(Exception("... exceeds maximum input length ...")) is True
    assert is_context_window_error(Exception("some other error")) is False


def test_prefilled_steps_present():
    assert set(PREFILLED_STEPS_CODE.keys()) == {("13", 5), ("62", 0), ("76", 2)}


# ----------------------------
# agent
# ----------------------------
class TestApp:
    def test_sanity(self):
        _agent()

    def test_config_defaults(self):
        assert _config().with_background is True

    @pytest.mark.asyncio
    async def test_responses_forwards_to_model(self):
        agent = _agent()
        agent.server_client.post = AsyncMock(return_value=_Resp(_model_json("x = 1"), cookies={"sid": "abc"}))
        body = NeMoGymResponseCreateParamsNonStreaming(input="hi")
        with patch.object(app, "raise_for_status", AsyncMock()):
            result = await agent.responses(_FakeRequest(), Response(), body)
        assert result.output_text == "```python\nx = 1\n```"

    @pytest.mark.asyncio
    async def test_run_builds_solutions_and_calls_verify(self):
        agent = _agent()
        captured = {}

        def _post(server_name, url_path, json, cookies):
            if url_path == "/v1/responses":
                return _Resp(_model_json("x = 1"))
            captured["verify"] = json
            return _Resp({"reward": 1.0})

        agent.server_client.post = AsyncMock(side_effect=_post)
        with patch.object(app, "raise_for_status", AsyncMock()):
            result = await agent.run(_FakeRequest(), _run_request(problem_id="1", n_steps=2))

        assert result == {"reward": 1.0}
        verify = captured["verify"]
        assert "response" in verify  # /verify requires a response field
        solutions = verify["solutions"]
        assert set(solutions.keys()) == {"1.1", "1.2"}
        assert "x = 1" in solutions["1.1"]

    @pytest.mark.asyncio
    async def test_run_skips_prefilled_step(self):
        # Problem "62" has a prefilled step at index 0 -> no model call, no solution entry for it.
        agent = _agent()
        captured = {}
        model_calls = 0

        def _post(server_name, url_path, json, cookies):
            nonlocal model_calls
            if url_path == "/v1/responses":
                model_calls += 1
                return _Resp(_model_json("x = 1"))
            captured["verify"] = json
            return _Resp({"reward": 0.0})

        agent.server_client.post = AsyncMock(side_effect=_post)
        with patch.object(app, "raise_for_status", AsyncMock()):
            await agent.run(_FakeRequest(), _run_request(problem_id="62", n_steps=2))

        assert model_calls == 1  # only the non-prefilled step is generated
        assert set(captured["verify"]["solutions"].keys()) == {"62.2"}

    @pytest.mark.asyncio
    async def test_run_context_window_sentinels_remaining_steps(self):
        agent = _agent()
        captured = {}

        def _post(server_name, url_path, json, cookies):
            if url_path == "/v1/responses":
                raise RuntimeError("... exceeds maximum input length ...")
            captured["verify"] = json
            return _Resp({"reward": 0.0})

        agent.server_client.post = AsyncMock(side_effect=_post)
        with patch.object(app, "raise_for_status", AsyncMock()):
            await agent.run(_FakeRequest(), _run_request(problem_id="1", n_steps=2))

        solutions = captured["verify"]["solutions"]
        assert solutions == {"1.1": "_ran_out_of_context_", "1.2": "_ran_out_of_context_"}

    @pytest.mark.asyncio
    async def test_run_reraises_non_context_error(self):
        agent = _agent()

        def _post(server_name, url_path, json, cookies):
            raise RuntimeError("boom")  # not a context-window error -> should propagate

        agent.server_client.post = AsyncMock(side_effect=_post)
        with patch.object(app, "raise_for_status", AsyncMock()), pytest.raises(RuntimeError, match="boom"):
            await agent.run(_FakeRequest(), _run_request(problem_id="1", n_steps=1))

    def test_compute_metrics_subtask_accuracy_is_substep_weighted(self):
        # Two problems: 1/2 and 3/4 passed -> 4/6, NOT the mean of ratios (0.5, 0.75).
        tasks = [
            [{"num_steps_passed": 1, "num_steps_total": 2}],
            [{"num_steps_passed": 3, "num_steps_total": 4}],
        ]
        assert _agent().compute_metrics(tasks) == {"subtask_accuracy": 4 / 6}

    def test_compute_metrics_no_steps(self):
        assert _agent().compute_metrics([]) == {"subtask_accuracy": 0.0}

    def test_get_key_metrics_includes_subtask_accuracy(self):
        agent_metrics = {"mean/reward": 0.1875, "max/reward": 1.0, "subtask_accuracy": 0.414}
        key = _agent().get_key_metrics(agent_metrics)
        assert key["subtask_accuracy"] == 0.414
        assert key["mean/reward"] == 0.1875
        assert "max/reward" not in key  # only mean/* + subtask_accuracy are headline
