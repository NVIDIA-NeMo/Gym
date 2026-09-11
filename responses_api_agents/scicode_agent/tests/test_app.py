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
import statistics
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Response

import responses_api_agents.scicode_agent.app as app
from nemo_gym.base_resources_server import AggregateMetricsRequest
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.reward_profile import compute_aggregate_metrics
from nemo_gym.server_utils import ServerClient
from responses_api_agents.scicode_agent.app import (
    TOKEN_USAGE_VERSION,
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


_PROMPT_FPATH = "benchmarks/scicode/prompts/background.yaml"


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
        body = NeMoGymResponseCreateParamsNonStreaming(input="hi", temperature=0.25)
        with patch.object(app, "raise_for_status", AsyncMock()):
            result = await agent.responses(_FakeRequest(), Response(), body)
        assert result.output_text == "```python\nx = 1\n```"
        request_json = agent.server_client.post.await_args.kwargs["json"]
        assert isinstance(request_json, dict)
        assert request_json["temperature"] == 0.25
        assert request_json["input"][0]["content"] == "hi"

    @pytest.mark.asyncio
    async def test_run_builds_solutions_and_calls_verify(self):
        agent = _agent()
        captured = {"model": []}

        def _post(server_name, url_path, json, cookies):
            if url_path == "/v1/responses":
                captured["model"].append(json)
                return _Resp(_model_json("x = 1"))
            captured["verify"] = json
            return _Resp({"reward": 1.0})

        agent.server_client.post = AsyncMock(side_effect=_post)
        body = _run_request(problem_id="1", n_steps=2)
        body.responses_create_params = NeMoGymResponseCreateParamsNonStreaming(
            input=[],
            max_output_tokens=4096,
            temperature=0.25,
            top_p=0.9,
        )
        with patch.object(app, "raise_for_status", AsyncMock()):
            result = await agent.run(_FakeRequest(), body)

        assert result == {"reward": 1.0}
        assert len(captured["model"]) == 2
        for request_json in captured["model"]:
            assert request_json["max_output_tokens"] == 4096
            assert request_json["temperature"] == 0.25
            assert request_json["top_p"] == 0.9
            assert request_json["input"][0]["role"] == "user"
            assert request_json["input"][0]["content"]
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


# ---------------------------------------------------------------------------
# Across-run variability (_across_run_stats via compute_metrics)
# ---------------------------------------------------------------------------


def _repeat(idx, passed, total):
    return {
        "_ng_rollout_index": idx,
        "num_steps_passed": passed,
        "num_steps_total": total,
        "problem_accuracy": passed == total,
    }


class TestAcrossRunStats:
    def test_three_runs_hand_computed(self):
        # Two problems x three repeats. Per-run problem accuracy is the mean of
        # problem_accuracy over problems; per-run subtask accuracy is the
        # sub-step-weighted pool - the same definitions as the headline metrics.
        tasks = [
            [_repeat(0, 2, 2), _repeat(1, 1, 2), _repeat(2, 0, 2)],
            [_repeat(0, 3, 4), _repeat(1, 4, 4), _repeat(2, 3, 4)],
        ]
        m = _agent().compute_metrics(tasks)
        # Pooled headline is unchanged: (2+1+0+3+4+3) / (3*2 + 3*4) = 13/18.
        assert m["subtask_accuracy"] == pytest.approx(13 / 18)
        problem_runs = [1 / 2, 1 / 2, 0.0]  # run means of problem_accuracy
        subtask_runs = [5 / 6, 5 / 6, 3 / 6]  # per-run pooled sub-step fractions
        assert m["mean/problem_accuracy/std_dev_across_runs"] == pytest.approx(statistics.stdev(problem_runs))
        assert m["subtask_accuracy/std_dev_across_runs"] == pytest.approx(statistics.stdev(subtask_runs))
        assert not any(k.endswith("std_err_across_runs") for k in m)

    def test_single_repeat_emits_only_pooled_metric(self):
        tasks = [[_repeat(0, 1, 2)], [_repeat(0, 3, 4)]]
        assert _agent().compute_metrics(tasks) == {"subtask_accuracy": 4 / 6}

    def test_rollout_index_alignment_not_arrival_order(self):
        ordered = [
            [_repeat(0, 2, 2), _repeat(1, 0, 2)],
            [_repeat(0, 4, 4), _repeat(1, 1, 4)],
        ]
        shuffled = [list(reversed(task)) for task in ordered]
        assert _agent().compute_metrics(shuffled) == _agent().compute_metrics(ordered)

    def test_uneven_repeat_counts_use_min_k(self):
        tasks = [
            [_repeat(0, 2, 2), _repeat(1, 0, 2), _repeat(2, 1, 2)],
            [_repeat(0, 4, 4), _repeat(1, 0, 4)],
        ]
        m = _agent().compute_metrics(tasks)
        # k = 2: subtask runs (2+4)/6 = 1.0 and (0+0)/6 = 0.0.
        assert m["subtask_accuracy/std_dev_across_runs"] == pytest.approx(statistics.stdev([1.0, 0.0]))

    def test_identical_runs_zero_std(self):
        tasks = [
            [_repeat(0, 1, 2), _repeat(1, 1, 2)],
            [_repeat(0, 4, 4), _repeat(1, 4, 4)],
        ]
        m = _agent().compute_metrics(tasks)
        assert m["mean/problem_accuracy/std_dev_across_runs"] == 0.0
        assert m["subtask_accuracy/std_dev_across_runs"] == 0.0

    def test_get_key_metrics_includes_across_run_stats(self):
        agent_metrics = {
            "mean/problem_accuracy": 0.19,
            "mean/problem_accuracy/std_dev_across_runs": 0.02,
            "subtask_accuracy": 0.41,
            "subtask_accuracy/std_dev_across_runs": 0.015,
            "std/reward": 0.4,
        }
        key = _agent().get_key_metrics(agent_metrics)
        assert "mean/problem_accuracy/std_dev_across_runs" in key
        assert "subtask_accuracy/std_dev_across_runs" in key
        assert "std/reward" not in key


def _usage(output_tokens, input_tokens=10, cached_tokens=2, reasoning_tokens=3):
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_tokens_details": {"cached_tokens": cached_tokens},
        "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
    }


async def _collect_usage(usages, problem_id="1", n_steps=None, response_overrides=None):
    """Exercise the step loop and capture the payload sent to verification."""
    agent = _agent()
    responses = iter(usages)

    def _post(server_name, url_path, json, cookies):
        if url_path == "/v1/responses":
            usage = next(responses)
            if isinstance(usage, Exception):
                raise usage
            return _Resp({**_model_json("x = 1"), "usage": usage, **(response_overrides or {})})
        return _Resp({**json, "reward": 0.0})

    agent.server_client.post = AsyncMock(side_effect=_post)
    with patch.object(app, "raise_for_status", AsyncMock()):
        return await agent.run(
            _FakeRequest(), _run_request(problem_id=problem_id, n_steps=len(usages) if n_steps is None else n_steps)
        )


def _aggregate(rows):
    agent = _agent()
    rows = [{"_ng_task_index": i, "_ng_rollout_index": 0, **row} for i, row in enumerate(rows)]
    return compute_aggregate_metrics(rows, agent.compute_metrics, agent.get_key_metrics)


class TestTokenAccounting:
    @pytest.mark.asyncio
    async def test_usage_sent_to_verifier_sums_all_steps_and_details(self):
        row = await _collect_usage([_usage(100), _usage(300, input_tokens=20)])
        assert row["token_usage_version"] == TOKEN_USAGE_VERSION
        assert row["response"]["usage"] == _usage(400, input_tokens=30, cached_tokens=4, reasoning_tokens=6)
        assert row["step_usage"] == [
            {"step_number": "1.1", "status": "generated", "usage": _usage(100)},
            {"step_number": "1.2", "status": "generated", "usage": _usage(300, input_tokens=20)},
        ]
        assert row["response"]["output"][0]["content"][0]["text"] == "```python\nx = 1\n```"

    @pytest.mark.asyncio
    async def test_unequal_step_counts_and_repeats_are_pooled(self):
        # Two attempts each: one-step problem (100, 200), three-step problem
        # (300+400+500, 600+700+800). Total 3600 / 4 problems / 8 steps.
        rows = []
        for task_idx, attempts in enumerate(([[100], [200]], [[300, 400, 500], [600, 700, 800]])):
            for repeat_idx, tokens in enumerate(attempts):
                row = await _collect_usage([_usage(t) for t in tokens])
                rows.append({**row, "_ng_task_index": task_idx, "_ng_rollout_index": repeat_idx})
        result = _aggregate(rows)
        for name, per_problem, per_subproblem in (("input", 20, 10), ("output", 900, 450), ("total", 920, 460)):
            assert result.key_metrics[f"mean/{name}_tokens_per_problem"] == per_problem
            assert result.key_metrics[f"mean/{name}_tokens_per_subproblem"] == per_subproblem
            assert result.key_metrics[f"mean/{name}_tokens"] == per_problem
        assert result.agent_metrics["num_generated_steps"] == 8
        assert result.agent_metrics["num_subproblems"] == 8
        assert result.agent_metrics["num_steps_with_usage"] == 8
        assert result.key_metrics["generation_coverage"] == 1.0
        assert result.key_metrics["token_usage_complete"] is True
        assert result.agent_metrics["token_usage_version"] == 1
        assert all(isinstance(value, (int, float)) for value in result.agent_metrics.values())
        assert [g["mean/output_tokens"] for g in result.group_level_metrics] == [150, 1650]

    @pytest.mark.asyncio
    async def test_prefilled_and_context_skipped_steps_are_not_generated(self):
        row = await _collect_usage(
            [_usage(100), RuntimeError("exceeds maximum input length")], problem_id="62", n_steps=4
        )
        assert [s["status"] for s in row["step_usage"]] == [
            "prefilled",
            "generated",
            "context_window_exceeded",
            "skipped",
        ]
        assert row["response"]["usage"]["output_tokens"] == 100
        for i in (0, 2, 3):
            assert row["step_usage"][i]["usage"] == _usage(0, input_tokens=0, cached_tokens=0, reasoning_tokens=0)
        result = _aggregate([row])
        metrics = result.key_metrics
        assert metrics["mean/output_tokens_per_problem"] == 100
        assert metrics["mean/output_tokens_per_subproblem"] == pytest.approx(100 / 3)
        assert metrics["generation_coverage"] == 1 / 3
        assert metrics["token_usage_complete"] is True
        assert result.agent_metrics["num_subproblems"] == 3

    @pytest.mark.asyncio
    @pytest.mark.parametrize("missing_index", [0, 1])
    async def test_missing_usage_stays_unknown_in_rollout_and_collection(self, missing_index):
        usages = [_usage(100), _usage(300)]
        usages[missing_index] = None
        unknown = await _collect_usage(usages)
        known = await _collect_usage([_usage(200)])
        assert unknown["response"]["usage"] is None
        assert unknown["step_usage"][missing_index]["usage"] is None
        result = _aggregate([unknown, {**known, "_ng_task_index": 1}])
        for name in ("input", "output", "total"):
            for suffix in ("", "_per_problem", "_per_subproblem"):
                assert result.key_metrics[f"mean/{name}_tokens{suffix}"] is None
        assert result.agent_metrics["num_generated_steps"] == 3
        assert result.agent_metrics["num_steps_with_usage"] == 2
        assert result.key_metrics["token_usage_complete"] is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("missing_usage", [False, True])
    async def test_aggregate_endpoint_suppresses_all_partial_token_statistics(self, missing_usage):
        rows = []
        for task_idx in range(2):
            for repeat_idx in range(2):
                usage = None if missing_usage and task_idx == repeat_idx == 0 else _usage(100)
                row = await _collect_usage([usage])
                rows.append({**row, "_ng_task_index": task_idx, "_ng_rollout_index": repeat_idx})
        body = AggregateMetricsRequest(verify_responses=rows)
        original = body.model_dump()
        result = await _agent().aggregate_metrics(body)
        assert body.model_dump() == original
        assert result.key_metrics["token_usage_complete"] is not missing_usage
        assert result.key_metrics["generation_coverage"] == 1.0
        for section in (result.agent_metrics, result.key_metrics):
            assert all(value is not None and not isinstance(value, str) for value in section.values())
        if missing_usage:
            assert "mean/output_tokens_per_problem" not in result.key_metrics

            def check_no_numeric_tokens(value):
                if isinstance(value, dict):
                    for key, item in value.items():
                        if "tokens" in key:
                            assert not isinstance(item, (int, float)), (key, item)
                        check_no_numeric_tokens(item)
                elif isinstance(value, list):
                    for item in value:
                        check_no_numeric_tokens(item)

            check_no_numeric_tokens(result.model_dump())
        else:
            assert result.key_metrics["mean/output_tokens_per_problem"] == 100
            assert result.agent_metrics["max/output_tokens"] == 100
            assert all(group["mean/output_tokens"] == 100 for group in result.group_level_metrics)

    @pytest.mark.asyncio
    async def test_unknown_reasoning_details_do_not_poison_known_output_count(self):
        row = await _collect_usage([_usage(100, reasoning_tokens=None), _usage(200)])
        assert row["response"]["usage"]["output_tokens"] == 300
        assert row["response"]["usage"]["output_tokens_details"]["reasoning_tokens"] is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("kind", ["empty", "prefilled", "context"])
    async def test_no_generations_zero_usage_with_fixed_subproblem_denominator(self, kind):
        if kind == "prefilled":
            row = await _collect_usage([], problem_id="62", n_steps=1)
        elif kind == "context":
            row = await _collect_usage([RuntimeError("exceeds maximum input length")], n_steps=2)
        else:
            row = await _collect_usage([])
        assert row["response"]["usage"]["output_tokens"] == 0
        assert all(step["usage"]["total_tokens"] == 0 for step in row["step_usage"])
        metrics = _aggregate([row]).key_metrics
        for name in ("input", "output", "total"):
            assert metrics[f"mean/{name}_tokens_per_problem"] == 0
            assert metrics[f"mean/{name}_tokens_per_subproblem"] == (0 if kind == "context" else None)
        assert metrics["generation_coverage"] == (0 if kind == "context" else None)

    @pytest.mark.asyncio
    async def test_truncated_generation_counts_all_usage(self):
        row = await _collect_usage(
            [_usage(262000, input_tokens=144)],
            response_overrides={"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
        )
        assert row["response"]["incomplete_details"] == {"reason": "max_output_tokens"}
        assert row["step_usage"][0]["status"] == "generated"
        metrics = _aggregate([row]).key_metrics
        for scope in ("problem", "subproblem"):
            assert metrics[f"mean/input_tokens_per_{scope}"] == 144
            assert metrics[f"mean/output_tokens_per_{scope}"] == 262000
            assert metrics[f"mean/total_tokens_per_{scope}"] == 262144

    @pytest.mark.asyncio
    async def test_correctness_does_not_filter_token_usage(self):
        row = await _collect_usage([_usage(1000)] * 3)
        one_correct = {**row, "step_results": [True, False, False], "num_steps_passed": 1, "num_steps_total": 3}
        two_correct = {**row, "step_results": [True, True, False], "num_steps_passed": 2, "num_steps_total": 3}
        first = _aggregate([one_correct]).key_metrics
        second = _aggregate([two_correct]).key_metrics
        assert first["subtask_accuracy"] == 1 / 3
        assert second["subtask_accuracy"] == 2 / 3
        assert first["mean/output_tokens_per_problem"] == second["mean/output_tokens_per_problem"] == 3000
        assert first["mean/output_tokens_per_subproblem"] == second["mean/output_tokens_per_subproblem"] == 1000

    @pytest.mark.asyncio
    @pytest.mark.parametrize("text", ["not Python code", "```python\nthis is ! invalid\n```", "```python\n", ""])
    async def test_unusable_code_does_not_stop_later_steps_or_drop_usage(self, text):
        output = _model_json("ignored")["output"]
        output[0]["content"][0]["text"] = text
        row = await _collect_usage([_usage(100), _usage(200)], response_overrides={"output": output})
        assert set(row["solutions"]) == {"1.1", "1.2"}
        assert [step["status"] for step in row["step_usage"]] == ["generated", "generated"]
        assert row["response"]["usage"]["output_tokens"] == 300
        metrics = _aggregate([row]).key_metrics
        assert metrics["mean/output_tokens_per_problem"] == 300
        assert metrics["mean/output_tokens_per_subproblem"] == 150

    def test_legacy_rollouts_do_not_gain_whole_problem_metrics(self):
        row = {"reward": 0.0, "response": {"usage": _usage(100)}}
        metrics = _aggregate([row]).key_metrics
        assert metrics["mean/output_tokens"] == 100
        assert "mean/output_tokens_per_problem" not in metrics
        assert "mean/output_tokens_per_subproblem" not in metrics

    @pytest.mark.asyncio
    async def test_mixed_accounting_versions_rejected(self):
        row = await _collect_usage([_usage(100)])
        old_row = {"reward": 0.0, "response": {"usage": _usage(200)}}
        with pytest.raises(ValueError, match="different token accounting versions"):
            _aggregate([row, old_row])
