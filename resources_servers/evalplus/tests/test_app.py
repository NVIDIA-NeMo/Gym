# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
import ray
from app import (
    EvalPlusResourcesServer,
    EvalPlusResourcesServerConfig,
    EvalPlusVerifyRequest,
    EvalPlusVerifyResponse,
    extract_code_strict,
)
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym.base_resources_server import AggregateMetricsRequest
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient


# ----------------------------
# Mock dataset / expected_output so tests don't require evalplus + HF
# ----------------------------
_MOCK_PROMPT = "def f(x):\n"
_MOCK_DATASET = {
    "HumanEval/0": {
        "task_id": "HumanEval/0",
        "prompt": _MOCK_PROMPT,
        "entry_point": "f",
        "base_input": [[1]],
        "plus_input": [[2]],
        "atol": 0,
        "canonical_solution": "    return x\n",
    }
}
_MOCK_EXPECTED_OUTPUT = {
    "HumanEval/0": {"base": [1], "plus": [2]},
}


def _make_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp",
        created_at=0.0,
        model="dummy",
        object="response",
        output=[
            {
                "id": "msg",
                "content": [
                    {"annotations": [], "text": text, "type": "output_text"},
                ],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )


class _FakeFuture:
    def __init__(self, result):
        self._result = result


def _ray_get_returns(result_dict):
    """Patch ray.get to return a preset result without launching workers."""

    def _get(future):
        return future._result if isinstance(future, _FakeFuture) else result_dict

    return _get


def _remote_returns(result_dict):
    """Patch check_correctness_remote.remote() so it returns a fake future."""

    def _remote(*args, **kwargs):
        return _FakeFuture(result_dict)

    return _remote


# ----------------------------
# Pure-function: code extractor
# ----------------------------
class TestExtractCode:
    def test_extracts_last_python_fence(self):
        text = "scratch ```python\nx=1\n``` then ```python\ny=2\n```"
        assert extract_code_strict(text) == "y=2"

    def test_falls_back_to_generic_fence(self):
        # Generic ``` fallback only kicks in when ```python is absent. Because
        # we use rfind for parity with Skills' preprocess_code, a single
        # generic fence pair has its closing fence as the "last" ``` and
        # nothing follows it -> strict mode returns "". This matches Skills.
        # When ```python is also present (the realistic case), the python
        # fence wins via the prior rfind branch.
        text = "no language tag ```\nz=3\n```"
        assert extract_code_strict(text) == ""

    def test_strict_mode_no_closing_fence(self):
        text = "```python\nincomplete"
        assert extract_code_strict(text) == ""

    def test_no_fence_returns_empty(self):
        assert extract_code_strict("plain text no fences") == ""

    def test_strips_carriage_returns(self):
        text = "```python\r\nprint(1)\r\n```"
        assert extract_code_strict(text) == "print(1)"

    def test_chooses_last_when_multiple_blocks(self):
        text = "Step 1: ```python\nx=1\n```\nStep 2: ```python\nfinal_answer=42\n```"
        assert extract_code_strict(text) == "final_answer=42"


# ----------------------------
# verify() — with patched dataset/runner
# ----------------------------
class TestApp:
    @pytest.fixture(scope="class")
    def evalplus_client(self) -> Generator[TestClient, None, None]:
        ray.init(num_cpus=1, ignore_reinit_error=True)
        with patch("app._load_dataset_and_expected", return_value=(_MOCK_DATASET, _MOCK_EXPECTED_OUTPUT)):
            server = EvalPlusResourcesServer(
                config=EvalPlusResourcesServerConfig(
                    host="0.0.0.0",
                    port=8080,
                    entrypoint="",
                    name="",
                    dataset="humaneval",
                    num_processes=1,
                ),
                server_client=MagicMock(spec=ServerClient),
            )
            app = server.setup_webserver()
            with TestClient(app) as client:
                yield client

    def _post_verify(self, client: TestClient, text: str, task_id: str = "HumanEval/0", **md):
        verify_req = EvalPlusVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "complete f"}]},
            response=_make_response(text),
            verifier_metadata={"task_id": task_id, **md},
        )
        return client.post(url="/verify", json=verify_req.model_dump())

    async def test_verify_pass_base_and_plus(self, evalplus_client):
        with (
            patch(
                "app.check_correctness_remote.remote",
                side_effect=_remote_returns({"base_status": "pass", "plus_status": "pass"}),
            ),
            patch("app.ray.get", side_effect=_ray_get_returns({"base_status": "pass", "plus_status": "pass"})),
        ):
            resp = self._post_verify(evalplus_client, "```python\ndef f(x):\n    return x\n```")
            res = EvalPlusVerifyResponse.model_validate(resp.json())
        assert res.reward == 1.0
        assert res.is_correct is True
        assert res.is_correct_plus is True

    async def test_verify_passes_base_fails_plus(self, evalplus_client):
        with (
            patch(
                "app.check_correctness_remote.remote",
                side_effect=_remote_returns({"base_status": "pass", "plus_status": "fail"}),
            ),
            patch("app.ray.get", side_effect=_ray_get_returns({"base_status": "pass", "plus_status": "fail"})),
        ):
            resp = self._post_verify(evalplus_client, "```python\ndef f(x):\n    return x if x < 2 else 0\n```")
            res = EvalPlusVerifyResponse.model_validate(resp.json())
        # Base reward is for the strict (plus) verdict
        assert res.reward == 0.0
        assert res.is_correct is True
        assert res.is_correct_plus is False

    async def test_verify_fail_base(self, evalplus_client):
        with (
            patch(
                "app.check_correctness_remote.remote",
                side_effect=_remote_returns({"base_status": "fail", "plus_status": "fail"}),
            ),
            patch("app.ray.get", side_effect=_ray_get_returns({"base_status": "fail", "plus_status": "fail"})),
        ):
            resp = self._post_verify(evalplus_client, "```python\ndef f(x):\n    return -x\n```")
            res = EvalPlusVerifyResponse.model_validate(resp.json())
        assert res.reward == 0.0
        assert res.is_correct is False
        assert res.is_correct_plus is False

    async def test_verify_no_code_block(self, evalplus_client):
        resp = self._post_verify(evalplus_client, "I will solve this by returning x.")
        res = EvalPlusVerifyResponse.model_validate(resp.json())
        assert res.reward == 0.0
        assert res.extracted_model_output == "I will solve this by returning x."
        assert not res.extracted_model_code

    async def test_verify_empty_output(self, evalplus_client):
        resp = self._post_verify(evalplus_client, "")
        res = EvalPlusVerifyResponse.model_validate(resp.json())
        assert res.reward == 0.0
        assert not res.extracted_model_code

    async def test_verify_unknown_task_id(self, evalplus_client):
        resp = self._post_verify(evalplus_client, "```python\nx=1\n```", task_id="HumanEval/9999")
        res = EvalPlusVerifyResponse.model_validate(resp.json())
        assert res.reward == 0.0
        assert "unknown task_id" in (res.metadata or {}).get("error", "")

    async def test_verify_missing_task_id(self, evalplus_client):
        verify_req = EvalPlusVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "complete f"}]},
            response=_make_response("```python\nx=1\n```"),
            verifier_metadata={},
        )
        resp = evalplus_client.post(url="/verify", json=verify_req.model_dump())
        res = EvalPlusVerifyResponse.model_validate(resp.json())
        assert res.reward == 0.0
        assert "missing task_id" in (res.metadata or {}).get("error", "")

    async def test_verify_prepends_prompt_when_missing(self, evalplus_client, monkeypatch):
        """Mirrors evalplus.evaluate.evaluate(): if the extracted code doesn't already
        start with problem['prompt'], prepend it before calling check_correctness.
        Without this, check_correctness gets a body-only fragment and fails.
        """
        captured = {}

        def _remote_capture(dataset, problem, solution, expected, *args):
            captured["dataset"] = dataset
            captured["solution"] = solution
            return _FakeFuture({"base_status": "pass", "plus_status": "pass"})

        with (
            patch("app.check_correctness_remote.remote", side_effect=_remote_capture),
            patch(
                "app.ray.get",
                side_effect=_ray_get_returns({"base_status": "pass", "plus_status": "pass"}),
            ),
        ):
            # Body-only completion (no `def` line, no imports) — Skills-style
            self._post_verify(evalplus_client, "```python\n    return x\n```")
        assert captured["solution"].startswith(_MOCK_PROMPT)
        assert captured["solution"] == _MOCK_PROMPT + "return x"

    async def test_verify_does_not_double_prepend(self, evalplus_client):
        """If extracted code already starts with problem['prompt'], we should NOT
        prepend again. Otherwise the resulting source has duplicated imports +
        function signature.
        """
        captured = {}

        def _remote_capture(dataset, problem, solution, expected, *args):
            captured["solution"] = solution
            return _FakeFuture({"base_status": "pass", "plus_status": "pass"})

        full_code = _MOCK_PROMPT + "    return x"
        with (
            patch("app.check_correctness_remote.remote", side_effect=_remote_capture),
            patch(
                "app.ray.get",
                side_effect=_ray_get_returns({"base_status": "pass", "plus_status": "pass"}),
            ),
        ):
            self._post_verify(evalplus_client, f"```python\n{full_code}\n```")
        assert captured["solution"] == full_code  # NOT _MOCK_PROMPT + full_code

    def test_verify_missing_response_validation_error(self):
        with pytest.raises(ValidationError):
            EvalPlusVerifyRequest(
                responses_create_params={"input": [{"role": "user", "content": "anything"}]},
                # response intentionally omitted
                verifier_metadata={"task_id": "HumanEval/0"},
            )


# ----------------------------
# Metrics
# ----------------------------
def _make_server() -> EvalPlusResourcesServer:
    with patch("app._load_dataset_and_expected", return_value=(_MOCK_DATASET, _MOCK_EXPECTED_OUTPUT)):
        return EvalPlusResourcesServer(
            config=EvalPlusResourcesServerConfig(
                host="127.0.0.1",
                port=12345,
                entrypoint="app.py",
                name="evalplus",
                dataset="humaneval",
                num_processes=1,
            ),
            server_client=MagicMock(spec=ServerClient),
        )


class TestComputeMetrics:
    def test_produces_pass_at_k_for_both_scores(self) -> None:
        server = _make_server()
        tasks = [
            [
                {"reward": 1.0, "is_correct": True, "is_correct_plus": True, "extracted_model_code": "x"},
                {"reward": 0.0, "is_correct": True, "is_correct_plus": False, "extracted_model_code": "x"},
            ],
            [
                {"reward": 1.0, "is_correct": True, "is_correct_plus": True, "extracted_model_code": "y"},
                {"reward": 1.0, "is_correct": True, "is_correct_plus": True, "extracted_model_code": "y"},
            ],
        ]
        m = server.compute_metrics(tasks)
        assert "pass@1/passing_base_tests" in m
        assert "pass@1/passing_plus_tests" in m
        assert "pass@2/passing_base_tests" in m
        assert "pass@2/passing_plus_tests" in m
        assert "majority@2/passing_base_tests" in m
        assert "pass@1[avg-of-2]/passing_base_tests" in m
        # Plus is strictly stricter: per-task pass@1 of plus <= base
        assert m["pass@1[avg-of-2]/passing_plus_tests"] <= m["pass@1[avg-of-2]/passing_base_tests"]

    def test_no_answer_tracked(self) -> None:
        server = _make_server()
        tasks = [
            [
                {"reward": 1.0, "is_correct": True, "is_correct_plus": True, "extracted_model_code": "x"},
                {"reward": 0.0, "is_correct": False, "is_correct_plus": False, "extracted_model_code": None},
            ],
        ]
        m = server.compute_metrics(tasks)
        assert "pass@1[avg-of-2]/no_answer" in m
        assert m["pass@1[avg-of-2]/no_answer"] == pytest.approx(50.0)


class TestLoadDatasetAndExpected:
    """Cover the dataset-loader fork without hitting HF / EvalPlus."""

    def test_humaneval_path(self):
        import app

        with (
            patch("evalplus.data.get_human_eval_plus", return_value={"HumanEval/0": {"task_id": "HumanEval/0"}}),
            patch("evalplus.data.get_human_eval_plus_hash", return_value="hash"),
            patch("evalplus.evaluate.get_groundtruth", return_value={"HumanEval/0": {"base": [], "plus": []}}),
        ):
            problems, expected = app._load_dataset_and_expected("humaneval", "default")
        assert "HumanEval/0" in problems
        assert "HumanEval/0" in expected

    def test_mbpp_path(self):
        import app

        with (
            patch(
                "evalplus.data.get_mbpp_plus",
                return_value={"Mbpp/0": {"canonical_solution": "x"}, "Mbpp/1": {"canonical_solution": ""}},
            ),
            patch("evalplus.data.get_mbpp_plus_hash", return_value="hash"),
            patch("evalplus.data.mbpp.mbpp_serialize_inputs", create=True),
            patch("evalplus.evaluate.get_groundtruth", return_value={"Mbpp/0": {}, "Mbpp/1": {}}) as gt,
        ):
            problems, expected = app._load_dataset_and_expected("mbpp", "default")
        # Only Mbpp/0 (with canonical_solution) is in the filter list passed to get_groundtruth.
        gt.assert_called_once()
        _, _, only_not_none = gt.call_args.args
        assert only_not_none == ["Mbpp/0"]

    def test_unsupported_dataset_raises(self):
        import app

        with pytest.raises(ValueError, match="Unsupported evalplus dataset"):
            app._load_dataset_and_expected("not-a-dataset", "default")


class TestRunnerImport:
    """Smoke-import the Ray-remote module so its stmts count toward coverage."""

    def test_import(self):
        from evalplus_integration import runner

        assert hasattr(runner, "check_correctness_remote")
        # Decorated with @ray.remote — the module-level wrapper has a `.remote` attr.
        assert hasattr(runner.check_correctness_remote, "remote")

    def test_runner_body_with_mocked_evalplus(self):
        """Exercise the @ray.remote function body locally by calling the
        underlying function (Ray's RemoteFunction exposes it as _function).
        evalplus.evaluate.check_correctness is mocked so we don't actually
        spawn subprocesses or need expected_output to be real.
        """
        import sys
        from unittest.mock import MagicMock

        # Stub the optional evalplus.evaluate.check_correctness import.
        fake_evaluate_mod = MagicMock()
        fake_evaluate_mod.check_correctness.return_value = {
            "base": ("pass", [True]),
            "plus": ("fail", [True, False]),
        }
        sys.modules.setdefault("evalplus", MagicMock())
        sys.modules["evalplus.evaluate"] = fake_evaluate_mod

        from evalplus_integration import runner

        # Ray's @ray.remote stores the underlying Python function on _function.
        underlying = runner.check_correctness_remote._function
        result = underlying(
            dataset="humaneval",
            problem={"task_id": "HumanEval/0"},
            solution="def f(): pass",
            expected_output={"base": [], "plus": []},
            min_time_limit=1.0,
            gt_time_limit_factor=4.0,
        )
        assert result["base_status"] == "pass"
        assert result["plus_status"] == "fail"
        assert result["base_details"] == [True]
        assert result["plus_details"] == [True, False]


class TestGetKeyMetrics:
    @pytest.mark.asyncio
    async def test_selects_headline_metrics(self) -> None:
        server = _make_server()
        responses = []
        for task_idx in range(4):
            for rollout_idx in range(3):
                base_pass = (task_idx + rollout_idx) % 2 == 0
                plus_pass = base_pass and rollout_idx < 2
                responses.append(
                    {
                        TASK_INDEX_KEY_NAME: task_idx,
                        ROLLOUT_INDEX_KEY_NAME: rollout_idx,
                        "reward": 1.0 if plus_pass else 0.0,
                        "is_correct": base_pass,
                        "is_correct_plus": plus_pass,
                        "extracted_model_code": f"x={task_idx}" if rollout_idx < 2 else None,
                    }
                )
        result = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=responses))
        km = result.key_metrics
        assert "pass@3/passing_base_tests" in km
        assert "pass@3/passing_plus_tests" in km
        assert "majority@3/passing_base_tests" in km
        assert "pass@1[avg-of-3]/passing_base_tests" in km
        assert "pass@1[avg-of-3]/passing_plus_tests" in km
        assert not any("std_dev" in k for k in km)
        assert not any("no_answer" in k for k in km)
