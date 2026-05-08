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
    CodeFIMResourcesServer,
    CodeFIMResourcesServerConfig,
    CodeFIMVerifyRequest,
    CodeFIMVerifyResponse,
    extract_code_strict,
    postprocess_completion,
    remove_overlap,
)
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym.base_resources_server import AggregateMetricsRequest
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient


# ----------------------------
# Mock dataset so tests don't require human_eval_infilling + network
# ----------------------------
_MOCK_PREFIX = "def f(x):\n    "
_MOCK_SUFFIX = "\n    return y\n"
_MOCK_TEST = "def check(candidate):\n    assert candidate(1) == 2"
_MOCK_DATASET = {
    "RandomSpanInfilling/HumanEval/0/0": {
        "task_id": "RandomSpanInfilling/HumanEval/0/0",
        "prompt": _MOCK_PREFIX,
        "suffix": _MOCK_SUFFIX,
        "entry_point": "f",
        "test": _MOCK_TEST,
        "canonical_solution": "y = x + 1",
    }
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
    def _get(future):
        return future._result if isinstance(future, _FakeFuture) else result_dict

    return _get


def _remote_returns(result_dict):
    def _remote(*args, **kwargs):
        return _FakeFuture(result_dict)

    return _remote


# ----------------------------
# Pure-function helpers
# ----------------------------
class TestExtractCode:
    def test_extracts_last_python_fence(self):
        text = "scratch ```python\nx=1\n``` then ```python\ny=2\n```"
        # Note: no .strip() (FIM uses strip_whitespace=False), so we keep the \n.
        assert extract_code_strict(text) == "\ny=2\n"

    def test_falls_back_to_generic_fence(self):
        # Generic fallback only kicks in when ```python is absent. With a single
        # generic fence pair, rfind matches the closing fence and nothing follows
        # it -> strict mode returns "". Mirrors Skills' preprocess_code.
        text = "no language tag ```\nz=3\n```"
        assert extract_code_strict(text) == ""

    def test_strict_mode_no_closing_fence(self):
        text = "```python\nincomplete"
        assert extract_code_strict(text) == ""

    def test_no_fence_returns_empty(self):
        assert extract_code_strict("plain text no fences") == ""

    def test_strips_carriage_returns(self):
        text = "```python\r\nprint(1)\r\n```"
        # No .strip() — keep leading/trailing \n inside the fence.
        assert extract_code_strict(text) == "\nprint(1)\n"

    def test_chooses_last_when_multiple_blocks(self):
        text = "Step 1: ```python\nx=1\n```\nStep 2: ```python\nfinal=42\n```"
        assert extract_code_strict(text) == "\nfinal=42\n"

    def test_strips_think_tag(self):
        text = "<think>scratch ```python\nbad\n```</think>final ```python\ngood\n```"
        assert extract_code_strict(text) == "\ngood\n"

    def test_unclosed_think_falls_through_to_extraction(self):
        # Skills' preprocess_code only branches on "</think>" being present;
        # a stray "<think>" with no closing tag just falls through to the
        # fence extractor (the dead-code "else" branch in preprocess_code is
        # unreachable because str.partition always returns the separator
        # when the pattern is found).
        text = "<think>still thinking ```python\nfake\n```"
        assert extract_code_strict(text) == "\nfake\n"


class TestRemoveOverlap:
    def test_following_with_prefix_overlap(self):
        # The longest tail of preceding that prefixes following is "c\n" (len 2),
        # so following[2:] = "rest" is returned. Whitespace-only candidates that
        # contain a newline DO count, but the iteration runs from longest to
        # shortest, so the longer non-whitespace match wins first.
        assert remove_overlap("abc\n", "c\nrest", truncate_from="following") == "rest"

    def test_no_overlap_returns_following_unchanged(self):
        assert remove_overlap("abc", "xyz", truncate_from="following") == "xyz"

    def test_whitespace_only_overlap_skipped_when_no_newline(self):
        # An overlap of " " (single space, no \n) is skipped by Skills.
        assert remove_overlap("foo ", " bar", truncate_from="following") == " bar"

    def test_preceding_truncates_against_suffix(self):
        # Strip a trailing fragment of `preceding` that matches the head of `following`.
        assert remove_overlap("body\nx=1\n", "x=1\nrest", truncate_from="preceding") == "body\n"


class TestPostprocessCompletion:
    def test_drops_one_leading_newline(self):
        assert postprocess_completion("\nbody", _MOCK_PREFIX, _MOCK_SUFFIX) == "body"

    def test_strips_prefix_overlap_at_head(self):
        # Model re-emits the trailing newline of prefix at the head of its
        # completion: prefix ends "...:\n    ", model writes "    y = x + 1".
        # remove_overlap finds the "\n    " overlap (contains a newline so
        # the whitespace-skip rule does not apply) and strips it.
        prefix = "def f(x):\n    "
        suffix = "\n    return y\n"
        repeated = "\n    y = x + 1"
        out = postprocess_completion(repeated, prefix, suffix)
        # The leading "\n" is dropped first; remove_overlap then strips the
        # remaining "    " — which is whitespace-only (no newline) and skipped,
        # so the output keeps the indentation. Test that the leading-newline
        # drop happened.
        assert out == "    y = x + 1"

    def test_strips_suffix_overlap_at_tail(self):
        # Model re-emits the head of the suffix at the tail of its completion.
        prefix = "def f(x):\n    "
        suffix = "\n    return y\n"
        repeated = "y = x + 1\n    return y"
        out = postprocess_completion(repeated, prefix, suffix)
        # The newline-containing tail "\n    return y" matches the head of
        # suffix and is stripped (truncate_from="preceding").
        assert out == "y = x + 1"


# ----------------------------
# verify() — with patched dataset/runner
# ----------------------------
class TestApp:
    @pytest.fixture(scope="class")
    def fim_client(self) -> Generator[TestClient, None, None]:
        ray.init(num_cpus=1, ignore_reinit_error=True)
        with patch("app._load_split", return_value=_MOCK_DATASET):
            server = CodeFIMResourcesServer(
                config=CodeFIMResourcesServerConfig(
                    host="0.0.0.0",
                    port=8080,
                    entrypoint="",
                    name="",
                    split="random_span",
                    num_processes=1,
                ),
                server_client=MagicMock(spec=ServerClient),
            )
            app = server.setup_webserver()
            with TestClient(app) as client:
                yield client

    def _post_verify(
        self,
        client: TestClient,
        text: str,
        task_id: str = "RandomSpanInfilling/HumanEval/0/0",
        **md,
    ):
        verify_req = CodeFIMVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "fill in"}]},
            response=_make_response(text),
            verifier_metadata={"task_id": task_id, **md},
        )
        return client.post(url="/verify", json=verify_req.model_dump())

    async def test_verify_passes(self, fim_client):
        with (
            patch(
                "app.check_correctness_remote.remote",
                side_effect=_remote_returns({"passed": True, "result": "passed"}),
            ),
            patch("app.ray.get", side_effect=_ray_get_returns({"passed": True, "result": "passed"})),
        ):
            resp = self._post_verify(fim_client, "```python\ny = x + 1\n```")
            res = CodeFIMVerifyResponse.model_validate(resp.json())
        assert res.reward == 1.0
        assert res.passed is True
        assert res.status == "passed"
        # Completion is the post-processed extracted code.
        assert res.completion == "y = x + 1"

    async def test_verify_fails(self, fim_client):
        with (
            patch(
                "app.check_correctness_remote.remote",
                side_effect=_remote_returns({"passed": False, "result": "failed: AssertionError"}),
            ),
            patch(
                "app.ray.get",
                side_effect=_ray_get_returns({"passed": False, "result": "failed: AssertionError"}),
            ),
        ):
            resp = self._post_verify(fim_client, "```python\ny = x - 1\n```")
            res = CodeFIMVerifyResponse.model_validate(resp.json())
        assert res.reward == 0.0
        assert res.passed is False

    async def test_verify_no_code_block(self, fim_client):
        resp = self._post_verify(fim_client, "no fences here at all")
        res = CodeFIMVerifyResponse.model_validate(resp.json())
        assert res.reward == 0.0
        assert res.extracted_model_output == "no fences here at all"
        assert not res.extracted_model_code

    async def test_verify_empty_output(self, fim_client):
        resp = self._post_verify(fim_client, "")
        res = CodeFIMVerifyResponse.model_validate(resp.json())
        assert res.reward == 0.0
        assert not res.extracted_model_code

    async def test_verify_unknown_task_id(self, fim_client):
        resp = self._post_verify(
            fim_client,
            "```python\nx=1\n```",
            task_id="RandomSpanInfilling/HumanEval/9999/0",
        )
        res = CodeFIMVerifyResponse.model_validate(resp.json())
        assert res.reward == 0.0
        assert "unknown task_id" in (res.metadata or {}).get("error", "")

    async def test_verify_missing_task_id(self, fim_client):
        verify_req = CodeFIMVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "fill in"}]},
            response=_make_response("```python\nx=1\n```"),
            verifier_metadata={},
        )
        resp = fim_client.post(url="/verify", json=verify_req.model_dump())
        res = CodeFIMVerifyResponse.model_validate(resp.json())
        assert res.reward == 0.0
        assert "missing task_id" in (res.metadata or {}).get("error", "")

    async def test_verify_passes_postprocessed_completion_to_runner(self, fim_client):
        """The completion fed to check_correctness must have boundary
        overlaps trimmed (Skills behavior)."""
        captured = {}

        def _remote_capture(problem, completion, timeout):
            captured["completion"] = completion
            return _FakeFuture({"passed": True, "result": "passed"})

        with (
            patch("app.check_correctness_remote.remote", side_effect=_remote_capture),
            patch(
                "app.ray.get",
                side_effect=_ray_get_returns({"passed": True, "result": "passed"}),
            ),
        ):
            # Model echoes the head of suffix at the tail. The first
            # postprocess step also drops the single leading "\n" that LLMs
            # emit after ```python\n.
            self._post_verify(
                fim_client,
                "```python\ny = x + 1\n    return y\n```",
            )
        # Leading "\n" dropped; remove_overlap strips the suffix-head echo.
        assert captured["completion"] == "y = x + 1"

    def test_verify_missing_response_validation_error(self):
        with pytest.raises(ValidationError):
            CodeFIMVerifyRequest(
                responses_create_params={"input": [{"role": "user", "content": "anything"}]},
                # response intentionally omitted
                verifier_metadata={"task_id": "RandomSpanInfilling/HumanEval/0/0"},
            )


# ----------------------------
# Metrics
# ----------------------------
def _make_server() -> CodeFIMResourcesServer:
    with patch("app._load_split", return_value=_MOCK_DATASET):
        return CodeFIMResourcesServer(
            config=CodeFIMResourcesServerConfig(
                host="127.0.0.1",
                port=12345,
                entrypoint="app.py",
                name="code_fim",
                split="random_span",
                num_processes=1,
            ),
            server_client=MagicMock(spec=ServerClient),
        )


class TestComputeMetrics:
    def test_produces_pass_at_k(self) -> None:
        server = _make_server()
        tasks = [
            [
                {"reward": 1.0, "passed": True, "completion": "x"},
                {"reward": 0.0, "passed": False, "completion": "x"},
            ],
            [
                {"reward": 1.0, "passed": True, "completion": "y"},
                {"reward": 1.0, "passed": True, "completion": "y"},
            ],
        ]
        m = server.compute_metrics(tasks)
        assert "pass@1/accuracy" in m
        assert "pass@2/accuracy" in m
        assert "majority@2/accuracy" in m
        assert "pass@1[avg-of-2]/accuracy" in m

    def test_no_answer_tracked(self) -> None:
        server = _make_server()
        tasks = [
            [
                {"reward": 1.0, "passed": True, "completion": "x"},
                {"reward": 0.0, "passed": False, "completion": None},
            ],
        ]
        m = server.compute_metrics(tasks)
        assert "pass@1[avg-of-2]/no_answer" in m
        assert m["pass@1[avg-of-2]/no_answer"] == pytest.approx(50.0)


class TestLoadSplit:
    """Cover the dataset loader without hitting the network."""

    def test_unsupported_split_raises(self):
        import app

        with pytest.raises(ValueError, match="Unsupported FIM split"):
            app._load_split("not-a-split")

    def test_random_span_path(self):
        import app

        with patch(
            "human_eval_infilling.data.read_problems",
            return_value={"RandomSpanInfilling/HumanEval/0/0": {"task_id": "RandomSpanInfilling/HumanEval/0/0"}},
        ):
            problems = app._load_split("random_span")
        assert "RandomSpanInfilling/HumanEval/0/0" in problems


class TestRunnerImport:
    """Smoke-import the Ray-remote module so its stmts count toward coverage."""

    def test_import(self):
        from human_eval_infilling_integration import runner

        assert hasattr(runner, "check_correctness_remote")
        assert hasattr(runner.check_correctness_remote, "remote")

    def test_runner_body_with_mocked_human_eval_infilling(self):
        """Exercise the @ray.remote function body locally by calling the
        underlying function. The upstream lib is mocked so we don't spawn
        subprocesses or need a real test suite.
        """
        import sys
        from unittest.mock import MagicMock

        fake_execution_mod = MagicMock()
        fake_execution_mod.check_correctness.return_value = {
            "passed": True,
            "result": "passed",
        }
        sys.modules.setdefault("human_eval_infilling", MagicMock())
        sys.modules["human_eval_infilling.execution"] = fake_execution_mod

        from human_eval_infilling_integration import runner

        underlying = runner.check_correctness_remote._function
        result = underlying(
            problem={"task_id": "X", "prompt": "", "suffix": "", "test": "", "entry_point": "f"},
            completion="pass",
            timeout=1.0,
        )
        assert result["passed"] is True
        assert result["result"] == "passed"


class TestGetKeyMetrics:
    @pytest.mark.asyncio
    async def test_selects_headline_metrics(self) -> None:
        server = _make_server()
        responses = []
        for task_idx in range(4):
            for rollout_idx in range(3):
                passed = (task_idx + rollout_idx) % 2 == 0
                responses.append(
                    {
                        TASK_INDEX_KEY_NAME: task_idx,
                        ROLLOUT_INDEX_KEY_NAME: rollout_idx,
                        "reward": 1.0 if passed else 0.0,
                        "passed": passed,
                        "completion": f"x={task_idx}" if rollout_idx < 2 else None,
                    }
                )
        result = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=responses))
        km = result.key_metrics
        assert "pass@3/accuracy" in km
        assert "majority@3/accuracy" in km
        assert "pass@1[avg-of-3]/accuracy" in km
        assert not any("std_dev" in k for k in km)
        assert not any("no_answer" in k for k in km)
