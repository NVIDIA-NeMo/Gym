# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
from app import (
    ComputeEvalResourcesServer,
    ComputeEvalResourcesServerConfig,
    ComputeEvalVerifyRequest,
)
from compute_eval.data.data_model import SourceFile

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def server():
    # Mock both ensure_cuda_nvcc (avoid the micromamba download in tests) and
    # get_nvcc_version (avoid PATH dependency on dev/CI machines). The
    # evaluate_solutions call itself is always patched in tests below.
    cfg = ComputeEvalResourcesServerConfig(num_processes=1, host="0.0.0.0", port=8080, entrypoint="", name="")
    with patch("app.ensure_cuda_nvcc", return_value=None), patch("app.get_nvcc_version", return_value="12.9.0"):
        return ComputeEvalResourcesServer(config=cfg, server_client=MagicMock(spec=ServerClient))


@pytest.fixture
def server_no_nvcc():
    cfg = ComputeEvalResourcesServerConfig(num_processes=1, host="0.0.0.0", port=8080, entrypoint="", name="")
    with patch("app.ensure_cuda_nvcc", return_value=None), patch("app.get_nvcc_version", return_value=None):
        yield cfg


class TestNvccCheck:
    def test_missing_nvcc_raises(self, server_no_nvcc):
        with patch("app.ensure_cuda_nvcc", return_value=None), patch("app.get_nvcc_version", return_value=None):
            with pytest.raises(RuntimeError, match="NVCC not found"):
                ComputeEvalResourcesServer(config=server_no_nvcc, server_client=MagicMock(spec=ServerClient))


_MINIMAL_PROBLEM = {
    "type": "cuda_cpp",
    "schema_version": "1.0",
    "task_id": "test-task",
    "date": "2026-01-01",
    "prompt": "Write a CUDA kernel that adds 1.",
    "metadata": {},
    "group": "cuda-kernels",
    "context_files": [],
    "test_files": [],
    "source_references": [],
    "build_command": "nvcc kernel.cu -o kernel",
    "test_command": "./kernel",
    "benchmark_command": "",
    "min_cuda_toolkit": "12.0",
    "compute_capability": "80",
    "requires_datacenter_gpu": False,
    "timeout_seconds": 30,
    "baseline_solution": None,
}


def _make_request(output_text: str, problem: dict | None = None, task_id: str = "test-task"):
    return ComputeEvalVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=NeMoGymResponse(
            id="r",
            created_at=0.0,
            model="m",
            object="response",
            output=[
                NeMoGymResponseOutputMessage(
                    id="msg",
                    content=[NeMoGymResponseOutputText(annotations=[], text=output_text, type="output_text")],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        ),
        verifier_metadata={
            "task_id": task_id,
            "problem": problem if problem is not None else _MINIMAL_PROBLEM,
        },
    )


# ---------------------------------------------------------------------------
# Pre-NVCC validation paths (don't actually invoke evaluate_solutions)
# ---------------------------------------------------------------------------


class TestVerifyEarlyReturns:
    """Inputs that should never reach evaluate_solutions."""

    @pytest.mark.asyncio
    async def test_empty_output_returns_zero(self, server):
        resp = await server.verify(_make_request(""))
        assert resp.reward == 0.0
        assert resp.error == "empty_output"

    @pytest.mark.asyncio
    async def test_whitespace_only_output(self, server):
        resp = await server.verify(_make_request("   \n  \n"))
        assert resp.reward == 0.0
        assert resp.error == "empty_output"

    @pytest.mark.asyncio
    async def test_invalid_problem_schema(self, server):
        resp = await server.verify(_make_request("```cuda\nfoo\n```", problem={"type": "bogus"}))
        assert resp.reward == 0.0
        assert resp.error is not None
        assert "problem schema validation failed" in resp.error

    @pytest.mark.asyncio
    async def test_no_fenced_code_blocks(self, server):
        # _parse_solution falls back to a raw text path when no fences exist;
        # for plain prose with no // file: header, the SourceFile candidate
        # is dropped and we return no_fenced_code_blocks.
        resp = await server.verify(_make_request("This is plain prose, no code."))
        assert resp.reward == 0.0
        assert resp.error == "no_fenced_code_blocks"


# ---------------------------------------------------------------------------
# Solution-parsing exercise (in-process; doesn't touch nvcc)
# ---------------------------------------------------------------------------


class TestParseAndPackage:
    """Verify the server correctly parses multi-file fenced blocks before
    handing off to evaluate_solutions. We patch evaluate_solutions to a no-op
    so this stays purely in-process and doesn't need a working CUDA toolchain.
    """

    @pytest.mark.asyncio
    async def test_single_file_solution(self, server):
        passing_graded = MagicMock(passed=True, skipped=False, build_output="ok", test_output="ok")
        with patch("app.evaluate_solutions", return_value=[passing_graded]) as patched:
            resp = await server.verify(
                _make_request(
                    "Sure, here's the kernel:\n```\n// file: kernel.cu\n"
                    "__global__ void add(int* a) { a[0] += 1; }\n```"
                )
            )
        assert patched.called
        call_kwargs = patched.call_args.kwargs
        sol = call_kwargs["solutions"][0]
        assert len(sol.files) == 1
        assert sol.files[0].path == "kernel.cu"
        assert resp.reward == 1.0
        assert resp.passed is True

    @pytest.mark.asyncio
    async def test_multi_file_solution(self, server):
        passing = MagicMock(passed=True, skipped=False, build_output="ok", test_output="ok")
        with patch("app.evaluate_solutions", return_value=[passing]) as patched:
            output = (
                "Two files:\n"
                "```\n// file: kernel.cu\n__global__ void k() {}\n```\n"
                "```\n// file: helper.cu\nint h() { return 0; }\n```\n"
            )
            resp = await server.verify(_make_request(output))
        sol = patched.call_args.kwargs["solutions"][0]
        paths = {f.path for f in sol.files}
        assert paths == {"kernel.cu", "helper.cu"}
        assert resp.reward == 1.0

    @pytest.mark.asyncio
    async def test_failed_grade_returns_zero_reward(self, server):
        failing = MagicMock(passed=False, skipped=False, build_output="error", test_output="")
        with patch("app.evaluate_solutions", return_value=[failing]):
            resp = await server.verify(_make_request("```\n// file: k.cu\n__global__ void k(){}\n```"))
        assert resp.reward == 0.0
        assert resp.passed is False
        assert resp.build_output == "error"

    @pytest.mark.asyncio
    async def test_evaluate_raises_returns_zero_with_error(self, server):
        with patch("app.evaluate_solutions", side_effect=RuntimeError("nvcc segfault")):
            resp = await server.verify(_make_request("```\n// file: k.cu\n__global__ void k(){}\n```"))
        assert resp.reward == 0.0
        assert resp.error is not None
        assert "evaluate_solutions raised" in resp.error


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_score_fn_pass(self):
        assert ComputeEvalResourcesServer._score_fn({"reward": 1.0}) == {"accuracy": 1.0}

    def test_score_fn_fail(self):
        assert ComputeEvalResourcesServer._score_fn({"reward": 0.0}) == {"accuracy": 0.0}

    def test_compute_metrics_shape(self, server):
        # One task with two rollouts: one pass, one fail. pass@1 = 50%, pass@2 = 100%
        # (compute_pass_majority_metrics reports percentages, not fractions).
        tasks = [
            [
                {"reward": 1.0, "extracted_model_output": "// file: a.cu\nok"},
                {"reward": 0.0, "extracted_model_output": "// file: a.cu\nbad"},
            ]
        ]
        metrics = server.compute_metrics(tasks)
        assert "pass@1[avg-of-2]/accuracy" in metrics
        assert metrics["pass@1[avg-of-2]/accuracy"] == pytest.approx(50.0)
        assert "pass@2/accuracy" in metrics
        assert metrics["pass@2/accuracy"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Sanity: SourceFile parsing (compute_eval pkg dependency)
# ---------------------------------------------------------------------------


class TestSourceFileParsing:
    """Sanity-check that the compute_eval._parse_solution helper still emits
    the SourceFile shape we expect — this catches dep version drift early."""

    def test_returns_source_file_objects(self):
        from compute_eval.generate_completions import _parse_solution

        files = _parse_solution("```\n// file: a.cu\nvoid f() {}\n```")
        assert len(files) == 1
        assert isinstance(files[0], SourceFile)
        assert files[0].path == "a.cu"
