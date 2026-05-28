# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import concurrent.futures
from typing import Annotated, Any, Dict, List, Optional

from compute_eval.data.data_model import (
    CudaCppProblem,
    CudaPythonProblem,
    FileSolution,
    PatchSolution,
)
from compute_eval.execution import evaluate_solutions
from compute_eval.generate_completions import _parse_solution
from compute_eval.utils.eval_utils import get_nvcc_version
from pydantic import Field, TypeAdapter
from setup_cuda_nvcc import ensure_cuda_nvcc

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.reward_profile import (
    compute_pass_majority_metrics,
    highest_k_metrics,
)


_PROBLEM_ADAPTER = TypeAdapter(Annotated[CudaCppProblem | CudaPythonProblem, Field(discriminator="type")])
_SOLUTION_ADAPTER = TypeAdapter(Annotated[FileSolution | PatchSolution, Field(discriminator="type")])


class ComputeEvalResourcesServerConfig(BaseResourcesServerConfig):
    # Bound concurrent NVCC compilations. nvcc is single-threaded per
    # process and IO-bound on the lustre mount, so 8 is a safe default
    # on an 8-GPU node.
    num_processes: int = 8


class ComputeEvalVerifyRequest(BaseVerifyRequest):
    verifier_metadata: Optional[Dict[str, Any]] = None


class ComputeEvalVerifyResponse(BaseVerifyResponse):
    extracted_model_output: Optional[str] = None
    extracted_solution: Optional[Dict[str, Any]] = None
    passed: bool = False
    skipped: bool = False
    build_output: Optional[str] = None
    test_output: Optional[str] = None
    task_id: Optional[str] = None
    error: Optional[str] = None


class ComputeEvalResourcesServer(SimpleResourcesServer):
    """compute-eval verifier.

    Mirrors Skills' ``ComputeEvalEvaluator``: parse fenced code blocks from
    the model output into a multi-file ``FileSolution``, then compile + run
    against hidden tests via ``compute_eval.execution.evaluate_solutions``.

    The Skills evaluator at HEAD has a bug (``graded.passed`` on a list
    return) that silently returns passed=False for every problem. This
    server implements the corrected version (``graded_list[0].passed``);
    the migration's recipe-side Skills tree carries the parallel fix at
    nemo_skills/evaluation/evaluator/compute_eval.py.
    """

    config: ComputeEvalResourcesServerConfig

    def model_post_init(self, context):
        # Auto-install the CUDA Toolkit via micromamba if nvcc isn't on PATH.
        # First boot installs to .cuda_nvcc/env (~5 min, cached on lustre);
        # subsequent boots short-circuit.
        ensure_cuda_nvcc()
        nvcc_version = get_nvcc_version()
        if not nvcc_version:
            raise RuntimeError("NVCC not found after auto-install. Check setup_cuda_nvcc logs.")
        # compute_eval's verify_source_references path uses a tree-sitter
        # Parser from tree_sitter_language_pack, which is a pyo3-backed Rust
        # type marked unsendable. asyncio.to_thread cycles through threads in
        # the default executor — a Parser created on thread A panics when
        # touched from thread B on a later call. Pin all evaluate_solutions
        # invocations to a single dedicated worker thread so the parser is
        # created and used exclusively on that one thread.
        self._eval_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="compute-eval-parser"
        )
        # Bound concurrent evaluate_solutions calls — they're all serialized
        # by the single-thread executor anyway, so the semaphore prevents
        # unbounded queue growth.
        self._semaphore = asyncio.Semaphore(self.config.num_processes)

    @staticmethod
    def _score_fn(r: dict) -> Dict[str, float]:
        return {"accuracy": float(r["reward"] > 0)}

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        return compute_pass_majority_metrics(
            tasks,
            score_fn=self._score_fn,
            answer_key="extracted_model_output",
        )[0]

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        key: Dict[str, Any] = {}
        for name in ("mean/input_tokens", "mean/output_tokens"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]
        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]", score_names=["accuracy"]))
        key.update(highest_k_metrics(agent_metrics, "pass@{k}", score_names=["accuracy"]))
        key.update(highest_k_metrics(agent_metrics, "majority@{k}", score_names=["accuracy"]))
        return key

    async def verify(self, body: ComputeEvalVerifyRequest) -> ComputeEvalVerifyResponse:
        model_out = body.response.output_text or ""
        meta = body.verifier_metadata or {}
        task_id = meta.get("task_id")

        if not model_out.strip():
            return ComputeEvalVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                task_id=task_id,
                error="empty_output",
            )

        try:
            problem = _PROBLEM_ADAPTER.validate_python(meta["problem"])
        except Exception as e:
            return ComputeEvalVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                extracted_model_output=model_out,
                task_id=task_id,
                error=f"problem schema validation failed: {e}",
            )

        try:
            files = _parse_solution(model_out)
        except Exception as e:
            return ComputeEvalVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                extracted_model_output=model_out,
                task_id=task_id,
                error=f"parse_solution failed: {e}",
            )

        if not files:
            return ComputeEvalVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                extracted_model_output=model_out,
                task_id=task_id,
                error="no_fenced_code_blocks",
            )

        try:
            solution = _SOLUTION_ADAPTER.validate_python(
                {"type": "file", "task_id": task_id, "files": [f.model_dump() for f in files]}
            )
        except Exception as e:
            return ComputeEvalVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                extracted_model_output=model_out,
                task_id=task_id,
                error=f"solution schema validation failed: {e}",
            )

        async with self._semaphore:
            try:
                loop = asyncio.get_running_loop()
                graded_list = await loop.run_in_executor(
                    self._eval_executor,
                    lambda: evaluate_solutions(
                        problem=problem,
                        solutions=[solution],
                        eval_mode="local",
                        profile_mode=None,
                    ),
                )
            except Exception as e:
                return ComputeEvalVerifyResponse(
                    **body.model_dump(),
                    reward=0.0,
                    extracted_model_output=model_out,
                    extracted_solution=solution.model_dump(),
                    task_id=task_id,
                    error=f"evaluate_solutions raised: {e}",
                )

        graded = graded_list[0]
        return ComputeEvalVerifyResponse(
            **body.model_dump(),
            reward=1.0 if graded.passed else 0.0,
            extracted_model_output=model_out,
            extracted_solution=solution.model_dump(),
            passed=graded.passed,
            skipped=graded.skipped,
            build_output=graded.build_output,
            test_output=graded.test_output,
            task_id=task_id,
        )


if __name__ == "__main__":
    ComputeEvalResourcesServer.run_webserver()
