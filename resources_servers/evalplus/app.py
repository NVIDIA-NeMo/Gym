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

"""EvalPlus resource server.

Verifies Python function-completion benchmarks (HumanEval+, MBPP+) by
running the model's extracted code against the EvalPlus base + plus test
inputs. Returns two named scores per task:

  - passing_base_tests: completion passes the base HumanEval / MBPP tests
  - passing_plus_tests: completion ALSO passes the EvalPlus extra tests

This mirrors NeMo-Skills' `eval_evalplus` evaluator (in
`nemo_skills/evaluation/evaluator/code.py`) which delegates to
`evalplus.evaluate.evaluate(...)` and reads back `base_status` and
`plus_status` per task.

The dataset is selected via `dataset: humaneval | mbpp` in the server
config. Both flow through the same `evalplus.evaluate.check_correctness`
entry point so adding MBPP+ as a separate benchmark requires only a new
benchmark dir (no server changes).
"""

from asyncio import Semaphore, get_running_loop
from typing import Any, Dict, List, Optional

import ray
from evalplus_integration.runner import check_correctness_remote

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.reward_profile import (
    compute_pass_majority_metrics,
    highest_k_metrics,
)


# ----------------------------
# Config
# ----------------------------
class EvalPlusResourcesServerConfig(BaseResourcesServerConfig):
    dataset: str  # "humaneval" or "mbpp"
    num_processes: int
    # Time-limit knobs are fed straight into evalplus.evaluate.check_correctness.
    # Skills' default in eval_evalplus(): min_time_limit=1, gt_time_limit_factor=4.0.
    min_time_limit: float = 1.0
    gt_time_limit_factor: float = 4.0
    # Optional EvalPlus-version pin (default: "default", as in Skills).
    evalplus_version: str = "default"


# ----------------------------
# Schemas
# ----------------------------


class EvalPlusRunRequest(BaseRunRequest):
    pass


class EvalPlusVerifyRequest(EvalPlusRunRequest, BaseVerifyRequest):
    verifier_metadata: Optional[Dict[str, Any]] = None


class EvalPlusVerifyResponse(BaseVerifyResponse):
    extracted_model_output: Optional[str] = None
    extracted_model_code: Optional[str] = None
    base_status: Optional[str] = None
    plus_status: Optional[str] = None
    is_correct: bool = False  # base tests passed
    is_correct_plus: bool = False  # base AND plus tests passed
    metadata: Optional[Dict[str, Any]] = None


# ----------------------------
# Code extraction
# ----------------------------
def extract_code_strict(completion: str, language: str = "python") -> str:
    """Match Skills' `preprocess_code` last-fence + strict-mode semantics.

    Skills additionally strips `<think>...</think>` reasoning preambles;
    we delegate that to the vLLM `--reasoning-parser` flag, so this function
    operates on already-clean output. If the model emits no closing fence,
    the extracted code is empty (strict mode), matching Skills.

    Picks the LAST occurrence of a fenced code block (preferring
    ```python over a generic ```) so chain-of-thought scratch code blocks
    don't shadow the final solution.
    """
    completion = completion.replace("\r", "")
    specific_fence = f"```{language}"
    generic_fence = "```"
    start = completion.rfind(specific_fence)
    fence_len = len(specific_fence)
    if start == -1:
        start = completion.rfind(generic_fence)
        fence_len = len(generic_fence)
    if start == -1:
        return ""
    rest = completion[start + fence_len :]
    end = rest.find(generic_fence)
    if end == -1:
        return ""
    return rest[:end].strip()


# ----------------------------
# Server
# ----------------------------
class EvalPlusResourcesServer(SimpleResourcesServer):
    config: EvalPlusResourcesServerConfig

    def model_post_init(self, context):
        self._semaphore: Semaphore = Semaphore(value=self.config.num_processes)
        self._dataset, self._expected_output = _load_dataset_and_expected(
            self.config.dataset, self.config.evalplus_version
        )

    @staticmethod
    def _score_fn(r: dict) -> Dict[str, float]:
        return {
            "passing_base_tests": float(r.get("is_correct", False)),
            "passing_plus_tests": float(r.get("is_correct_plus", False)),
        }

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Compute pass@k / majority@k for both base and plus verdicts."""
        metrics, _, _, _ = compute_pass_majority_metrics(
            tasks,
            score_fn=self._score_fn,
            answer_key="extracted_model_code",
        )
        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Headline metrics: pass@1[avg-of-k], pass@k, majority@k for base + plus."""
        key: Dict[str, Any] = {}
        for name in ("mean/input_tokens", "mean/output_tokens"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]
        score_names = ["passing_base_tests", "passing_plus_tests"]
        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]", score_names=score_names))
        key.update(highest_k_metrics(agent_metrics, "pass@{k}", score_names=score_names))
        key.update(highest_k_metrics(agent_metrics, "majority@{k}", score_names=score_names))
        return key

    async def verify(self, body: EvalPlusVerifyRequest) -> EvalPlusVerifyResponse:
        model_out = body.response.output_text
        verifier_metadata = body.verifier_metadata or {}
        task_id = verifier_metadata.get("task_id")

        if task_id is None:
            return EvalPlusVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                metadata={"error": "missing task_id in verifier_metadata"},
            )
        if task_id not in self._dataset:
            return EvalPlusVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                metadata={"error": f"unknown task_id {task_id!r} for dataset {self.config.dataset!r}"},
            )

        if not model_out or not model_out.strip():
            return EvalPlusVerifyResponse(
                **body.model_dump(),
                reward=0.0,
            )

        code = extract_code_strict(model_out, language="python")
        if not code:
            return EvalPlusVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                extracted_model_output=model_out,
            )

        problem = self._dataset[task_id]
        expected_output = self._expected_output[task_id]

        # Match `evalplus.evaluate.evaluate()`'s completion-style behaviour:
        # if the JSONL row has `completion` (no `solution`), it prepends
        # `problem["prompt"]` before calling check_correctness. The prompt
        # carries the imports and the function signature; the model usually
        # emits just the body (or the body + a redundant `def`), so without
        # this prepend the candidate hits NameError on type annotations like
        # `List[float]` and check_correctness reports a hard fail with empty
        # details. Skills' `eval_evalplus` evaluator goes through `evaluate()`
        # and gets this prepend for free; check_correctness does NOT.
        solution = code if code.startswith(problem["prompt"]) else problem["prompt"] + code

        async with self._semaphore:
            loop = get_running_loop()
            future = check_correctness_remote.remote(
                self.config.dataset,
                problem,
                solution,
                expected_output,
                self.config.min_time_limit,
                self.config.gt_time_limit_factor,
            )
            result = await loop.run_in_executor(None, ray.get, future)

        is_correct = result.get("base_status") == "pass"
        is_correct_plus = is_correct and result.get("plus_status") == "pass"

        return EvalPlusVerifyResponse(
            **body.model_dump(),
            reward=1.0 if is_correct_plus else 0.0,
            extracted_model_output=model_out,
            extracted_model_code=code,
            base_status=result.get("base_status"),
            plus_status=result.get("plus_status"),
            is_correct=is_correct,
            is_correct_plus=is_correct_plus,
            metadata={
                "base_details": result.get("base_details"),
                "plus_details": result.get("plus_details"),
            },
        )


# ----------------------------
# Dataset / expected-output loading
# ----------------------------
def _load_dataset_and_expected(dataset: str, version: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Load EvalPlus dataset + groundtruth expected outputs.

    Mirrors `evalplus.evaluate.evaluate`'s setup path so the per-task
    `check_correctness` calls see identical inputs to the batch evaluator
    Skills uses. `get_groundtruth` is cached on disk by EvalPlus, so this
    is a one-time cost at server startup.
    """
    if dataset == "humaneval":
        from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash

        problems = get_human_eval_plus(version=version)
        problems_hash = get_human_eval_plus_hash(version=version)
        # HumanEval has expected outputs for every task (no None entries).
        tasks_only_output_not_none: List[str] = []
    elif dataset == "mbpp":
        from evalplus.data import get_mbpp_plus, get_mbpp_plus_hash
        from evalplus.data.mbpp import mbpp_serialize_inputs  # noqa: F401  (registers serializer)

        problems = get_mbpp_plus(version=version)
        problems_hash = get_mbpp_plus_hash(version=version)
        # MBPP+: a handful of tasks have no canonical solution; evalplus
        # filters them. Mirror its convention.
        tasks_only_output_not_none = [tid for tid, p in problems.items() if p.get("canonical_solution")]
    else:
        raise ValueError(f"Unsupported evalplus dataset: {dataset!r}")

    from evalplus.evaluate import get_groundtruth

    expected_output = get_groundtruth(problems, problems_hash, tasks_only_output_not_none)
    return problems, expected_output


if __name__ == "__main__":
    EvalPlusResourcesServer.run_webserver()
