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

from asyncio import Semaphore, get_running_loop
from time import time
from typing import Any, Dict, List, Optional, Union

import ray
from lcb_integration.compute_code_generation_metrics import check_correctness_remote
from lcb_integration.extraction_utils import LMStyle, extract_code
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.reward_profile import (
    add_avg_sample_std_dev,
    compute_pass_majority_metrics,
    compute_subset_metrics,
    highest_k_metrics,
)


# ----------------------------
# Config
# ----------------------------
class CompCodingResourcesServerConfig(BaseResourcesServerConfig):
    num_processes: int
    unit_test_timeout_secs: int
    debug: bool
    reasoning_format_penalty: float = -0.2


# ----------------------------
# Schemas
# ----------------------------


# This is LiveCodeBench format
class UnitTests(BaseModel):
    inputs: List[str]
    outputs: List[str]
    fn_name: Optional[str] = None


class CompCodingRunRequest(BaseRunRequest):
    pass


class CompCodingVerifyRequest(CompCodingRunRequest, BaseVerifyRequest):
    verifier_metadata: Optional[Dict[str, Any]] = None


class CompCodingVerifyResponse(BaseVerifyResponse):
    extracted_model_output: Optional[str] = None
    extracted_model_code: Optional[str] = None
    result: Optional[List[Union[int, bool]]] = None
    metadata: Optional[Dict[str, Any]] = None
    unit_tests_time_taken: Optional[float] = None
    reasoning_format_violation_rate: float = 0.0
    difficulty: Optional[str] = None


# ----------------------------
# Server
# ----------------------------
class CompCodingResourcesServer(SimpleResourcesServer):
    config: CompCodingResourcesServerConfig

    def model_post_init(self, context):
        self._semaphore: Semaphore = Semaphore(value=self.config.num_processes)

    @staticmethod
    def _has_reasoning_format_violation(response) -> bool:
        open_tag = "<think>"
        close_tag = "</think>"

        # Final answer (output_text) should not contain any think tags.
        final_answer = response.output_text or ""
        if open_tag in final_answer or close_tag in final_answer:
            return True

        # Reasoning content should not have more than 1 think tag.
        reasoning = ""
        for item in response.output or []:
            if getattr(item, "type", None) == "reasoning":
                for summary in getattr(item, "summary", []) or []:
                    text = getattr(summary, "text", None)
                    if isinstance(text, str):
                        reasoning += text
        if reasoning.count(open_tag) > 1 or reasoning.count(close_tag) > 1:
            return True

        return False

    @staticmethod
    def _code_score_fn(r: dict) -> Dict[str, float]:
        return {"accuracy": float(r["reward"] > 0)}

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Compute code generation metrics: pass@k, majority@k, per-sample statistics.

        Produces overall metrics (with avg_sample_std_dev) and per-difficulty-subset metrics.
        """
        metrics, all_score_dicts, score_names, max_k = compute_pass_majority_metrics(
            tasks,
            score_fn=self._code_score_fn,
            answer_key="extracted_model_code",
            return_internals=True,
        )
        add_avg_sample_std_dev(metrics, all_score_dicts, score_names, max_k)
        metrics.update(compute_subset_metrics(tasks, "difficulty", self._code_score_fn, "extracted_model_code"))
        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Select headline metrics for code generation benchmarks."""
        key: Dict[str, Any] = {}

        for name in ("mean/input_tokens", "mean/output_tokens"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]

        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]", score_names=["accuracy"]))
        key.update(highest_k_metrics(agent_metrics, "pass@{k}", score_names=["accuracy"]))
        key.update(highest_k_metrics(agent_metrics, "majority@{k}", score_names=["accuracy"]))

        # Per-difficulty-subset headlines
        for prefix in {k.split("/pass@")[0] for k in agent_metrics if "/pass@" in k and k[0].islower()}:
            key.update(highest_k_metrics(agent_metrics, f"{prefix}/pass@1[avg-of-{{k}}]", score_names=["accuracy"]))

        return key

    async def verify(self, body: CompCodingVerifyRequest) -> CompCodingVerifyResponse:
        model_out = body.response.output_text
        difficulty = (body.verifier_metadata or {}).get("difficulty")

        if not model_out or not model_out.strip():
            # A response existed but had no usable text -> model failure
            return CompCodingVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                difficulty=difficulty,
            )

        tests = UnitTests.model_validate(body.verifier_metadata["unit_tests"])

        # 3) extract code (code fence or raw)
        code = extract_code(model_out, LMStyle.OpenAIChat)
        if not code:
            return CompCodingVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                extracted_model_output=model_out,
                difficulty=difficulty,
            )

        # 4) run (no sandbox)
        async with self._semaphore:
            loop = get_running_loop()

            """
            Sample looks like this:
            ```json
            {
                "input_output": "{\"inputs\": [...], ...}",
            }
            ```
            `input_output` looks like this:
            ```json
            {
                "inputs": [
                    "6\n4 13 2 3 2 6",
                    ...
                ],
                "outputs": [
                    "4 30 2 13 2 13",
                    ...
                ],
                "fn_name": null
            }
            ```
            """

            # We can directly measure here since we are inside the semaphore.
            start_time = time()

            task_args = (
                {"input_output": tests.model_dump_json()},  # sample
                code,  # generation
                self.config.unit_test_timeout_secs,  # timeout
                self.config.debug,  # debug
            )

            future = check_correctness_remote.remote(*task_args)
            result, metadata = await loop.run_in_executor(None, ray.get, future)

            unit_tests_time_taken = time() - start_time

        has_violation = self._has_reasoning_format_violation(body.response)

        return CompCodingVerifyResponse(
            **body.model_dump(),
            reward=(
                self.config.reasoning_format_penalty
                if has_violation
                else (1.0 if all(r == True for r in result) else 0.0)
            ),
            extracted_model_output=model_out,
            extracted_model_code=code,
            result=result,
            metadata=metadata,
            unit_tests_time_taken=unit_tests_time_taken,
            reasoning_format_violation_rate=1.0 if has_violation else 0.0,
            difficulty=difficulty,
        )


if __name__ == "__main__":
    CompCodingResourcesServer.run_webserver()
