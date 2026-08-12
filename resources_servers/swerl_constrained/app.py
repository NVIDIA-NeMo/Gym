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
"""SWE constrained resources server.

Extends swerl_gen with agentic-if constraint rewards. Constraints are declared
per-task in ``metadata.constraints`` using the agentic-if schema::

    constraints: [{"type": "<canonical id>", "params": {...}}, ...]

Canonical ids and verifiers live in the agentic-if repo
(instruction_pool/rubrics/) — this server imports them rather than vendoring,
so constraint semantics have a single source of truth. Grading is
trajectory-based: scope filtering, injection awareness, N/A handling, and
per-step partial credit all follow agentic-if grade_constraints().

Reward is the shaped multiplicative formula from agentic-if reward.py::

    reward = task_reward * (1 + alpha * constraint_reward)

task_reward = 0 always yields reward 0 (no constraint reward hacking), and
constraint pressure never zeroes the task gradient on hard SWE tasks. When no
constraint had a gradeable step (any_graded=False), the constraint term is
dropped — "not measured" is not perfect compliance.

Optional metadata keys: ``constraint_alpha`` (default from config),
``grading_mode`` ("fraction" | "binary"), ``step_aggregation`` ("mean" | "all"),
``injection_mode``, ``injection_step``.
"""

import base64
import json
import logging
import time
from asyncio import Semaphore
from typing import Any, Optional

from fastapi import FastAPI

from nemo_gym.base_resources_server import (
    BaseMultiRewardVerifyResponse,
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    SimpleResourcesServer,
)
from resources_servers.swerl_constrained.eval.agentic_if_bridge import (
    coerce_constraint_declarations,
    load_grading_core,
)
from resources_servers.swerl_gen.eval.process_patch import (
    extract_pred_patch,
    extract_pred_patch_relaxed_formatting,
)
from resources_servers.swerl_gen.eval.singularity_utils import compute_score


log = logging.getLogger(__name__)

DEFAULT_CONSTRAINT_ALPHA = 1.0  # FORMAT mode default (agentic-if reward.py)


class SWEConstrainedResourcesServerConfig(BaseResourcesServerConfig):
    num_processes: int = 1
    sandbox_timeout: int = 600
    debug: bool = False
    relaxed_formatting: bool = False
    # Path to the agentic-if checkout (absolute, or relative to the Gym repo
    # root). AGENTIC_IF_REPO env var takes precedence.
    agentic_if_repo: Optional[str] = None
    constraint_alpha: float = DEFAULT_CONSTRAINT_ALPHA


class SWEConstrainedRunRequest(BaseRunRequest):
    instance: dict[str, Any]
    metadata: dict[str, Any] = {}
    dataset_name: Optional[str] = None
    dataset_split: Optional[str] = None
    mode: str = "eval"


class SWEConstrainedVerifyRequest(SWEConstrainedRunRequest, BaseVerifyRequest):
    pass


class SWEConstrainedVerifyResponse(BaseMultiRewardVerifyResponse):
    # Inherited from BaseMultiRewardVerifyResponse:
    #   reward: float
    #   reward_components: dict[str, float]
    task_reward: float
    constraint_reward: Optional[float] = None
    # False when no constraint had a gradeable in-scope step: "format not
    # measured", NOT perfect compliance. The constraint term is dropped from
    # the reward in that case.
    constraint_graded: bool = False
    constraint_alpha: float = DEFAULT_CONSTRAINT_ALPHA
    model_patch: Optional[str] = None
    model_output: Optional[str] = None
    verification_result: Optional[dict[str, Any]] = None
    verification_time: Optional[float] = None
    # Per-constraint pass/fail, partial-credit scores, applicability, and
    # human-readable violations from agentic-if grade_constraints().
    constraint_results: dict[str, bool] = {}
    constraint_scores: dict[str, float] = {}
    constraint_applicable: dict[str, bool] = {}
    violations: list[str] = []


def _extract_last_assistant_text(body: BaseVerifyRequest) -> str:
    texts: list[str] = []
    for o in body.response.output:
        if getattr(o, "type", None) == "message" and getattr(o, "role", None) == "assistant":
            content = getattr(o, "content", None)
            if isinstance(content, list):
                for c in content:
                    t = getattr(c, "text", None)
                    if isinstance(t, str):
                        texts.append(t)
            elif isinstance(content, str):
                texts.append(content)
    return "\n".join(texts).strip()


class SWEConstrainedResourcesServer(SimpleResourcesServer):
    config: SWEConstrainedResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        return super().setup_webserver()

    def model_post_init(self, context):
        self._semaphore: Semaphore = Semaphore(value=self.config.num_processes)
        # Fail fast at startup if the agentic-if checkout is missing.
        (
            self._parse_trajectory,
            self._grade_constraints,
            self._compute_reward,
            self._injection_mode_cls,
        ) = load_grading_core(self.config.agentic_if_repo)

    def _grade(self, body: SWEConstrainedVerifyRequest) -> Any:
        constraints = coerce_constraint_declarations(body.metadata.get("constraints", []))
        steps = self._parse_trajectory(body.response.output)
        return self._grade_constraints(
            steps,
            constraints,
            injection_mode=self._injection_mode_cls(
                body.metadata.get("injection_mode", self._injection_mode_cls.SYSTEM_PROMPT)
            ),
            injection_step=int(body.metadata.get("injection_step", 0)),
            grading_mode=body.metadata.get("grading_mode", "fraction"),
            step_aggregation=body.metadata.get("step_aggregation", "mean"),
        )

    def _build_response(
        self,
        body: SWEConstrainedVerifyRequest,
        task_reward: float,
        grading: Any,
        alpha: float,
        **extra: Any,
    ) -> SWEConstrainedVerifyResponse:
        if grading.any_graded:
            shaped = self._compute_reward(task_reward, grading.reward, alpha=alpha)
            reward = shaped.total
            constraint_reward: Optional[float] = grading.reward
        else:
            reward = task_reward
            constraint_reward = None

        reward_components = {"task": task_reward}
        if grading.any_graded:
            reward_components["constraint"] = grading.reward
            for name, score in grading.constraint_scores.items():
                if grading.constraint_applicable.get(name):
                    reward_components[f"constraint_{name}"] = score

        return SWEConstrainedVerifyResponse(
            **body.model_dump(),
            reward=reward,
            reward_components=reward_components,
            task_reward=task_reward,
            constraint_reward=constraint_reward,
            constraint_graded=grading.any_graded,
            constraint_alpha=alpha,
            constraint_results=grading.constraint_results,
            constraint_scores=grading.constraint_scores,
            constraint_applicable=grading.constraint_applicable,
            violations=grading.violations,
            **extra,
        )

    async def verify(self, body: SWEConstrainedVerifyRequest) -> SWEConstrainedVerifyResponse:
        alpha = float(body.metadata.get("constraint_alpha", self.config.constraint_alpha))

        # Constraint grading runs on the trajectory regardless of patch
        # extraction: with the shaped formula, task_reward=0 zeroes the total,
        # but the per-constraint diagnostics stay available for RL logging.
        grading = self._grade(body)

        predict_str = _extract_last_assistant_text(body)
        if not predict_str:
            log.debug("Zero task reward (empty model output)")
            return self._build_response(body, 0.0, grading, alpha)

        try:
            if self.config.relaxed_formatting:
                extracted = extract_pred_patch_relaxed_formatting(
                    json.loads(body.metadata.get("relevant_file_contents", "{}")),
                    predict_str,
                    body.metadata.get("remove_repo_name", False),
                )
            else:
                extracted = extract_pred_patch(
                    json.loads(body.metadata.get("relevant_file_contents", "{}")),
                    predict_str,
                    body.metadata.get("remove_repo_name", False),
                )
        except Exception:
            extracted = None

        if extracted is None:
            log.debug("Zero task reward (patch extraction failed)")
            return self._build_response(body, 0.0, grading, alpha, model_output=predict_str)

        model_patch: str = extracted["model_patch"]

        extra_info = {"instance_info": body.instance, "image": body.metadata.get("image", "")}
        extra_info_b64 = base64.b64encode(json.dumps(extra_info).encode()).decode()

        async with self._semaphore:
            start = time.time()
            future = compute_score.remote(
                extra_info_b64,
                model_patch,
                None,  # repro_test_info_base64
                "eval",
                self.config.sandbox_timeout,
                self.config.debug,
            )
            task_reward_raw, verification_result = await future
            verification_time = time.time() - start

        return self._build_response(
            body,
            float(task_reward_raw),
            grading,
            alpha,
            model_patch=model_patch,
            model_output=predict_str,
            verification_result=verification_result,
            verification_time=verification_time,
        )


if __name__ == "__main__":
    SWEConstrainedResourcesServer.run_webserver()
