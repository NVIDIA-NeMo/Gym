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

Extends swerl_gen with constraint rewards. Constraints are declared per-task
inside ``metadata.constraints`` (list of constraint names) and an optional
``metadata.constraint_weight`` (float, default 0.3).

Final reward = task_reward * (1 - w) + mean(constraint_scores) * w

Supported constraints (deterministic, no LLM call needed):
  - minimal_editing       : penalizes patches much larger than the golden reference
  - no_hardcoded_secrets  : penalizes patches that introduce literal credentials
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
from resources_servers.swerl_gen.eval.process_patch import (
    extract_pred_patch,
    extract_pred_patch_relaxed_formatting,
)
from resources_servers.swerl_gen.eval.singularity_utils import compute_score
from resources_servers.swerl_constrained.eval.constraints import (
    CONSTRAINT_REGISTRY,
    run_constraints,
)


log = logging.getLogger(__name__)

DEFAULT_CONSTRAINT_WEIGHT = 0.3


class SWEConstrainedResourcesServerConfig(BaseResourcesServerConfig):
    num_processes: int = 1
    sandbox_timeout: int = 600
    debug: bool = False
    relaxed_formatting: bool = False


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
    constraint_reward: float
    model_patch: Optional[str] = None
    model_output: Optional[str] = None
    verification_result: Optional[dict[str, Any]] = None
    verification_time: Optional[float] = None
    constraint_details: dict[str, Any] = {}


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

    async def verify(self, body: SWEConstrainedVerifyRequest) -> SWEConstrainedVerifyResponse:
        constraints: list[str] = body.metadata.get("constraints", [])
        constraint_weight: float = float(body.metadata.get("constraint_weight", DEFAULT_CONSTRAINT_WEIGHT))
        golden_patch: Optional[str] = body.instance.get("patch")

        unknown = [c for c in constraints if c not in CONSTRAINT_REGISTRY]
        if unknown:
            log.warning("Unknown constraints will be skipped: %s", unknown)

        # --- extract model output ---
        predict_str = _extract_last_assistant_text(body)
        if not predict_str:
            return self._zero_reward(body, constraint_weight, constraints, "empty model output")

        # --- extract patch ---
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
            return self._zero_reward(body, constraint_weight, constraints, "patch extraction failed")

        model_patch: str = extracted["model_patch"]

        # --- sandbox task evaluation ---
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

        task_reward = float(task_reward_raw)

        # --- constraint evaluation ---
        constraint_results = run_constraints(constraints, model_patch, golden_patch)
        valid_scores = [r["score"] for r in constraint_results.values() if r["score"] is not None]
        constraint_reward = sum(valid_scores) / len(valid_scores) if valid_scores else 1.0

        # --- combine ---
        if constraints:
            reward = task_reward * (1.0 - constraint_weight) + constraint_reward * constraint_weight
        else:
            reward = task_reward

        reward_components = {"task": task_reward, "constraint": constraint_reward}
        for name, result in constraint_results.items():
            if result["score"] is not None:
                reward_components[f"constraint_{name}"] = result["score"]

        return SWEConstrainedVerifyResponse(
            **body.model_dump(),
            reward=reward,
            reward_components=reward_components,
            task_reward=task_reward,
            constraint_reward=constraint_reward,
            model_patch=model_patch,
            model_output=predict_str,
            verification_result=verification_result,
            verification_time=verification_time,
            constraint_details=constraint_results,
        )

    def _zero_reward(
        self,
        body: SWEConstrainedVerifyRequest,
        constraint_weight: float,
        constraints: list[str],
        reason: str,
    ) -> SWEConstrainedVerifyResponse:
        log.debug("Zero reward (%s)", reason)
        return SWEConstrainedVerifyResponse(
            **body.model_dump(),
            reward=0.0,
            reward_components={"task": 0.0, "constraint": 0.0},
            task_reward=0.0,
            constraint_reward=0.0,
            constraint_details={c: {"score": None, "detail": {"error": reason}} for c in constraints},
        )


if __name__ == "__main__":
    SWEConstrainedResourcesServer.run_webserver()
