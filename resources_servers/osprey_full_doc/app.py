# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Optional

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.reward_profile import compute_pass_majority_metrics, highest_k_metrics
from resources_servers.osprey_full_doc.common.evaluator import (
    WRONG_PREDICTION_TYPES,
    evaluate_response,
)


class OspreyFullDocResourcesServerConfig(BaseResourcesServerConfig):
    pass


class OspreyFullDocVerifyRequest(BaseVerifyRequest):
    response: Optional[NeMoGymResponse] = None
    doc_name: Optional[str] = None
    line_item_name: Optional[str] = None
    ground_truth: Optional[Any] = None


class OspreyFullDocVerifyResponse(OspreyFullDocVerifyRequest, BaseVerifyResponse):
    is_correct: bool
    wrong_prediction_type: Optional[str]
    clean_prediction: Optional[Any] = None


class OspreyFullDocResourcesServer(SimpleResourcesServer):
    config: OspreyFullDocResourcesServerConfig

    async def verify(self, body: OspreyFullDocVerifyRequest) -> OspreyFullDocVerifyResponse:
        result = evaluate_response(
            response=body.response,
            ground_truth=body.ground_truth,
        )
        return OspreyFullDocVerifyResponse(
            **body.model_dump(),
            reward=float(result.score),
            is_correct=result.is_correct,
            wrong_prediction_type=result.wrong_prediction_type,
            clean_prediction=result.clean_prediction,
        )

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        metrics, _, _, _ = compute_pass_majority_metrics(
            tasks,
            score_fn=lambda result: {"accuracy": result["reward"]},
        )

        flat_results = [result for task in tasks for result in task]
        total = len(flat_results)
        if not total:
            return metrics

        correct_count = sum(1 for result in flat_results if result.get("is_correct"))
        metrics["rollout_count/correct"] = correct_count
        metrics["rollout_pct/correct"] = 100.0 * correct_count / total

        for wrong_prediction_type in WRONG_PREDICTION_TYPES:
            count = sum(1 for result in flat_results if result.get("wrong_prediction_type") == wrong_prediction_type)
            slug = wrong_prediction_type.lower().replace(" ", "_")
            metrics[f"rollout_count/{slug}"] = count
            metrics[f"rollout_pct/{slug}"] = 100.0 * count / total

        return metrics

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        keys = highest_k_metrics(agent_metrics, "pass@{k}")
        keys.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]"))
        for name in (
            "rollout_pct/correct",
            "rollout_pct/api_error",
            "rollout_pct/extraction_error",
            "rollout_pct/false_negative",
            "rollout_pct/false_positive",
            "rollout_pct/incorrect_value",
        ):
            if name in agent_metrics:
                keys[name] = agent_metrics[name]
        return keys


if __name__ == "__main__":
    OspreyFullDocResourcesServer.run_webserver()
