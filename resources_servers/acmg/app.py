# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic ACMG/AMP variant classification verifier using the dataset's pinned scorer."""

from pathlib import Path
from typing import Any, ClassVar

from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verification import VerifierFixture
from resources_servers.acmg.scoring import final_answer_text, metrics, score
from resources_servers.acmg.task_data import TaskData


HEADLINE_METRICS = ("label_accuracy", "criterion_f1", "self_consistency", "official_reward")


class AcmgResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS


class AcmgVerifyRequest(BaseVerifyRequest):
    verifier_metadata: TaskData


class AcmgVerifyResponse(BaseVerifyResponse, AcmgVerifyRequest):
    components: dict[str, float | None]
    predicted_label: str | None
    predicted_criteria: dict[str, str] | None
    returned_n: int
    returned_acceptable: int
    required_n: int
    required_hit: int
    self_consistency_assessable: bool


class AcmgVerifier:
    async def verify(self, body: AcmgVerifyRequest) -> AcmgVerifyResponse:
        task = body.verifier_metadata
        text = final_answer_text(body.response.output_text or "")
        result = score(text, task.reference.model_dump(), task.arm)
        return AcmgVerifyResponse(**body.model_dump(), **result)


class AcmgResourcesServer(AcmgVerifier, SimpleResourcesServer):
    # Each verification is a pure function of the request: no session state, nothing to export.
    ray_enabled = False
    config: AcmgResourcesServerConfig

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        return metrics([row for task in tasks for row in task])

    def compute_repeat_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        return self.compute_metrics(tasks)

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in agent_metrics.items()
            if key.startswith("acmg/") and key.rsplit("/", 1)[-1] in HEADLINE_METRICS
        }


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=AcmgVerifier,
    request_model=AcmgVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
)


if __name__ == "__main__":
    AcmgResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = AcmgResourcesServer.run_webserver()  # noqa: F401
