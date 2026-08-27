# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PrimeVul paired vulnerability classification as a NeMo Gym resources server."""

from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.verifier_fixture import VerifierFixture
from resources_servers.primevul.metrics import aggregate_paired, score_verdict
from resources_servers.primevul.verdict import parse_verdict


def final_assistant_text(response: NeMoGymResponse) -> str:
    """The model's final assistant text.

    Scans backwards for the last `message` item rather than reading `response.output_text`,
    which concatenates every message in the trajectory. For the single-turn reference harness the
    two agree; for a multi-turn harness they do not, and the verdict is what the agent said last.
    """
    for item in reversed(response.output or []):
        # Output message items are always the assistant's; the other item types in a trajectory
        # are tool calls, tool outputs and reasoning, none of which carry the verdict.
        if getattr(item, "type", None) != "message":
            continue
        texts = [
            part.text
            for part in getattr(item, "content", None) or []
            if getattr(part, "type", None) == "output_text" and getattr(part, "text", None)
        ]
        if texts:
            return "".join(texts)
    return response.output_text


class PrimeVulResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS


class PrimeVulVerifierMetadata(BaseModel):
    """Required gold labels for one member of a vulnerable/fixed pair."""

    id: str
    pair_id: str
    gold_is_vulnerable: bool


class PrimeVulVerifyRequest(BaseVerifyRequest):
    verifier_metadata: PrimeVulVerifierMetadata


class PrimeVulVerifyResponse(BaseVerifyResponse):
    """Per-row reward and the fields required for paired aggregation."""

    correct: bool
    parse_error: bool
    pair_id: str
    pred_is_vulnerable: bool | None = None
    gold_is_vulnerable: bool


class PrimeVulVerifier:
    async def verify(self, body: PrimeVulVerifyRequest) -> PrimeVulVerifyResponse:
        verdict = parse_verdict(final_assistant_text(body.response))
        return PrimeVulVerifyResponse(
            **body.model_dump(),
            **score_verdict(verdict, body.verifier_metadata.model_dump()),
        )


class PrimeVulResourcesServer(PrimeVulVerifier, SimpleResourcesServer):
    config: PrimeVulResourcesServerConfig

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        return aggregate_paired(tasks)

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        """Promote paired accuracy and diagnostics over per-row reward."""
        keys = (
            "mean/paired_accuracy",
            "mean/pairwise_vulnerable_rate",
            "mean/pairwise_benign_rate",
            "mean/pairwise_reversed_rate",
            "mean/parse_error_rate",
            "mean/binary_accuracy",
            "mean/binary_f1",
            "mean/reward",
        )
        return {key: agent_metrics[key] for key in keys if key in agent_metrics}


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=PrimeVulVerifier,
    request_model=PrimeVulVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
)


if __name__ == "__main__":
    PrimeVulResourcesServer.run_webserver()
