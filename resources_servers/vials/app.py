# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct VIALS answer verification using Gym model-server judging."""

import re
from pathlib import Path
from typing import ClassVar

from pydantic import ConfigDict, Field, PrivateAttr, model_validator

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.server_utils import ServerClient
from nemo_gym.verifier_fixture import VerifierFixture
from resources_servers.vials.grading import VialsJudge
from resources_servers.vials.upstream import extract_answer_block
from resources_servers.vials.verifier_fixture import create_fixture_verifier, invoke_fixture


class VialsResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS
    judge_model_server: ModelServerRef
    multimodal_fallback: bool = True
    max_concurrent_judges: int = Field(default=4, gt=0)


class VialsVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    question: str = Field(min_length=1)
    expected_answer: str = Field(min_length=1)
    answer_lower_bound: float | None = Field(default=None, allow_inf_nan=False)
    answer_upper_bound: float | None = Field(default=None, allow_inf_nan=False)

    @model_validator(mode="after")
    def ordered_bounds(self) -> "VialsVerifyRequest":
        if (
            self.answer_lower_bound is not None
            and self.answer_upper_bound is not None
            and self.answer_lower_bound > self.answer_upper_bound
        ):
            raise ValueError("answer_lower_bound must not exceed answer_upper_bound")
        return self


class VialsVerifyResponse(VialsVerifyRequest, BaseVerifyResponse):
    candidate_answer: str | None = None
    judge_model: str
    judge_reasoning: str
    correct: bool
    judge_unverifiable: bool = False
    judge_multimodal: bool = False
    judge_unparseable: bool = False
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0


class VialsVerifier:
    """Service-independent verification, also exercised by offline fixtures."""

    def __init__(
        self,
        server_client: ServerClient,
        *,
        judge_server_name: str = "vials_judge_model",
        multimodal_fallback: bool = True,
        max_concurrent_judges: int = 4,
    ) -> None:
        self.judge_server_name = judge_server_name
        self.multimodal_fallback = multimodal_fallback
        self._judge = VialsJudge(server_client, server_name=judge_server_name, max_concurrent=max_concurrent_judges)

    async def verify(self, body: VialsVerifyRequest) -> VialsVerifyResponse:
        """Extract a visible answer and grade it, retaining references outside policy input."""
        text = re.sub(r"<(think|thinking)>.*?(?:</\1>|$)", "", body.response.output_text, flags=re.I | re.S)
        candidate = extract_answer_block(text)
        verdict = {"correct": False, "judge_reasoning": "No final answer was extracted."}
        failure = {}
        if candidate is not None:
            item = {
                "Question": body.question,
                "GTFA": body.expected_answer,
                "LeadBoundLower": body.answer_lower_bound,
                "LeadBoundUpper": body.answer_upper_bound,
            }
            inputs = body.responses_create_params.model_dump(mode="json", exclude_none=True)["input"]
            try:
                verdict = await self._judge.grade(
                    candidate,
                    item,
                    inputs if isinstance(inputs, list) else [],
                    multimodal_fallback=self.multimodal_fallback,
                )
            except ValueError as error:
                # Invalid fallback evidence is a task/verifier failure, not a wrong policy answer.
                verdict = {"correct": False, "judge_reasoning": str(error)}
                failure = {"mask_sample": True, "failure_kind": "verifier_error", "failure_reason": str(error)}
        # Transport JudgeError intentionally propagates to Gym's verify-endpoint failsafe,
        # which records the original response in the retryable judge-failures sidecar.
        return VialsVerifyResponse.model_validate(
            body.model_dump()
            | {"judge_model": self.judge_server_name}
            | verdict
            | {"reward": float(verdict["correct"]), "candidate_answer": candidate}
            | failure
        )


class VialsResourcesServer(SimpleResourcesServer):
    config: VialsResourcesServerConfig
    _verifier: VialsVerifier = PrivateAttr()

    def model_post_init(self, context: object) -> None:
        super().model_post_init(context)
        self._verifier = VialsVerifier(
            self.server_client,
            judge_server_name=self.config.judge_model_server.name,
            multimodal_fallback=self.config.multimodal_fallback,
            max_concurrent_judges=self.config.max_concurrent_judges,
        )

    async def verify(self, body: VialsVerifyRequest) -> VialsVerifyResponse:
        return await self._verifier.verify(body)


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=create_fixture_verifier,
    request_model=VialsVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
    invoke=invoke_fixture,
)


if __name__ == "__main__":
    VialsResourcesServer.run_webserver()
