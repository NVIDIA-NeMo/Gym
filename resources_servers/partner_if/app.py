# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared text IF verification, independent of task sources and native task rewards."""

import asyncio
from functools import partial
from pathlib import Path
from typing import ClassVar

from fastapi import HTTPException
from pydantic import Field, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.verifier_fixture import VerifierFixture
from resources_servers.partner_if.checks import register_modules
from resources_servers.partner_if.evidence import EvidenceBundle
from resources_servers.partner_if.grading_contract import Evaluation
from resources_servers.partner_if.grading_service import GradeRequest, GradeResponse, contract_hash, grade
from resources_servers.partner_if.llm import PROTOCOL
from resources_servers.partner_if.metrics import PartnerIFMetricsMixin
from resources_servers.partner_if.native_evidence import visible_events
from resources_servers.partner_if.receipt import scoring_fields
from resources_servers.partner_if.schema import IFResult, VerifierMetadata, validate_input
from resources_servers.partner_if.verifier_fixture import create_fixture_server


class PartnerIFResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS
    judge_model_server: ModelServerRef
    judge_max_output_tokens: int = Field(default=8192, gt=0)
    judge_temperature: float = Field(default=0.0, ge=0, le=2)
    judge_max_concurrency: int = Field(default=8, gt=0)
    allow_provisional_rubrics: bool = False
    # Importable modules whose CHECKERS add task-specific verifier IDs (see checks.register_modules).
    task_check_modules: list[str] = Field(default_factory=list)


class VerifyRequest(BaseVerifyRequest):
    verifier_metadata: VerifierMetadata


class VerifyResponse(VerifyRequest, BaseVerifyResponse):
    if_result: IFResult
    judge_protocol: str = PROTOCOL
    judge_response: NeMoGymResponse | None = None
    judge_request: dict | None = None
    grading_result: Evaluation | None = None


# Simple Agent marks a provider-completed reply that has neither a message nor a call as incomplete,
# with this reason. For one-response grading the reply is complete: its omission is gradable behavior.
AGENT_RELABELED_COMPLETIONS = frozenset({"incomplete_reasoning", "empty_output"})


def is_complete(response: NeMoGymResponse) -> bool:
    if response.incomplete_details or response.error:
        return False
    reason = (response.metadata or {}).get("ng_termination_reason")
    return response.status == "completed" or (
        response.status == "incomplete" and reason in AGENT_RELABELED_COMPLETIONS
    )


def visible_answer(response: NeMoGymResponse) -> list[dict]:
    if not is_complete(response):
        raise ValueError("Actor response is not complete")
    messages = []
    for item in response.output:
        if item.type == "reasoning":
            continue  # API reasoning is not user-visible output.
        if item.type != "message" or item.status != "completed":
            raise ValueError("Tool/nontext or incomplete output requires another evidence adapter")
        if any(part.type not in {"output_text", "refusal"} for part in item.content):
            raise ValueError("Unsupported answer content")
        messages.append(item.model_dump(exclude_none=True))
    if not any(
        (part.get("text") or part.get("refusal") or "").strip() for message in messages for part in message["content"]
    ):
        raise ValueError("No visible answer evidence")
    return messages


def tool_response_evidence(params: dict, response: NeMoGymResponse) -> EvidenceBundle:
    """Capture one new response, including unexecuted calls, for the shared judge.

    A completed omission or malformed argument is candidate behavior, not a capture
    failure. Frozen history stays context; reasoning and executed tool results are
    never substituted for the next response being graded.
    """
    if not is_complete(response):
        raise ValueError("Actor response is not complete")
    for item in response.output:
        if item.type == "reasoning":
            continue
        if item.type not in {"message", "function_call"}:
            raise ValueError("Next-call evidence cannot include executed tool results")
        if item.status not in {None, "completed"}:
            raise ValueError("Actor output item is not complete")
        if item.type == "message" and item.role != "assistant":
            raise ValueError("New response messages must belong to the assistant")
    events = visible_events([item.model_dump(exclude_none=True) for item in response.output])
    return EvidenceBundle(
        context=params["input"],
        instructions=params.get("instructions"),
        tools=params.get("tools") or [],
        answer="\n".join(event.text for event in events if event.kind == "assistant"),
        trajectory=events,
    )


class PartnerIFResourcesServer(PartnerIFMetricsMixin, SimpleResourcesServer):
    config: PartnerIFResourcesServerConfig
    _judge_semaphore: asyncio.Semaphore = PrivateAttr()

    def model_post_init(self, context):
        super().model_post_init(context)
        register_modules(self.config.task_check_modules)
        self._judge_semaphore = asyncio.Semaphore(self.config.judge_max_concurrency)

    def setup_webserver(self):
        app = super().setup_webserver()
        app.post("/grade")(self.grade_evidence)
        return app

    async def grade_evidence(self, body: GradeRequest) -> GradeResponse:
        if not body.contract.audit_signoff and not self.config.allow_provisional_rubrics:
            raise HTTPException(422, "Rubric audit unsigned; enable provisional development scoring explicitly")
        return await grade(
            body,
            server_client=self.server_client,
            server_name=self.config.judge_model_server.name,
            semaphore=self._judge_semaphore,
            max_tokens=self.config.judge_max_output_tokens,
            temperature=self.config.judge_temperature,
        )

    async def verify(self, body: VerifyRequest) -> VerifyResponse:
        contract = body.verifier_metadata.partner_if
        identity = dict(
            task_id=contract.task_id,
            rubric_version=contract.rubric_version,
            rubric_hash=contract_hash(contract),
            total=sum(r.dimension == "if" for r in contract.rubrics),
        )
        try:
            if not contract.audit_signoff and not self.config.allow_provisional_rubrics:
                raise ValueError("Rubric audit unsigned; explicitly enable provisional development scoring")
            params = body.responses_create_params.model_dump(exclude_none=True)
            validate_input(params, evidence_mode=contract.evidence_mode)
            if contract.evidence_mode == "tool_call_response":
                bundle = tool_response_evidence(params, body.response)
            else:
                answer = visible_answer(body.response)
                bundle = EvidenceBundle(
                    context=params["input"],
                    instructions=params.get("instructions"),
                    tools=params.get("tools") or [],
                    answer="\n".join(
                        part.get("text") or part.get("refusal") or ""
                        for message in answer
                        for part in message["content"]
                    ),
                )
        except ValueError as error:
            return VerifyResponse(
                **body.model_dump(),
                reward=0,
                mask_sample=True,
                failure_kind="partner_if:unscorable",
                failure_reason=str(error),
                if_result=IFResult(**identity, status="unscorable", passed=0, items=[], reason=str(error)),
            )

        receipt = await self.grade_evidence(GradeRequest(contract=contract, evidence=bundle))
        return VerifyResponse(**body.model_dump(), **scoring_fields(contract, receipt))


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=partial(create_fixture_server, PartnerIFResourcesServer, PartnerIFResourcesServerConfig),
    request_model=VerifyRequest,
    cases_path=Path(__file__).parent / "tests/verifier_cases.jsonl",
)


if __name__ == "__main__":
    PartnerIFResourcesServer.run_webserver()
