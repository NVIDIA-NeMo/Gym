# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared IF verification of replies and live-agent captures, independent of task sources and native task rewards."""

import asyncio
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import ClassVar

from fastapi import HTTPException
from pydantic import Field, JsonValue, model_validator

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verifier_fixture import VerifierFixture
from resources_servers.partner_if.checks import register_modules
from resources_servers.partner_if.evidence import EvidenceBundle
from resources_servers.partner_if.grading_contract import (
    NO_ANSWER,
    TRUNCATED,
    UNSCORABLE,
    UNSIGNED_CONTRACT,
    Evaluation,
    GradingContract,
)
from resources_servers.partner_if.grading_service import grade, unanswered
from resources_servers.partner_if.llm import MAX_JUDGE_INPUT_CHARS, PROTOCOL
from resources_servers.partner_if.metrics import PartnerIFMetricsMixin
from resources_servers.partner_if.native_evidence import LiveAgentCapture, live_agent_evidence, visible_events
from resources_servers.partner_if.receipt import masked_fields, scoring_fields
from resources_servers.partner_if.schema import IFResult, VerifierMetadata, validate_input
from resources_servers.partner_if.verifier_fixture import create_fixture_server


class PartnerIFGradingConfig(BaseResourcesServerConfig):
    """The shared grader's settings, which every partner_if server has."""

    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        default_factory=lambda: NeMoGymResponseCreateParamsNonStreaming(
            input=[], max_output_tokens=8192, temperature=0
        )
    )
    judge_max_concurrency: int = Field(default=8, gt=0)
    # Characters of the judge's user message (the serialized rubrics and evidence); size it to the judge's
    # context window. A larger input is not sent: the checks that gate reward are judged alone, and any item
    # left out is evidence_too_large.
    judge_max_input_chars: int = Field(default=MAX_JUDGE_INPUT_CHARS, gt=0)
    allow_provisional_rubrics: bool = False
    # Importable modules whose CHECKERS add task-specific verifier IDs (see checks.register_modules).
    task_check_modules: list[str] = Field(default_factory=list)


class PartnerIFResourcesServerConfig(PartnerIFGradingConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS
    # Grade live_agent rows on the verification_input their agent sends. Rows never carry it, so such a
    # server reports reverify mode UNSUPPORTED.
    accept_verification_input: bool = False


UNSIGNED = "Rubric audit unsigned; enable allow_provisional_rubrics for provisional development scoring"


class VerifyRequest(BaseVerifyRequest):
    verifier_metadata: VerifierMetadata
    # Sent only for live_agent rows and graded with the response; rows never carry it.
    verification_input: LiveAgentCapture | None = Field(default=None, exclude=True)


class VerifyResponse(VerifyRequest, BaseVerifyResponse):
    if_result: IFResult
    judge_protocol: str = PROTOCOL
    judge_response: NeMoGymResponse | None = None
    judge_request: dict | None = None
    grading_result: Evaluation | None = None
    # Whether grading ran to completion, as Gym's evidence gate reads it: a masked rollout's did not.
    evaluation_completed: bool | None = None

    @model_validator(mode="after")
    def completion_follows_masking(self) -> "VerifyResponse":
        if self.evaluation_completed is None:
            self.evaluation_completed = not self.mask_sample
        return self


# Simple Agent marks a provider-completed reply that has neither a message nor a call as incomplete,
# with this reason. The reply is complete: it gives no answer.
AGENT_RELABELED_COMPLETIONS = frozenset({"incomplete_reasoning", "empty_output"})


class NoAnswer(Exception):
    """The reply gives nothing to grade, which is the model's own failure rather than a capture gap."""

    def __init__(self, failure_kind: str, reason: str):
        super().__init__(reason)
        self.failure_kind = failure_kind


def output_cap_cutoff(response: NeMoGymResponse) -> bool:
    details = response.incomplete_details
    return response.error is None and details is not None and details.reason == "max_output_tokens"


def is_complete(response: NeMoGymResponse) -> bool:
    if response.incomplete_details or response.error:
        return False
    reason = (response.metadata or {}).get("ng_termination_reason")
    return response.status == "completed" or (
        response.status == "incomplete" and reason in AGENT_RELABELED_COMPLETIONS
    )


def check_finished(response: NeMoGymResponse) -> None:
    """A reply cut off by the output-token cap is the model's failure; any other unfinished reply is unscorable."""
    if output_cap_cutoff(response):
        raise NoAnswer(TRUNCATED, "The reply was cut off by the output-token cap")
    if not is_complete(response):
        raise ValueError("Actor response is not complete")


def visible_answer(response: NeMoGymResponse) -> list[dict]:
    check_finished(response)
    messages, called = [], False
    for item in response.output:
        if item.type == "reasoning":
            continue  # API reasoning is not user-visible output.
        if item.type == "function_call":
            called = True
            continue
        if item.type != "message" or item.status != "completed":
            raise ValueError("Nontext or incomplete output requires another evidence adapter")
        if any(part.type not in {"output_text", "refusal"} for part in item.content):
            raise ValueError("Unsupported answer content")
        messages.append(item.model_dump(exclude_none=True))
    if called:
        raise NoAnswer(NO_ANSWER, "The reply calls a tool where a text answer is expected")
    if not any(
        (part.get("text") or part.get("refusal") or "").strip() for message in messages for part in message["content"]
    ):
        raise NoAnswer(NO_ANSWER, "The reply has no visible answer")
    return messages


def tool_response_evidence(params: dict, response: NeMoGymResponse) -> EvidenceBundle:
    """Capture one new response, including unexecuted calls, for the shared judge.

    A malformed argument is candidate behavior, not a capture failure. Frozen history stays
    context; reasoning and executed tool results are never substituted for the next response
    being graded.
    """
    check_finished(response)
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
    if not any(event.kind == "tool_call" or event.text.strip() for event in events):
        raise NoAnswer(NO_ANSWER, "The reply has neither a visible answer nor a tool call")
    return EvidenceBundle(
        context=params["input"],
        instructions=params.get("instructions"),
        tools=params.get("tools") or [],
        answer="\n".join(event.text for event in events if event.kind == "assistant"),
        trajectory=events,
    )


class PartnerIFGradingMixin:
    """The grading boundary every partner_if server shares: its config, its startup and its scoring policy."""

    config: PartnerIFGradingConfig

    def model_post_init(self, context) -> None:
        super().model_post_init(context)
        register_modules(self.config.task_check_modules)
        self._judge_semaphore = asyncio.Semaphore(self.config.judge_max_concurrency)

    async def grade_response(
        self, contract: GradingContract, build_evidence: Callable[[], EvidenceBundle]
    ) -> dict[str, JsonValue]:
        """The Gym verify fields for one rollout. In order: an unsigned contract is masked; a reply that gives
        nothing to grade fails every item without the judge (`build_evidence` raises NoAnswer, TRUNCATED when
        the output-token cap cut the reply off); evidence that cannot be assessed (ValueError) is masked; any
        other evidence is graded."""
        if not contract.audit_signoff and not self.config.allow_provisional_rubrics:
            return masked_fields(contract, UNSIGNED_CONTRACT, UNSIGNED)
        try:
            evidence = build_evidence()
        except NoAnswer as missing:
            receipt = unanswered(contract, missing.failure_kind, str(missing))
        except ValueError as error:
            return masked_fields(contract, UNSCORABLE, str(error))
        else:
            receipt = await grade(
                contract,
                evidence,
                server_client=self.server_client,
                server_name=self.config.judge_model_server.name,
                semaphore=self._judge_semaphore,
                responses_create_params=self.config.judge_responses_create_params,
                max_input_chars=self.config.judge_max_input_chars,
            )
        return scoring_fields(contract, receipt)


class PartnerIFResourcesServer(PartnerIFGradingMixin, PartnerIFMetricsMixin, SimpleResourcesServer):
    # Verification keeps no session state, so a checkpoint can run it again after a crash instead of waiting for it.
    checkpoint_mode = "stateless"
    checkpoint_verify = "replay"

    config: PartnerIFResourcesServerConfig

    async def get_reverify_mode(self) -> ReverifyMode:
        # A live agent's capture is not in the row, so reverify cannot grade the row again.
        if self.config.accept_verification_input:
            return ReverifyMode.UNSUPPORTED
        return await super().get_reverify_mode()

    async def verify(self, body: VerifyRequest) -> VerifyResponse:
        contract = body.verifier_metadata.partner_if
        if contract.evidence_mode == "live_agent" and not self.config.accept_verification_input:
            raise HTTPException(422, "live_agent rows need a server with accept_verification_input: true")

        def evidence() -> EvidenceBundle:
            params = body.responses_create_params.model_dump(exclude_none=True)
            if contract.evidence_mode == "live_agent":
                output = [item.model_dump(exclude_none=True) for item in body.response.output]
                bundle = live_agent_evidence(params, output, body.verification_input)
                # After the capture checks, so a capture fault is masked even in a reply that was cut off. Harbor
                # reports each run it grades as completed, including an OpenHands run stopped at its iteration cap.
                check_finished(body.response)
                return bundle
            validate_input(params, evidence_mode=contract.evidence_mode)
            if contract.evidence_mode == "tool_call_response":
                return tool_response_evidence(params, body.response)
            answer = visible_answer(body.response)
            return EvidenceBundle(
                context=params["input"],
                instructions=params.get("instructions"),
                tools=params.get("tools") or [],
                answer="\n".join(
                    part.get("text") or part.get("refusal") or "" for message in answer for part in message["content"]
                ),
            )

        return VerifyResponse(**body.model_dump(), **await self.grade_response(contract, evidence))


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=partial(create_fixture_server, PartnerIFResourcesServer, PartnerIFResourcesServerConfig),
    request_model=VerifyRequest,
    cases_path=Path(__file__).parent / "tests/verifier_cases.jsonl",
)


if __name__ == "__main__":
    PartnerIFResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = PartnerIFResourcesServer.run_webserver()
