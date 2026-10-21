# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scale-SWE lifecycle with visible-agent evidence and the shared IF grader.

The native SWE verdict remains a separate diagnostic; it never supplies IF points.
This adapter neither implements tools nor changes the agent harness: the agent's own
config sets the actor's system prompt and any tool-description additions, and this
server's config repeats them for the judge.
"""

import asyncio

from fastapi import HTTPException, Request
from pydantic import Field, JsonValue, PrivateAttr

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.partner_if.app import UNSIGNED, VerifyResponse
from resources_servers.partner_if.checks import register_modules
from resources_servers.partner_if.evidence import EvidenceBundle
from resources_servers.partner_if.grading_contract import UNSIGNED_CONTRACT
from resources_servers.partner_if.grading_service import GradeRequest, grade
from resources_servers.partner_if.metrics import PartnerIFMetricsMixin
from resources_servers.partner_if.native_evidence import visible_events
from resources_servers.partner_if.receipt import masked_fields, scoring_fields
from resources_servers.partner_if.schema import VerifierMetadata
from resources_servers.scale_swe.app import (
    ScaleSWEResourcesServer,
    ScaleSWEResourcesServerConfig,
    ScaleSWESeedSessionRequest,
    ScaleSWESeedSessionResponse,
    ScaleSWEVerifyRequest,
)


class SWEIFConfig(ScaleSWEResourcesServerConfig):
    judge_model_server: ModelServerRef
    # The system prompt the agent's config gives the actor; the judge reads it as the instructions.
    actor_system_prompt: str = Field(min_length=1)
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        default_factory=lambda: NeMoGymResponseCreateParamsNonStreaming(
            input=[], max_output_tokens=32768, temperature=0
        )
    )
    judge_max_concurrency: int = Field(default=2, gt=0)
    allow_provisional_rubrics: bool = False
    # Text the agent's config appends to tool descriptions, by tool ID; shown to the judge with the evidence.
    tool_description_additions: dict[str, str] = Field(default_factory=dict)
    task_check_modules: list[str] = Field(default_factory=list)


class SWEIFSeedResponse(ScaleSWESeedSessionResponse):
    workdir: str


class SWEIFVerifyRequest(ScaleSWEVerifyRequest):
    verifier_metadata: VerifierMetadata
    task_name: str | None = None


class SWEIFVerifyResponse(VerifyResponse):
    # Scale-SWE's own verdict; absent when the rollout is masked before verification.
    task_outcome: dict[str, JsonValue] | None = None
    task_name: str | None = None


def build_swe_evidence(
    *,
    context: list[dict[str, JsonValue]],
    system_prompt: str,
    output: list[dict[str, JsonValue]],
    complete: bool,
) -> EvidenceBundle:
    """Keep tool calls/results and public messages, never private reasoning."""
    bundle = EvidenceBundle(context=context, instructions=system_prompt)
    if not complete:
        bundle.trajectory_error = "SWE actor did not produce a complete trajectory"
        return bundle
    try:
        bundle.trajectory = visible_events(output)
        bundle.answer = "\n".join(event.text for event in bundle.trajectory if event.kind == "assistant")
    except ValueError as error:
        bundle.trajectory_error = str(error)
    return bundle


class SWEIFResourcesServer(PartnerIFMetricsMixin, ScaleSWEResourcesServer):
    config: SWEIFConfig
    _judge_semaphore: asyncio.Semaphore = PrivateAttr()

    def model_post_init(self, context: object) -> None:
        super().model_post_init(context)
        if self.config.num_workers not in (None, 1):
            raise ValueError("Scale-SWE sandbox sessions are process-local and require num_workers=1")
        if self.config.is_verifying_golden_patch:
            raise ValueError("Golden patches are not agent instruction-following trajectories")
        register_modules(self.config.task_check_modules)
        self._judge_semaphore = asyncio.Semaphore(self.config.judge_max_concurrency)

    async def seed_session(self, request: Request, body: ScaleSWESeedSessionRequest) -> SWEIFSeedResponse:
        result = await super().seed_session(request, body)
        # OpenCode reconnects by ID. Its SDK handle must retain the task directory.
        return SWEIFSeedResponse(**result.model_dump(), workdir=body.workdir)

    async def verify(self, request: Request, body: SWEIFVerifyRequest) -> SWEIFVerifyResponse:
        contract = body.verifier_metadata.partner_if
        if contract.evidence_mode != "live_agent":
            raise HTTPException(422, "SWE evidence requires a shared live-agent grading contract")
        if not contract.audit_signoff and not self.config.allow_provisional_rubrics:
            # No verification sandbox starts; the actor's is released as Scale-SWE's verify would release it.
            session_id = request.session[SESSION_ID_KEY]
            self._session_id_to_base_commit.pop(session_id, None)
            self._session_id_to_pristine_untracked.pop(session_id, None)
            await self._stop_sandbox(self._session_id_to_sandbox.pop(session_id, None))
            return SWEIFVerifyResponse.model_validate(
                body.model_dump() | masked_fields(contract, UNSIGNED_CONTRACT, UNSIGNED)
            )
        outcome = await super().verify(request, body)
        response = body.response
        evidence = build_swe_evidence(
            context=body.responses_create_params.model_dump(mode="json", exclude_none=True)["input"],
            system_prompt=self.config.actor_system_prompt,
            output=[item.model_dump(mode="json", exclude_none=True) for item in response.output],
            complete=response.status in (None, "completed") and not response.error and not response.incomplete_details,
        )
        evidence.tools = [
            {"name": name, "description_addition": text}
            for name, text in self.config.tool_description_additions.items()
        ]
        receipt = await grade(
            GradeRequest(contract=contract, evidence=evidence),
            server_client=self.server_client,
            server_name=self.config.judge_model_server.name,
            semaphore=self._judge_semaphore,
            responses_create_params=self.config.judge_responses_create_params,
        )
        return SWEIFVerifyResponse.model_validate(
            body.model_dump()
            | scoring_fields(contract, receipt)
            | {"task_outcome": outcome.model_dump(mode="json", exclude={"response", "responses_create_params"})}
        )


if __name__ == "__main__":
    SWEIFResourcesServer.run_webserver()
