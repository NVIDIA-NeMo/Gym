# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scale-SWE lifecycle with visible-agent evidence and the shared IF grader.

The native SWE verdict remains a separate diagnostic; it never supplies IF points.
This adapter neither implements tools nor changes the agent harness: the agent's own
config sets the actor's system prompt and any tool-description additions, and this
server's config repeats them for the judge.
"""

from typing import ClassVar

from fastapi import HTTPException, Request
from pydantic import Field, JsonValue

from nemo_gym.base_resources_server import ReverifyMode
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.partner_if.app import (
    NoAnswer,
    PartnerIFGradingConfig,
    PartnerIFGradingMixin,
    VerifyResponse,
    check_finished,
)
from resources_servers.partner_if.evidence import EvidenceBundle
from resources_servers.partner_if.grading_contract import NO_ANSWER
from resources_servers.partner_if.metrics import PartnerIFMetricsMixin
from resources_servers.partner_if.native_evidence import visible_events
from resources_servers.partner_if.schema import VerifierMetadata
from resources_servers.scale_swe.app import (
    ScaleSWEResourcesServer,
    ScaleSWEResourcesServerConfig,
    ScaleSWESeedSessionRequest,
    ScaleSWESeedSessionResponse,
    ScaleSWEVerifyRequest,
)


class SWEIFConfig(PartnerIFGradingConfig, ScaleSWEResourcesServerConfig):
    # Verification needs the actor's live sandbox, so reverify cannot run it again.
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.UNSUPPORTED
    # The system prompt the agent's config gives the actor; the judge reads it as the instructions.
    actor_system_prompt: str = Field(min_length=1)
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        default_factory=lambda: NeMoGymResponseCreateParamsNonStreaming(
            input=[], max_output_tokens=32768, temperature=0
        )
    )
    judge_max_concurrency: int = Field(default=2, gt=0)
    # Text the agent's config appends to tool descriptions, by tool ID; shown to the judge with the evidence.
    tool_description_additions: dict[str, str] = Field(default_factory=dict)


class SWEIFSeedResponse(ScaleSWESeedSessionResponse):
    workdir: str


class SWEIFVerifyRequest(ScaleSWEVerifyRequest):
    verifier_metadata: VerifierMetadata
    task_name: str | None = None


class SWEIFVerifyResponse(VerifyResponse):
    # Scale-SWE's own verdict.
    task_outcome: dict[str, JsonValue] | None = None
    task_name: str | None = None


def build_swe_evidence(
    *,
    context: list[dict[str, JsonValue]],
    system_prompt: str,
    output: list[dict[str, JsonValue]],
) -> EvidenceBundle:
    """Keep tool calls/results and public messages, never private reasoning."""
    bundle = EvidenceBundle(context=context, instructions=system_prompt)
    try:
        bundle.trajectory = visible_events(output)
    except ValueError as error:
        bundle.trajectory_error = str(error)
        return bundle
    if not bundle.trajectory:
        raise NoAnswer(NO_ANSWER, "The trajectory has no visible events")
    bundle.answer = "\n".join(event.text for event in bundle.trajectory if event.kind == "assistant")
    return bundle


def ends_in_assistant_message(response: NeMoGymResponse) -> bool:
    """Whether the visible output, which leaves out reasoning, ends in an assistant message."""
    visible = [item for item in response.output if item.type != "reasoning"]
    return bool(visible) and visible[-1].type == "message" and visible[-1].role == "assistant"


class SWEIFResourcesServer(PartnerIFGradingMixin, PartnerIFMetricsMixin, ScaleSWEResourcesServer):
    config: SWEIFConfig

    def model_post_init(self, context: object) -> None:
        super().model_post_init(context)
        if self.config.is_verifying_golden_patch:
            raise ValueError("Golden patches are not agent instruction-following trajectories")

    async def seed_session(self, request: Request, body: ScaleSWESeedSessionRequest) -> SWEIFSeedResponse:
        result = await super().seed_session(request, body)
        # OpenCode reconnects by ID. Its SDK handle must retain the task directory.
        return SWEIFSeedResponse(**result.model_dump(), workdir=body.workdir)

    async def verify(self, request: Request, body: SWEIFVerifyRequest) -> SWEIFVerifyResponse:
        contract = body.verifier_metadata.partner_if
        if contract.evidence_mode != "live_agent":
            raise HTTPException(422, "SWE evidence requires a shared live-agent grading contract")
        # Scale-SWE's verify records the native verdict and releases the actor's sandbox.
        outcome = await super().verify(request, body)
        response = body.response
        if response.status is None and ends_in_assistant_message(response):
            # OpenCode sets a status only on a run the output-token cap cut off, and sends no other sign that a run
            # finished, so a run counts as finished when it ends in an assistant message. Any other run, such as one
            # whose transcript export failed or one interrupted mid-step, keeps no status, and check_finished masks it.
            response = response.model_copy(update={"status": "completed"})

        def evidence() -> EvidenceBundle:
            check_finished(response)
            bundle = build_swe_evidence(
                context=body.responses_create_params.model_dump(mode="json", exclude_none=True)["input"],
                system_prompt=self.config.actor_system_prompt,
                output=[item.model_dump(mode="json", exclude_none=True) for item in response.output],
            )
            bundle.tools = [
                {"name": name, "description_addition": text}
                for name, text in self.config.tool_description_additions.items()
            ]
            return bundle

        return SWEIFVerifyResponse.model_validate(
            body.model_dump()
            | await self.grade_response(contract, evidence)
            | {"task_outcome": outcome.model_dump(mode="json", exclude={"response", "responses_create_params"})}
        )


if __name__ == "__main__":
    SWEIFResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = SWEIFResourcesServer.run_webserver()
