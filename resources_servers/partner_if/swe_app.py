# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scale-SWE lifecycle with visible-agent evidence and the shared IF grader.

The native SWE verdict remains a separate diagnostic; it never supplies IF points.
This adapter neither implements tools nor changes the agent harness.
"""

import asyncio
import json

from fastapi import HTTPException, Request
from pydantic import Field, JsonValue, PrivateAttr

from nemo_gym.config_types import ModelServerRef
from resources_servers.partner_if.app import VerifyResponse
from resources_servers.partner_if.checks import register_modules
from resources_servers.partner_if.evidence import EvidenceBundle
from resources_servers.partner_if.grading_service import GradeRequest, grade
from resources_servers.partner_if.metrics import PartnerIFMetricsMixin
from resources_servers.partner_if.native_evidence import visible_events
from resources_servers.partner_if.receipt import scoring_fields
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
    actor_system_prompt: str = Field(min_length=1)
    judge_max_output_tokens: int = Field(default=32768, gt=0)
    judge_temperature: float = Field(default=0.0, ge=0, le=2)
    judge_max_concurrency: int = Field(default=2, gt=0)
    allow_provisional_rubrics: bool = False
    tool_description_additions: dict[str, str] = Field(default_factory=dict)
    task_check_modules: list[str] = Field(default_factory=list)


DESCRIPTION_PLUGIN_PATH = "/tmp/nemo-gym-tool-description-additions.mjs"


def description_plugin(additions: dict[str, str]) -> str:
    """Trusted host configuration changes descriptions, never schemas or execution."""
    return (
        "const additions = "
        + json.dumps(additions, ensure_ascii=True)
        + ";\n"
        + """
export const ToolDescriptionAdditions = async () => ({
  "tool.definition": async (input, output) => {
    if (Object.hasOwn(additions, input.toolID)) {
      output.description = (output.description || "") + "\\n\\n" + additions[input.toolID];
    }
  }
});
"""
    )


class SWEIFSeedResponse(ScaleSWESeedSessionResponse):
    workdir: str


class SWEIFVerifyRequest(ScaleSWEVerifyRequest):
    verifier_metadata: VerifierMetadata
    task_name: str | None = None


class SWEIFVerifyResponse(VerifyResponse):
    task_outcome: dict[str, JsonValue]
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

    async def _create_sandbox(self, body, files=None):
        # Only the actor receives the description-only plugin. The pristine grader
        # sandbox receives the upstream held-out verification assets unchanged.
        if files is None and self.config.tool_description_additions:
            files = {DESCRIPTION_PLUGIN_PATH: description_plugin(self.config.tool_description_additions)}
        return await super()._create_sandbox(body, files=files)

    async def seed_session(self, request: Request, body: ScaleSWESeedSessionRequest) -> SWEIFSeedResponse:
        result = await super().seed_session(request, body)
        # OpenCode reconnects by ID. Its SDK handle must retain the task directory.
        return SWEIFSeedResponse(**result.model_dump(), workdir=body.workdir)

    async def verify(self, request: Request, body: SWEIFVerifyRequest) -> SWEIFVerifyResponse:
        contract = body.verifier_metadata.partner_if
        if contract.evidence_mode != "live_agent":
            raise HTTPException(422, "SWE evidence requires a shared live-agent grading contract")
        if not contract.audit_signoff and not self.config.allow_provisional_rubrics:
            raise HTTPException(422, "Rubric audit unsigned; opt into provisional scoring explicitly")
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
            max_tokens=self.config.judge_max_output_tokens,
            temperature=self.config.judge_temperature,
        )
        return SWEIFVerifyResponse.model_validate(
            body.model_dump()
            | scoring_fields(contract, receipt)
            | {"task_outcome": outcome.model_dump(mode="json", exclude={"response", "responses_create_params"})}
        )


if __name__ == "__main__":
    SWEIFResourcesServer.run_webserver()
