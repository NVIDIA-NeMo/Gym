# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Captured IF generation with explicit native scoring and unreviewed masking."""

from pathlib import Path
from typing import ClassVar

from omegaconf import OmegaConf
from pydantic import ConfigDict, Field

from nemo_gym.client import ServerClient
from nemo_gym.config import AggregateMetrics, AggregateMetricsRequest, BaseServerConfig
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
    observe_programmatic_grading,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verification import VerifierFixture
from resources_servers.captured_instruction_following.metrics import summarize
from resources_servers.captured_instruction_following.scoring import canonical_sha256, score_native
from resources_servers.captured_instruction_following.task_data import CapturedMetadata


class CapturedInstructionFollowingConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS
    expected_items: int | None = Field(default=None, gt=0)


class VerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    benchmark: str = Field(min_length=1)
    orig_trajectory_id: str = Field(min_length=1)
    task_group_id: str = Field(min_length=1)
    verifier_metadata: CapturedMetadata


class VerifyResponse(VerifyRequest, BaseVerifyResponse):
    scoring_status: str
    evaluation_completed: bool
    scoring_reason: str
    native_comparison: dict | None = None


class CapturedInstructionFollowingResourcesServer(SimpleResourcesServer):
    config: CapturedInstructionFollowingConfig
    checkpoint_mode = "stateless"
    checkpoint_verify = "replay"

    async def verify(self, body: VerifyRequest) -> VerifyResponse:
        return observe_programmatic_grading(lambda: self._score(body))

    def _score(self, body: VerifyRequest) -> VerifyResponse:
        metadata = body.verifier_metadata
        fields = body.model_dump()

        def result(**outcome):
            return VerifyResponse.model_validate(fields | outcome)

        # The resource binds the evaluated request and identity. Source-request
        # and private-baseline digests are validated against the source artifact
        # by the offline importer; this server never opens a private file.
        identity = metadata.source_identity.model_dump()
        actual_identity = {key: getattr(body, key) for key in identity}
        request = body.responses_create_params.model_dump(mode="json", exclude_unset=True)
        if actual_identity != identity or canonical_sha256(request) != metadata.evaluation_request_sha256:
            return result(
                reward=0,
                mask_sample=True,
                scoring_status="not_scored",
                evaluation_completed=False,
                scoring_reason="input_or_identity_binding_mismatch",
                failure_kind="captured_instruction_following:input_mismatch",
                failure_reason="Evaluated request or source identity differs from the prepared row",
            )
        if metadata.evaluation_mode == "single_turn_text_replay":
            return result(
                reward=0,
                mask_sample=True,
                scoring_status="not_scored",
                evaluation_completed=False,
                scoring_reason="criteria_require_review",
                failure_kind="captured_instruction_following:criteria_require_review",
                failure_reason="Captured text task has no reviewed scoring criteria; this is a generation-only sample",
            )
        response = body.response
        if response.error is not None:
            return result(
                reward=0,
                mask_sample=True,
                scoring_status="not_scored",
                evaluation_completed=False,
                scoring_reason="provider_error",
                failure_kind="captured_instruction_following:provider_error",
                failure_reason="Provider returned an error rather than a candidate action",
            )
        first_call = next((item for item in response.output if item.type == "function_call"), None)
        if (
            response.incomplete_details is not None
            or response.status != "completed"
            or (first_call is not None and first_call.status not in {None, "completed"})
        ):
            return result(
                reward=0,
                mask_sample=True,
                scoring_status="not_scored",
                evaluation_completed=False,
                scoring_reason="incomplete_candidate_response",
                failure_kind="captured_instruction_following:incomplete_response",
                failure_reason="An incomplete candidate response is not a measured next-call grade",
            )
        comparison = score_native(
            [item.model_dump(mode="json") for item in response.output],
            metadata.expected_function,
            metadata.expected_params,
            metadata.comparison_profile,
        )
        return result(
            reward=comparison["reward"],
            mask_sample=False,
            scoring_status="scored",
            evaluation_completed=True,
            scoring_reason=comparison["reason"],
            native_comparison=comparison,
        )

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        aggregate = await super().aggregate_metrics(body)
        summary = summarize(body.verify_responses, self.config.expected_items)
        aggregate.agent_metrics.update(summary)
        aggregate.key_metrics.update(summary)
        return aggregate


def fixture_server() -> CapturedInstructionFollowingResourcesServer:
    """An in-process real resource instance; verification sends no requests."""
    return CapturedInstructionFollowingResourcesServer(
        config=CapturedInstructionFollowingConfig(
            host="127.0.0.1", port=0, entrypoint="app.py", name="captured_fixture"
        ),
        server_client=ServerClient(
            head_server_config=BaseServerConfig(host="127.0.0.1", port=0), global_config_dict=OmegaConf.create({})
        ),
    )


async def reseed_fixture(server: CapturedInstructionFollowingResourcesServer, body: VerifyRequest) -> None:
    """Exercise the stateless seed acknowledgement before repeated grading."""
    await server.seed_session(BaseSeedSessionRequest())


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=fixture_server,
    request_model=VerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "fixtures" / "verifier_cases.jsonl",
    reseed=reseed_fixture,
)


if __name__ == "__main__":
    CapturedInstructionFollowingResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = CapturedInstructionFollowingResourcesServer.run_webserver()  # noqa: F401
