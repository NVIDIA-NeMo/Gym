# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Own the replay's multi-step ordering evaluation and its native agent sessions."""

import json
from pathlib import Path

from pydantic import Field

from environment_servers.single_agent_turn.app import SessionHandles
from environment_servers.single_agent_turn_legacy.app import (
    SingleAgentTurnLegacyEnvironmentServer,
    SingleAgentTurnLegacyEnvironmentServerConfig,
)
from nemo_gym.agents import AGENT_ACTIVATION_HEADER, AgentSeedSessionRequest, AgentSeedSessionResponse
from nemo_gym.client import get_response_json, raise_boundary_failure, rollout_path_prefix
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseUsage
from nemo_gym.server_runtime import (
    CleanupContext,
    SingleAgentTurnRequest,
    SingleAgentTurnResponse,
    SingleAgentTurnResult,
)
from nemo_gym.trajectory import AgentObservationBundle, TrajectoryRecord, VerifierInvocation
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.partner_if.receipt import masked_fields
from resources_servers.partner_if.schema import VerifierMetadata
from responses_api_agents.ngc_recorded_context.prefix import ordering_outcome
from responses_api_agents.ngc_recorded_context.replay import write_json
from responses_api_agents.ngc_recorded_context.runtime_contract import OrderingContract


class OrderingEnvironmentConfig(SingleAgentTurnLegacyEnvironmentServerConfig):
    ordering_criterion_id: str | None = None
    probe_namespace: str | None = None
    max_parent_turns: int = Field(default=6, ge=1, le=32)
    max_policy_calls: int = Field(default=8, ge=1, le=64)


def cookie_values(response) -> dict[str, str]:
    return {key: value.value for key, value in response.cookies.items()}


class OrderingEnvironment(SingleAgentTurnLegacyEnvironmentServer):
    config: OrderingEnvironmentConfig
    checkpoint_mode = "restart_only"

    async def run(self, request: SingleAgentTurnRequest, cleanup: CleanupContext) -> SingleAgentTurnResponse:
        task_input = request.task.task_input
        data = task_input.task_data
        contract = VerifierMetadata.model_validate(data["verifier_metadata"]).partner_if
        if len(contract.rubrics) != 1 or contract.prerequisites:
            raise ValueError("Ordering evaluation requires one rubric and no scored prerequisites")
        if (
            self.config.ordering_criterion_id is not None
            and contract.rubrics[0].id != self.config.ordering_criterion_id
        ):
            raise ValueError("Task rubric does not match the configured ordering criterion")
        resources = SessionHandles.new()
        resource_cleanup = cleanup.register_cleanup(
            "ordering resources session", lambda: self._close_resources(request, resources)
        )
        await self._seed_resources(request, resources)
        invocations, turns, tools, gaps, verifier_invocations, usages = [], [], [], [], [], []
        previous, previous_sha, latest, graded = None, None, None, None
        completed = False
        review_count = 0
        final_diagnostics = {}
        for parent_turn in range(1, self.config.max_parent_turns + 1):
            handles = SessionHandles.new()
            handles.resources_session_id = resources.resources_session_id
            handles.resources_cookies = resources.resources_cookies
            closed = False
            close_reply = None

            async def close_agent():
                nonlocal closed, close_reply
                result = await self._close_agent(request, handles)
                if result.cleanup_confirmed is not True:
                    raise RuntimeError("Replay agent cleanup was not confirmed; verification is forbidden")
                closed = True
                close_reply = result
                return result

            agent_cleanup = cleanup.register_cleanup(f"ordering agent step {parent_turn}", close_agent)
            try:
                seed = await self.server_client.post(
                    server_name=self.config.agent_server.name,
                    url_path="/v1/agent_sessions",
                    json=AgentSeedSessionRequest(
                        agent_session_id=handles.agent_session_id,
                        episode_id=request.episode_id,
                        task_id=request.task.task_id,
                    ).model_dump(mode="json"),
                )
                await raise_boundary_failure(seed)
                seeded = AgentSeedSessionResponse.model_validate(await get_response_json(seed))
                if seeded.agent_session_id != handles.agent_session_id:
                    raise ValueError("Seed returned a different session identity")
                handles.agent_cookies = cookie_values(seed)
                activation = task_input.responses_create_params.model_dump(
                    mode="json", exclude_none=True, exclude_unset=True
                ) | {"parent_turn_limit": parent_turn}
                if previous is not None:
                    activation.update(saved_prefix=previous, saved_prefix_sha256=previous_sha)
                result = await self.server_client.post(
                    server_name=self.config.agent_server.name,
                    url_path=rollout_path_prefix(request.episode_id.capture_key) + "/v1/responses",
                    json=activation,
                    cookies=handles.agent_cookies,
                    headers={AGENT_ACTIVATION_HEADER: "1"},
                )
                await raise_boundary_failure(result)
                latest = NeMoGymResponse.model_validate(await get_response_json(result))
                native = latest.model_extra or {}
                declared = OrderingContract.model_validate(native["replay_ordering_contract"])
                if (
                    contract.task_id != declared.task_id
                    or data.get("task_name") not in declared.task_names
                    or contract.rubric_version != declared.rubric_version
                    or contract.rubrics[0].id != declared.criterion_id
                    or self.config.probe_namespace not in (None, declared.probe_namespace)
                ):
                    raise ValueError("Task and rubric identity differ from the frozen replay contract")
                source_trajectory = TrajectoryRecord.model_validate(native["_ng_trajectory"])
                invocations.extend(source_trajectory.invocations)
                turns.extend(source_trajectory.turns)
                tools.extend(source_trajectory.tool_calls)
                gaps.extend(source_trajectory.gaps)
                usages.append(latest.usage)
                # Close before every verify, including a failing or incomplete prefix.
                await agent_cleanup.close()
                if close_reply.agent_observations is None:
                    raise ValueError("Native agent close returned no observation bundle")
            finally:
                if not closed:
                    await agent_cleanup.close()
            root = Path(native["replay_artifacts_dir"])
            diagnostics = native["replay_diagnostics"]
            previous, previous_sha = str(root), native["saved_prefix_sha256"]
            terminal = diagnostics["termination"]
            final_diagnostics = dict(diagnostics)
            if terminal not in {"turn_limit", "policy_call_limit", "first_answer_draft", "final_response"}:
                graded = task_input.task_data | masked_fields(
                    contract,
                    "partner_if:replay_" + terminal,
                    "See the native replay capture; no completed ordering score",
                )
                break
            capture = native["verification_input"]
            prefix_request = request.model_copy(
                update={
                    "task": request.task.model_copy(
                        update={
                            "task_input": task_input.model_copy(
                                update={"task_data": data | {"verification_input": capture}}
                            )
                        }
                    )
                }
            )
            if not closed:
                raise RuntimeError("Verification attempted before agent close")
            verification = await self._verify(prefix_request, resources, latest, None)
            graded = verification.model_dump(mode="json")
            review_count += 1
            verifier_invocations.extend(
                VerifierInvocation.model_validate(v) for v in (graded.get("ng_verifier_invocations") or [])
            )
            write_json(root / "native-prefix.grading-result.json", graded)
            write_json(
                root / "native-prefix.input.json",
                {"output": latest.model_dump(mode="json")["output"], "capture": capture},
            )
            try:
                outcome = ordering_outcome(
                    graded,
                    json.loads((root / "events.json").read_text()),
                    probe_namespace=declared.probe_namespace,
                    expected_criterion_id=declared.criterion_id,
                )
            except ValueError:
                final_diagnostics["termination"] = "ordering_judge_error"
                break
            if outcome is not None:
                completed = True
                final_diagnostics["termination"] = outcome
                break
            if terminal in {"first_answer_draft", "final_response"}:
                final_diagnostics["termination"] = "ordering_undecided"
                break
            if diagnostics["model_calls"] >= self.config.max_policy_calls:
                final_diagnostics["termination"] = "policy_call_limit"
                break
        if latest is None:
            raise RuntimeError("The ordering environment produced no agent step")
        final_diagnostics.update(
            evaluation_completed=completed,
            evaluation_boundary="ordering_prefix_v1",
            prefix_verifications=review_count,
            native_sessions=True,
        )
        for invocation in invocations:
            invocation.verification_skipped = review_count == 0
        observations = AgentObservationBundle(
            source="ngc_recorded_context.native", records=[*invocations, *tools], gaps=gaps
        )
        trajectory = TrajectoryRecord(
            task_id=request.task.task_id.task_id,
            rollout_id=request.episode_id.capture_key,
            invocations=invocations,
            verifier_invocations=verifier_invocations,
            turns=turns,
            tool_calls=tools,
            gaps=gaps,
        )
        latest = latest.model_copy(
            update={
                "status": "completed" if completed else "incomplete",
                "usage": NeMoGymResponseUsage.sum_from_list(usages)
                if usages and all(value is not None for value in usages)
                else None,
            }
        )
        if not completed:
            graded = data | masked_fields(
                contract,
                "partner_if:replay_" + final_diagnostics["termination"],
                "No decisive ordering event or a technical exclusion; see saved prefix receipts",
            )
        graded.update(
            response=latest.model_dump(mode="json"),
            responses_create_params=task_input.responses_create_params.model_dump(mode="json"),
            verifier_metadata=data["verifier_metadata"],
            evaluation_completed=completed,
            replay_artifacts_dir=previous,
            replay_diagnostics=final_diagnostics,
            ng_agent_observations=observations.model_dump(mode="json"),
            ng_trajectory=trajectory.model_dump(mode="json"),
            ng_verifier_invocations=[v.model_dump(mode="json") for v in verifier_invocations],
        )
        write_json(Path(previous) / "native-episode-result.json", graded)
        await resource_cleanup.close()
        return SingleAgentTurnResponse(
            episode_id=request.episode_id,
            task_id=request.task.task_id,
            result=SingleAgentTurnResult.model_validate(graded),
        )


if __name__ == "__main__":
    OrderingEnvironment.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = OrderingEnvironment.run_webserver()
