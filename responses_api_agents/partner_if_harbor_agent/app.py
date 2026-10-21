# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a Harbor task with OpenHands and send the captured evidence to partner_if's `/verify`.

OpenHands calls the run's Gym model server, so the provider key never enters the sandbox.
"""

import asyncio
import json
import os
import re
from pathlib import Path

from fastapi import Body
from harbor.models.job.config import JobConfig
from harbor.models.trial.paths import TrialPaths
from pydantic import JsonValue

from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest, ModelServerRef, ResourcesServerRef
from nemo_gym.failure_kinds import AGENT_RUN_ERROR
from nemo_gym.global_config import ROLLOUT_ID_KEY_NAME
from nemo_gym.rollout_collection import NG_FAILURE_CLASS_KEY
from nemo_gym.sandbox.agent_tools import sandbox_server_url
from nemo_gym.server_utils import get_response_json, is_nemo_gym_fastapi_entrypoint, raise_for_status
from resources_servers.partner_if.evidence import MAX_EVIDENCE_CHARS
from resources_servers.partner_if.grading_service import contract_hash
from resources_servers.partner_if.native_evidence import LiveAgentCapture
from resources_servers.partner_if.package import RepoPath, TaskBundle
from resources_servers.partner_if.schema import VerifierMetadata
from responses_api_agents.harbor_agent_general.app import (
    HarborAgent,
    HarborAgentConfig,
    HarborRunRequest,
    HarborVerifyResponse,
)
from responses_api_agents.partner_if_harbor_agent.environment import OPENSANDBOX_API_KEY, ArtifactEnvironment


class PartnerIFHarborConfig(HarborAgentConfig):
    model_server: ModelServerRef
    task_bundles_dir: RepoPath
    resources_server: ResourcesServerRef


# OpenHands stops at max_iterations with this error. Like a step cap, it ends the episode, and the episode is graded.
ITERATION_CAP = "RuntimeError: Agent reached maximum iteration."


def validate_openhands_completion(trial_dir: Path) -> None:
    """Harbor's tee pipeline can exit zero after an OpenHands controller error."""
    sessions = list((TrialPaths(trial_dir).agent_dir / "sessions").glob("*/events"))
    if len(sessions) != 1:
        raise RuntimeError("Cannot establish OpenHands completion state: expected one event stream")
    state, reason = None, ""
    for path in sorted(sessions[0].glob("*.json"), key=lambda item: int(item.stem)):
        event = json.loads(path.read_text())
        if event.get("observation") == "agent_state_changed":
            state = event.get("extras", {}).get("agent_state")
            reason = str(event.get("extras", {}).get("reason") or "")
    if state == "error" and reason.startswith(ITERATION_CAP):
        return
    if state != "finished":
        raise RuntimeError(f"Invalid OpenHands completion state {state!r}: {reason[:1000]}")


# JSON escapes a character to at most 12 bytes (a surrogate pair), so a receipt this size holds any text of
# MAX_EVIDENCE_CHARS characters; partner_if refuses longer text as evidence_too_large.
MAX_RECEIPT_BYTES = 12 * MAX_EVIDENCE_CHARS + 65_536


def artifact_receipt(trial_dir: Path) -> dict[str, JsonValue]:
    """The verifier's capture receipt. A receipt that cannot be read is a capture error, which leaves
    only the items that need the artifact unscorable."""
    capture = TrialPaths(trial_dir).verifier_dir / "artifact.json"
    try:
        if capture.stat().st_size > MAX_RECEIPT_BYTES:
            return {"error": "Artifact capture receipt exceeds size limit"}
        receipt = json.loads(capture.read_text())
    except (OSError, ValueError) as error:
        return {"error": f"Cannot read the artifact capture receipt: {error}"}
    return receipt if isinstance(receipt, dict) else {"error": "Artifact capture receipt is not a JSON object"}


class PartnerIFHarborAgent(HarborAgent):
    # The inherited Harbor job worker executes through Ray.
    ray_enabled = True
    config: PartnerIFHarborConfig

    def model_post_init(self, context) -> None:
        super().model_post_init(context)
        # Fail at startup rather than on every rollout.
        sandbox_server_url(self.config.model_server.name, require_reachable=True)
        environment = self.config.job.environment.import_path
        if environment == f"{ArtifactEnvironment.__module__}:{ArtifactEnvironment.__name__}" and not (
            os.environ.get(OPENSANDBOX_API_KEY) or self.config.harbor_worker_env.get(OPENSANDBOX_API_KEY)
        ):
            raise ValueError(f"{OPENSANDBOX_API_KEY} must be set for the Harbor workers")

    def job_config_for_run(self, body: HarborRunRequest) -> JobConfig:
        job = super().job_config_for_run(body)
        # The rollout prefix keeps OpenHands' model calls correlated with this run.
        run = body.model_dump(by_alias=True) | {ROLLOUT_ID_KEY_NAME: body.capture_rollout_id}
        url = sandbox_server_url(self.config.model_server.name, require_reachable=True)
        base_url = f"{self.base_url_for_run(url, run)}/v1"
        # The job is this run's own copy, with the single agent the config allows.
        [agent] = job.agents
        # Harbor's OpenHands prefers LLM_BASE_URL in its env, then in the worker's, over api_base.
        if agent.env.get("LLM_BASE_URL", base_url) != base_url:
            raise ValueError("LLM_BASE_URL is set by this agent: OpenHands must call the Gym model server")
        agent.kwargs["api_base"] = base_url
        agent.env["LLM_BASE_URL"] = base_url
        return job

    def success_response(self, body: HarborRunRequest, trial_dir: Path) -> HarborVerifyResponse:
        validate_openhands_completion(trial_dir)
        response = super().success_response(body, trial_dir)
        return response.model_copy(update={"partner_if_artifact": artifact_receipt(trial_dir)})

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        """Proxy aggregate_metrics to the resources server, which computes partner_if's metrics."""
        response = await self.server_client.post(
            server_name=self.config.resources_server.name, url_path="/aggregate_metrics", json=body
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))

    async def run(self, body: HarborRunRequest) -> HarborVerifyResponse:
        # Validate host-selected task identity before allocating a sandbox.
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", body.task_name):
            raise ValueError("Unsafe native task name")
        task = TaskBundle.model_validate_json(
            await asyncio.to_thread((self.config.task_bundles_dir / f"{body.task_name}.json").read_text)
        )
        contract = VerifierMetadata.model_validate(body.model_dump().get("verifier_metadata", {})).partner_if
        if contract_hash(contract) != contract_hash(task.contract):
            raise ValueError("Native task bundle does not match the row contract")
        response = await super().run(body)
        data = response.model_dump(mode="json")
        if data.get(NG_FAILURE_CLASS_KEY):
            # Harbor or OpenHands failed: an outage for the failures file, not a score.
            return HarborVerifyResponse.model_validate(
                data
                | {
                    "mask_sample": True,
                    "failure_kind": AGENT_RUN_ERROR,
                    "failure_reason": data.get("error", "Harbor execution failed"),
                }
            )
        capture = LiveAgentCapture(
            input=response.responses_create_params.model_dump(mode="json", exclude_none=True)["input"],
            documents=task.documents,
            artifacts={"submission": data["partner_if_artifact"]},
        )
        # The resources server grades the capture and returns the score fields; a failed judge call comes
        # back as a judge_failed row for the failures file. Dumped by alias, the body keeps the _ng_* index
        # keys, from which ServerClient derives this run's rollout prefix; checkpointing requires it.
        verified = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=body.model_dump(mode="json", by_alias=True)
            | {"response": data["response"], "verification_input": capture.model_dump(mode="json")},
        )
        await raise_for_status(verified)
        return HarborVerifyResponse.model_validate(data | await get_response_json(verified))


if __name__ == "__main__":
    PartnerIFHarborAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = PartnerIFHarborAgent.run_webserver()
