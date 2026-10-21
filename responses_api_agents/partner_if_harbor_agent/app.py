# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a Harbor task with OpenHands and grade the captured evidence through partner_if's `/grade` endpoint.

OpenHands calls the run's Gym model server, so the provider key never enters the sandbox.
"""

import asyncio
import json
import logging
import re
from pathlib import Path

from harbor.models.job.config import JobConfig
from pydantic import field_validator

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.sandbox.agent_tools import sandbox_server_url
from nemo_gym.server_utils import raise_for_status
from resources_servers.partner_if.evidence import EvidenceBundle
from resources_servers.partner_if.grading_service import GradeRequest, GradeResponse, contract_hash
from resources_servers.partner_if.native_evidence import captured_artifact, visible_events
from resources_servers.partner_if.package import RepoPath, TaskBundle, resolve_repo_path
from resources_servers.partner_if.receipt import scoring_fields
from resources_servers.partner_if.schema import VerifierMetadata
from responses_api_agents.harbor_agent_general.app import (
    HarborAgent,
    HarborAgentConfig,
    HarborRunRequest,
    HarborVerifyResponse,
)


logger = logging.getLogger(__name__)


class PartnerIFHarborConfig(HarborAgentConfig):
    model_server: ModelServerRef
    task_bundles_dir: RepoPath
    grading_server: ResourcesServerRef

    @field_validator("harbor_dataset", mode="after")
    @classmethod
    def resolve_dataset_path(cls, dataset):
        path = getattr(dataset, "path", None)
        if path is None or Path(path).is_absolute():
            return dataset
        return dataset.model_copy(update={"path": resolve_repo_path(Path(path))})

    def build_job_config(self, task_name: str, job_name: str) -> JobConfig:
        job = super().build_job_config(task_name, job_name)
        api_base = f"{sandbox_server_url(self.model_server.name, require_reachable=True)}/v1"
        agents = [agent.model_copy(update={"kwargs": {**agent.kwargs, "api_base": api_base}}) for agent in job.agents]
        return job.model_copy(update={"agents": agents})


def validate_openhands_completion(trial_dir: Path) -> None:
    """Harbor's tee pipeline can exit zero after an OpenHands controller error."""
    sessions = list((trial_dir / "agent/sessions").glob("*/events"))
    if len(sessions) != 1:
        raise RuntimeError("Cannot establish OpenHands completion state: expected one event stream")
    state, reason = None, ""
    for path in sorted(sessions[0].glob("*.json"), key=lambda item: int(item.stem)):
        event = json.loads(path.read_text())
        if event.get("observation") == "agent_state_changed":
            state = event.get("extras", {}).get("agent_state")
            reason = str(event.get("extras", {}).get("reason") or "")
    limits = (
        "RuntimeError: Agent reached maximum iteration.",
        "RuntimeError: Agent reached maximum budget for conversation.",
    )
    if state == "error" and reason.startswith(limits):
        raise RuntimeError(f"OpenHands execution budget exhausted: {reason}")
    if state != "finished":
        raise RuntimeError(f"Invalid OpenHands completion state {state!r}: {reason[:1000]}")


class PartnerIFHarborAgent(HarborAgent):
    # The inherited Harbor job worker executes through Ray.
    ray_enabled = True
    config: PartnerIFHarborConfig

    def success_response(self, body: HarborRunRequest, trial_dir: Path) -> HarborVerifyResponse:
        validate_openhands_completion(trial_dir)
        response = super().success_response(body, trial_dir)
        capture = trial_dir / "verifier/artifact.json"
        if capture.stat().st_size > 2_000_000:
            raise ValueError("Artifact capture receipt exceeds size limit")
        artifact = json.loads(capture.read_text())
        return response.model_copy(update={"partner_if_artifact": artifact})

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
        if data.get("_ng_failure_class"):
            return response.model_copy(
                update={
                    "mask_sample": True,
                    "failure_kind": data["_ng_failure_class"],
                    "failure_reason": data.get("error", "Harbor execution failed"),
                }
            )
        try:
            params = data["responses_create_params"]
            if params["input"] != body.responses_create_params.model_dump(mode="json")["input"]:
                raise ValueError("Harbor task instruction differs from the frozen row")
            events = visible_events(data["response"]["output"])
            evidence = EvidenceBundle(
                context=params["input"],
                instructions=params.get("instructions"),
                tools=params.get("tools") or [],
                documents=task.documents,
                trajectory=events,
                answer="\n".join(event.text for event in events if event.kind == "assistant"),
                artifacts={"submission": captured_artifact(data["partner_if_artifact"])},
            )
            request = GradeRequest(contract=task.contract, evidence=evidence)
            remote = await self.server_client.post(
                server_name=self.config.grading_server.name, url_path="/grade", json=request.model_dump(mode="json")
            )
            await raise_for_status(remote)
            receipt = GradeResponse.model_validate(await remote.json())
            return response.model_copy(update=scoring_fields(task.contract, receipt))
        except Exception as error:
            # Boundary: capture/contract/judge failures are never an ordinary actor zero.
            logger.exception("Shared grading failed for native task %s", body.task_name)
            return response.model_copy(
                update={
                    "reward": 0.0,
                    "mask_sample": True,
                    "failure_kind": "partner_if:unscorable",
                    "failure_reason": str(error),
                    "_ng_failure_class": "partner_if_unscorable",
                }
            )


if __name__ == "__main__":
    PartnerIFHarborAgent.run_webserver()
