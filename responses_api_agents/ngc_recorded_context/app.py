# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Historical saved-capture grading; live evaluation uses native_agent.py."""

import asyncio
import json
import time
from pathlib import Path
from uuid import uuid4

from fastapi import Body, HTTPException, Request
from pydantic import ConfigDict, PrivateAttr, SecretStr

from nemo_gym.agents import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.client import get_response_json, raise_for_status
from nemo_gym.config import ResourcesServerRef
from nemo_gym.core import NeMoGymResponse
from nemo_gym.server_runtime import BaseRunRequest, BaseVerifyResponse
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.partner_if.native_evidence import LiveAgentCapture
from resources_servers.partner_if.receipt import masked_fields
from resources_servers.partner_if.schema import VerifierMetadata
from responses_api_agents.ngc_recorded_context.replay import ReplayConfig, execute, load_runtime, write_json
from responses_api_agents.ngc_recorded_context.saved_prefix import SavedPolicyPrefix


class ReplayAgentConfig(BaseResponsesAPIAgentConfig):
    model: str
    inference_api_key: SecretStr
    resources_server: ResourcesServerRef
    replay: ReplayConfig
    policy_base_url: str = "https://inference-api.nvidia.com/v1"
    resume_prefix: Path | None = None
    resume_prefix_sha256: str | None = None
    regrade_saved_prefix: Path | None = None
    regrade_saved_prefix_sha256: str | None = None


class ReplayRunRequest(BaseRunRequest):
    task_name: str
    verifier_metadata: dict


class ReplayRunResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


def response_from_output(output: list[dict], model: str, *, completed: bool = True) -> NeMoGymResponse:
    """Represent captured replay actions without inventing model-call records."""
    return NeMoGymResponse.model_validate(
        {
            "id": "resp_" + uuid4().hex,
            "created_at": time.time(),
            "object": "response",
            "model": model,
            "output": output,
            "status": "completed" if completed else "incomplete",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        }
    )


def resume_settings_changes(previous: dict[str, object], replay: ReplayConfig) -> dict[str, dict[str, object]]:
    """Permit declared ceiling increases while preserving actor-visible settings."""
    for key in ("coverage_profile", "max_tokens", "temperature", "top_p", "timeout_s"):
        if previous[key] != getattr(replay, key):
            raise ValueError(f"Resume changes actor execution setting: {key}")
    if previous.get("evaluation_boundary", "first_answer") != replay.evaluation_boundary:
        raise ValueError("Resume changes evaluation boundary; use a separately declared saved regrade")
    changes = {}
    for key in ("max_new_turns", "max_policy_calls", "max_delegate_turns"):
        before, after = previous[key], getattr(replay, key)
        if not isinstance(before, int) or after < before:
            raise ValueError(f"Resume decreases or corrupts execution ceiling: {key}")
        if before != after:
            changes[key] = {"from": before, "to": after}
    for key in ("temporal_history", "capture_stdout", "shell_control_flow", "missing_runtime_reads"):
        before_coverage = previous.get(key, False)
        after_coverage = getattr(replay, key)
        if before_coverage and not after_coverage:
            raise ValueError(f"Resume removes fixture coverage: {key}")
        if before_coverage != after_coverage:
            changes[key] = {"from": before_coverage, "to": after_coverage}
    return changes


class ReplayAgent(SimpleResponsesAPIAgent):
    config: ReplayAgentConfig
    _semaphore: asyncio.Semaphore = PrivateAttr(default_factory=lambda: asyncio.Semaphore(1))

    def model_post_init(self, context: object) -> None:
        from responses_api_agents.ngc_recorded_context.setup_ripgrep import ensure_ripgrep

        ensure_ripgrep()
        super().model_post_init(context)
        if self.config.num_workers != 1:
            raise ValueError("Recorded replay requires one worker for bounded concurrency")

    async def responses(self, request: Request, body=Body()):
        raise HTTPException(405, "Use /run for the bounded recorded-context replay")

    async def run(self, request: Request, body: ReplayRunRequest = Body()) -> ReplayRunResponse:
        contract = VerifierMetadata.model_validate(body.verifier_metadata).partner_if
        runtime = load_runtime(self.config.replay)
        if contract.task_id != runtime.ordering.task_id or body.task_name not in runtime.ordering.task_names:
            raise HTTPException(422, "Task identity differs from the declared replay contract")
        params = body.responses_create_params.model_dump(exclude_none=True, exclude_unset=True)
        secret = self.config.inference_api_key.get_secret_value()

        if self.config.replay.evaluation_boundary == "ordering_prefix_v1":
            if (
                contract.rubric_version != runtime.ordering.rubric_version
                or len(contract.rubrics) != 1
                or contract.rubrics[0].id != runtime.ordering.criterion_id
                or contract.prerequisites
            ):
                raise HTTPException(422, "Ordering boundary requires the declared prefix rubric")
        if self.config.resume_prefix is not None and self.config.regrade_saved_prefix is not None:
            raise HTTPException(422, "Choose resumption or a separate saved-prefix regrade")
        prefix = None
        fresh_calls = 0

        async def request_api(payload, timeout_s):
            if prefix is not None:
                saved = prefix.take(payload)
                if saved is not None:
                    return saved
            if self.config.regrade_saved_prefix is not None:
                raise ValueError(
                    "Saved-prefix regrade exhausted its recorded responses; live actor calls are forbidden"
                )
            raise ValueError("Live actor calls are disabled in the audit entrypoint; use native_agent.py")

        async def verify_capture(response: NeMoGymResponse, captured: LiveAgentCapture) -> dict:
            verification = body.model_dump(exclude={"capture_rollout_id"}) | {
                "response": response.model_dump(),
                "verification_input": captured.model_dump(),
            }
            result = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=verification,
                cookies=request.cookies,
            )
            await raise_for_status(result)
            return await get_response_json(result)

        async def review_prefix(output: list[dict], captured: LiveAgentCapture, turn: int) -> dict:
            response = response_from_output(output, self.config.model)
            response.metadata = {
                "evaluation_boundary": "ordering_prefix_v1",
                "scope": "captured ordering decision; no final answer is implied",
                "new_response_turn": str(turn),
            }
            return await verify_capture(response, captured)

        async with self._semaphore:
            try:
                if self.config.regrade_saved_prefix is not None:
                    prefix = SavedPolicyPrefix(
                        self.config.regrade_saved_prefix, self.config.regrade_saved_prefix_sha256, self.config.model
                    )
                    if self.config.replay.evaluation_boundary != "ordering_prefix_v1":
                        raise ValueError("Saved regrading requires the explicit ordering-prefix boundary")
                    previous = prefix.identity["settings"]
                    for key in (
                        "coverage_profile",
                        "max_tokens",
                        "temperature",
                        "top_p",
                        "timeout_s",
                        "max_delegate_turns",
                    ):
                        if previous[key] != getattr(self.config.replay, key):
                            raise ValueError(f"Saved regrade changes actor execution setting: {key}")
                    for key in ("temporal_history", "capture_stdout", "shell_control_flow", "missing_runtime_reads"):
                        if previous.get(key, False) != getattr(self.config.replay, key):
                            raise ValueError(f"Saved regrade changes fixture coverage: {key}")
                elif self.config.resume_prefix is not None:
                    prefix = SavedPolicyPrefix(
                        self.config.resume_prefix, self.config.resume_prefix_sha256, self.config.model
                    )
                    resume_settings_changes(prefix.identity["settings"], self.config.replay)
                root, output, captured, diagnostics = await execute(
                    self.config.replay,
                    params,
                    model=self.config.model,
                    secret=secret,
                    request_api=request_api,
                    review_prefix=review_prefix
                    if self.config.replay.evaluation_boundary == "ordering_prefix_v1"
                    else None,
                    replay_id=(prefix.identity.get("session_metadata") or {}).get("session_id")
                    if prefix is not None
                    else None,
                )
                if self.config.regrade_saved_prefix is not None:
                    write_json(
                        root / "saved-regrade.json",
                        {
                            "source_capture": str(prefix.directory),
                            "source_sha256": self.config.regrade_saved_prefix_sha256,
                            "available_saved_policy_calls": len(prefix.requests),
                            "consumed_saved_policy_calls": prefix.consumed,
                            "new_actor_requests": fresh_calls,
                            "old_evaluation_boundary": prefix.identity["settings"].get(
                                "evaluation_boundary", "first_answer"
                            ),
                            "new_evaluation_boundary": self.config.replay.evaluation_boundary,
                            "rubric_version": contract.rubric_version,
                            "original_evidence_unchanged": True,
                        },
                        secret,
                    )
                    if fresh_calls != 0:
                        raise ValueError("Saved regrade made an unexpected actor request")
                elif prefix is not None:
                    if prefix.consumed != len(prefix.requests):
                        raise ValueError("Resumption did not consume the complete saved prefix")
                    write_json(
                        root / "resumption.json",
                        {
                            "source_capture": str(prefix.directory),
                            "source_sha256": self.config.resume_prefix_sha256,
                            "reused_policy_calls": prefix.consumed,
                            "fresh_policy_calls": fresh_calls,
                            "changed_execution_contract": resume_settings_changes(
                                prefix.identity["settings"], self.config.replay
                            ),
                            "old_max_delegate_turns": prefix.identity["settings"]["max_delegate_turns"],
                            "new_max_delegate_turns": self.config.replay.max_delegate_turns,
                        },
                    )
            except Exception as error:
                return ReplayRunResponse.model_validate(
                    body.model_dump()
                    | masked_fields(
                        contract, "partner_if:replay_capture_error", str(error).replace(secret, "[REDACTED]")
                    )
                    | {
                        "response": response_from_output([], self.config.model, completed=False),
                        "evaluation_completed": False,
                    }
                )
            complete = diagnostics["evaluation_completed"]
            response = response_from_output(output, self.config.model, completed=complete)
            response.metadata = {
                "replay_artifacts_dir": str(root),
                "actor_capture": "raw Chat Completions requests/responses",
                "evaluation_boundary": self.config.replay.evaluation_boundary,
            }
            if not complete:
                graded = (
                    body.model_dump()
                    | masked_fields(
                        contract,
                        "partner_if:replay_" + diagnostics["termination"],
                        "See saved replay capture; no completed score",
                    )
                    | {"response": response.model_dump(), "evaluation_completed": False}
                )
            elif self.config.replay.evaluation_boundary == "ordering_prefix_v1":
                graded = json.loads(
                    (root / f"prefix-{diagnostics['prefix_grading_turn']:02d}.grading-result.json").read_text()
                )
                graded.update(response=response.model_dump(), evaluation_completed=True)
            else:
                try:
                    graded = await verify_capture(response, captured)
                except Exception as error:
                    graded = (
                        body.model_dump()
                        | masked_fields(
                            contract, "partner_if:replay_verification_error", str(error).replace(secret, "[REDACTED]")
                        )
                        | {"response": response.model_dump(), "evaluation_completed": False}
                    )
            write_json(root / "grading-result.json", graded, secret)
            return ReplayRunResponse.model_validate(
                graded | {"replay_artifacts_dir": str(root), "replay_diagnostics": diagnostics}
            )


if __name__ == "__main__":
    ReplayAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = ReplayAgent.run_webserver()
