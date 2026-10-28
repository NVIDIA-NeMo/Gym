# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multimodal evaluation with configurable prompts, budgets and failure evidence."""

import json
import shutil
from pathlib import Path
from time import time
from typing import Any, Literal
from uuid import uuid4

from aiohttp import ClientResponseError
from fastapi import HTTPException, Request
from pydantic import Field, model_validator

from nemo_gym.agent_outcome import AGENT_OUTCOME_METADATA_KEY, AgentOutcome
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseUsage
from nemo_gym.rollout_observability import AgentObservationBundle, TrajectoryRecord
from responses_api_agents.opencode_agent.artifacts import _parse_opencode_session, parse_opencode_session
from responses_api_agents.opencode_agent.observability import scope_opencode_trajectory
from responses_api_agents.opencode_multimodal_agent.app import (
    OpenCodeMultimodalAgent,
    OpenCodeMultimodalAgentConfig,
    OpenCodeMultimodalRunRequest,
    OpenCodeMultimodalVerifyResponse,
)
from responses_api_agents.opencode_multimodal_agent.eval_budget import EvalTurnBudget
from responses_api_agents.opencode_multimodal_agent.eval_failures import episode_failure_kind, terminal_model_result
from responses_api_agents.opencode_multimodal_agent.eval_sandbox import EvalSandboxConfig
from responses_api_agents.opencode_multimodal_agent.system_prompt import system_prompt_plugin


class OpenCodeMultimodalEvalConfig(OpenCodeMultimodalAgentConfig):
    """Opt-in eval policy; training defaults remain on the native entrypoint."""

    final_answer_reserve: int | None = Field(default=None, gt=0)
    strip_reasoning_for_grading: bool = False
    build_system_prompt: str | None = Field(default=None, min_length=1)
    # Evaluation may omit the turn cap; training keeps its required positive budget.
    max_turns: int | None = Field(default=8, gt=0)
    max_images: int | None = Field(default=33, gt=0)
    # Reserve room for a final answer after the stock soft step threshold.
    # Zero places the advisory threshold at the full trajectory limit.
    final_answer_turn_headroom: int = Field(default=2, ge=0)
    sandbox: EvalSandboxConfig
    preserve_image_format: bool = True
    # Optional image naming convention for checkpoints trained with image_0.png.
    input_image_prefix: Literal["input-", "image_"] = "input-"
    command_exec_timeout: int = Field(default=120, gt=0)
    compaction_auto: bool = False
    use_default_image_resize: bool = False
    turn_budget: EvalTurnBudget | None = None
    capture_message_ids: bool = False

    @model_validator(mode="after")
    def validate_prompt_mode(self) -> "OpenCodeMultimodalEvalConfig":
        if self.build_system_prompt is not None and self.system_prompt is not None:
            raise ValueError("A build system prompt override must omit the user-message prefix")
        if self.turn_budget is not None and self.max_turns != self.turn_budget.hard_limit:
            raise ValueError("max_turns must equal regular_turns + grace_turns for an enforced eval budget")
        return self


class OpenCodeMultimodalEvalAgent(OpenCodeMultimodalAgent):
    """Reuse native execution, artifact parsing and verification in a private filesystem."""

    config: OpenCodeMultimodalEvalConfig

    async def _run_opencode(
        self,
        instruction: str,
        system_prompt: str | None,
        *,
        rollout_id: str | None = None,
        collect_observations: bool = True,
        trajectory: TrajectoryRecord | None = None,
        artifact_path: Path | None = None,
        timeout: float | None = None,
    ) -> tuple[list[Any], dict[str, int], str, AgentObservationBundle]:
        try:
            return await super()._run_opencode(
                instruction,
                system_prompt,
                rollout_id=rollout_id,
                collect_observations=collect_observations,
                trajectory=trajectory,
                artifact_path=artifact_path,
                timeout=timeout,
            )
        except RuntimeError as error:
            if not self.config.capture_failure_artifacts or episode_failure_kind(str(error)) is None:
                raise
            detail: dict[str, Any] = {"failure_reason": str(error)}
            if artifact_path is not None and artifact_path.is_file():
                # Use persisted provider evidence, never HTTP-like text in tool output.
                # A timeout or process crash must not inherit an earlier provider error.
                if str(error).startswith("OpenCode failed (timeout=False, exit=1):"):
                    native_result = terminal_model_result(artifact_path)
                    cleanup = {"return_code": 1, "timed_out": False, "error": None}
                    if self._reported_model_error(native_result, cleanup):
                        outcome = self.classify_outcome(native_result)
                        if outcome.failure_kind != "unknown":
                            detail[AGENT_OUTCOME_METADATA_KEY] = outcome.model_dump_json()
                failed_trajectory = TrajectoryRecord(task_id="unscoped", rollout_id=rollout_id or "unscoped")
                observations = _parse_opencode_session(
                    artifact_path,
                    rollout_id or "multimodal",
                    failed_trajectory,
                    model_ref=self.config.model_server,
                )
                _, usage = parse_opencode_session(artifact_path)
                detail["usage"] = NeMoGymResponseUsage(
                    input_tokens=usage["input_tokens"],
                    output_tokens=usage["output_tokens"],
                    total_tokens=usage["input_tokens"] + usage["output_tokens"],
                    input_tokens_details={"cached_tokens": 0},
                    output_tokens_details={"reasoning_tokens": usage.get("reasoning_tokens", 0)},
                ).model_dump(mode="json")
                detail["ng_agent_observations"] = observations.model_dump(mode="json")
                detail["ng_trajectory"] = failed_trajectory.model_dump(mode="json")
            # A structured error survives the HTTP self-call. FastAPI's generic
            # 500 body otherwise loses both the failure diagnosis and its evidence.
            raise HTTPException(status_code=500, detail=detail) from error

    async def run(self, request: Request, body: OpenCodeMultimodalRunRequest) -> OpenCodeMultimodalVerifyResponse:
        try:
            return await super().run(request, body)
        except ClientResponseError as exc:
            # Only the agent self-call can produce a scored failure here. A failed
            # seed/verify request must still reach the retryable failures sidecar.
            path = str(exc.request_info.real_url).split("?", 1)[0]
            reason = getattr(exc, "response_content", b"").decode(errors="replace")
            evidence = {}
            usage = None
            metadata = None
            if self.config.capture_failure_artifacts and reason.startswith("{"):
                parsed = json.loads(reason).get("detail")
                if isinstance(parsed, dict) and isinstance(parsed.get("failure_reason"), str):
                    reason = parsed["failure_reason"]
                    usage = parsed.get("usage")
                    if AGENT_OUTCOME_METADATA_KEY in parsed:
                        outcome = AgentOutcome.model_validate_json(parsed[AGENT_OUTCOME_METADATA_KEY])
                        metadata = {AGENT_OUTCOME_METADATA_KEY: outcome.model_dump_json()}
                    if "ng_agent_observations" in parsed:
                        evidence["ng_agent_observations"] = parsed["ng_agent_observations"]
                    rollout_id = self.rollout_id_from_run(body)
                    if "ng_trajectory" in parsed and rollout_id is not None:
                        evidence["ng_trajectory"] = scope_opencode_trajectory(
                            TrajectoryRecord.model_validate(parsed["ng_trajectory"]),
                            body,
                            rollout_id,
                        ).model_dump(mode="json")
            kind = episode_failure_kind(reason)
            if exc.status != 500 or not path.endswith("/v1/responses") or kind is None:
                raise
            response = NeMoGymResponse(
                id=f"resp_failed_{uuid4().hex}",
                created_at=int(time()),
                model=self.config.model,
                object="response",
                output=[],
                status="failed",
                error={"code": "server_error", "message": reason},
                usage=usage,
                metadata=metadata,
                tool_choice=body.responses_create_params.tool_choice,
                tools=body.responses_create_params.tools,
                parallel_tool_calls=body.responses_create_params.parallel_tool_calls,
            )
            # A normal persisted zero row makes collection complete and gates this
            # task on resume. Keep the diagnosis, but never invent a final answer.
            return OpenCodeMultimodalVerifyResponse.model_validate(
                body.model_dump()
                | {
                    "response": response,
                    "reward": 0.0,
                    "resolved": False,
                    "mask_sample": False,
                    "failure_kind": kind,
                    "failure_reason": reason,
                    "scorer": "agent-failure",
                    "finished_naturally": False,
                }
                | evidence
            )

    def _build_opencode_config(self, rollout_id: str | None = None) -> dict[str, Any]:
        config = super()._build_opencode_config(rollout_id)
        policy = json.loads(Path(__file__).with_name("eval-policy.json").read_text())
        config.update(policy)
        config["compaction"] = {"auto": self.config.compaction_auto, "prune": False}
        if self.config.use_default_image_resize:
            config.pop("attachment", None)
        build = config["agent"]["build"]
        if self.config.max_turns is None or self.config.turn_budget is not None:
            # With an enforced budget the model gateway owns warnings, tool
            # restrictions and the hard stop. Stock steps is only a soft hint.
            build.pop("steps", None)
            build.pop("maxSteps", None)
        else:
            # Stock OpenCode can make an additional assistant call after its
            # steps threshold. Leave room for the text-only final response;
            # the session parser still enforces the full max_turns budget.
            build["steps"] = max(1, self.config.max_turns - self.config.final_answer_turn_headroom)
            build.pop("maxSteps", None)
        build.pop("permission", None)
        if not self.config.subagents_enabled:
            config["tools"]["task"] = False
            config["permission"]["task"] = "deny"
        if self.config.build_system_prompt is not None:
            build["prompt"] = self.config.build_system_prompt
            config["plugin"] = ["file:///testbed/.eval-system-prompt.mjs"]
        if self.config.capture_message_ids:
            config.setdefault("plugin", []).append("file:///testbed/.eval-correlation.mjs")
        config["provider"]["nemo"]["options"].update({"timeout": False, "chunkTimeout": 600000})
        return config

    def _write_opencode_config(self, work_dir: Path, rollout_id: str | None = None) -> None:
        super()._write_opencode_config(work_dir, rollout_id)
        if self.config.capture_message_ids:
            shutil.copyfile(Path(__file__).with_name("eval-correlation.mjs"), work_dir / ".eval-correlation.mjs")
        if self.config.build_system_prompt is None:
            return
        # Stock agent.prompt alone still appends environment/model identity text.
        (work_dir / ".eval-system-prompt.mjs").write_text(system_prompt_plugin(self.config.build_system_prompt))

    def _input_image_name(self, source: Path, index: int) -> str:
        return f"{self.config.input_image_prefix}{index}{source.suffix}"


if __name__ == "__main__":
    OpenCodeMultimodalEvalAgent.run_webserver()
