# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One activation runs bounded frozen prefixes; task Resources owns every grading decision."""

import asyncio
import copy
import json
import re
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter, time
from typing import ClassVar, Literal

from aiohttp import ClientTimeout
from fastapi import Body, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from nemo_gym.agents import (
    AgentCloseSessionResponse,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.client import request as http_request
from nemo_gym.config import ModelServerRef, ResourcesServerRef
from nemo_gym.core import (
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseUsage,
)
from nemo_gym.models import ResponsesConverter
from nemo_gym.server_runtime import BaseRunRequest, BaseVerifyResponse, DirectHTTPToolAccess
from nemo_gym.trajectory import (
    AgentInvocation,
    AgentObservationBundle,
    ModelCallRef,
    ObservationGap,
    TrajectoryRecord,
    TrajectoryToolCall,
    TrajectoryTurn,
    VerifierInvocation,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.partner_if.replay_protocol import (
    PrefixReviewRequest,
    PrefixReviewResponse,
    canonical_replay_response,
    replay_request_sha256,
)
from responses_api_agents.ngc_recorded_context.app import response_from_output
from responses_api_agents.ngc_recorded_context.replay import ReplayConfig, capture, execute, load_runtime, write_json
from responses_api_agents.ngc_recorded_context.saved_prefix import (
    SavedPolicyPrefix,
    prefix_digest,
    verification_digest,
)
from responses_api_agents.ngc_recorded_context.setup_ripgrep import ensure_ripgrep


class NativeReplayAgentConfig(BaseResponsesAPIAgentConfig):
    num_workers: Literal[1] = 1
    model_server: ModelServerRef
    resources_server: ResourcesServerRef
    model: str
    replay: ReplayConfig


class ReplayTaskContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_data: dict[str, JsonValue]


class ReplayAgentContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    partner_if_replay: ReplayTaskContext


class ReplayActivation(NeMoGymResponseCreateParamsNonStreaming):
    parent_turn_limit: int = Field(default=1, ge=1, le=32)
    saved_prefix: Path | None = None
    saved_prefix_sha256: str | None = None


@dataclass
class ReplaySession(AgentSessionState):
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    request_bytes: str | None = None
    reply: NeMoGymResponse | None = None
    records: list[AgentInvocation | TrajectoryToolCall] = field(default_factory=list)
    gaps: list[ObservationGap] = field(default_factory=list)
    created_at: float = field(default_factory=time)
    started_monotonic: float = field(default_factory=perf_counter)
    active_task: asyncio.Task[NeMoGymResponse] | None = None
    resource_access: DirectHTTPToolAccess | None = None
    resource_cookies: dict[str, str] = field(default_factory=dict)
    task_data: dict[str, JsonValue] = field(default_factory=dict)
    trajectory: TrajectoryRecord | None = None
    prefix_trajectories: list[TrajectoryRecord] = field(default_factory=list)
    verifier_invocations: list[VerifierInvocation] = field(default_factory=list)
    prefix_history: list[dict[str, JsonValue]] = field(default_factory=list)
    outer_invocation: AgentInvocation | None = None
    outer_agent_session_id: str | None = None


def summed_usage(values: list[dict]) -> NeMoGymResponseUsage | None:
    if not values or any(
        not all(isinstance(v.get(k), int) for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
        for v in values
    ):
        return None

    def detail(group: str, key: str) -> int | None:
        counts = [(v.get(group) or {}).get(key) for v in values]
        return sum(counts) if all(isinstance(value, int) for value in counts) else None

    return NeMoGymResponseUsage.model_validate(
        {
            "input_tokens": sum(v["prompt_tokens"] for v in values),
            "output_tokens": sum(v["completion_tokens"] for v in values),
            "total_tokens": sum(v["total_tokens"] for v in values),
            "input_tokens_details": {"cached_tokens": detail("prompt_tokens_details", "cached_tokens")},
            "output_tokens_details": {"reasoning_tokens": detail("completion_tokens_details", "reasoning_tokens")},
        }
    )


def decode_gym_reasoning(raw: dict) -> dict:
    """Undo Gym's leading think-tag transport wrapper without editing the raw capture."""
    decoded = copy.deepcopy(raw)
    for choice in decoded.get("choices") or []:
        message = choice.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            continue
        parts, offset = [], 0
        while match := ResponsesConverter.THINK_TAG_PATTERN.match(content, offset):
            parts.append(match.group(1))
            offset = match.end()
        if parts:
            if message.get("reasoning_content"):
                raise ValueError("Ambiguous duplicated reasoning transport")
            message["reasoning_content"] = "".join(parts)
            message["content"] = content[offset:] or None
    return decoded


class NativeReplayAgent(SimpleResponsesAPIAgent):
    config: NativeReplayAgentConfig
    agent_context_model: ClassVar[type[BaseModel]] = ReplayAgentContext

    def model_post_init(self, context: object) -> None:
        ensure_ripgrep()
        super().model_post_init(context)

    async def run(self, request: Request, body: BaseRunRequest = Body()) -> BaseVerifyResponse:
        raise HTTPException(409, "Run this replay through single_agent_turn and agent sessions")

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> ReplaySession:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", body.agent_session_id):
            raise AgentSeedRejectedError("Replay session IDs must be safe single path components")
        if body.continuation is not None:
            raise AgentSeedRejectedError(
                "Cross-activation continuation is unsupported; one activation owns the replay prefixes"
            )
        if body.sandbox_access is not None:
            raise AgentSeedRejectedError("The data-only replay does not consume a task sandbox")
        accesses = self.effective_tool_accesses(body)
        expected = self.config.resources_server.name + ".direct_http"
        grants = [a for a in accesses if isinstance(a, DirectHTTPToolAccess) and a.name == expected]
        unsupported = [a.name for a in accesses if a.required and a not in grants]
        if unsupported:
            raise AgentSeedRejectedError("Replay required external grants are unsupported: " + ", ".join(unsupported))
        if len(grants) != 1 or not grants[0].cookies:
            raise AgentSeedRejectedError(
                "Replay requires its configured Resources direct HTTP grant and session cookie"
            )
        context = self.parse_agent_context(body)
        assert isinstance(context, ReplayAgentContext)
        # Gym may assign a positional task ID to a flat registry row. Keep that
        # transport identity; the pinned actor input and the environment's rubric
        # contract validate which benchmark task this session actually executes.
        load_runtime(self.config.replay)
        return ReplaySession(
            request=body,
            resource_access=grants[0].model_copy(deep=True),
            resource_cookies=dict(grants[0].cookies),
            task_data=copy.deepcopy(context.partner_if_replay.task_data),
        )

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        if not isinstance(state, ReplaySession):
            raise TypeError("Expected replay session")
        if state.active_task is not None and not state.active_task.done():
            state.active_task.cancel()
            with suppress(asyncio.CancelledError):
                await state.active_task
        async with state.lock:
            return AgentCloseSessionResponse(
                agent_session_id=state.request.agent_session_id,
                cleanup_confirmed=True,
                resources_cookies=dict(state.resource_cookies),
                trajectory=self._episode_trajectory(state) if state.outer_invocation is not None else state.trajectory,
                agent_observations=AgentObservationBundle(
                    source="ngc_recorded_context", records=state.records, gaps=state.gaps
                ),
            )

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        values = body.model_dump(mode="json", exclude_none=True, exclude_unset=True)
        controls = {"parent_turn_limit", "saved_prefix", "saved_prefix_sha256"}
        metadata_controls = {
            "ngc_replay_parent_turn_limit",
            "ngc_replay_saved_prefix",
            "ngc_replay_saved_prefix_sha256",
        }
        if controls & values.keys() or metadata_controls & (body.metadata or {}).keys():
            raise HTTPException(422, "Replay continuation controls are internal to the activation")
        session_id = self._agent_session_id_from_request(request)
        if session_id is None:
            raise HTTPException(409, "Seed an agent session before activation")
        state = self._require_agent_session(session_id)
        if not isinstance(state, ReplaySession):
            raise TypeError("Expected replay session")
        capture_key = state.request.episode_id.capture_key
        if request.path_params.get("rollout_id") != capture_key:
            raise HTTPException(409, "Activation rollout does not match its seeded session")
        if request.headers.get("x-ng-agent-activation") != "1":
            raise HTTPException(422, "Each replay step is one activation, indexed 1")
        request_bytes = body.model_dump_json(exclude_none=True)
        async with state.lock:
            if state.request_bytes is not None:
                if state.request_bytes != request_bytes:
                    raise HTTPException(409, "This activation was already bound to different input")
                if state.reply is not None:
                    return state.reply.model_copy(deep=True)
                raise HTTPException(409, "The previous activation did not return; start a new episode attempt")
            state.request_bytes = request_bytes
            state.active_task = asyncio.current_task()
            try:
                state.reply = await self._execute_episode(request, state, body)
                return state.reply.model_copy(deep=True)
            finally:
                state.active_task = None

    def _episode_trajectory(self, state: ReplaySession) -> TrajectoryRecord:
        invocations = [state.outer_invocation] if state.outer_invocation is not None else []
        invocations.extend(item for part in state.prefix_trajectories for item in part.invocations)
        unique: dict[str, VerifierInvocation] = {}
        for verifier in state.verifier_invocations:
            previous = unique.get(verifier.invocation_id)
            if previous is not None and previous != verifier:
                raise ValueError("Replay review returned conflicting verifier identities")
            unique[verifier.invocation_id] = verifier
        return TrajectoryRecord(
            task_id=state.request.task_id.task_id,
            rollout_id=state.request.episode_id.capture_key,
            invocations=invocations,
            verifier_invocations=list(unique.values()),
            turns=[item for part in state.prefix_trajectories for item in part.turns],
            tool_calls=[item for part in state.prefix_trajectories for item in part.tool_calls],
            gaps=list(state.gaps),
        )

    def _retain_prefix(self, state: ReplaySession, prefix: ReplaySession) -> None:
        outer = state.request.agent_session_id
        scoped: dict[str, AgentInvocation] = {}
        for record in prefix.records:
            if isinstance(record, AgentInvocation):
                scoped[record.invocation_id] = record.model_copy(
                    deep=True,
                    update={
                        "harness_session_id": outer,
                        "parent_invocation_id": record.parent_invocation_id or outer,
                    },
                )
        state.records.extend(
            scoped.get(record.invocation_id, record) if isinstance(record, AgentInvocation) else record
            for record in prefix.records
        )
        state.gaps.extend(prefix.gaps)
        if prefix.trajectory is not None:
            state.prefix_trajectories.append(
                prefix.trajectory.model_copy(update={"invocations": list(scoped.values())})
            )

    async def _review_prefix(self, state: ReplaySession, body: PrefixReviewRequest) -> PrefixReviewResponse:
        assert state.resource_access is not None
        result = await http_request(
            "POST",
            str(state.resource_access.base_url).rstrip("/") + "/replay/prefix-review",
            json=body.model_dump(mode="json", exclude_unset=True),
            cookies=state.resource_cookies,
            headers={
                **{
                    key: value for key, value in state.resource_access.headers.items() if key.lower() != "x-session-id"
                },
                "x-session-id": state.request.agent_session_id,
            },
            timeout=ClientTimeout(total=self.config.replay.timeout_s),
            _max_num_tries=1,
            _max_connection_retries=0,
            _resend_after_disconnect=False,
            _retry_other_errors=False,
        )
        try:
            if result.status != 200:
                raise RuntimeError(f"Replay prefix review failed with HTTP {result.status}")
            review = PrefixReviewResponse.model_validate(await result.json(content_type=None))
            state.resource_cookies.update({key: value.value for key, value in result.cookies.items()})
        finally:
            result.release()
        expected_hash = replay_request_sha256(body)
        if review.request_sha256 != expected_hash:
            raise ValueError("Replay review belongs to different input")
        expected = canonical_replay_response(
            body.response,
            review.receipt_id,
            expected_hash,
            masked=review.mask_sample and not review.evaluation_completed,
        )
        if review.response != expected:
            raise ValueError("Replay review changed the frozen response beyond its canonical receipt")
        return review

    async def _execute_episode(
        self, request: Request, state: ReplaySession, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        params = body.model_dump(mode="json", exclude_none=True, exclude_unset=True)
        root = AgentInvocation(
            invocation_id=state.request.agent_session_id,
            activation_id=1,
            harness_session_id=state.request.agent_session_id,
            in_session=True,
            status="unknown",
        )
        state.outer_invocation = root
        state.records.append(root)
        responses: list[NeMoGymResponse] = []
        previous: NeMoGymResponse | None = None
        try:
            for index in range(1, self.config.replay.max_new_turns + 1):
                activation = dict(params) | {"parent_turn_limit": index}
                if previous is not None:
                    extra = previous.model_extra or {}
                    if not isinstance(extra.get("saved_prefix_sha256"), str):
                        raise ValueError("Replay review requested continuation without a resumable prefix")
                    activation.update(
                        saved_prefix=extra["replay_artifacts_dir"],
                        saved_prefix_sha256=extra["saved_prefix_sha256"],
                    )
                prefix = ReplaySession(
                    request=state.request.model_copy(
                        deep=True, update={"agent_session_id": f"{state.request.agent_session_id}.prefix-{index}"}
                    ),
                    outer_agent_session_id=state.request.agent_session_id,
                )
                try:
                    response = await self._execute_step(request, prefix, ReplayActivation.model_validate(activation))
                finally:
                    self._retain_prefix(state, prefix)
                responses.append(response)
                usage = (
                    NeMoGymResponseUsage.sum_from_list([part.usage for part in responses])
                    if all(part.usage is not None for part in responses)
                    else None
                )
                extra = response.model_extra or {}
                entry = {
                    "prefix_index": index,
                    "replay_artifacts_dir": extra.get("replay_artifacts_dir"),
                    "replay_verification_sha256": extra.get("replay_verification_sha256"),
                    "saved_prefix_sha256": extra.get("saved_prefix_sha256"),
                }
                state.prefix_history.append(entry)
                root.duration_ms = (perf_counter() - state.started_monotonic) * 1000
                response = response.model_copy(
                    update={
                        "usage": usage,
                        "_ng_trajectory": self._episode_trajectory(state).model_dump(mode="json"),
                        "partner_if_replay_history": copy.deepcopy(state.prefix_history),
                    }
                )
                try:
                    review = await self._review_prefix(
                        state,
                        PrefixReviewRequest(
                            episode_id=state.request.episode_id,
                            task_id=state.request.task_id,
                            agent_session_id=state.request.agent_session_id,
                            prefix_index=index,
                            task_data=copy.deepcopy(state.task_data),
                            responses_create_params=NeMoGymResponseCreateParamsNonStreaming.model_validate(params),
                            response=response,
                        ),
                    )
                    state.verifier_invocations.extend(review.ng_verifier_invocations)
                    if review.next_parent_turn_limit is not None and (
                        review.next_parent_turn_limit != index + 1 or index == self.config.replay.max_new_turns
                    ):
                        raise ValueError("Replay review requested an invalid or over-budget prefix")
                except Exception as error:
                    # A missing or invalid review is not permission to replay either actor or judge work.
                    # No terminal receipt is invented: normal final verification masks this retained capture.
                    root.status, root.error_type = "failed", type(error).__name__
                    root.duration_ms = (perf_counter() - state.started_monotonic) * 1000
                    state.gaps.append(
                        ObservationGap(
                            code="replay_prefix_review_failed",
                            invocation_id=root.invocation_id,
                            detail="Prefix review failed: " + type(error).__name__,
                        )
                    )
                    return response.model_copy(
                        update={
                            "status": "incomplete",
                            "replay_review_error": type(error).__name__,
                            "_ng_trajectory": self._episode_trajectory(state).model_dump(mode="json"),
                        }
                    )
                entry.update(receipt_id=review.receipt_id, request_sha256=review.request_sha256)
                if review.next_parent_turn_limit is None:
                    root.status = "completed"
                    root.duration_ms = (perf_counter() - state.started_monotonic) * 1000
                    return review.response
                previous = review.response
            raise RuntimeError("Replay did not receive a terminal Resources assessment")
        except BaseException as error:
            root.status = "failed" if isinstance(error, Exception) else "incomplete"
            root.error_type = type(error).__name__
            root.duration_ms = (perf_counter() - state.started_monotonic) * 1000
            raise

    async def _execute_step(self, request: Request, state: ReplaySession, body: ReplayActivation) -> NeMoGymResponse:
        config = self.config.replay
        params = body.model_dump(
            exclude_none=True, exclude_unset=True, exclude={"parent_turn_limit", "saved_prefix", "saved_prefix_sha256"}
        )
        prefix = None
        if body.saved_prefix is not None:
            source = body.saved_prefix.resolve()
            if not source.is_relative_to(config.artifacts_dir.resolve()):
                raise HTTPException(422, "A live step may resume only captures in its configured artifact directory")
            prefix = SavedPolicyPrefix(source, body.saved_prefix_sha256, self.config.model)
            for key in (
                "coverage_profile",
                "max_tokens",
                "temperature",
                "top_p",
                "timeout_s",
                "max_delegate_turns",
                "temporal_history",
                "capture_stdout",
                "shell_control_flow",
                "gym_reasoning_decode",
                "missing_runtime_reads",
            ):
                if prefix.identity["settings"].get(key) != getattr(config, key):
                    raise HTTPException(422, "Saved step changes its actor execution contract: " + key)
        capture_key = state.request.episode_id.capture_key
        task_id = state.request.task_id.task_id
        root_id = state.request.agent_session_id
        invocations = {
            root_id: AgentInvocation(
                invocation_id=root_id, activation_id=1, harness_session_id=root_id, in_session=True, status="unknown"
            )
        }
        state.records.append(invocations[root_id])
        started = {root_id: state.started_monotonic}
        turns, tools, cookies, usages = [], [], {}, []
        owners, child_spawns = {}, {}
        fresh = 0

        async def model_call(payload, timeout, context):
            nonlocal fresh
            if prefix is not None:
                saved = prefix.take(payload)
                if saved is not None:
                    return saved
            fresh += 1
            usage_index = len(usages)
            usages.append({})  # Dispatch without a complete receipt leaves this call's token usage unknown.
            group = context.parts[0] if len(context.parts) > 1 else None
            owner = root_id if group is None else root_id + ":" + group
            if owner not in invocations:
                invocation = AgentInvocation(
                    invocation_id=owner,
                    parent_invocation_id=root_id,
                    spawned_by_tool_call_id=child_spawns.get(group),
                    activation_id=1,
                    harness_session_id=root_id,
                    in_session=True,
                    status="unknown",
                )
                invocations[owner] = invocation
                state.records.append(invocation)
                started[owner] = perf_counter()
            invocation = invocations[owner]
            observed_at = time()
            endpoint = self.resolve_model_base_url(self.config.model_server.name, capture_key) + "/chat/completions"
            result = await http_request(
                "POST",
                endpoint,
                json=payload,
                headers={"x-session-id": owner},
                cookies=cookies,
                timeout=ClientTimeout(total=timeout),
                _max_num_tries=1,
                _max_connection_retries=0,
                _resend_after_disconnect=False,
                _retry_other_errors=False,
            )
            try:
                status, raw = result.status, await result.json(content_type=None)
                cookies.update({k: v.value for k, v in result.cookies.items()})
            finally:
                result.release()
            if status == 200 and self.config.replay.gym_reasoning_decode:
                raw = decode_gym_reasoning(raw)
            if status == 200 and raw.get("id"):
                ref = ModelCallRef(model_ref=self.config.model_server, response_id=raw["id"])
                invocation.model_calls.append(ref)
                choice = (raw.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                calls = message.get("tool_calls") or []
                for index, call in enumerate(calls, 1):
                    invocation.conversation.append(
                        NeMoGymResponseFunctionToolCall(
                            type="function_call",
                            call_id=call["id"],
                            name=call["function"]["name"],
                            arguments=call["function"]["arguments"],
                            status="completed",
                        )
                    )
                    owners[(group or "parent", call["id"])] = owner
                    if call["function"]["name"] == "Agent" and group is None:
                        parent_turn = int(context.name.split("-")[1])
                        child_spawns[f"delegate-{parent_turn:02d}-{index:02d}"] = call["id"]
                turns.append(
                    TrajectoryTurn(
                        invocation_id=owner,
                        activation_id=1,
                        source_message_id=raw["id"],
                        source_model_ref=self.config.model_server,
                        task_id=task_id,
                        rollout_id=capture_key,
                        turn_no=1 + sum(t.invocation_id == owner for t in turns),
                        timestamp=observed_at,
                        question=copy.deepcopy(payload["messages"]),
                        answer=copy.deepcopy(message),
                        reasoning_content=message.get("reasoning_content"),
                        resolved=True,
                        step_count=0,
                        model_calls=[ref],
                    )
                )
                usages[usage_index] = raw.get("usage") or {}
            else:
                invocation.status = "failed"
                invocation.error_type = f"model_http_{status}"
            return status, raw

        def observed_tool(event):
            owner = owners.get((event["conversation_id"], event["tool_call_id"]))
            if owner is None:
                return  # Reconstructed tools from the saved prefix are not new actions.
            observation = TrajectoryToolCall(
                invocation_id=owner,
                tool_call_id=event["tool_call_id"],
                tool_name=event["function"]["name"],
                started_at=event["started_at"],
                completed_at=event["completed_at"],
                duration_ms=event["duration_ms"],
                timing_source="executor",
                status=event["status"],
                error_type=event.get("error_type"),
                output=event["result"],
            )
            invocations[owner].conversation.append(
                NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=event["tool_call_id"],
                    output=event["result"],
                )
            )
            tools.append(observation)
            state.records.append(observation)
            for turn in reversed(turns):
                if turn.invocation_id == owner:
                    turn.step_count += 1
                    break
            if owner == root_id and event["kind"] in {"delegate_result", "delegate_partial", "fixture_gap"}:
                child = next(
                    (v for v in invocations.values() if v.spawned_by_tool_call_id == event["tool_call_id"]), None
                )
                if child is not None:
                    child.status = "completed" if event["kind"] == "delegate_result" else "incomplete"
                    child.duration_ms = (perf_counter() - started[child.invocation_id]) * 1000
                    if event["kind"] == "delegate_partial":
                        child.stop_reason = "max_iterations"

        try:
            step_config = config.model_copy(
                update={
                    "evaluation_boundary": "first_answer",
                    "max_new_turns": body.parent_turn_limit,
                    "artifacts_dir": config.artifacts_dir / root_id,
                }
            )
            replay_id = (
                (prefix.identity.get("session_metadata") or {}).get("session_id") if prefix is not None else None
            )
            root, output, _, diagnostics = await execute(
                step_config,
                params,
                model=self.config.model,
                secret="",
                request_api=None,
                request_api_context=model_call,
                observe_tool=observed_tool,
                replay_id=replay_id,
            )
            if prefix is not None and prefix.consumed != len(prefix.requests):
                raise ValueError("Step did not consume the complete saved prefix")
            events = json.loads((root / "events.json").read_text())
            drafts = json.loads((root / "drafts.json").read_text())
            final = json.loads((root / "terminal_reply.json").read_text())["content"]
            captured = capture(
                params, output, events, drafts, final, prefix=True, answer_paths=load_runtime(config).answer_paths
            )
            write_json(root / "ordering-verification-input.json", captured.model_dump())
            write_json(
                root / "native-step.json",
                {
                    "new_policy_calls": fresh,
                    "reused_policy_calls": prefix.consumed if prefix else 0,
                    "parent_turn_limit": body.parent_turn_limit,
                    "prefix_index": body.parent_turn_limit,
                    "responses_create_params": params,
                    "episode_id": state.request.episode_id.model_dump(mode="json"),
                    "task_id": state.request.task_id.model_dump(mode="json"),
                    "agent_session_id": state.outer_agent_session_id or state.request.agent_session_id,
                    "invocation_id": root_id,
                    "ordering_contract": load_runtime(config).ordering.model_dump(),
                },
            )
            terminal = diagnostics["termination"]
            root_invocation = invocations[root_id]
            root_invocation.status = (
                "completed"
                if terminal in {"turn_limit", "policy_call_limit", "first_answer_draft", "final_response"}
                else "failed"
            )
            if root_invocation.status == "failed":
                root_invocation.error_type = terminal
            root_invocation.duration_ms = (perf_counter() - state.started_monotonic) * 1000
            for invocation in invocations.values():
                if invocation.status == "unknown":
                    invocation.status = root_invocation.status
                if invocation.duration_ms is None:
                    invocation.duration_ms = (perf_counter() - started[invocation.invocation_id]) * 1000
            trajectory = TrajectoryRecord(
                task_id=task_id,
                rollout_id=capture_key,
                invocations=list(invocations.values()),
                turns=turns,
                tool_calls=tools,
                gaps=state.gaps,
            )
            reply = response_from_output(output, self.config.model)
            reply.usage = summed_usage(usages)
            if reply.usage is None:
                state.gaps.append(
                    ObservationGap(
                        code="usage_unavailable",
                        invocation_id=root_id,
                        detail="One or more newly executed calls omitted complete token counts",
                    )
                )
                trajectory.gaps = list(state.gaps)
            reply = reply.model_copy(
                update={
                    "verification_input": captured.model_dump(),
                    "replay_ordering_contract": load_runtime(config).ordering.model_dump(),
                    "replay_artifacts_dir": str(root),
                    "replay_verification_sha256": verification_digest(root),
                    "replay_diagnostics": diagnostics,
                    "saved_prefix_sha256": prefix_digest(root)
                    if terminal in {"turn_limit", "policy_call_limit", "first_answer_draft", "final_response"}
                    else None,
                    "_ng_trajectory": trajectory.model_dump(mode="json"),
                }
            )
            state.trajectory = trajectory
            write_json(root / "native-trajectory.json", trajectory.model_dump(mode="json"))
            return reply
        except BaseException as error:
            for invocation in invocations.values():
                if invocation.status == "unknown":
                    invocation.status = "failed" if isinstance(error, Exception) else "incomplete"
                    invocation.error_type = type(error).__name__
                    invocation.duration_ms = (perf_counter() - started[invocation.invocation_id]) * 1000
            state.trajectory = TrajectoryRecord(
                task_id=task_id,
                rollout_id=capture_key,
                invocations=list(invocations.values()),
                turns=turns,
                tool_calls=tools,
                gaps=state.gaps,
            )
            raise


if __name__ == "__main__":
    NativeReplayAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = NativeReplayAgent.run_webserver()
