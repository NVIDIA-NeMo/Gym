# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recorded-context task mechanics and grading behind Gym's stock Gymnasium agent."""

import asyncio
import copy
import hashlib
import json
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter, time
from typing import ClassVar
from uuid import uuid4

from aiohttp import ClientTimeout
from fastapi import FastAPI, HTTPException, Request
from pydantic import Field, PrivateAttr, TypeAdapter, model_validator
from starlette.responses import Response

from nemo_gym.client import SESSION_ID_KEY, get_response_json, raise_for_status
from nemo_gym.config import ModelServerRef
from nemo_gym.core import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseUsage,
    accumulate_response_usage,
)
from nemo_gym.models import ChatCompletionsConverter
from nemo_gym.server_runtime import (
    ResourcesSessionConflictError,
    ReverifyMode,
    observe_llm_judge_grading,
    observe_verifier,
)
from nemo_gym.trajectory import (
    AgentInvocation,
    AgentObservationBundle,
    ModelCallRef,
    ObservationGap,
    ToolCallObservation,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verification import judge_failsafe, record_parsed_judge_verdict
from resources_servers.gymnasium import (
    EnvResetRequest,
    EnvResetResponse,
    EnvStepRequest,
    EnvStepResponse,
    GymnasiumServer,
)
from resources_servers.partner_if.app import PartnerIFGradingConfig, PartnerIFGradingMixin, VerifyRequest
from resources_servers.partner_if.grading_contract import GradingContract
from resources_servers.partner_if.metrics import PartnerIFMetricsMixin
from resources_servers.partner_if.native_evidence import LiveAgentCapture, live_agent_evidence, visible_events
from resources_servers.partner_if.receipt import masked_fields
from resources_servers.partner_if.recorded_context.assets import (
    ReplayConfig,
    capture,
    load_actor_params,
    load_inputs,
    load_runtime,
    probe_identity,
)
from resources_servers.partner_if.recorded_context.delegation import explore
from resources_servers.partner_if.recorded_context.fixtures import Fixtures
from resources_servers.partner_if.recorded_context.observation_hooks import tool_observation
from resources_servers.partner_if.recorded_context.prefix import ordering_outcome
from resources_servers.partner_if.recorded_context.setup_ripgrep import ensure_ripgrep
from resources_servers.partner_if.schema import VerifierMetadata


RESULT_KEY = "partner_if_recorded_context_result"
RESULT_BINDING_KEY = "recorded_context_result_sha256"
_TRANSPORT_FIELDS = frozenset(
    {
        "responses_create_params",
        "response",
        "agent_ref",
        "task_source",
        "task_index",
        "rollout_index",
        "_ng_rollout_id",
        "_ng_attempt_index",
        "_ng_step_request_id",
        "_ng_skip_token_capture",
        "_ng_model_sampling",
        "agent_history_mode",
    }
)


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def recorded_context_result_sha256(result: dict) -> str:
    """Detect accidental mutation of a Resources-owned final report; this is not an authentication signature."""
    return _sha({key: value for key, value in result.items() if key != RESULT_BINDING_KEY})


def recorded_context_task_data(payload: dict) -> dict:
    """Keep task data separate from the stock agent's collector and step transport fields."""
    return {key: copy.deepcopy(value) for key, value in payload.items() if key not in _TRANSPORT_FIELDS}


class RecordedContextResourcesServerConfig(PartnerIFGradingConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.UNSUPPORTED
    replay: ReplayConfig
    policy_model_server: ModelServerRef
    policy_model: str
    max_parent_turns: int = Field(default=6, ge=1, le=6)

    @model_validator(mode="after")
    def bounded_contract(self):
        if self.replay.max_policy_calls > 8 or self.replay.max_delegate_turns > 6:
            raise ValueError("Recorded-context evaluation allows at most eight policy calls and six Explore turns")
        return self


@dataclass
class _Episode:
    params: dict
    task_data: dict
    contract: GradingContract
    case: dict
    bundle: dict
    simulator: Fixtures | None
    messages: list[dict]
    schemas: dict
    reset_sha256: str
    invocation_id: str = field(default_factory=lambda: "recorded-context-" + uuid4().hex)
    tool_observations: list[dict] = field(default_factory=list)
    children: dict[str, str] = field(default_factory=dict)
    output: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    child_receipts: dict[str, object] = field(default_factory=dict)
    model_calls: list[dict] = field(default_factory=list)
    verifier_invocations: list[dict] = field(default_factory=list)
    response_ids: set[str] = field(default_factory=set)
    tool_ids: set[str] = field(default_factory=set)
    parent_turns: int = 0
    policy_calls: int = 0
    child_calls: int = 0
    usage: NeMoGymResponseUsage | None = None
    usage_known: bool = True
    last_response: NeMoGymResponse | None = None
    step_receipts: dict[str, tuple[str, EnvStepResponse]] = field(default_factory=dict)
    in_progress: tuple[str, str] | None = None
    active_task: asyncio.Task | None = None
    closed_at: float | None = None


class RecordedContextResourcesServer(PartnerIFGradingMixin, PartnerIFMetricsMixin, GymnasiumServer):
    """The task owns its simulator and judge; the existing Gymnasium agent owns parent inference."""

    config: RecordedContextResourcesServerConfig
    checkpoint_mode = "restart_only"
    ray_enabled = False
    _episodes: dict[str, _Episode] = PrivateAttr(default_factory=dict)
    _locks: dict[str, asyncio.Lock] = PrivateAttr(default_factory=dict)

    def model_post_init(self, context: object) -> None:
        ensure_ripgrep()
        super().model_post_init(context)

    def register_protocol_routes(self, app: FastAPI) -> None:
        super().register_protocol_routes(app)
        app.post("/close")(self._close_endpoint)

    def _lock(self, session_id: str) -> asyncio.Lock:
        return self._locks.setdefault(session_id, asyncio.Lock())

    def _prune(self) -> None:
        cutoff = time() - self.config.session_close_retry_window_seconds
        for key, episode in list(self._episodes.items()):
            if episode.closed_at is not None and episode.closed_at < cutoff:
                self._episodes.pop(key)
                self._locks.pop(key, None)

    @staticmethod
    def _session_id(request: Request) -> str:
        value = request.session.get(SESSION_ID_KEY)
        if not isinstance(value, str) or not value:
            raise HTTPException(422, "Recorded-context execution requires a Resources session cookie")
        return value

    async def _reset_endpoint(self, body: EnvResetRequest, request: Request) -> EnvResetResponse:
        self._prune()
        session_id = self._session_id(request)
        params = load_actor_params(self.config.replay)
        expected_params = NeMoGymResponseCreateParamsNonStreaming.model_validate(params).model_dump(
            mode="json", exclude_none=True
        )
        if body.responses_create_params.model_dump(mode="json", exclude_none=True) != expected_params:
            raise HTTPException(422, "Actor input differs from the pinned recorded-context parameters")
        task_data = recorded_context_task_data(body.model_extra or {})
        harness = task_data.get("recorded_context_harness")
        expected_harness = {
            "version": "gymnasium-v1",
            "max_parent_turns": self.config.max_parent_turns,
            "max_policy_calls": self.config.replay.max_policy_calls,
            "max_delegate_turns": self.config.replay.max_delegate_turns,
            "temperature": self.config.replay.temperature,
            "top_p": self.config.replay.top_p,
            "max_tokens": self.config.replay.max_tokens,
            "wire_format": "gym-responses-to-chat",
        }
        if harness is not None and harness != expected_harness:
            raise HTTPException(422, "Recorded-context harness metadata differs from the configured evaluation")
        binding = _sha({"params": params, "task_data": task_data})
        async with self._lock(session_id):
            previous = self._episodes.get(session_id)
            if previous is not None:
                if previous.closed_at is not None or previous.reset_sha256 != binding:
                    raise ResourcesSessionConflictError(
                        "This recorded-context session is closed or bound to different task data"
                    )
            else:
                try:
                    contract = VerifierMetadata.model_validate(task_data["verifier_metadata"]).partner_if
                    case, bundle = load_inputs(self.config.replay, params)
                    runtime = load_runtime(self.config.replay)
                    if (
                        contract.evidence_mode != "live_agent"
                        or len(contract.rubrics) != 1
                        or contract.prerequisites
                        or contract.task_id != runtime.ordering.task_id
                        or task_data.get("task_name") not in runtime.ordering.task_names
                        or contract.rubric_version != runtime.ordering.rubric_version
                        or contract.rubrics[0].id != runtime.ordering.criterion_id
                    ):
                        raise ValueError("Task and rubric identity differ from the pinned recorded-context contract")
                    simulator = Fixtures(bundle, session_id=uuid4().hex)
                    messages = copy.deepcopy(case["messages"])
                    simulator.sync_transcript(messages)
                except (OSError, ValueError, TypeError, KeyError) as error:
                    raise HTTPException(422, str(error)) from error
                self._episodes[session_id] = _Episode(
                    params=params,
                    task_data=task_data,
                    contract=contract,
                    case=case,
                    bundle=bundle,
                    simulator=simulator,
                    messages=messages,
                    schemas={tool["function"]["name"]: tool["function"]["parameters"] for tool in case["tools"]},
                    reset_sha256=binding,
                    tool_ids={call["id"] for message in messages for call in message.get("tool_calls", [])},
                )
        return EnvResetResponse(
            observation=None, info={"supports_explicit_close": True, "supports_step_idempotency": True}
        )

    async def _step_endpoint(self, body: EnvStepRequest, request: Request) -> EnvStepResponse:
        session_id = self._session_id(request)
        step_id = (body.model_extra or {}).get("_ng_step_request_id")
        if not isinstance(step_id, str) or not step_id or len(step_id) > 128:
            raise HTTPException(422, "A recorded-context step requires its stable Gymnasium request ID")
        params = body.responses_create_params.model_dump(mode="json", exclude_none=True)
        task_data = recorded_context_task_data(body.model_extra or {})
        binding = _sha({"params": params, "task_data": task_data, "response": body.response.model_dump(mode="json")})
        async with self._lock(session_id):
            episode = self._episodes.get(session_id)
            if episode is None:
                raise ResourcesSessionConflictError("Recorded-context session was not reset or expired")
            expected_params = NeMoGymResponseCreateParamsNonStreaming.model_validate(episode.params).model_dump(
                mode="json", exclude_none=True
            )
            if params != expected_params or task_data != episode.task_data:
                raise ResourcesSessionConflictError("Frozen actor parameters or task data changed during the episode")
            cached = episode.step_receipts.get(step_id)
            if cached is not None:
                if cached[0] != binding:
                    raise ResourcesSessionConflictError(
                        "This step request ID was already bound to a different response"
                    )
                return cached[1].model_copy(deep=True)
            if episode.closed_at is not None:
                raise ResourcesSessionConflictError("Recorded-context session is already closed")
            if episode.in_progress is not None:
                # The previous dispatch may have reached the judge. Never grant it a hidden second attempt.
                reply = self._terminal(
                    episode,
                    self._masked(
                        episode,
                        "review_uncertain",
                        "The preceding step did not record a result; start a new episode attempt",
                    ),
                    "review_uncertain",
                )
                episode.step_receipts[step_id] = (binding, reply)
                self._release(episode)
                self.checkpoint_session_ended(request)
                return reply.model_copy(deep=True)
            episode.in_progress = (step_id, binding)
            episode.active_task = asyncio.current_task()
            try:
                reply = await self._perform_step(episode, body.response)
                episode.step_receipts[step_id] = (binding, reply)
                episode.in_progress = None
                if reply.terminated or reply.truncated:
                    self._release(episode)
                    self.checkpoint_session_ended(request)
                return reply.model_copy(deep=True)
            finally:
                episode.active_task = None

    def _masked(self, episode: _Episode, kind: str, reason: str) -> dict:
        return masked_fields(episode.contract, "partner_if:recorded_context_" + kind, reason) | {
            "evaluation_completed": False,
            "grading_result": None,
        }

    @staticmethod
    def _add_usage(episode: _Episode, usage: NeMoGymResponseUsage | None) -> None:
        if usage is None:
            episode.usage_known = False
        else:
            episode.usage = accumulate_response_usage(episode.usage, usage)

    def _canonical_response(self, episode: _Episode, *, masked: bool) -> NeMoGymResponse:
        assert episode.last_response is not None
        return NeMoGymResponse.model_validate(
            episode.last_response.model_dump(mode="json")
            | {
                "output": copy.deepcopy(episode.output),
                "usage": episode.usage.model_dump(mode="json")
                if episode.usage_known and episode.usage is not None
                else None,
                "status": "incomplete" if masked else "completed",
            }
        )

    async def _perform_step(self, episode: _Episode, action: NeMoGymResponse) -> EnvStepResponse:
        episode.last_response = action.model_copy(deep=True)
        if (
            episode.parent_turns >= self.config.max_parent_turns
            or episode.policy_calls >= self.config.replay.max_policy_calls
        ):
            return self._terminal(
                episode,
                self._masked(episode, "policy_call_limit", "A parent request arrived after its configured budget"),
                "policy_call_limit",
            )
        episode.parent_turns += 1
        episode.policy_calls += 1
        self._add_usage(episode, action.usage)
        episode.model_calls.append(
            {
                "role": "parent",
                "response_id": action.id,
                "model": action.model,
                "model_ref": self.config.policy_model_server.model_dump(mode="json"),
            }
        )
        episode.output.extend(item.model_dump(mode="json", exclude_none=True) for item in action.output)
        if (
            action.model != self.config.policy_model
            or not action.id
            or action.id in episode.response_ids
            or action.status != "completed"
            or action.error is not None
            or action.incomplete_details is not None
        ):
            return self._terminal(
                episode,
                self._masked(
                    episode,
                    "invalid_response",
                    "The parent response is incomplete, repeated, or from a different model",
                ),
                "invalid_response",
            )
        episode.response_ids.add(action.id)
        try:
            current = visible_events([item.model_dump(mode="json", exclude_none=True) for item in action.output])
            if not any(
                event.kind == "tool_call" or event.kind == "assistant" and event.text.strip() for event in current
            ):
                raise ValueError("The parent response has no visible assistant action")
            if any(
                item.type not in {"reasoning", "message", "function_call"}
                or getattr(item, "status", None) not in {None, "completed"}
                for item in action.output
            ):
                raise ValueError("The parent response contains unsupported or incomplete output")
            tool_outputs, gap = await self._execute_tools(episode, action)
        except (ValueError, TypeError, KeyError) as error:
            return self._terminal(episode, self._masked(episode, "invalid_capture", str(error)), "invalid_capture")
        if gap is not None:
            return self._terminal(
                episode,
                self._masked(episode, gap, "Recorded-context tool coverage did not settle this prefix"),
                gap,
                tool_outputs,
            )
        captured = capture(
            episode.params,
            episode.output,
            episode.events,
            episode.simulator.drafts,
            action.output_text,
            prefix=True,
            answer_paths=episode.bundle["runtime"]["answer_paths"],
        )
        judged = await self._judge_prefix(episode, captured)
        episode.history.append(
            {
                "prefix_index": episode.parent_turns,
                "capture": captured.model_dump(mode="json"),
                "judge": copy.deepcopy(judged),
            }
        )
        try:
            outcome = ordering_outcome(
                judged,
                episode.events,
                probe_namespace=episode.bundle["runtime"]["ordering"]["probe_namespace"],
                expected_criterion_id=episode.contract.rubrics[0].id,
            )
        except ValueError:
            return self._terminal(
                episode, judged | {"mask_sample": True, "evaluation_completed": False}, "judge_error", tool_outputs
            )
        if outcome is not None:
            return self._terminal(episode, judged | {"evaluation_completed": True}, outcome, tool_outputs)
        if not any(item.type == "function_call" for item in action.output) or episode.simulator.drafts:
            kind = "ordering_undecided"
        elif episode.policy_calls >= self.config.replay.max_policy_calls:
            kind = "policy_call_limit"
        elif episode.parent_turns >= self.config.max_parent_turns:
            kind = "parent_turn_limit"
        else:
            return EnvStepResponse(reward=0, info={"tool_outputs": tool_outputs})
        return self._terminal(
            episode,
            self._masked(episode, kind, "No decisive ordering event before the final response or configured limit"),
            kind,
            tool_outputs,
        )

    async def _execute_tools(self, episode: _Episode, action: NeMoGymResponse) -> tuple[list[dict], str | None]:
        assert episode.simulator is not None
        visible_events([item.model_dump(mode="json", exclude_none=True) for item in action.output])
        calls = [item for item in action.output if item.type == "function_call"]
        assistant = {"role": "assistant", "content": action.output_text or None}
        if calls:
            assistant["tool_calls"] = [
                {"id": call.call_id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}
                for call in calls
            ]
        episode.messages.append(assistant)
        episode.simulator.sync_transcript(episode.messages)
        outputs, gap = [], None
        for index, call in enumerate(calls, 1):
            if not call.call_id or call.call_id in episode.tool_ids:
                raise ValueError("Missing or duplicate parent tool call identity")
            episode.tool_ids.add(call.call_id)
            value = {
                "id": call.call_id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            observed_at, observed_monotonic = time(), perf_counter()
            result, kind = episode.simulator.dispatch(value, episode.schemas)
            if call.name == "Agent" and kind != "invalid_call":
                worker_directory = Path(f"delegate-{episode.parent_turns:02d}-{index:02d}")
                episode.children[str(worker_directory)] = call.call_id
                result, kind = await explore(
                    json.loads(call.arguments),
                    state=episode.simulator,
                    case=episode.case,
                    config=self.config.replay,
                    model=self.config.policy_model,
                    directory=worker_directory,
                    invoke=lambda payload, timeout, prefix: self._invoke_child(episode, payload, timeout, prefix),
                    write=lambda path, receipt: episode.child_receipts.__setitem__(str(path), copy.deepcopy(receipt)),
                    observe_tool=lambda observation: self._record_tool_observation(episode, observation, worker=True),
                )
            self._record_tool_observation(
                episode, tool_observation(value, result, kind, episode.simulator, observed_at, observed_monotonic)
            )
            event = {
                "turn": episode.parent_turns,
                "tool_call_id": call.call_id,
                "function": value["function"],
                "kind": kind,
                "result": result,
            }
            event["verified_probe"] = probe_identity(event, episode.bundle)
            episode.events.append(event)
            outputs.append({"call_id": call.call_id, "output": result})
            if kind in {"fixture_gap", "invalid_call"}:
                gap = kind
        # Calls chosen in one response all precede every returned result, including in judge evidence.
        episode.output.extend({"type": "function_call_output", **output} for output in outputs)
        episode.messages.extend(
            {"role": "tool", "tool_call_id": output["call_id"], "content": output["output"]} for output in outputs
        )
        episode.simulator.sync_transcript(episode.messages)
        return outputs, gap

    @staticmethod
    def _record_tool_observation(episode: _Episode, observation: dict, *, worker: bool = False) -> None:
        owner = episode.invocation_id
        if worker:
            owner += "." + observation["conversation_id"]
        episode.tool_observations.append(
            ToolCallObservation(
                invocation_id=owner,
                tool_call_id=observation["tool_call_id"],
                tool_name=observation["function"]["name"],
                started_at=observation["started_at"],
                completed_at=observation["completed_at"],
                duration_ms=observation["duration_ms"],
                timing_source="executor",
                status=observation["status"],
                error_type=observation.get("error_type"),
            ).model_dump(mode="json")
        )

    def _agent_observations(self, episode: _Episode) -> dict:
        def references(role: str, directory: str | None = None) -> list[ModelCallRef]:
            return [
                ModelCallRef(model_ref=self.config.policy_model_server, response_id=record["response_id"])
                for record in episode.model_calls
                if record["role"] == role
                and record.get("response_id")
                and (directory is None or record.get("receipt", "").startswith(directory + "/"))
            ]

        records = [
            AgentInvocation(
                invocation_id=episode.invocation_id,
                status="unknown",
                model_calls=references("parent"),
                conversation=episode.params["input"] + episode.output,
            )
        ]
        gaps = []
        converter = ChatCompletionsConverter(return_token_id_information=False)
        for directory, call_id in episode.children.items():
            worker_id = episode.invocation_id + "." + directory
            conversation = episode.child_receipts.get(directory + "/conversation.json")
            receipt = episode.child_receipts.get(directory + "/result.json") or {}
            try:
                if not isinstance(conversation, list):
                    raise ValueError("Missing Explore conversation")
                items = converter.chat_completions_messages_to_responses_items(copy.deepcopy(conversation))
            except (ValueError, TypeError, KeyError, NotImplementedError):
                items = []
                gaps.append(
                    ObservationGap(code="recorded_context_child_conversation_unavailable", invocation_id=worker_id)
                )
            records.append(
                AgentInvocation(
                    invocation_id=worker_id,
                    parent_invocation_id=episode.invocation_id,
                    spawned_by_tool_call_id=call_id,
                    status="completed" if receipt.get("completed") is True else "incomplete",
                    model_calls=references("Explore", directory),
                    conversation=items,
                )
            )
        return AgentObservationBundle(
            source="partner_if_recorded_context", records=[*records, *episode.tool_observations], gaps=gaps
        ).model_dump(mode="json")

    async def _invoke_child(self, episode: _Episode, payload: dict, timeout: int, prefix: Path) -> tuple[int, dict]:
        if episode.policy_calls >= self.config.replay.max_policy_calls:
            raise RuntimeError("The shared parent and Explore policy budget is exhausted")
        episode.policy_calls += 1
        episode.child_calls += 1
        known_before_dispatch = episode.usage_known
        episode.usage_known = False
        response = await self.server_client.post(
            server_name=self.config.policy_model_server.name,
            url_path="/v1/chat/completions",
            json=payload,
            timeout=ClientTimeout(total=timeout),
            _max_num_tries=1,
            _max_connection_retries=0,
            _resend_after_disconnect=False,
            _retry_other_errors=False,
        )
        try:
            await raise_for_status(response)
            raw = await get_response_json(response)
        finally:
            response.release()
        if not isinstance(raw, dict):
            raise ValueError("Explore policy response is not an object")
        usage = raw.get("usage")
        if isinstance(usage, dict) and all(
            type(usage.get(key)) is int for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        ):
            episode.usage_known = known_before_dispatch
            self._add_usage(
                episode,
                NeMoGymResponseUsage.model_validate(
                    {
                        "input_tokens": usage["prompt_tokens"],
                        "output_tokens": usage["completion_tokens"],
                        "total_tokens": usage["total_tokens"],
                        "input_tokens_details": {
                            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                        },
                        "output_tokens_details": {
                            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
                        },
                    }
                ),
            )
        else:
            self._add_usage(episode, None)
        episode.model_calls.append(
            {
                "role": "Explore",
                "response_id": raw.get("id"),
                "model": raw.get("model"),
                "model_ref": self.config.policy_model_server.model_dump(mode="json"),
                "receipt": str(prefix),
            }
        )
        episode.child_receipts[str(prefix) + ".model-server-response.json"] = copy.deepcopy(raw)
        # The Chat model-server endpoint wraps its reasoning in leading think tags. Keep it out of worker findings.
        decoded = copy.deepcopy(raw)
        for choice in decoded.get("choices") or []:
            message = choice.get("message") or {}
            content = message.get("content")
            if not isinstance(content, str):
                continue
            pieces, offset = [], 0
            while match := ChatCompletionsConverter.THINK_TAG_PATTERN.match(content, offset):
                pieces.append(match.group(1))
                offset = match.end()
            if pieces:
                if message.get("reasoning_content"):
                    raise ValueError("Duplicated Explore reasoning transport")
                message["reasoning_content"] = "".join(pieces)
                message["content"] = content[offset:] or None
        return 200, decoded

    async def _judge_prefix(self, episode: _Episode, captured: LiveAgentCapture) -> dict:
        response = self._canonical_response(episode, masked=False)
        body = VerifyRequest.model_validate(
            episode.task_data
            | {"responses_create_params": episode.params, "response": response.model_dump(mode="json")}
        )

        async def calculate(body: VerifyRequest) -> dict:
            async def grade() -> dict:
                result = await self.grade_response(
                    episode.contract,
                    lambda: live_agent_evidence(episode.params, copy.deepcopy(episode.output), captured),
                )
                if (
                    result.get("judge_response") is not None
                    and (result.get("grading_result") or {}).get("status") == "scored"
                ):
                    record_parsed_judge_verdict()
                return result | {
                    "response": body.response.model_dump(mode="json"),
                    "evaluation_completed": not result.get("mask_sample", False),
                }

            return await observe_llm_judge_grading(grade)

        config = getattr(self.server_client, "global_config_dict", {})
        observed = observe_verifier(
            judge_failsafe(calculate),
            enabled=TypeAdapter(bool).validate_python(config.get("observability_enabled", False)),
            verifier_name=self.config.name,
            uninstrumented_models=True,
        )
        raw = await observed(body)
        result = json.loads(raw.body) if isinstance(raw, Response) else raw
        if "if_result" not in result:
            result = (
                result
                | masked_fields(
                    episode.contract,
                    result.get("failure_kind") or "partner_if:recorded_context_judge_error",
                    result.get("failure_reason") or "The shared judge did not return a grading receipt",
                )
                | {"evaluation_completed": False}
            )
        episode.verifier_invocations.extend(copy.deepcopy(result.get("ng_verifier_invocations") or []))
        return result

    def _terminal(
        self, episode: _Episode, fields: dict, termination: str, tool_outputs: list[dict] | None = None
    ) -> EnvStepResponse:
        completed = fields.get("evaluation_completed") is True and fields.get("mask_sample") is False
        masked = not completed
        canonical = self._canonical_response(episode, masked=masked)
        diagnostics = {
            "schema_version": "partner-if-recorded-context-v1",
            "termination": termination,
            "task_data_sha256": _sha(episode.task_data),
            "parent_model_calls": episode.parent_turns,
            "delegate_model_calls": episode.child_calls,
            "model_calls": episode.policy_calls,
            "automatic_retries": 0,
            "model_call_records": copy.deepcopy(episode.model_calls),
            "prefix_history": copy.deepcopy(episode.history),
            "events": copy.deepcopy(episode.events),
            "child_receipts": copy.deepcopy(episode.child_receipts),
            "usage_complete": episode.usage_known,
        }
        result = (
            copy.deepcopy(episode.task_data)
            | copy.deepcopy(fields)
            | {
                "responses_create_params": copy.deepcopy(episode.params),
                "response": canonical.model_dump(mode="json"),
                "evaluation_completed": completed,
                "mask_sample": masked,
                "reward": float(fields.get("reward", 0)) if completed else 0.0,
                "ng_verifier_invocations": copy.deepcopy(episode.verifier_invocations),
                "ng_agent_observations": self._agent_observations(episode),
                "recorded_context": diagnostics,
            }
        )
        result[RESULT_BINDING_KEY] = recorded_context_result_sha256(result)
        return EnvStepResponse(
            reward=result["reward"],
            terminated=completed,
            truncated=masked,
            mask_sample=masked,
            failure_kind=result.get("failure_kind"),
            failure_reason=result.get("failure_reason"),
            info={"tool_outputs": tool_outputs or [], RESULT_KEY: result},
        )

    @staticmethod
    def _release(episode: _Episode) -> None:
        episode.closed_at = episode.closed_at or time()
        episode.simulator = None
        episode.bundle = {}
        episode.case = {}
        episode.messages = []
        episode.schemas = {}

    async def close_session(self, session_id: str | None) -> None:
        """Release simulator state on close and checkpoint retirement."""
        if session_id is not None and session_id in self._episodes:
            async with self._lock(session_id):
                if episode := self._episodes.get(session_id):
                    self._release(episode)
        await super().close_session(session_id)

    async def _close_endpoint(self, request: Request) -> dict:
        session_id = self._session_id(request)
        episode = self._episodes.get(session_id)
        if (
            episode is not None
            and episode.active_task is not None
            and episode.active_task is not asyncio.current_task()
        ):
            episode.active_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await episode.active_task
        await self.close_session(session_id)
        self.checkpoint_session_ended(request)
        return {"closed": True}

    async def step(self, action: NeMoGymResponse, metadata: dict, session_id: str | None = None):
        raise RuntimeError("Use the typed Gymnasium /step boundary")


if __name__ == "__main__":
    RecordedContextResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = RecordedContextResourcesServer.run_webserver()
