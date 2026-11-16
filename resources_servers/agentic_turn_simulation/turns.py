# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One simulated turn: its session (`TurnSimulation.new_session`), the tool calls the simulator answers
(`simulate_tool_call`), and its score (`verify`, code gates then the judges).

The ``agentic_turn_simulation`` resources server and the ``agent_arena`` environment server both run turns through a
`TurnSimulation`. It keeps no sessions itself: each server holds the `TurnSession` of each of its turns.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable

from nemo_gym.client import ServerClient, get_response_json
from nemo_gym.core import NeMoGymChatCompletion, NgFailureEvent, failure_kinds
from nemo_gym.metrics import MASK_SAMPLE_FIELD
from nemo_gym.server_runtime import observe_llm_judge_grading
from nemo_gym.verification import call_judge, is_judge_failure, record_parsed_judge_verdict
from nemo_gym.verification.judge import _is_retryable_judge_error
from resources_servers.agentic_turn_simulation.foreign_script import foreign_script_letters
from resources_servers.agentic_turn_simulation.harnesses import HARNESSES, Harness, select_harness
from resources_servers.agentic_turn_simulation.items import as_dict, content_text, parse_args
from resources_servers.agentic_turn_simulation.judge import (
    VERDICT_FIELDS,
    build_system_prompt,
    code_gate,
    combine,
    evaluation_schema,
    load_prompt,
    parse_verdict,
    usable,
)
from resources_servers.agentic_turn_simulation.obstacles import (
    Obstacle,
    ObstacleReport,
    arm,
    obstacle_metrics,
)
from resources_servers.agentic_turn_simulation.render import (
    JUDGE_RENDER,
    SIMULATOR_SYSTEM_PROMPT_CHARS,
    agent_system_prompt,
    format_tools,
    render_conversation,
    render_system_prompt,
)
from resources_servers.agentic_turn_simulation.schemas import (
    AgenticTurnSimulationRunRequest,
    AgenticTurnSimulationSettings,
    AgenticTurnSimulationVerifyRequest,
    AgenticTurnSimulationVerifyResponse,
    AnswerSource,
    HarnessResponse,
    JudgeSpec,
    SimulateToolCallRequest,
    SimulateToolCallResponse,
)
from resources_servers.agentic_turn_simulation.simulator import (
    holds_simulator_failure,
    simulation_history,
    simulator_failure,
    simulator_user_message,
)
from resources_servers.agentic_turn_simulation.workspace import Workspace


LOG = logging.getLogger(__name__)

FAILURE_KIND_PREFIX = "agentic_turn_simulation"  # the namespace of this server's failure kinds
# The output items of a policy turn that are scored: its reasoning is never shown to the judge.
SCORED_ITEM_TYPES = ("message", "function_call", "function_call_output")
# A reply cut off by its output budget is retried with the budget doubled, starting from the params' max_tokens
# (or max_completion_tokens), else from these.
SIMULATOR_DEFAULT_MAX_TOKENS = 16_384
JUDGE_DEFAULT_MAX_TOKENS = 20_000
# Responses statuses of a request the model server did not complete (Responses-native servers pass them on, partial
# output and all). "incomplete" is not one: a reply cut off by its output budget is the policy's.
FAILED_RESPONSE_STATUSES = frozenset({"failed", "cancelled", "queued", "in_progress"})
# A cap on concurrent model calls; nullcontext when uncapped.
ConcurrencyLimit = asyncio.Semaphore | nullcontext[None]


@dataclass
class TurnSession:
    harness: Harness
    # The row's history as tool items.
    prefix: list[dict[str, Any]]
    # The tools a call may name.
    tool_names: frozenset[str]
    workspace: Workspace | None
    # What the simulator's user message starts with for every call of the session (so it forms a shared prompt
    # prefix): the row's <agent_system_prompt> block and the <tools> block, each when the harness shows it.
    simulator_preamble: str = ""
    created: float = field(default_factory=time.monotonic)
    # How many of the session's calls were answered each way (AnswerSource).
    answered_by: Counter[str] = field(default_factory=Counter)
    # The row's armed obstacle, if any.
    obstacle: Obstacle | None = None
    # The row's verifier_metadata.
    metadata: dict[str, Any] = field(default_factory=dict)


def _prepared_call(harness: Harness, call: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    """The call as its tool runs it: with its arguments in the form the harness's tool takes them
    (`Harness.prepare_arguments`), for the workspace and the simulator."""
    arguments = parse_args(call)
    if arguments is None:
        return call
    prepared = harness.prepare_arguments(call.get("name") or "", arguments, metadata)
    return call if prepared == arguments else {**call, "arguments": json.dumps(prepared, ensure_ascii=False)}


def _input_items(body: AgenticTurnSimulationRunRequest) -> list[dict[str, Any]]:
    rcp = body.responses_create_params
    if isinstance(rcp.input, str):
        return [{"type": "message", "role": "user", "content": rcp.input}]
    return [as_dict(i) for i in rcp.input or []]


def _row_tools(body: AgenticTurnSimulationRunRequest) -> list[dict[str, Any]]:
    return [as_dict(t) for t in body.responses_create_params.tools or []]


def _response_metadata(body: AgenticTurnSimulationVerifyRequest) -> dict[str, Any]:
    """The policy response's metadata, or {}."""
    return (body.response.metadata or {}) if body.response is not None else {}


def _policy_output(body: AgenticTurnSimulationVerifyRequest) -> list[dict[str, Any]]:
    """The policy turn's scored items (no reasoning), as the agent's response holds them."""
    output = body.response.output if body.response is not None else []
    return [d for d in (as_dict(o) for o in output or []) if d.get("type") in SCORED_ITEM_TYPES]


def _executed_calls(items: list[dict[str, Any]], stop_tools: tuple[str, ...]) -> int:
    """Calls in the tool items that got a result, stop tools excluded (a stop call never runs)."""
    answered = {x.get("call_id") for x in items if x.get("type") == "function_call_output"}
    return sum(
        1
        for x in items
        if x.get("type") == "function_call" and x.get("call_id") in answered and x.get("name") not in stop_tools
    )


def _final_reply_failed(body: AgenticTurnSimulationVerifyRequest) -> bool:
    """Whether the turn's last model reply is a failed request's rather than the policy's: the model server marked it
    unfinished (an `error`, or a status in FAILED_RESPONSE_STATUSES: failed, cancelled, queued or in_progress; the
    agent's response carries the last reply's), or it held no items at all, not even reasoning (the output ends on a
    tool result, or is empty). completed, incomplete and a missing status are the policy's."""
    response = body.response
    if response is None:
        return True
    if response.error is not None or response.status in FAILED_RESPONSE_STATUSES:
        return True
    output = response.output or []
    return not output or as_dict(output[-1]).get("type") == "function_call_output"


def _final_reply_unscored(body: AgenticTurnSimulationVerifyRequest) -> bool:
    """Whether the turn ends on a reply of the policy's that holds no scored item (only reasoning): the scored turn
    does not show that the policy replied after its last item."""
    return not _final_reply_failed(body) and as_dict(body.response.output[-1]).get("type") not in SCORED_ITEM_TYPES


def _row_system_prompt(body: AgenticTurnSimulationRunRequest, prefix: list[dict[str, Any]]) -> str:
    """The row's own system prompt: its `instructions`, then its system and developer messages."""
    instructions = body.responses_create_params.instructions
    parts = [instructions] if isinstance(instructions, str) and instructions.strip() else []
    text = agent_system_prompt(prefix)
    return "\n\n".join(parts + ([text] if text else []))


def _masked(failure_kind: str, reason: str) -> dict[str, Any]:
    return {MASK_SAMPLE_FIELD: True, "failure_kind": failure_kind, "failure_reason": reason}


def _grow_max_tokens(params: dict[str, Any], default: int, cap: int) -> None:
    """Double the output budget (max_tokens, else max_completion_tokens, else `default`), up to `cap`."""
    current = params.get("max_tokens") or params.get("max_completion_tokens") or default
    params["max_tokens"] = min(current * 2, cap)
    params.pop("max_completion_tokens", None)


def _without_rejected_param(params: dict[str, Any], error: str) -> bool:
    """Drop or relax the parameter a 400 error (lowercased) names, at most once per parameter; False when the
    error names none that can change."""
    schema_rejected = "response_format" in error or "json_schema" in error
    if schema_rejected and params.get("response_format", {}).get("type") == "json_schema":
        params["response_format"] = {"type": "json_object"}
    elif "reasoning" in error and "reasoning_effort" in params:
        params.pop("reasoning_effort")
    elif "temperature" in error and "temperature" in params:
        params.pop("temperature")
    else:
        return False
    return True


class TurnSimulation:
    """Sets turns up, answers their tool calls and scores them, with one server's settings and model servers."""

    def __init__(self, config: AgenticTurnSimulationSettings, server_client: ServerClient) -> None:
        self.config = config
        self.server_client = server_client
        sim_limit, judge_limit = self.config.tool_simulator_max_concurrency, self.config.judge_max_concurrency
        self._simulator_semaphore: ConcurrencyLimit = asyncio.Semaphore(sim_limit) if sim_limit else nullcontext()
        self._judge_semaphore: ConcurrencyLimit = asyncio.Semaphore(judge_limit) if judge_limit else nullcontext()
        # The simulator's system prompt, per harness name.
        self._simulator_prompts: dict[str, str] = {}
        note = load_prompt("tool-simulator-system-prompt-note.md")
        for name, harness in HARNESSES.items():
            prompt = load_prompt(harness.simulator_prompt)
            if self.shows_system_prompt(harness) and not harness.show_system_prompt:
                # Made to show a block its prompt does not describe: explain the block.
                prompt = f"{prompt}\n\n{note}\n"
            self._simulator_prompts[name] = prompt

    def shows_system_prompt(self, harness: Harness) -> bool:
        """Whether the judge and the simulator see the row's own system prompt (the config overrides)."""
        forced = self.config.show_agent_system_prompt
        return harness.show_system_prompt if forced is None else forced

    def _obstacle(
        self, body: AgenticTurnSimulationRunRequest, harness: Harness, tool_names: frozenset[str]
    ) -> Obstacle | None:
        """The row's armed obstacle, or None (always None when scoring the recorded turn)."""
        if self.config.score_source != "policy":
            return None
        return arm(
            self.config.obstacle_injection,
            harness=harness,
            tool_names=tool_names,
            input_items=_input_items(body),
            metadata=body.verifier_metadata or {},
        )

    # --- session --------------------------------------------------------------------------------------------

    def new_session(self, body: AgenticTurnSimulationRunRequest) -> TurnSession:
        """The row's session: its harness, its history replayed into the workspace, the simulator's preamble and the
        armed obstacle."""
        harness = select_harness(body.verifier_metadata)
        prefix = harness.to_tool_items(_input_items(body))
        tools = _row_tools(body)
        workspace = harness.new_workspace(body.verifier_metadata or {})
        if workspace is not None:
            workspace.replay(prefix)
        preamble = ""
        if self.shows_system_prompt(harness):
            preamble += render_system_prompt(_row_system_prompt(body, prefix), SIMULATOR_SYSTEM_PROMPT_CHARS)
        if harness.simulator_sees_tools:
            preamble += format_tools(harness.described_tools(tools, prefix)) + "\n\n"
        tool_names = harness.tool_names(tools)
        return TurnSession(
            harness=harness,
            prefix=prefix,
            tool_names=tool_names,
            workspace=workspace,
            simulator_preamble=preamble,
            obstacle=self._obstacle(body, harness, tool_names),
            metadata=body.verifier_metadata or {},
        )

    @staticmethod
    def describe_harness(state: TurnSession) -> HarnessResponse:
        return HarnessResponse(stop_tools=list(state.harness.stop_tools), protocol=state.harness.protocol)

    # --- tool simulation ------------------------------------------------------------------------------------

    async def simulate_tool_call(
        self, state: TurnSession | None, body: SimulateToolCallRequest
    ) -> SimulateToolCallResponse:
        if state is None:
            return SimulateToolCallResponse(output=simulator_failure("session not found"), source="failed")
        harness = state.harness
        call = {"type": "function_call", "call_id": body.call_id, "name": body.name, "arguments": body.arguments}
        turn = harness.to_tool_items(body.turn)
        call_id, more_follow = harness.locate_call(turn, body.call_id)
        if state.obstacle is not None:
            # A failed call is not answered, so the workspace never learns from it; the simulator sees a note in
            # place of each earlier failure.
            injected, turn = state.obstacle.at_call(turn, call)
            if injected is not None:
                state.answered_by["injected"] += 1
                return SimulateToolCallResponse(
                    output=harness.frame(call, injected, more_follow=more_follow), source="injected"
                )
        before = simulation_history(turn, call_id)
        try:
            output, source = await self._answer(state, _prepared_call(harness, call, state.metadata), before)
        except Exception as exc:  # noqa: BLE001 - a crash while answering is a failed simulation (masked), not a 500
            LOG.exception("answering a %s call failed", body.name)
            state.answered_by["failed"] += 1
            output, source = simulator_failure(f"{type(exc).__name__}: {exc}"), "failed"
        return SimulateToolCallResponse(output=harness.frame(call, output, more_follow=more_follow), source=source)

    async def _answer(
        self, state: TurnSession, call: dict[str, Any], before: list[dict[str, Any]]
    ) -> tuple[str, AnswerSource]:
        harness = state.harness
        exact = harness.exact_answer(
            call, history=state.prefix + before, workspace=state.workspace, tool_names=state.tool_names
        )
        if exact is not None:
            state.answered_by["exact"] += 1
            return exact, "exact"
        conversation = simulator_user_message(call, state.prefix + before, state.workspace)
        output, error = await self._simulate_with_llm(state.simulator_preamble + conversation, harness)
        if output is None:
            state.answered_by["failed"] += 1
            return simulator_failure(error), "failed"
        if state.workspace is not None:
            state.workspace.observe(call, output)
        state.answered_by["llm"] += 1
        return output, "llm"

    async def _post_chat(
        self, model_server: str, params: dict[str, Any], semaphore: ConcurrencyLimit
    ) -> tuple[int, dict[str, Any] | str]:
        """POST /v1/chat/completions: (status, the response JSON), or (status, the error body) for HTTP >= 400."""
        async with semaphore:
            response = await self.server_client.post(
                server_name=model_server, url_path="/v1/chat/completions", json=params
            )
            if response.status >= 400:
                return response.status, (await response.content.read()).decode(errors="replace")
            return response.status, await get_response_json(response)

    async def _backoff(self, attempt: int) -> None:
        await asyncio.sleep(min(2**attempt, self.config.retry_backoff_max_seconds))

    async def _simulate_with_llm(self, user_message: str, harness: Harness) -> tuple[str | None, str]:
        """(the simulated output, "") or (None, the last error) once every attempt failed."""
        params = self.config.tool_simulator_chat_completions_create_params.model_dump(mode="json", exclude_none=True)
        params["messages"] = [
            {"role": "system", "content": self._simulator_prompts[harness.name]},
            {"role": "user", "content": user_message},
        ]
        error = "no attempt"
        for attempt in range(self.config.tool_simulator_max_attempts):
            try:
                status, reply = await self._post_chat(
                    self.config.tool_simulator_model_server.name, params, self._simulator_semaphore
                )
                if status >= 400:
                    error = f"HTTP {status}: {reply[:200]}"
                    # Unlike the judge's (`_without_rejected_param`), a simulator 400 only ever relaxes
                    # reasoning_effort, and one about the context length is retried like any other failure.
                    if status == 400 and "reasoning" in reply.lower() and "reasoning_effort" in params:
                        params.pop("reasoning_effort")
                        continue  # retried at once
                else:
                    choice = (reply.get("choices") or [{}])[0]
                    message, finish = choice.get("message") or {}, choice.get("finish_reason")
                    output = harness.parse_reply(message, finish)
                    if output is not None:
                        return output, ""
                    content = message.get("content") or ""
                    error = f"unparseable (finish={finish}): {content[:200]!r}"
                    if finish == "length" or not content.strip():
                        _grow_max_tokens(
                            params, SIMULATOR_DEFAULT_MAX_TOKENS, self.config.tool_simulator_max_tokens_cap
                        )
            except Exception as exc:  # noqa: BLE001 - a failed simulation is reported to the policy and masked
                error = f"{type(exc).__name__}: {exc}"
            LOG.warning("tool simulator attempt %d failed: %s", attempt + 1, error)
            await self._backoff(attempt)
        return None, error

    # --- judge ----------------------------------------------------------------------------------------------

    def _judge_params(self, judge: JudgeSpec, system: str, conversation: str) -> dict[str, Any]:
        base = self.config.judge_chat_completions_create_params.model_dump(mode="json", exclude_none=True)
        # A null override removes the key (JudgeSpec.overrides).
        params = {key: value for key, value in (base | judge.overrides).items() if value is not None}
        params["messages"] = [{"role": "system", "content": system}, {"role": "user", "content": conversation}]
        if "response_format" not in judge.overrides:  # a null override sends none
            schema = {"name": "Audit", "strict": True, "schema": evaluation_schema()}
            params.setdefault("response_format", {"type": "json_schema", "json_schema": schema})
        return params

    async def _call_judge(self, judge: JudgeSpec, system: str, conversation: str) -> dict[str, Any] | None:
        """The judge's verdict: the first usable one, else the last one parsed (None when none was).

        The call goes through call_judge (one attempt each; this loop owns the retries). A judge that could not be
        reached on any attempt, or that rejects the conversation as too long, raises a judge_failed event rather
        than dropping out of the vote. An unusable verdict is retried, then left to the vote.
        """
        params = self._judge_params(judge, system, conversation)
        name = judge.model_server.name
        verdict = None
        failure: NgFailureEvent | None = None
        for attempt in range(self.config.judge_max_attempts):
            try:
                async with self._judge_semaphore:
                    reply = await call_judge(
                        self.server_client,
                        server_name=name,
                        url_path="/v1/chat/completions",
                        json=params,
                        response_model=NeMoGymChatCompletion,
                        max_attempts=1,
                    )
            except NgFailureEvent as exc:
                if not is_judge_failure(exc):
                    raise
                failure = exc
                cause = exc.__cause__ or exc
                LOG.warning("judge %s call failed (attempt %d): %s", name, attempt + 1, exc)
                bad_request = getattr(cause, "status", None) == 400  # retried as before, once nothing can be relaxed
                if bad_request:
                    content = getattr(cause, "response_content", b"")
                    error = (content.decode(errors="replace") if isinstance(content, bytes) else str(content)).lower()
                    if "context" in error and ("length" in error or "maximum" in error):
                        raise  # the conversation does not fit this judge; retrying will not help
                    if _without_rejected_param(params, error):
                        continue  # retried at once
                if not bad_request and not _is_retryable_judge_error(cause):
                    raise
            else:
                failure = None
                choice = reply.choices[0] if reply.choices else None
                verdict = parse_verdict(choice.message.content if choice is not None else None)
                if usable(verdict):
                    record_parsed_judge_verdict()
                    return verdict
                if choice is not None and choice.finish_reason == "length":
                    _grow_max_tokens(params, JUDGE_DEFAULT_MAX_TOKENS, self.config.judge_max_tokens_cap)
                LOG.warning("judge %s gave no usable verdict (attempt %d)", name, attempt + 1)
            await self._backoff(attempt)
        if failure is not None:
            raise failure  # the last attempt could not reach the judge: it is lost, not unusable
        return verdict

    async def _judge(
        self,
        body: AgenticTurnSimulationVerifyRequest,
        harness: Harness,
        items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any] | None]:
        metadata = body.verifier_metadata or {}
        prefix = harness.to_tool_items(_input_items(body))
        requester = harness.requester(metadata)
        render = JUDGE_RENDER.model_copy(update={"user_label": requester.label})
        # The history span [lo, hi) the user's message corrects (indices into the prefix as tool items), which the
        # judge sees marked as <rejected_turn>. The range is keyed by the rubric that uses it (steerability).
        corrected = (metadata.get("corrected_turn_item_range") or {}).get("steerability")
        conversation = render_conversation(prefix, items, render, corrected)
        shows_prompt = self.shows_system_prompt(harness)
        if shows_prompt:
            conversation = render_system_prompt(_row_system_prompt(body, prefix)) + conversation
        system = build_system_prompt(
            harness.described_tools(tools, prefix),
            template=harness.judge_prompt,
            requester=requester.label,
            requester_note=load_prompt(requester.note_file) if requester.note_file else "",
            environment=load_prompt(harness.judge_environment) if harness.judge_environment else "",
            system_prompt_note=load_prompt(self.config.judge_system_prompt_note_file) if shows_prompt else "",
        )
        results = await asyncio.gather(
            *(self._call_judge(j, system, conversation) for j in self.config.judges), return_exceptions=True
        )
        for result in results:  # a judge lost to a failed call fails the row, whatever the others voted
            if isinstance(result, BaseException):
                raise result
        return list(results)

    # --- verify ---------------------------------------------------------------------------------------------

    def _scored_items(self, body: AgenticTurnSimulationVerifyRequest, harness: Harness) -> list[dict[str, Any]]:
        """The turn to score: the policy's, or the original model's recorded one (score_source)."""
        if self.config.score_source == "reference_turn":
            turn = [dict(x) for x in (body.verifier_metadata or {}).get("reference_turn") or []]
        else:
            turn = _policy_output(body)
        return harness.scored_turn(turn)

    def _mask_reason(
        self,
        body: AgenticTurnSimulationVerifyRequest,
        items: list[dict[str, Any]],
        *,
        stop_reason: str | None,
        steps: int | None,
    ) -> dict[str, Any] | None:
        """Why the turn's outcome says nothing about the policy, or None."""
        if self.config.score_source == "reference_turn":
            if items and items[-1].get("type") == "function_call_output":
                return _masked(
                    f"{FAILURE_KIND_PREFIX}:recording_cut_off", "the recording ends right after a tool result"
                )
            return None
        if stop_reason == "context_overflow" and steps == 0:
            return _masked(
                f"{FAILURE_KIND_PREFIX}:context_overflow",
                "the conversation history does not fit the policy's context (overflow before the first step)",
            )
        if stop_reason in self.config.masked_stop_reasons:
            return _masked(f"{FAILURE_KIND_PREFIX}:{stop_reason}", f"the turn was cut off ({stop_reason})")
        for item in items:
            if item.get("type") == "function_call_output":
                text = str(item.get("output") or "")
            elif item.get("type") == "message" and item.get("role") == "user":
                # Cline's results travel in user messages; one the server could not frame stays a message.
                text = content_text(item.get("content"))
            else:
                continue
            if holds_simulator_failure(text):
                return _masked(f"{FAILURE_KIND_PREFIX}:simulator_failed", text[:300])
        return None

    def _response_fields(
        self,
        body: AgenticTurnSimulationVerifyRequest,
        harness: Harness,
        items: list[dict[str, Any]],
        tool_names: frozenset[str],
        state: TurnSession | None,
    ) -> dict[str, Any]:
        """The verify response's fields that do not depend on the outcome."""
        response_metadata = _response_metadata(body)
        steps = response_metadata.get("steps")  # Responses metadata values are strings
        stop_reason = response_metadata.get("stop_reason")
        # The policy's turn even when the recorded one is scored, like stop_reason and steps.
        policy_items = items if self.config.score_source == "policy" else harness.scored_turn(_policy_output(body))
        obstacle = self._obstacle(body, harness, tool_names)
        report, failed_results = (
            obstacle.outcome(items, replied_after=_final_reply_unscored(body))
            if obstacle is not None
            else (ObstacleReport(), frozenset())
        )
        # An injected failure is not the shell's.
        shell_items = [x for i, x in enumerate(items) if i not in failed_results]
        instructions = body.responses_create_params.instructions
        letters = (
            foreign_script_letters(
                items,
                history=_input_items(body),
                stop_tools=harness.stop_tools,
                instructions=instructions if isinstance(instructions, str) else None,
            )
            if self.config.score_source == "policy"
            else None  # a recorded turn is not measured
        )
        fields = {
            "score_source": self.config.score_source,
            "stop_reason": stop_reason,
            "steps": int(steps) if isinstance(steps, str) and steps.isdigit() else None,
            "ended_on_stop_tool": stop_reason if stop_reason in harness.stop_tools else None,
            "executed_calls": _executed_calls(policy_items, harness.stop_tools),
            "foreign_script_letters": letters,
            "shell_failures": harness.shell_failures(shell_items, tool_names) if harness.shell_failures else None,
            "simulated_calls": dict(state.answered_by) if state is not None else {},
            "categories": list((body.verifier_metadata or {}).get("categories") or []),
            "harness": harness.name,
        }
        if self.config.obstacle_injection.enabled:
            fields["obstacle"] = report.model_dump()
        return fields

    async def verify(
        self, body: AgenticTurnSimulationVerifyRequest, state: TurnSession | None
    ) -> AgenticTurnSimulationVerifyResponse:
        """Score the turn from the request alone: the session (None when there is none) only reports
        `simulated_calls`."""
        harness = select_harness(body.verifier_metadata)
        tools = _row_tools(body)
        tool_names = harness.tool_names(tools)
        items = self._scored_items(body, harness)
        common = self._response_fields(body, harness, items, tool_names, state)

        def respond(reward: float, **outcome: Any) -> AgenticTurnSimulationVerifyResponse:
            return AgenticTurnSimulationVerifyResponse(**body.model_dump(), reward=reward, **common, **outcome)

        # A turn cut off by the wall-clock cap, a row too long for the policy, a failed simulation or a turn no judge
        # could score says nothing about the policy. It is masked (no training signal) rather than scored 0, which
        # would teach the policy that the environment's failure was its own.
        masked = self._mask_reason(body, items, stop_reason=common["stop_reason"], steps=common["steps"])
        if masked is not None:
            return respond(0.0, **masked)
        policy_turn = self.config.score_source == "policy"
        metadata = body.verifier_metadata or {}
        gate = code_gate(
            items,
            tool_names,
            policy_turn=policy_turn,
            malformed_argument_keys=harness.malformed_argument_keys,
            schemas={t.get("name"): t.get("parameters") for t in tools},
            gates_missing_arguments=harness.gates_missing_arguments,
            protocol=harness.protocol,
            prepare_arguments=lambda name, arguments: harness.prepare_arguments(name, arguments, metadata),
        )
        stop_reason = common["stop_reason"]
        unfinished = stop_reason in self.config.unfinished_stop_reasons
        if stop_reason == "empty_output" and _final_reply_failed(body):
            unfinished = False  # a failed request's placeholder, not the policy's empty reply: judged as before
        if gate is None and policy_turn and unfinished:
            gate = "unfinished_turn"
        letters, min_letters = common["foreign_script_letters"], self.config.foreign_script_gate_min_letters
        if gate is None and letters is not None and min_letters is not None and letters >= min_letters:
            gate = "foreign_script"
        if gate is not None:
            return respond(0.0, code_gate=gate, correct_ladder="no")

        async def grade() -> AgenticTurnSimulationVerifyResponse:
            verdicts = await self._judge(body, harness, items, tools)
            combined = combine(verdicts)
            judged = {key: combined[key] for key in (*VERDICT_FIELDS, "correct_ladder")} | {"judge_verdicts": verdicts}
            if combined["n_usable"] == 0:
                return respond(
                    0.0, **judged, **_masked(failure_kinds.JUDGE_FAILED, "no judge returned a usable verdict")
                )
            return respond(1.0 if combined["correct_ladder"] == "yes" else 0.0, **judged)

        return await observe_llm_judge_grading(grade)

    # --- aggregate metrics ----------------------------------------------------------------------------------

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        # Gym passes only the measured rollouts and counts masked ones itself (coverage/masked_rollouts); count
        # failure_kind in the rollouts file for the reasons.
        rows = [r for task in tasks for r in task]
        if not rows:
            return {}
        metrics: dict[str, Any] = {"stop_reasons": dict(Counter(r.get("stop_reason") or "none" for r in rows))}
        metrics.update(obstacle_metrics(rows))

        def share(pred: Callable[[dict[str, Any]], bool]) -> float:
            return sum(1 for r in rows if pred(r)) / len(rows)

        metrics["turn_score"] = share(lambda r: r.get("reward") == 1.0)
        metrics["steerable_rate"] = share(lambda r: r.get("addresses_concern") == "yes")
        metrics["code_gate_rate"] = share(lambda r: r.get("code_gate") is not None)
        metrics["questionable_request_rate"] = share(lambda r: r.get("request_questionable") == "questionable")
        # Over the rows whose harness measures it (None: no shell tool with a known result format).
        shell = [r["shell_failures"] for r in rows if r.get("shell_failures") is not None]
        if shell:
            metrics["shell_failures_per_turn"] = sum(shell) / len(shell)
        failed = [r for r in rows if r.get("reward") != 1.0]
        blocked = Counter(r.get("code_gate") or r.get("blocked_by") or "steer=no" for r in failed)
        metrics["failure_attribution"] = {k: v / len(failed) for k, v in blocked.most_common()} if failed else {}
        per_category: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            for category in r.get("categories") or ["uncategorized"]:
                per_category[category].append(float(r.get("reward") or 0.0))
        metrics["turn_score_by_category"] = {c: sum(v) / len(v) for c, v in sorted(per_category.items())}
        return {**metrics, **_breakdowns(rows, self.config.metric_breakdown_keys)}

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        keys = ["turn_score", "steerable_rate", "code_gate_rate"]
        return {k: agent_metrics[k] for k in keys if k in agent_metrics}


def _metadata_value(row: dict[str, Any], dotted_key: str) -> str | None:
    """The row's verifier_metadata value at a dotted key, as a group name (JSON for non-strings), or None."""
    value: Any = row.get("verifier_metadata") or {}
    for part in dotted_key.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value)


def _breakdowns(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    """turn_score per value of each key in `keys` that some row's verifier_metadata carries (rows without it count
    as "none"). Keys no row carries add nothing."""
    out: dict[str, Any] = {}
    for key in keys:
        values = [_metadata_value(r, key) for r in rows]
        if all(v is None for v in values):
            continue
        passed: dict[str, list[bool]] = defaultdict(list)
        for r, v in zip(rows, values):
            passed["none" if v is None else v].append(r.get("reward") == 1.0)
        out[f"turn_score_by_{key.replace('.', '_')}"] = {g: sum(p) / len(p) for g, p in sorted(passed.items())}
    return out
