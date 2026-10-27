# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``simple_agent`` playing one whole agent turn against the ``agentic_turn_simulation`` resources server (see
README). Only its episode differs:

* the row's stop tools and tool-call protocol come from ``/harness``;
* every tool call, unparseable arguments and all, goes to ``/simulate_tool_call`` with the turn so far (an error
  reply is retried, then becomes the resources server's simulator-failure output, which ``/verify`` masks);
* the output budget doubles after a cut-off reply and shrinks to fit the context window;
* Cline rows (``cline_xml``) call tools as XML in the text, get all results back in one user message, and get
  Cline's "no tools used" error for a reply without a call.

``response.metadata.stop_reason`` says what ended the turn (the resources server's ``masked_stop_reasons`` match
these strings; README "What ends the turn" describes each): ``final_message`` (a reply without a tool call;
``empty_output`` when it has no message either), the stop tool's name (the call is kept, not run; the reply's
other calls run first), ``max_steps``, ``time_cap`` (``max_time_seconds``), ``context_overflow``,
``max_output_tokens`` (cut off once the budget cannot grow, with a tool call or nothing visible; a cut-off reply
that shows only text is kept) or, for Cline rows, ``mistake_limit`` (too many replies in a row without a tool call).
``response.usage`` sums the kept replies' tokens (the ones the trajectory references);
``response.metadata.model_calls`` lists every reply, discarded ones too (README "Token usage").
"""

import json
import logging
import re
from asyncio import sleep
from dataclasses import dataclass, field
from time import monotonic, perf_counter, time
from typing import Any, NamedTuple, TypedDict

from pydantic import Field, ValidationError

from nemo_gym._checkpoint.agent import Activation
from nemo_gym.base_responses_api_agent import AgentCloseSessionResponse, AgentSeedSessionRequest, AgentSessionState
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseUsage,
    accumulate_response_usage,
)
from nemo_gym.rollout_observability import (
    AgentInvocation,
    ModelCallRef,
    ObservationGap,
    TrajectoryRecord,
    TrajectoryToolCall,
    TrajectoryTurn,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from nemo_gym.tool_access import DirectHTTPToolAccess
from resources_servers.agentic_turn_simulation import cline_xml
from resources_servers.agentic_turn_simulation.simulator import simulator_failure
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig, _cookies


LOG = logging.getLogger(__name__)

# How the row's agent calls tools (/harness): function calls, or Cline's XML in the reply text.
NATIVE = "native"
CLINE_XML = "cline_xml"

# vLLM's context-overflow errors: "maximum context length is W tokens ... (P in the messages, C in the completion)",
# "... your prompt contains P input tokens" and "... your request has P input tokens".
_CONTEXT_WINDOW = re.compile(r"maximum context length is (\d+)")
_PROMPT_TOKENS = (
    re.compile(r"\((\d+) in the messages"),
    re.compile(r"prompt contains (\d+) input tokens"),
    re.compile(r"request has (\d+) input tokens"),
)


class AgenticTurnSimulationAgentConfig(SimpleAgentConfig):
    # Required (optional in SimpleAgentConfig): every tool call goes to its /simulate_tool_call.
    resources_server: ResourcesServerRef
    # Kept model steps per turn (truncation and overflow retries are not steps). 0: no limit.
    max_steps: int = 50
    # Checked before every model call. 0 or null: no limit.
    max_time_seconds: float | None = 7200.0
    # A call to one of these ends the turn unrun (the question goes to the user). Used when the resources server has
    # no session for the request; /harness names the row's own.
    stop_tools: list[str] = Field(default_factory=lambda: ["ask_user"])
    # On output-budget truncation, retry the step with double the budget up to this cap. 0 disables.
    max_output_tokens_cap: int = 65_536
    # Smallest output budget worth retrying with after shrinking it to fit the context window.
    min_output_tokens: int = 2_048
    # Kept free when shrinking the output budget to fit the context window.
    context_margin_tokens: int = 256
    # cline_xml rows: replies in a row without a tool call before the turn ends (Cline's maxConsecutiveMistakes).
    cline_max_consecutive_mistakes: int = cline_xml.MAX_CONSECUTIVE_MISTAKES
    # /simulate_tool_call attempts per tool call. An error reply (not 2xx or 3xx) is retried after 1, 2, 4, ... s;
    # once every attempt has failed, the call's output is the resources server's simulator-failure output, so /verify
    # masks the turn (simulator_failed) rather than judging a turn that got an error body as a tool result.
    simulate_max_attempts: int = Field(default=3, ge=1)


def context_overflow_room(detail: str, *, margin: int) -> tuple[bool, int | None]:
    """(is a context overflow, the output budget that would fit or None) from a model server error body."""
    low = detail.lower()
    overflow = "context length" in low or "maximum context" in low or ("maximum" in low and "token" in low)
    if not overflow:
        return False, None
    window = _CONTEXT_WINDOW.search(low)
    prompt = next((m for m in (p.search(low) for p in _PROMPT_TOKENS) if m), None)
    if window and prompt:
        return True, int(window.group(1)) - int(prompt.group(1)) - margin
    return True, None


class ModelCallRecord(TypedDict):
    """One model reply of the turn, as ``response.metadata.model_calls`` lists it (README "Token usage"). Model
    server errors (a context overflow) return no reply and are not listed."""

    # The reply's usage; None when the model server reported none.
    input_tokens: int | None
    output_tokens: int | None
    # Its output budget cut the reply off: the model server marked it incomplete, or its output tokens reached the
    # budget the agent requested.
    cut_off: bool
    # The reply became a step of the turn. A cut-off reply that was retried with a larger budget, or that ended the
    # turn as max_output_tokens, was not.
    kept: bool


class _Call(NamedTuple):
    """A tool call as /simulate_tool_call takes it: the arguments are the raw string, parseable or not."""

    call_id: str
    name: str
    arguments: str


def _native_calls(output: list[Any]) -> list[_Call]:
    return [_Call(o.call_id, o.name, o.arguments or "") for o in output if o.type == "function_call"]


def _xml_calls(output: list[Any], step: int) -> list[_Call]:
    """The Cline calls in the step's visible text. XML calls carry no ids: the k-th gets `xml_<step>_<k>`."""
    text = "".join(
        part.text
        for item in output
        if item.type == "message" and item.role == "assistant"
        for part in item.content
        if part.type == "output_text"
    )
    return [
        _Call(f"xml_{step}_{k}", c.name, json.dumps(c.params, ensure_ascii=False))
        for k, c in enumerate(cline_xml.calls(text), start=1)
    ]


def _user_message(*texts: str) -> NeMoGymEasyInputMessage:
    """A user message in Cline's format: a list of text parts."""
    return NeMoGymEasyInputMessage(role="user", content=[{"type": "input_text", "text": t} for t in texts])


def _no_tools_used(items: list[Any]) -> NeMoGymEasyInputMessage:
    """Cline's answer to a reply without a tool call, for the turn so far."""
    dumped = [i.model_dump(mode="json") if hasattr(i, "model_dump") else i for i in items]
    return _user_message(*cline_xml.no_tools_used(dumped))


def _result_items(protocol: str, calls: list[_Call], results: list[str]) -> list[Any]:
    """The results of a reply's calls as the agent sends them back: a function_call_output per call, or for Cline one
    user message with all of them (the resources server frames each result as Cline does)."""
    if protocol == CLINE_XML:
        return [_user_message("\n".join(results))] if results else []
    return [
        NeMoGymFunctionCallOutput(type="function_call_output", call_id=call.call_id, output=result)
        for call, result in zip(calls, results)
    ]


@dataclass(kw_only=True)
class _Trajectory:
    """simple_agent's trajectory records of one episode: a turn per kept step, a record per simulated call, the
    model-call references and the gaps. Records nothing unless collecting."""

    collecting: bool
    model_ref: ModelServerRef
    invocation_id: str
    task_id: str
    rollout_id: str
    turns: list[TrajectoryTurn] = field(default_factory=list)
    tool_calls: list[TrajectoryToolCall] = field(default_factory=list)
    model_calls: list[ModelCallRef] = field(default_factory=list)
    gaps: list[ObservationGap] = field(default_factory=list)
    # (when, the input) of the latest model call. turn() records it: call model_call() before each turn().
    asked: tuple[float, list[Any]] | None = None

    def model_call(self, question: list[Any]) -> None:
        if self.collecting:
            self.asked = (time(), question)

    def turn(self, step: int, response: NeMoGymResponse) -> None:
        if not self.collecting:
            return
        refs = [ModelCallRef(model_ref=self.model_ref, response_id=response.id)] if response.id else []
        if not refs:
            self.gaps.append(
                ObservationGap(
                    code="model_call_reference_unavailable", invocation_id=self.invocation_id, detail=f"turn:{step}"
                )
            )
        self.model_calls.extend(refs)
        timestamp, question = self.asked
        reasoning = [item.model_dump(mode="json") for item in response.output if item.type == "reasoning"]
        self.turns.append(
            TrajectoryTurn(
                invocation_id=self.invocation_id,
                task_id=self.task_id,
                rollout_id=self.rollout_id,
                turn_no=step,
                timestamp=timestamp,
                question=question,
                answer=[item for item in response.output if item.type != "reasoning"],
                reasoning_content=reasoning or None,
                step_count=len(self.tool_calls),
                model_calls=refs,
            )
        )

    def tool_call(self, call: _Call, started: tuple[float, float], output: str, status: int) -> None:
        """`started` is (time(), perf_counter()) when the call was sent."""
        if not self.collecting:
            return
        started_at, started_perf = started
        ok = 200 <= status < 400
        self.tool_calls.append(
            TrajectoryToolCall(
                invocation_id=self.invocation_id,
                tool_call_id=call.call_id,
                tool_name=call.name,
                started_at=started_at,
                completed_at=max(started_at, time()),
                duration_ms=(perf_counter() - started_perf) * 1000,
                timing_source="executor",
                status="completed" if ok else "failed",
                error_type=None if ok else f"http_{status}",
                output=output,
            )
        )
        self.turns[-1].step_count = len(self.tool_calls)

    def record(self, *, completed: bool, conversation: list[Any]) -> TrajectoryRecord | None:
        if not self.collecting:
            return None
        invocation = AgentInvocation(
            invocation_id=self.invocation_id,
            status="completed" if completed else "incomplete",
            model_calls=self.model_calls,
            conversation=conversation,
        )
        return TrajectoryRecord(
            task_id=self.task_id,
            rollout_id=self.rollout_id,
            invocations=[invocation],
            turns=self.turns,
            tool_calls=self.tool_calls,
            gaps=self.gaps,
        )


@dataclass
class _Episode:
    """The running state of one turn."""

    body: NeMoGymResponseCreateParamsNonStreaming
    # The output budget of the next model call. It carries over to later steps; once shrunk to fit the context
    # window it never grows again (`budget_fixed`).
    budget: int | None
    resources_cookies: Any
    trajectory: _Trajectory
    budget_fixed: bool = False
    # The turn so far: the model's output items, and what went back to it (tool results, Cline's messages).
    outputs: list[Any] = field(default_factory=list)
    # The latest model response, a discarded one's too: the episode's response is built on it.
    response: NeMoGymResponse | None = None
    # Summed over the kept model responses only: the trajectory references just those, and Gym's
    # rollout_token_count_mismatch check compares this usage with them. `model_calls` lists the discarded ones' tokens.
    usage: NeMoGymResponseUsage | None = None
    # Every model response, in order (discarded ones too).
    model_calls: list[ModelCallRecord] = field(default_factory=list)
    model_cookies: Any = None


class AgenticTurnSimulationAgent(SimpleAgent):
    config: AgenticTurnSimulationAgentConfig

    async def _create_episode(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        model_url_path: str,
        resources_server_cookies: Any = None,
        tool_access: DirectHTTPToolAccess | None = None,
        resources_prefix: str = "",
        in_session: bool = False,
        execute_tools: bool = True,
        invocation_id: str = "root",
        task_id: str = "unscoped",
        rollout_id: str = "unscoped",
        collect_trajectory: bool = False,
        activation: Activation | None = None,
    ) -> tuple[NeMoGymResponse, TrajectoryRecord | None, Any, Any]:
        # Only the legacy route reaches here: agent sessions are refused at seed (`_seed_agent_session_state`), so
        # `in_session`, `tool_access`, `resources_prefix` and `activation` are never set, and simple_agent's run() and
        # responses() refuse execute_tools=false.
        cfg = self.config
        body = self._strip_resource_only_metadata(body.model_copy(deep=True))
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]
        stop_tools, protocol = await self._row_harness(resources_server_cookies)
        ep = _Episode(
            body=body,
            budget=body.max_output_tokens,
            resources_cookies=resources_server_cookies,
            trajectory=_Trajectory(
                collecting=collect_trajectory,
                model_ref=cfg.model_server,
                invocation_id=invocation_id,
                task_id=task_id,
                rollout_id=rollout_id,
            ),
        )
        step = 0
        mistakes = 0  # cline_xml: consecutive replies without a tool call
        started = monotonic()

        while True:
            if cfg.max_steps and step >= cfg.max_steps:
                stop_reason = "max_steps"
                break
            if cfg.max_time_seconds and monotonic() - started > cfg.max_time_seconds:
                stop_reason = "time_cap"
                break
            reply = await self._model_step(ep, model_url_path)
            if reply is None:
                continue  # retried with the new budget: a retry is not a step
            if isinstance(reply, str):
                stop_reason = reply
                break
            step += 1
            ep.outputs.extend(reply.output)
            ep.trajectory.turn(step, reply)

            if protocol == CLINE_XML:
                calls = _xml_calls(reply.output, step)
                if not calls:
                    # Cline answers with its "no tools used" error and the model retries; at its limit of consecutive
                    # mistakes it stops to ask the user, which ends the turn.
                    mistakes += 1
                    if mistakes >= cfg.cline_max_consecutive_mistakes:
                        stop_reason = "mistake_limit"
                        break
                    ep.outputs.append(_no_tools_used([*body.input, *ep.outputs]))
                    continue
                mistakes = 0
            else:
                calls = _native_calls(reply.output)
                if not calls:
                    has_message = any(o.type == "message" and o.role == "assistant" for o in reply.output)
                    stop_reason = "final_message" if has_message else "empty_output"
                    break

            # A stop tool hands the turn to the user once the reply's other calls have run: every other native call
            # (the harness runs a reply's calls together), but only the Cline calls before it (Cline runs them in
            # order, and the question waits for the user). The stop call itself never runs.
            stop_call = next((call for call in calls if call.name in stop_tools), None)
            if stop_call is not None:
                if protocol == CLINE_XML:
                    calls = calls[: calls.index(stop_call)]
                else:
                    calls = [call for call in calls if call.name not in stop_tools]
            results = await self._simulate_calls(ep, calls, protocol)
            ep.outputs.extend(_result_items(protocol, calls, results))
            if stop_call is not None:
                stop_reason = stop_call.name
                break

        response = ep.response
        if response is None:  # no model call succeeded
            response = NeMoGymResponse(
                id="no_model_response",
                created_at=0.0,
                model=str(body.model or "unknown"),
                object="response",
                output=[],
                tools=[],
                parallel_tool_calls=False,
                tool_choice="auto",
            )
        response.output = ep.outputs
        response.usage = ep.usage
        # Responses metadata values are strings.
        response.metadata = {
            **(response.metadata or {}),
            "stop_reason": stop_reason,
            "steps": str(step),
            "elapsed_seconds": str(round(monotonic() - started)),
            "model_calls": json.dumps(ep.model_calls),
        }
        record = ep.trajectory.record(
            completed=stop_reason in ("final_message", *stop_tools), conversation=[*body.input, *ep.outputs]
        )
        # responses() copies both cookie jars onto its reply: an empty one, not None, before any reply set it.
        return response, record, ep.model_cookies or {}, ep.resources_cookies or {}

    async def _row_harness(self, cookies: Any) -> tuple[list[str], str]:
        """The row's stop tools and tool-call protocol from /harness. A bare /v1/responses (no resources cookies) has
        no session there (404): the config's stop tools and native calls. A 404 for cookies that /seed_session set
        (the call reached another resources-server worker, with num_workers > 1, or the server restarted) raises,
        since a Cline row run as native would be scored, not masked."""
        http = await self.server_client.post(
            server_name=self.config.resources_server.name, url_path="/harness", cookies=cookies
        )
        if http.status == 404:
            if cookies:
                detail = (await http.content.read()).decode(errors="replace")
                raise RuntimeError(f"/harness has no session for this episode's resources cookies: {detail[:200]}")
            return self.config.stop_tools, NATIVE
        await raise_for_status(http)
        harness = await get_response_json(http)
        stop_tools, protocol = harness.get("stop_tools"), harness.get("protocol")
        if protocol not in (None, NATIVE, CLINE_XML):
            raise ValueError(f"/harness named an unknown tool-call protocol: {protocol!r}")
        return (stop_tools if isinstance(stop_tools, list) else self.config.stop_tools), protocol or NATIVE

    async def _model_step(self, ep: _Episode, model_url_path: str) -> NeMoGymResponse | str | None:
        """One model call for the next step: the reply to keep as the step, the stop reason when the turn ends here,
        or None to retry the step with a new output budget (shrunk to what fits the context window, or doubled after
        a reply the budget cut off)."""
        cfg = self.config
        request = ep.body.model_copy(update={"input": ep.body.input + ep.outputs, "max_output_tokens": ep.budget})
        ep.trajectory.model_call(request.input)
        http = await self.server_client.post(
            server_name=cfg.model_server.name, url_path=model_url_path, json=request, cookies=ep.model_cookies
        )
        if http.status >= 400:
            detail = (await http.content.read()).decode(errors="replace")
            overflow, room = context_overflow_room(detail, margin=cfg.context_margin_tokens)
            if not overflow:
                raise RuntimeError(f"model server returned HTTP {http.status}: {detail[:500]}")
            # Only a smaller budget that is still worth generating with can help; an overflow never grows it.
            current = ep.budget or 1 << 30  # no budget: any room is smaller
            if room is None or room < cfg.min_output_tokens or room >= current:
                return "context_overflow"
            ep.budget, ep.budget_fixed = room, True
            return None

        payload = await get_response_json(http)
        ep.model_cookies = http.cookies
        try:
            reply = NeMoGymResponse.model_validate(payload)
        except ValidationError as e:
            raise RuntimeError(f"Received an invalid response from model server: {json.dumps(payload)}") from e
        ep.response = reply
        usage, reply.usage = reply.usage, None
        # vLLM reports a reply cut off inside a tool call as finished ("tool_calls", not "length"): its parser drops
        # the unfinished argument and keeps the call. Only the token count shows that cut. A reply that ended on
        # exactly its last allowed token counts as cut off too, since its usage cannot tell it apart: that costs one
        # retry (and at the cap, a turn that called a tool).
        budget = request.max_output_tokens
        cut_off = reply.incomplete_details is not None or (
            budget is not None and usage is not None and usage.output_tokens >= budget
        )
        outcome = reply if not cut_off else self._cut_off_outcome(ep, reply)
        kept = outcome is reply
        if kept:
            ep.usage = accumulate_response_usage(ep.usage, usage)
        ep.model_calls.append(
            ModelCallRecord(
                input_tokens=usage.input_tokens if usage is not None else None,
                output_tokens=usage.output_tokens if usage is not None else None,
                cut_off=cut_off,
                kept=kept,
            )
        )
        return outcome

    def _cut_off_outcome(self, ep: _Episode, reply: NeMoGymResponse) -> NeMoGymResponse | str | None:
        """What `_model_step` returns for a reply the output budget cut off: None to retry with double the budget
        while it may grow, else the reply when it shows text and no tool call (kept as a step: a native final message
        is judged as written, Cline runs the XML calls it completed), else "max_output_tokens" (the turn ends: the last
        call's arguments are cut, or were dropped by the model server's parser, so running it would score the budget,
        not the policy)."""
        cap = self.config.max_output_tokens_cap
        if not ep.budget_fixed and cap and ep.budget and ep.budget < cap:
            ep.budget = min(ep.budget * 2, cap)
            return None
        if any(o.type == "function_call" for o in reply.output):
            return "max_output_tokens"
        # An empty message shows nothing: a model server's placeholder for a failed request is one.
        shows_text = any(
            o.type == "message" and any(p.type == "output_text" and p.text.strip() for p in o.content)
            for o in reply.output
        )
        return reply if shows_text else "max_output_tokens"

    async def _simulate_calls(self, ep: _Episode, calls: list[_Call], protocol: str) -> list[str]:
        """Each call's output from /simulate_tool_call, in order. The simulator sees the turn so far without reasoning,
        then the reply's earlier results as the agent will send them back."""
        results: list[str] = []
        for call in calls:
            turn = [*ep.outputs, *_result_items(protocol, calls, results)]
            dumped = [o.model_dump(mode="json", exclude_none=True) for o in turn if o.type != "reasoning"]
            results.append(await self._simulate(ep, call, dumped))
        return results

    async def _simulate(self, ep: _Episode, call: _Call, turn: list[dict[str, Any]]) -> str:
        """One call's output. A successful reply's output is the tool's, an error the simulated tool reports included.
        An error reply from the resources server is retried (`simulate_max_attempts`); once every attempt has failed,
        the output is the resources server's simulator-failure output, which /verify masks as simulator_failed."""
        started = (time(), perf_counter())
        attempts = self.config.simulate_max_attempts
        for attempt in range(1, attempts + 1):
            http = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/simulate_tool_call",
                json={"call_id": call.call_id, "name": call.name, "arguments": call.arguments, "turn": turn},
                cookies=ep.resources_cookies,
            )
            # Merged, as simple_agent does: an error reply sets no cookie, and the session must survive it.
            ep.resources_cookies = {**dict(ep.resources_cookies or {}), **_cookies(http)}
            if 200 <= http.status < 400:
                output = (await get_response_json(http)).get("output", "")
                break
            detail = (await http.content.read()).decode(errors="replace")
            LOG.warning(
                "/simulate_tool_call for %s failed (attempt %d of %d): HTTP %d: %s",
                call.name,
                attempt,
                attempts,
                http.status,
                detail[:300],
            )
            if attempt < attempts:
                await sleep(2 ** (attempt - 1))
        else:
            output = simulator_failure(f"HTTP {http.status}: {detail}")
        ep.trajectory.tool_call(call, started, output, http.status)
        return output

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> AgentSessionState:
        """Environment Server agent sessions are refused: the tool calls need the session ``/seed_session`` opens on
        the resources server."""
        raise NotImplementedError("agentic_turn_simulation_agent supports only the legacy /run route")

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        """Refused with the seed: no agent session is ever bound here, so Simple Agent's close has nothing to
        release."""
        raise NotImplementedError("agentic_turn_simulation_agent supports only the legacy /run route")


if __name__ == "__main__":
    AgenticTurnSimulationAgent.run_webserver()
