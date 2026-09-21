# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ToolAlignBench agent harness.

Port of ``callOpenRouterWithPromptBasedTools`` (``runner/src/openrouter/prompt-based.ts``) and the
per-document outer loop in ``executeScenario`` (``runner/src/run.ts``) from
https://github.com/aryankeluskar/ToolAlignBench (MIT, (c) 2026 Aryan Keluskar).

``simple_agent`` cannot run this benchmark. ToolAlignBench never sends a native ``tools`` array --
tools are documented in the system prompt and calls are parsed back out of the reply text -- so
``simple_agent``, which only reacts to native ``function_call`` items, would treat the very first
text-only reply as terminal and end the rollout after one model call.

The episode shape this reproduces:

* One rollout walks **four escalating documents**, each appended as a new ``user`` turn.
* Within a document, up to ``max_steps`` model calls: parse tool calls out of the reply, execute
  them, hand the results back, repeat until the model replies without calling anything.
* Tools are **stubs** -- ``{"success": true, "message": "<name> executed successfully"}`` -- which
  ignore their arguments and hold no state. They execute here rather than in the resources server
  because that is where upstream runs them, and it keeps the verifier a pure function of the
  stored trace (so ``gym eval reverify`` works).
* Between documents the tool-result turns are **discarded**: only the user documents and the
  assistant's prose (with tool-call markup stripped) carry forward. This is upstream behaviour and
  it is load-bearing -- the model cannot see what it did two documents ago.

Each parsed text call is normalised into a real ``function_call`` item plus a
``function_call_output`` in the emitted trace, so the verifier reads an ordinary Gym trace.
"""

from __future__ import annotations

import json
from time import monotonic, time
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

from fastapi import Request, Response
from pydantic import ConfigDict, Field, ValidationError

from nemo_gym.base_resources_server import (
    AggregateMetrics,
    AggregateMetricsRequest,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
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
from responses_api_agents.toolalignbench_agent.xml_tool_calls import (
    TOOL_CALL_FORMAT_XML,
    ExtractedToolCall,
    create_tool_call_fingerprint,
    detect_tool_call_loop,
    extract_tool_calls,
    format_tool_result,
    has_unparsed_tool_call_markup,
    normalize_arguments,
    strip_tool_call_markers,
)


# Synthetic assistant turns upstream substitutes when it stops a document early. Verbatim, because
# they enter the conversation the model sees on the next document.
LOOP_DETECTED_MESSAGE = "Tool call loop detected. Stopping execution to prevent timeout."
ALL_DUPLICATES_MESSAGE = "All requested tool calls have already been executed. No new actions to take."

TOOL_RESULTS_PREFIX = "Tool execution results:\n\n"


class ToolAlignBenchAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef

    max_steps: int = Field(
        default=10,
        description="Model calls allowed per document. Upstream passes maxSteps: 10.",
    )
    timeout_seconds: float = Field(
        default=600.0,
        description=(
            "Wall-clock budget for the whole episode, matching upstream's TIMEOUT_SECONDS. On "
            "expiry the episode stops early and the rollout is flagged rather than failed."
        ),
    )
    tool_call_format: str = Field(
        default=TOOL_CALL_FORMAT_XML,
        description="Tool-call syntax to parse and document. Upstream hardcodes 'xml'.",
    )
    parse_reasoning_text: bool = Field(
        default=False,
        description=(
            "Also scan reasoning items for tool calls. Upstream ignores reasoning entirely, so "
            "enabling this can only raise measured tool use and breaks comparability. Requires "
            "uses_reasoning_parser on the model server for the text to reach us at all."
        ),
    )
    harvest_native_tool_calls: bool = Field(
        default=True,
        description=(
            "Count native function_call items the model emits unprompted as tool calls. Upstream "
            "cannot see these (it only reads message content), so a model that emits native calls "
            "would otherwise look perfectly aligned while acting."
        ),
    )


class ToolAlignBenchAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    domain: Optional[str] = None
    # Documents 2..N. Document 1 is the user turn already in responses_create_params.input.
    remaining_documents: List[str] = Field(default_factory=list)
    # Tools offered for this domain. A call to anything else gets upstream's "not found" stub.
    tool_names: List[str] = Field(default_factory=list)


class ToolAlignBenchAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class ToolAlignBenchAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


def _message_text(item: Any) -> str:
    """Concatenate the text parts of one assistant message item."""
    content = getattr(item, "content", None)
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(part.text for part in content if getattr(part, "type", None) == "output_text")


def _reasoning_text(item: Any) -> str:
    """Concatenate a reasoning item's summary text."""
    summary = getattr(item, "summary", None) or []
    texts = [getattr(part, "text", "") or "" for part in summary]
    content = getattr(item, "content", None) or []
    texts.extend(getattr(part, "text", "") or "" for part in content)
    return "\n".join(text for text in texts if text)


def _assistant_text(output: Sequence[Any], include_reasoning: bool) -> str:
    """The reply text to parse tool calls out of.

    Upstream reads ``choice.message.content`` only, so reasoning is excluded by default.
    """
    parts = []
    for item in output:
        item_type = getattr(item, "type", None)
        if item_type == "message" and getattr(item, "role", None) == "assistant":
            parts.append(_message_text(item))
        elif include_reasoning and item_type == "reasoning":
            parts.append(_reasoning_text(item))
    return "\n".join(part for part in parts if part)


def _text_message_item(text: str) -> NeMoGymResponseOutputMessage:
    """Build an assistant message trace item."""
    return NeMoGymResponseOutputMessage(
        id=f"msg_{uuid4().hex}",
        content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
        role="assistant",
        status="completed",
        type="message",
    )


def _stub_tool_result(tool_name: str, allowed_tool_names: Sequence[str]) -> Dict[str, Any]:
    """Execute one stub tool. Arguments are ignored, exactly as upstream ignores them.

    An unknown name yields upstream's error payload rather than raising, so a hallucinated tool
    name stays visible to the model as a failed call.
    """
    if not allowed_tool_names or tool_name in allowed_tool_names:
        return {"success": True, "message": f"{tool_name} executed successfully"}
    return {"success": False, "error": f"Tool '{tool_name}' not found"}


def _task_id_from_run(body: ToolAlignBenchAgentRunRequest) -> str:
    """Resolve the task id for the trajectory.

    This must mirror ``_trajectory_identity`` in ``nemo_gym/rollout_collection.py`` exactly -- same
    keys, same order. The collector rebuilds the identity from the row and records a
    ``producer_trajectory_identity_mismatch`` gap when the agent disagrees, so reaching for the
    more readable ToolAlignBench row ``id`` here would poison every rollout with a spurious gap.
    The readable id stays available as the row's own ``id`` field.
    """
    extra = body.model_extra or {}
    for key in ("task_id", "problem_id", "instance_id", "_ng_task_index"):
        if extra.get(key) is not None:
            return str(extra[key])
    return "unknown"


class ToolAlignBenchAgent(SimpleResponsesAPIAgent):
    config: ToolAlignBenchAgentConfig

    async def _call_model(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        cookies: Any,
    ) -> Tuple[NeMoGymResponse, Any]:
        """One model call. Upstream sends only {model, messages} -- no sampling parameters."""
        response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path="/v1/responses",
            json=body,
            cookies=cookies,
        )
        await raise_for_status(response)
        response_json = await get_response_json(response)
        try:
            parsed = NeMoGymResponse.model_validate(response_json)
        except ValidationError as e:
            raise RuntimeError(f"Received an invalid response from model server: {json.dumps(response_json)}") from e
        return parsed, response.cookies

    def _harvest_native_tool_calls(self, output: Sequence[Any]) -> Tuple[List[Any], List[ExtractedToolCall]]:
        """Split native ``function_call`` items out of a model response.

        They are removed from the passthrough items and re-added as parsed calls, so they are
        counted exactly once.
        """
        if not self.config.harvest_native_tool_calls:
            return list(output), []
        kept: List[Any] = []
        harvested: List[ExtractedToolCall] = []
        for item in output:
            if getattr(item, "type", None) == "function_call":
                arguments = item.arguments or "{}"
                harvested.append(
                    ExtractedToolCall(
                        name=item.name,
                        arguments=arguments,
                        raw_arguments=normalize_arguments(arguments),
                        source="native",
                    )
                )
            else:
                kept.append(item)
        return kept, harvested

    async def _run_episode(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        row_metadata: Dict[str, Any],
        initial_cookies: Any,
        *,
        task_id: str = "unscoped",
        rollout_id: str = "unscoped",
        collect_trajectory: bool = False,
    ) -> Tuple[NeMoGymResponse, Any, Dict[str, Any], Optional[TrajectoryRecord]]:
        """Walk every document, returning the aggregated trace, cookies, diagnostics, trajectory.

        ``collect_trajectory`` mirrors ``simple_agent``: the trajectory is only assembled when
        model-call capture is on and the rollout has an id, since it exists to be joined against
        the captured request/response payloads. Each document becomes its own ``AgentInvocation``
        -- ``TrajectoryTurn`` forbids extra fields, so the per-document grouping that makes this
        benchmark legible has to live in the invocation id rather than on the turn.
        """
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        allowed_tool_names: List[str] = list(row_metadata.get("tool_names") or [])
        remaining_documents: List[str] = list(row_metadata.get("remaining_documents") or [])
        tool_call_format = self.config.tool_call_format

        # Turns that survive across documents: the user documents plus the assistant's prose.
        carried_input: List[Any] = list(body.input)
        trace_items: List[Any] = []
        usage = None
        model_cookies = initial_cookies
        last_response: Optional[NeMoGymResponse] = None

        deadline = monotonic() + self.config.timeout_seconds
        counters = {
            "num_documents": 1 + len(remaining_documents),
            "num_documents_completed": 0,
            "num_model_calls": 0,
            "num_tool_calls_executed": 0,
            "num_duplicate_tool_calls_skipped": 0,
            "num_unknown_tool_calls": 0,
            "num_unparsed_tool_call_replies": 0,
            "num_native_tool_calls": 0,
        }
        flags = {"episode_timed_out": False, "hit_max_steps": False, "model_incomplete": False}

        invocations: List[AgentInvocation] = []
        turns: List[TrajectoryTurn] = []
        tool_records: List[TrajectoryToolCall] = []
        trajectory_gaps: List[ObservationGap] = []

        for document_index in range(1, counters["num_documents"] + 1):
            if monotonic() > deadline:
                flags["episode_timed_out"] = True
                break

            if document_index > 1:
                document = remaining_documents[document_index - 2]
                carried_input.append(NeMoGymEasyInputMessage(role="user", content=document))

            # Working history for this document: the carried turns plus this document's own
            # tool-call and tool-result turns, which are dropped again once the document ends.
            working_input: List[Any] = list(carried_input)
            assistant_prose: List[str] = []
            executed_fingerprints: set[str] = set()
            recent_fingerprints: List[str] = []

            invocation_id = f"document-{document_index}"
            document_model_calls: List[ModelCallRef] = []
            document_turn_no = 0
            document_status = "completed"

            for step in range(1, self.config.max_steps + 1):
                if monotonic() > deadline:
                    flags["episode_timed_out"] = True
                    document_status = "incomplete"
                    break

                turn_timestamp = time()
                model_response, model_cookies = await self._call_model(
                    body.model_copy(update={"input": working_input}),
                    model_cookies,
                )
                counters["num_model_calls"] += 1
                last_response = model_response
                usage = accumulate_response_usage(usage, model_response.usage)

                if collect_trajectory:
                    document_turn_no += 1
                    turn_model_calls: List[ModelCallRef] = []
                    if model_response.id:
                        model_call_ref = ModelCallRef(
                            model_ref=self.config.model_server, response_id=model_response.id
                        )
                        document_model_calls.append(model_call_ref)
                        turn_model_calls.append(model_call_ref)
                    else:
                        trajectory_gaps.append(
                            ObservationGap(
                                code="model_call_reference_unavailable",
                                invocation_id=invocation_id,
                                detail=f"turn:{document_turn_no}",
                            )
                        )
                    reasoning = [
                        item.model_dump(mode="json")
                        for item in model_response.output
                        if getattr(item, "type", None) == "reasoning"
                    ] or None
                    turns.append(
                        TrajectoryTurn(
                            invocation_id=invocation_id,
                            task_id=task_id,
                            rollout_id=rollout_id,
                            turn_no=document_turn_no,
                            timestamp=turn_timestamp,
                            question=working_input,
                            answer=[
                                item for item in model_response.output if getattr(item, "type", None) != "reasoning"
                            ],
                            reasoning_content=reasoning,
                            step_count=len(tool_records),
                            model_calls=turn_model_calls,
                        )
                    )

                passthrough_items, native_calls = self._harvest_native_tool_calls(model_response.output)
                counters["num_native_tool_calls"] += len(native_calls)
                trace_items.extend(passthrough_items)

                reply_text = _assistant_text(model_response.output, self.config.parse_reasoning_text)
                extracted = native_calls + extract_tool_calls(reply_text, tool_call_format)

                if model_response.incomplete_details is not None or model_response.error is not None:
                    flags["model_incomplete"] = True
                    assistant_prose.append(reply_text)
                    break

                if not extracted:
                    # The normal exit: the model answered without calling anything.
                    if has_unparsed_tool_call_markup(reply_text):
                        counters["num_unparsed_tool_call_replies"] += 1
                    assistant_prose.append(reply_text)
                    break

                unique_calls: List[ExtractedToolCall] = []
                for call in extracted:
                    fingerprint = create_tool_call_fingerprint(call.name, call.raw_arguments)
                    if fingerprint in executed_fingerprints:
                        counters["num_duplicate_tool_calls_skipped"] += 1
                        continue
                    unique_calls.append(call)
                    executed_fingerprints.add(fingerprint)
                    recent_fingerprints.append(fingerprint)

                # Upstream checks for a loop before checking whether anything new remains, so the
                # order of these two branches decides which synthetic message is recorded.
                # Unreachable in practice, here and upstream: only *newly executed* fingerprints
                # are appended and each executes at most once per document, so no half-window can
                # repeat. Ported for parity; `detect_tool_call_loop` is unit-tested directly.
                if detect_tool_call_loop(recent_fingerprints):  # pragma: no cover
                    assistant_prose.append(LOOP_DETECTED_MESSAGE)
                    trace_items.append(_text_message_item(LOOP_DETECTED_MESSAGE))
                    break

                if not unique_calls:
                    assistant_prose.append(ALL_DUPLICATES_MESSAGE)
                    trace_items.append(_text_message_item(ALL_DUPLICATES_MESSAGE))
                    break

                assistant_prose.append(strip_tool_call_markers(reply_text, tool_call_format))
                # The model sees its own raw reply, markup included.
                working_input.append(NeMoGymEasyInputMessage(role="assistant", content=reply_text))

                tool_results: List[str] = []
                for index, call in enumerate(unique_calls):
                    call_id = f"call_{document_index}_{step}_{index}_{call.source}"
                    normalized_arguments = normalize_arguments(call.arguments)
                    result = _stub_tool_result(call.name, allowed_tool_names)
                    if not result["success"]:
                        counters["num_unknown_tool_calls"] += 1
                    counters["num_tool_calls_executed"] += 1

                    trace_items.append(
                        NeMoGymResponseFunctionToolCall(
                            arguments=json.dumps(normalized_arguments),
                            call_id=call_id,
                            name=call.name,
                            type="function_call",
                        )
                    )
                    trace_items.append(
                        NeMoGymFunctionCallOutput(
                            type="function_call_output",
                            call_id=call_id,
                            output=json.dumps(result),
                        )
                    )
                    tool_results.append(format_tool_result(call_id, call.name, result, tool_call_format))

                    if collect_trajectory:
                        # Stubs are fabricated in-process, so the timing is a formality; the
                        # value here is the tool name and output per turn. `timing_source` says
                        # "harness" rather than "executor" because nothing was really executed.
                        executed_at = time()
                        tool_records.append(
                            TrajectoryToolCall(
                                invocation_id=invocation_id,
                                tool_call_id=call_id,
                                tool_name=call.name,
                                started_at=executed_at,
                                completed_at=executed_at,
                                duration_ms=0.0,
                                timing_source="harness",
                                status="completed" if result["success"] else "failed",
                                error_type=None if result["success"] else "tool_not_found",
                                output=json.dumps(result),
                            )
                        )

                if collect_trajectory and turns:
                    turns[-1].step_count = len(tool_records)

                working_input.append(
                    NeMoGymEasyInputMessage(
                        role="user",
                        content=TOOL_RESULTS_PREFIX + "\n\n".join(tool_results),
                    )
                )

                if step == self.config.max_steps:
                    flags["hit_max_steps"] = True
                    document_status = "incomplete"

            if not flags["episode_timed_out"]:
                counters["num_documents_completed"] += 1

            if collect_trajectory:
                invocations.append(
                    AgentInvocation(
                        invocation_id=invocation_id,
                        status=document_status,
                        model_calls=document_model_calls,
                        # This document's full working history, tool-result turns included -- the
                        # conversation the model actually saw, before it is pruned for the next
                        # document.
                        conversation=list(working_input),
                    )
                )

            # Carry only the prose forward; the tool-call and tool-result turns are dropped.
            aggregated_prose = "\n\n".join(part for part in assistant_prose if part)
            carried_input.append(NeMoGymEasyInputMessage(role="assistant", content=aggregated_prose))

            if flags["episode_timed_out"]:
                break

        if last_response is None:
            raise RuntimeError("ToolAlignBench episode produced no model calls")

        aggregated = last_response.model_copy(deep=True)
        aggregated.output = trace_items
        aggregated.usage = usage

        rollout_info: Dict[str, Any] = dict(counters)
        rollout_info.update(flags)

        trajectory = None
        if collect_trajectory:
            trajectory = TrajectoryRecord(
                task_id=task_id,
                rollout_id=rollout_id,
                invocations=invocations,
                turns=turns,
                tool_calls=tool_records,
                gaps=trajectory_gaps,
            )
        return aggregated, model_cookies, rollout_info, trajectory

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        """Run an episode from ``input`` alone -- i.e. document 1 only, with no carry-over."""
        aggregated, model_cookies, _, _ = await self._run_episode(
            body=body,
            row_metadata={},
            initial_cookies=request.cookies,
        )
        for key, value in (*request.cookies.items(), *model_cookies.items()):
            response.set_cookie(key, value)
        return aggregated

    async def run(
        self,
        request: Request,
        body: ToolAlignBenchAgentRunRequest,
    ) -> ToolAlignBenchAgentVerifyResponse:
        cookies = request.cookies

        seed_session_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_session_response)
        resources_server_cookies = seed_session_response.cookies

        rollout_id = self.rollout_id_from_run(body)
        collect_trajectory = self._model_call_capture_enabled() and isinstance(rollout_id, str)
        aggregated, _, rollout_info, trajectory = await self._run_episode(
            body=body.responses_create_params,
            row_metadata=body.model_dump(exclude={"responses_create_params"}),
            initial_cookies=cookies,
            task_id=_task_id_from_run(body),
            rollout_id=rollout_id or "unscoped",
            collect_trajectory=collect_trajectory,
        )

        verify_request = ToolAlignBenchAgentVerifyRequest.model_validate(
            body.model_dump() | {"response": aggregated.model_dump()}
        )
        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
            cookies=resources_server_cookies,
        )
        await raise_for_status(verify_response)
        verify_response_json = await get_response_json(verify_response)

        # Layer the verifier's fields over the request, then the harness diagnostics. Starting from
        # the request means the row is complete even if a verifier does not echo it back, and the
        # diagnostics are ints and bools so they also surface in `rollout_infos` when profiling.
        result = verify_request.model_dump() | verify_response_json | rollout_info
        if trajectory is not None:
            # `is_misaligned` is the verifier's own outcome, so it is the honest per-episode
            # resolution signal for the last turn.
            resolved = result.get("is_misaligned")
            if isinstance(resolved, bool) and trajectory.turns:
                trajectory.turns[-1].resolved = not resolved
            else:
                trajectory.gaps.append(ObservationGap(code="resolution_unavailable", invocation_id="document-1"))
            result["ng_trajectory"] = trajectory.model_dump(mode="json")
        return ToolAlignBenchAgentVerifyResponse.model_validate(result)

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        """Proxy aggregate_metrics to the resources server."""
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))


if __name__ == "__main__":
    ToolAlignBenchAgent.run_webserver()
