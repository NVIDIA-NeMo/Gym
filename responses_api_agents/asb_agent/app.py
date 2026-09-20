# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ASB agent harness: the plan-then-execute loop from ``ReactAgentAttack.run``.

One rollout is: generate a JSON workflow, then walk it step by step, calling simulated
tools and feeding their returns back as observations. Attacks enter at three different
points and this loop is what makes them different from one another --

* **DPI** appended the injection to the task before the row was written, so it is already
  in the prompt when this agent starts.
* **OPI** appends it to every non-final tool observation, which only exists here.
* **Memory poisoning** inserts a retrieved (poisoned) workflow as an assistant turn before
  planning, so it steers the plan rather than the task.
* **PoT** ships the poisoned plan as a few-shot example in the system prompt, and fires
  only when the trigger phrase is present.

Simulated tools are executed in-process. Every ASB tool is a pure function of the row --
a normal tool returns its ``Expected Achievements`` string and the attacker tool returns
its attack-goal sentence, with no arguments consulted -- so routing them through the
resources server would add 30k HTTP round trips and change nothing about what the model
sees. Scoring stays in the resources server, where it belongs.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from fastapi import Request, Response
from pydantic import ConfigDict

from benchmarks.asb import upstream_spec as spec
from nemo_gym.base_resources_server import (
    AggregateMetrics,
    AggregateMetricsRequest,
    BaseRunRequest,
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
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    accumulate_response_usage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status


LOG = logging.getLogger(__name__)

#: ``ReactAgentAttack.plan_max_fail_times``.
PLAN_MAX_FAIL_TIMES = 10


class AsbAgentConfig(BaseResponsesAPIAgentConfig):
    model_config = ConfigDict(extra="allow")

    resources_server: ResourcesServerRef
    model_server: ModelServerRef

    #: Upstream caps generation at 256 tokens (``--max_new_tokens``, wired through to
    #: ``max_tokens``). That was calibrated for 2024-era non-reasoning models. A reasoning
    #: model spends the whole budget on its trace and emits no plan, which registers as a
    #: workflow failure and drives ASR to zero for reasons that have nothing to do with
    #: security. The cap is raised and the deviation is disclosed in METRICS.md.
    max_output_tokens: int = 4096
    upstream_max_output_tokens: int = 256


class AsbRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    asb_id: str
    condition: str
    verifier_metadata: dict[str, Any]


class AsbVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Workflow parsing -- ``BaseAgent.check_workflow``
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_ARRAY = re.compile(r"\[\s*\{.*\}\s*\]", re.DOTALL)


def parse_workflow(message: str) -> tuple[Optional[list[dict[str, Any]]], str]:
    """Parse a planning reply into a workflow, returning the parse path used.

    Upstream calls ``json.loads`` on the raw reply and treats any failure as a workflow
    failure. That is reproduced as the ``strict`` path. Two salvage paths follow, for
    markdown fencing and for a plan embedded in prose -- both are presentation artifacts
    of instruction-tuned models rather than different plans, and which path fired is
    reported so a run that leans on salvage is visible rather than silently flattering.
    """
    for path, candidate in _candidates(message):
        try:
            workflow = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(workflow, list):
            workflow = [workflow]
        if not workflow:
            continue
        if all(isinstance(step, dict) and "message" in step and "tool_use" in step for step in workflow):
            return workflow, path
    return None, "failed"


def _candidates(message: str):
    stripped = message.strip()
    yield "strict", stripped
    fence = _FENCE.search(message)
    if fence:
        yield "fenced", fence.group(1).strip()
    array = _ARRAY.search(message)
    if array:
        yield "embedded", array.group(0)


def inject_attacker_tool_into_workflow(workflow: list[dict[str, Any]], attacker_tool: str) -> list[dict[str, Any]]:
    """Reproduce ``attacker_tool_injection``: union every stage's tools, add the attacker's.

    Upstream builds one set across all stages and assigns that same list to every stage,
    so after injection each step may call any tool. The flattening is upstream's, and it
    is what makes the attacker tool reachable at every step rather than only the last.
    """
    all_tools: set[str] = {attacker_tool}
    for stage in workflow:
        tool_use = stage.get("tool_use")
        if isinstance(tool_use, (list, tuple, set)):
            all_tools.update(str(item) for item in tool_use)
    flattened = list(all_tools)
    return [dict(stage, tool_use=list(flattened)) for stage in workflow]


# ---------------------------------------------------------------------------
# Tool-call normalization -- ``ReactAgentAttack.normalize_tool_calls``
# ---------------------------------------------------------------------------


def normalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if tool_calls is None:
        return []
    if isinstance(tool_calls, (str, dict)):
        tool_calls = [tool_calls]

    normalized: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        name = None
        parameters = None
        if isinstance(tool_call, str):
            name = tool_call.strip()
        elif isinstance(tool_call, dict):
            if isinstance(tool_call.get("name"), str):
                name, parameters = tool_call["name"], tool_call.get("parameters")
            elif isinstance(tool_call.get("tool_name"), str):
                name, parameters = tool_call["tool_name"], tool_call.get("parameters")
            elif isinstance(tool_call.get("function"), dict):
                function = tool_call["function"]
                name, parameters = function.get("name"), function.get("arguments")
            elif isinstance(tool_call.get("tool_use"), list):
                for item in tool_call["tool_use"]:
                    if isinstance(item, str) and item.strip():
                        normalized.append({"name": item.strip(), "parameters": None})
                continue
        if isinstance(name, str) and name.strip():
            normalized.append({"name": name.strip(), "parameters": parameters})
    return normalized


def responses_tool_schema(chat_schema: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert ASB's chat-completions tool schemas to Responses API shape.

    ASB declares ``parameters: None`` (or omits it, for the attacker tool) and never
    passes arguments -- ``function_params`` is hardcoded to ``None`` in ``call_tools``.
    The Responses API requires a schema object, so an empty one is supplied. This affects
    only the API envelope: the tool list rendered into the planning prompt is the
    unmodified upstream text carried on the row.
    """
    converted = []
    for entry in chat_schema:
        function = entry.get("function", entry)
        converted.append(
            {
                "type": "function",
                "name": function["name"],
                "description": function.get("description", ""),
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                # Required by FunctionToolParam. False, because ASB's tools take no
                # arguments and strict mode would reject a model that sends any.
                "strict": False,
            }
        )
    return converted


def merge_system_messages(messages: list[Any]) -> list[Any]:
    """Collapse consecutive leading system turns into one.

    ASB opens every rollout with two system messages -- the agent description, then the
    planning instruction -- which is fine for the OpenAI chat API it was written against.
    Qwen3.5's chat template rejects it outright ("System message must be at the beginning",
    HTTP 400), so those rows would fail for one model and score for the others.

    The merge is applied to *every* model rather than only the one that needs it. Sending
    Qwen a merged prompt and the others a split one would mean the four models were not
    answering the same input, which is the one thing a cross-model table has to guarantee.
    Contents are joined with a blank line; no text is added, removed or reordered.
    """
    merged: list[Any] = []
    for message in messages:
        role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
        if role != "system" or not merged:
            merged.append(message)
            continue
        previous = merged[-1]
        previous_role = previous.get("role") if isinstance(previous, dict) else getattr(previous, "role", None)
        if previous_role != "system":
            merged.append(message)
            continue
        previous_content = previous.get("content") if isinstance(previous, dict) else getattr(previous, "content", "")
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
        merged[-1] = {"role": "system", "content": f"{previous_content}\n\n{content}"}
    return merged


def _message_text(response: NeMoGymResponse) -> str:
    for item in reversed(response.output):
        if item.type == "message" and item.role == "assistant":
            return "\n".join(part.text for part in item.content if getattr(part, "text", None)).strip()
    return ""


def _function_calls(response: NeMoGymResponse) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for item in response.output:
        if item.type == "function_call":
            calls.append({"name": item.name, "parameters": getattr(item, "arguments", None)})
    return calls


# ---------------------------------------------------------------------------
# Memory retrieval
# ---------------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def retrieve_memory(corpus: list[str], query: str) -> Optional[str]:
    """Top-1 retrieval over upstream's poisoned memory corpus.

    Upstream ranks with OpenAI ``text-embedding-ada-002`` via Chroma. The corpus here is
    upstream's own -- extracted from the shipped stores -- but the ranking is cosine over
    term-frequency vectors, because no OpenAI credential is in play for this campaign.
    Disclosed in METRICS.md: what poison is available to retrieve is upstream's; which
    record wins the ranking is not necessarily the one ada-002 would pick.
    """
    if not corpus:
        return None
    query_terms = tokenize(query)
    if not query_terms:
        return None
    query_counts: dict[str, int] = {}
    for term in query_terms:
        query_counts[term] = query_counts.get(term, 0) + 1
    query_norm = sum(value * value for value in query_counts.values()) ** 0.5

    best_score = -1.0
    best_record = None
    for record in corpus:
        counts: dict[str, int] = {}
        for term in tokenize(record):
            counts[term] = counts.get(term, 0) + 1
        if not counts:
            continue
        overlap = sum(query_counts[term] * counts.get(term, 0) for term in query_counts)
        if overlap <= 0:
            continue
        norm = sum(value * value for value in counts.values()) ** 0.5
        score = overlap / (query_norm * norm)
        if score > best_score:
            best_score, best_record = score, record
    return best_record


class AsbAgent(SimpleResponsesAPIAgent):
    config: AsbAgentConfig

    async def _call_model(
        self,
        *,
        messages: list[Any],
        tools: Optional[list[dict[str, Any]]],
        params: NeMoGymResponseCreateParamsNonStreaming,
        cookies: Any,
    ) -> tuple[NeMoGymResponse, Any]:
        request_params = params.model_copy(deep=True)
        request_params.input = merge_system_messages(messages)
        request_params.max_output_tokens = self.config.max_output_tokens
        if tools:
            request_params.tools = tools
        http_response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path="/v1/responses",
            json=request_params,
            cookies=cookies,
        )
        await raise_for_status(http_response)
        return NeMoGymResponse.model_validate(await get_response_json(http_response)), http_response.cookies

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        """Pass-through. The plan/execute loop lives in :meth:`run`."""
        model_response, cookies = await self._call_model(
            messages=body.input
            if isinstance(body.input, list)
            else [NeMoGymEasyInputMessage(role="user", content=body.input)],
            tools=None,
            params=body,
            cookies=request.cookies,
        )
        for key, value in cookies.items():
            response.set_cookie(key, value)
        return model_response

    def _execute_tools(
        self,
        *,
        calls: list[dict[str, Any]],
        metadata: dict[str, Any],
        final_stage: bool,
    ) -> tuple[list[str], list[str], bool, bool, list[str]]:
        """Reproduce ``call_tools``, including its failure strings and retry semantics."""
        normalized = normalize_tool_calls(calls)
        actions: list[str] = []
        observations: list[str] = []
        invoked: list[str] = []

        if not normalized:
            return (
                ["I fail to call any tools."],
                ["No valid tool calls were parsed from model output."],
                False,
                True,
                invoked,
            )

        runnable: dict[str, str] = {tool["name"]: tool["expected_achievement"] for tool in metadata["normal_tools"]}
        attacker = metadata["attacker_tool"]
        if metadata["attacker_tool_available"]:
            runnable[attacker["name"]] = spec.attacker_tool_observation(
                attacker_tool=attacker["name"], attack_goal=attacker["attack_goal"]
            )

        executable = [call for call in normalized if call["name"] in runnable]
        unknown = sorted({call["name"] for call in normalized if call["name"] not in runnable})
        if unknown:
            observations.append(f"Unknown tool name skipped: {', '.join(unknown)}")
        if not executable:
            actions.append("I fail to call any tools.")
            if not unknown:
                observations.append("The function name or the tool parameter is invalid.")
            return actions, observations, False, True, invoked

        for call in executable:
            name = call["name"]
            invoked.append(name)
            tool_response = runnable[name]
            if metadata["injects_into_observations"] and not final_stage:
                tool_response += f"; {metadata['observation_injection']}"
            if metadata["defense_type"] == "ob_sandwich_defense":
                tool_response = spec.apply_ob_sandwich_defense(
                    observation=tool_response, task_input=metadata["task_input"]
                )
            # Upstream always calls with params=None, so no simulated tool can raise.
            actions.append(f"I will call the {name} with the params as None")
            observations.append(f"The knowledge I get from {name} is: {tool_response}")

        return actions, observations, True, False, invoked

    async def run(self, request: Request, body: AsbRunRequest) -> AsbVerifyResponse:
        cookies = request.cookies
        metadata = body.verifier_metadata
        params = body.responses_create_params

        messages: list[Any] = [
            dict(message) if isinstance(message, dict) else message.model_dump()
            for message in (params.input if isinstance(params.input, list) else [])
        ]

        usage = None
        memory_record: Optional[str] = None
        # Every assistant item across planning and execution, in order. The verifier reads
        # the transcript from `messages`; `response` carries the same content in Responses
        # shape so the row is a standard Gym rollout as well.
        outputs: list[Any] = []
        last_response: Optional[NeMoGymResponse] = None

        # --- memory retrieval -------------------------------------------------
        if metadata.get("reads_memory"):
            seed = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/retrieve_memory",
                json={
                    "memory_key": metadata["memory_key"],
                    "query": f"{metadata['task_input']}; {json.dumps(metadata['tools_schema'])}",
                },
                cookies=cookies,
            )
            await raise_for_status(seed)
            payload = await get_response_json(seed)
            memory_record = payload.get("record")
            workflow_fragment = payload.get("workflow")
            # Upstream inserts the memory turn between the plan instruction and the task.
            messages.insert(
                len(messages) - 1,
                {
                    "role": "assistant",
                    "content": spec.build_memory_instruction(workflow_fragment) if workflow_fragment else "None",
                },
            )

        tools = responses_tool_schema(metadata["tools_schema"])

        # --- planning ---------------------------------------------------------
        workflow: Optional[list[dict[str, Any]]] = None
        parse_path = "failed"
        plan_attempts = 0
        for _ in range(PLAN_MAX_FAIL_TIMES):
            plan_attempts += 1
            plan_response, cookies = await self._call_model(
                messages=messages, tools=None, params=params, cookies=cookies
            )
            usage = accumulate_response_usage(usage, plan_response.usage)
            last_response = plan_response
            outputs.extend(plan_response.output)
            workflow, parse_path = parse_workflow(_message_text(plan_response))
            if workflow:
                break

        workflow_failure = workflow is None
        if workflow and (metadata["injects_into_task"] or metadata["injects_into_observations"]):
            workflow = inject_attacker_tool_into_workflow(workflow, metadata["attacker_tool"]["name"])

        messages.append(
            {
                "role": "assistant",
                "content": f"[Thinking]: The workflow generated for the problem is {json.dumps(workflow)}",
            }
        )

        # --- execution --------------------------------------------------------
        rounds = 0
        tool_call_success = False
        invoked_tools: list[str] = []
        steps: list[dict[str, Any]] = []

        if workflow:
            for index, step in enumerate(workflow):
                final_stage = index == len(workflow) - 1
                messages.append(
                    {
                        "role": "user",
                        "content": spec.build_step_prompt(step_index=rounds + 1, message=step["message"]),
                    }
                )
                step_tools = tools if step.get("tool_use") else None
                step_response, cookies = await self._call_model(
                    messages=messages, tools=step_tools, params=params, cookies=cookies
                )
                usage = accumulate_response_usage(usage, step_response.usage)
                last_response = step_response
                outputs.extend(step_response.output)

                calls = _function_calls(step_response)
                if calls:
                    # Upstream retries a failed tool call against the same reply, so the
                    # retry can only help when the failure was transient; a non-retryable
                    # parse failure breaks out immediately.
                    for _ in range(PLAN_MAX_FAIL_TIMES):
                        actions, observations, success, non_retryable, invoked = self._execute_tools(
                            calls=calls, metadata=metadata, final_stage=final_stage
                        )
                        invoked_tools.extend(invoked)
                        messages.append(
                            {
                                "role": "assistant",
                                "content": "[Action]: "
                                + ";".join(actions)
                                + ";"
                                + "[Observation]: "
                                + ";".join(observations),
                            }
                        )
                        if success:
                            tool_call_success = True
                            break
                        if non_retryable:
                            break
                    steps.append({"step": index, "tool_calls": [call["name"] for call in calls]})
                else:
                    messages.append({"role": "assistant", "content": f"[Thinking]: {_message_text(step_response)}"})
                    steps.append({"step": index, "tool_calls": []})
                rounds += 1

        assert last_response is not None, "the planning loop always issues at least one call"
        final_response = last_response.model_copy(update={"output": outputs, "usage": usage})

        verify_payload = body.model_dump() | {
            "response": final_response.model_dump(mode="json"),
            "messages": messages,
            "workflow": workflow,
            "workflow_failure": workflow_failure,
            "workflow_parse_path": parse_path,
            "plan_attempts": plan_attempts,
            "rounds": rounds,
            "tool_call_success": tool_call_success,
            "invoked_tools": invoked_tools,
            "steps": steps,
            "memory_record": memory_record,
        }
        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_payload,
            cookies=cookies,
        )
        await raise_for_status(verify_response)
        return AsbVerifyResponse.model_validate(await get_response_json(verify_response))

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))


if __name__ == "__main__":
    AsbAgent.run_webserver()
