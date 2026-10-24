# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single canonical IF run contract; authoring is complete before this boundary."""

import re
from collections.abc import Mapping
from copy import deepcopy
from typing import Literal

from pydantic import Field, JsonValue, SerializerFunctionWrapHandler, StrictInt, model_serializer, model_validator

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from resources_servers.agentic_if.if_grading.contracts import Contract, EventReceipt, ResolvedCheck
from resources_servers.agentic_if.task_variants.builder import actor_input_digest, content_text, digest
from resources_servers.agentic_if.task_variants.catalog import BUILTIN_ARGUMENTS
from resources_servers.agentic_if.task_variants.guards import ReadFirstGuard
from resources_servers.agentic_if.task_variants.interfaces import installed_interfaces
from resources_servers.agentic_if.task_variants.tasks import backend_task_digest
from resources_servers.agentic_if.task_variants.toolsets import configurable_tools


class ToolConfiguration(Contract):
    """Installed implementations plus literal names and descriptions, never tool code."""

    tool_names: dict[str, str] | None = None
    tool_descriptions: dict[str, str] = Field(default_factory=dict)
    tool_parameter_descriptions: dict[str, dict[str, str]] = Field(default_factory=dict)
    tool_interfaces: dict[str, str] = Field(default_factory=dict)
    read_first: ReadFirstGuard | None = None

    @model_serializer(mode="wrap")
    def serialize(self, handler: SerializerFunctionWrapHandler) -> dict:
        result = handler(self)
        # An absent guard is left out of the dump, so the receipts of rows sealed without one (and a continuation's
        # source_tools_sha256) stay valid.
        if self.read_first is None:
            result.pop("read_first", None)
        return result


class ArgumentMatch(Contract):
    """Match an execution argument using the shared Python/JavaScript regex subset."""

    field: str = Field(min_length=1)
    regex: str

    @model_validator(mode="after")
    def validate_pattern(self) -> "ArgumentMatch":
        try:
            re.compile(self.regex)
        except re.error as error:
            raise ValueError(f"event argument regex does not compile: {error}") from error
        if re.search(r"\(\?(?![:=!]|<[=!])|\\[AZaNU]|(?:[?*+]|\})\+", self.regex):
            raise ValueError("event argument regex must use the shared Python/JavaScript subset")
        return self


class ToolMatch(Contract):
    """The call an event fires on: its tool, and optionally an argument pattern."""

    tool: str = Field(min_length=1)
    arg: ArgumentMatch | None = None


class ToolEvent(Contract):
    """One scheduled tool failure: the matching call does not run and the model sees `text` as its error.

    Firing is recorded separately in evidence.
    """

    id: str = Field(min_length=1)
    match: ToolMatch
    nth: StrictInt = Field(default=1, ge=1)
    action: Literal["fail"]
    text: str = Field(min_length=1)


class ContinuationSpec(Contract):
    """A frozen OpenCode session prefix followed by the row's literal continuation messages.

    The author chooses the cut before export. Restore imports recorded history and
    workspace state; it never infers a cut or fabricates a new user message at runtime.
    """

    mode: Literal["restore"]
    prefix_export: dict[str, JsonValue]
    # Always empty: sealed receipts hash it as [], and a prefix's planted events cannot be restored.
    prefix_events: list[EventReceipt] = Field(default_factory=list)
    workspace_diff: str
    source_task_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_tools_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_prefix(self) -> "ContinuationSpec":
        if self.prefix_events:
            raise ValueError("continuation prefix_events must be empty: a prefix's planted events cannot be restored")
        info = self.prefix_export.get("info")
        messages = self.prefix_export.get("messages")
        if not isinstance(info, dict) or not info.get("id") or not isinstance(messages, list) or not messages:
            raise ValueError("continuation prefix requires an OpenCode session ID and messages")
        for message in messages:
            if not isinstance(message, dict) or not isinstance(message.get("info"), dict):
                raise ValueError("continuation prefix has an invalid message")
            parts = message.get("parts")
            if not isinstance(parts, list):
                raise ValueError("continuation prefix message requires a parts list")
            for part in parts:
                if not isinstance(part, dict):
                    raise ValueError("continuation prefix has an invalid message part")
                if part.get("type") == "tool":
                    state = part.get("state")
                    if not isinstance(state, dict) or state.get("status") not in {"completed", "error"}:
                        raise ValueError("continuation prefix contains an unfinished tool call")
                    if part.get("tool") == "task":
                        raise ValueError("continuation prefix cannot yet restore delegated child state")
        if messages[-1]["info"].get("role") != "assistant":
            raise ValueError("continuation prefix must end at an assistant-message boundary")
        return self


class InputReceipt(Contract):
    """Content-integrity receipts, not authentication or permission to execute."""

    actor_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    interfaces_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RuntimeSpec(Contract):
    """The sole IF envelope accepted for execution, independent of its author."""

    schema_version: Literal[1] = 1
    harness: Literal["opencode"]
    task_backend: Literal["swe", "tb21"]
    tools: ToolConfiguration = Field(default_factory=ToolConfiguration)
    workspace_files: dict[str, str] = Field(default_factory=dict)
    continuation: ContinuationSpec | None = None
    events: list[ToolEvent] = Field(default_factory=list)
    checks: list[ResolvedCheck] = Field(default_factory=list)
    provenance: dict[str, JsonValue] = Field(default_factory=dict)
    receipt: InputReceipt | None = None

    @model_validator(mode="after")
    def validate_capabilities(self) -> "RuntimeSpec":
        for path in self.workspace_files:
            absolute = path.startswith("/")
            parts = path.removeprefix("/").split("/")
            if "\\" in path or "\0" in path or any(part in {"", ".", "..", ".git"} for part in parts):
                raise ValueError("workspace file paths must be canonical sandbox paths outside Git internals")
            if any(
                ("/" if absolute else "") + "/".join(parts[:index]) in self.workspace_files
                for index in range(1, len(parts))
            ):
                raise ValueError("workspace file and directory paths collide")
        selected = configurable_tools(self.tools.tool_names)
        if self.tools.read_first is not None:
            self.tools.read_first.validate_tools(selected)
        names = [(self.tools.tool_names or {}).get(tool, tool) for tool in selected]
        if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name) for name in names):
            raise ValueError("invalid tool alias")
        if len(names) != len(set(names)):
            raise ValueError("tool aliases must be unique")
        if self.tools.tool_names is not None:
            if ("edit" in selected) != ("write" in selected):
                raise ValueError("edit and write must be selected together (OpenCode checks both under 'edit')")
        targets = (
            set(self.tools.tool_descriptions)
            | set(self.tools.tool_parameter_descriptions)
            | set(self.tools.tool_interfaces)
        )
        if not targets <= selected:
            raise ValueError(f"tool configuration targets tools not exposed: {sorted(targets - selected)}")
        if self.tools.tool_interfaces:
            installed_interfaces(self.tools.tool_interfaces)
        event_ids = [event.id for event in self.events]
        check_ids = [check.id for check in self.checks]
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("event IDs must be unique")
        if len(set(check_ids)) != len(check_ids):
            raise ValueError("check IDs must be unique")
        for event in self.events:
            if event.match.tool not in selected:
                raise ValueError("event must target an exposed tool")
            if event.match.arg:
                required, optional = BUILTIN_ARGUMENTS.get(event.match.tool, (frozenset(), frozenset()))
                if event.match.arg.field not in required | optional:
                    raise ValueError(f"event matches an unknown executable argument of {event.match.tool}")
        for check in self.checks:
            if check.scope.activation == "event" and check.scope.event_id not in event_ids:
                raise ValueError("check references an undeclared event")
            if check.scope.activation == "continuation_start" and self.continuation is None:
                raise ValueError("continuation-scoped check requires a continuation")
        return self


def _validate_row(row: Mapping[str, object], spec: RuntimeSpec) -> list[dict]:
    params = row.get("responses_create_params")
    if isinstance(params, Mapping) and params.get("instructions") is not None:
        raise ValueError("use a literal system input message, not a second instructions prompt")
    messages = params.get("input") if isinstance(params, Mapping) else None
    if not isinstance(messages, list) or not messages:
        raise ValueError("responses_create_params.input must contain literal messages")
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"system", "user"}:
            raise ValueError("initial input accepts system/user messages; recorded history belongs to continuation")
        content_text(message.get("content"))
    if not any(message["role"] == "user" for message in messages):
        raise ValueError("input requires a user message")
    systems = [message for message in messages if message["role"] == "system"]
    if len(systems) > 1:
        raise ValueError("input accepts at most one resolved system message")
    if any(content_text(message.get("content")) == "" for message in systems):
        # opencode_agent stages no instructions for an empty one, so OpenCode would keep its own system prompt.
        raise ValueError("a system message needs text; omit it to keep OpenCode's own system prompt")
    _validate_sources(row, spec)
    return messages


def _validate_sources(row: Mapping[str, object], spec: RuntimeSpec) -> None:
    """The task source, and a continuation's source receipts: checks that need no actor request."""
    if any(key in row for key in ("task_ref", "dataset_ref")):
        raise ValueError("package task references (task_ref, dataset_ref) are not a supported task source")
    if spec.continuation:
        if spec.continuation.source_task_sha256 != backend_task_digest(dict(row)):
            raise ValueError("continuation source task receipt does not match")
        if spec.continuation.source_tools_sha256 != digest(spec.tools.model_dump(mode="json")):
            raise ValueError("continuation source tools receipt does not match")


def _receipt(row: Mapping[str, object], spec: RuntimeSpec, messages: list[dict]) -> InputReceipt:
    # Bind every model execution setting, normalizing only the Responses API defaults.
    # Actor text has its own representation-independent receipt.
    model_parameters = NeMoGymResponseCreateParamsNonStreaming.model_validate(
        row["responses_create_params"]
    ).model_dump(mode="json", exclude_defaults=True, exclude={"input"})
    return InputReceipt(
        actor_sha256=actor_input_digest(messages),
        task_sha256=backend_task_digest(dict(row)),
        configuration_sha256=digest(
            {
                "configuration": spec.model_dump(mode="json", exclude={"receipt"}),
                "model_parameters": model_parameters,
            }
        ),
        interfaces_sha256=digest(installed_interfaces(spec.tools.tool_interfaces)),
    )


def seal_runtime_row(row: Mapping[str, object], spec: RuntimeSpec) -> dict[str, object]:
    """Freeze resolved authoring output without editing actor text or task metadata."""
    if "agentic_if" in row or spec.receipt is not None:
        raise ValueError("seal a fresh row and configuration, not an already sealed run")
    # Re-validate even mutable Pydantic instances rather than trusting their origin.
    spec = RuntimeSpec.model_validate(spec.model_dump(mode="json"))
    messages = _validate_row(row, spec)
    spec.receipt = _receipt(row, spec, messages)
    result = deepcopy(dict(row))
    result["agentic_if"] = spec.model_dump(mode="json")
    return result


def validate_runtime(row: Mapping[str, object]) -> RuntimeSpec:
    """Validate the sole input format and its receipts before allocating resources."""
    spec = RuntimeSpec.model_validate(row.get("agentic_if"))
    messages = _validate_row(row, spec)
    if spec.receipt is None or spec.receipt != _receipt(row, spec, messages):
        raise ValueError("IF input receipt does not match actor, task, or configuration")
    return spec


def validate_seed_runtime(row: Mapping[str, object]) -> RuntimeSpec:
    """Validate a sealed row's configuration and the receipts that do not depend on the actor's request.

    An Environment Server seeds with the task data only: the actor's ``responses_create_params`` reach the Resources
    Server at verify, where ``validate_runtime`` checks every receipt.
    """
    spec = RuntimeSpec.model_validate(row.get("agentic_if"))
    _validate_sources(row, spec)
    if spec.receipt is None:
        raise ValueError("IF input receipt is missing")
    if spec.receipt.task_sha256 != backend_task_digest(dict(row)):
        raise ValueError("IF input receipt does not match the task")
    if spec.receipt.interfaces_sha256 != digest(installed_interfaces(spec.tools.tool_interfaces)):
        raise ValueError("IF input receipt does not match the installed interfaces")
    return spec
