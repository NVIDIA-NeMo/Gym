# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stirrup rollout primitives for the Archipelago environment sandbox."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import shutil
import zipfile
from contextlib import suppress
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, get_args, get_origin


LOGGER = logging.getLogger(__name__)


FILESYSTEM_ROOT = Path("/filesystem")
APPS_DATA_ROOT = Path("/.apps_data")
MCP_ROOT = Path("/app/mcp_servers")
MCP_TOOL_TIMEOUT_SECONDS = 60
TOOL_OUTPUT_TOKEN_BUDGET = 24_000
TOOL_OUTPUT_ESTIMATED_CHARACTERS_PER_TOKEN = 4
TOOL_OUTPUT_HEAD_CHARACTERS = 20_000
TOOL_OUTPUT_TAIL_CHARACTERS = 5_000
PARTIAL_RESULT_CHECKPOINT_INTERVAL_SECONDS = 1.0

_STANDARD_SERVERS: dict[str, tuple[str, str, str]] = {
    "pdfs": ("pdfs", "pdf_server", "APP_PDF_ROOT"),
    "sheets": ("spreadsheets", "sheets_server", "APP_SHEETS_ROOT"),
    "docs": ("documents", "docs_server", "APP_DOCS_ROOT"),
    "presentations": ("presentations", "slides_server", "APP_SLIDES_ROOT"),
    "code": ("code", "code_execution_server", "APP_FS_ROOT"),
    "mail": ("mail", "mail_server", "APP_MAIL_DATA_ROOT"),
    "chat": ("chat", "chat_server", "APP_CHAT_DATA_ROOT"),
    "calendar": ("calendar", "calendar_server", "APP_CALENDAR_DATA_ROOT"),
}

SYSTEM_PROMPT = """You are an autonomous workplace agent operating in an Archipelago environment.

The task filesystem is rooted at /filesystem. Workplace applications and file operations are available only through MCP-backed tools. Your initial toolbelt is deliberately small:
- list_tools: list MCP tools that can be added and show which are active.
- inspect_tool: inspect one MCP tool's description and argument schema.
- add_tool: add an inspected MCP tool to your active toolbelt.
- remove_tool: remove an MCP tool you no longer need.
- todo_write: replace the todo list or merge updates by todo ID.
- finish: submit the final answer and completion status.

Use todo_write to plan and track the work. Before a completed finish submission is accepted, every todo must be completed or cancelled. Calling finish is the only way to submit a final answer. Plain assistant text does not submit the task. Use status="completed" only when the task is actually complete; use status="incomplete" when it cannot be completed.
"""


def truncate_tool_text(text: str) -> str:
    """Apply the 24k-token budget and 20k-head/5k-tail excerpt policy."""
    estimated_tokens = (
        len(text) + TOOL_OUTPUT_ESTIMATED_CHARACTERS_PER_TOKEN - 1
    ) // TOOL_OUTPUT_ESTIMATED_CHARACTERS_PER_TOKEN
    if estimated_tokens <= TOOL_OUTPUT_TOKEN_BUDGET:
        return text
    excerpt_characters = TOOL_OUTPUT_HEAD_CHARACTERS + TOOL_OUTPUT_TAIL_CHARACTERS
    removed = len(text) - excerpt_characters
    marker = f"\n\n[... {removed} characters truncated ...]\n\n"
    return text[:TOOL_OUTPUT_HEAD_CHARACTERS] + marker + text[-TOOL_OUTPUT_TAIL_CHARACTERS:]


def mcp_call_arguments(params: Any) -> dict[str, Any]:
    """Forward only concrete MCP arguments; omitted optional fields must stay omitted."""
    return params.model_dump(exclude_none=True)


def _annotation_wants_structured_data(annotation: Any) -> bool:
    """True when a field annotation expects a container or nested model, through Optional/Union/Annotated."""
    if annotation in (list, dict, set, tuple):
        return True
    origin = get_origin(annotation)
    if origin in (list, dict, set, frozenset, tuple):
        return True
    if origin is not None:
        return any(_annotation_wants_structured_data(arg) for arg in get_args(annotation) if arg is not type(None))
    return isinstance(annotation, type) and hasattr(annotation, "model_fields")


def _nested_model_class(annotation: Any) -> Any:
    """Resolve an annotation to a pydantic-model class (duck-typed), or None."""
    if isinstance(annotation, type) and hasattr(annotation, "model_fields"):
        return annotation
    if get_origin(annotation) is not None:
        for arg in get_args(annotation):
            if arg is type(None):
                continue
            found = _nested_model_class(arg)
            if found is not None:
                return found
    return None


def _keys_known_to_model(data: dict[str, Any], model_cls: Any) -> bool:
    """True when every key in data names a field (or alias) of model_cls.

    MCP-generated models default to extra="ignore", so validation alone would
    accept arbitrary keys and silently drop them; repairs must not manufacture
    a "valid" call out of args the tool would never see.
    """
    fields = getattr(model_cls, "model_fields", None)
    if not isinstance(fields, dict):
        return False
    known = set(fields)
    for field in fields.values():
        for alias in (getattr(field, "alias", None), getattr(field, "validation_alias", None)):
            if isinstance(alias, str):
                known.add(alias)
    return set(data) <= known


def coerce_tool_arguments(params_model: Any, arguments: str) -> str | None:
    """Return a coerced JSON arguments string that validates against params_model, or None.

    Empty/whitespace arguments are normalized to "{}" before evaluation (mirroring
    stirrup's own normalization in Agent.run_tool). Arguments that already validate
    return None (no change needed). Two repairs are attempted, and a candidate is
    accepted only if it validates against params_model:
    - unwrap top-level string values that JSON-decode to an object/array where the
      field annotation expects structured data, and
    - wrap flat arguments into the model's single required nested-model field.
    Dict payloads destined for a nested model must only use keys that model knows
    (see _keys_known_to_model). Never raises; any unexpected error returns None.
    """
    try:
        normalized = arguments if arguments and arguments.strip() else "{}"
        try:
            params_model.model_validate_json(normalized)
            return None
        except Exception:
            pass
        try:
            given = json.loads(normalized)
        except ValueError:
            return None
        if not isinstance(given, dict):
            return None
        fields = getattr(params_model, "model_fields", None)
        if not isinstance(fields, dict):
            return None

        unwrapped = dict(given)
        changed = False
        for key, value in given.items():
            field = fields.get(key)
            if field is None or not isinstance(value, str):
                continue
            if not _annotation_wants_structured_data(getattr(field, "annotation", None)):
                continue
            try:
                parsed = json.loads(value)
            except ValueError:
                continue
            if not isinstance(parsed, (dict, list)):
                continue
            nested = _nested_model_class(getattr(field, "annotation", None))
            if isinstance(parsed, dict) and nested is not None and not _keys_known_to_model(parsed, nested):
                continue
            unwrapped[key] = parsed
            changed = True

        candidates: list[dict[str, Any]] = []
        if changed:
            candidates.append(unwrapped)
        required = [name for name, field in fields.items() if field.is_required()]
        if len(required) == 1 and required[0] not in given:
            wrapper = required[0]
            wrapper_model = _nested_model_class(getattr(fields[wrapper], "annotation", None))
            if wrapper_model is not None:
                if _keys_known_to_model(given, wrapper_model):
                    candidates.append({wrapper: given})
                if changed and _keys_known_to_model(unwrapped, wrapper_model):
                    candidates.append({wrapper: unwrapped})

        for candidate in candidates:
            encoded = json.dumps(candidate)
            try:
                params_model.model_validate_json(encoded)
            except Exception:
                continue
            return encoded
        return None
    except Exception:
        return None


def format_tool_argument_validation_error(exc: Any, arguments: str) -> str:
    """Render pydantic validation detail so the model can self-correct when coercion cannot repair the args."""
    errors = "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or '<root>'}: {error['msg']} (type={error.get('type', '?')})"
        for error in exc.errors()
    )
    preview = (arguments or "")[:500]
    return f"Tool arguments are not valid: {errors}. Submitted arguments (first 500 chars): {preview!r}"


# vLLM's glm47 tool-call parser string-encodes arguments for MCP tools whose
# JSON schema wraps params in a bare-$ref property with no inline "type"
# ({"request": "{\"code\": ...}"} or flat {"code": ...}). The typing bug is
# serving-side, so the model cannot self-correct no matter how the error is
# phrased; run_tool must repair the arguments instead (cf. vLLM PR #41801).
# Finish tools are exempt: Agent.step() re-validates the ORIGINAL tool_call
# for finish tools (stirrup agent.py:1261) outside any try/except, so a
# coerced copy that only run_tool sees would crash step().
def install_tool_argument_coercion(agent_cls: Any) -> None:
    """Patch agent_cls.run_tool to coerce parser-mangled tool arguments and surface validation detail."""
    original_run_tool = agent_cls.run_tool
    if getattr(original_run_tool, "_apex_tool_arg_patch", False):
        return

    async def run_tool_with_argument_coercion(self: Any, tool_call: Any, run_metadata: Any) -> Any:
        finish_tools = getattr(self, "_finish_tools", None) or {}
        tool = self._active_tools.get(tool_call.name)
        if tool is not None and tool_call.name not in finish_tools:
            coerced = coerce_tool_arguments(tool.parameters, tool_call.arguments or "")
            if coerced is not None:
                tool_call = tool_call.model_copy(update={"arguments": coerced})
        result_msg = await original_run_tool(self, tool_call, run_metadata)
        if not getattr(result_msg, "args_was_valid", True) and result_msg.content == "Tool arguments are not valid":
            tool = self._active_tools.get(tool_call.name)
            if tool is not None:
                args = tool_call.arguments if tool_call.arguments and tool_call.arguments.strip() else "{}"
                try:
                    tool.parameters.model_validate_json(args)
                except Exception as exc:
                    if hasattr(exc, "errors"):
                        with suppress(Exception):
                            detailed = format_tool_argument_validation_error(exc, tool_call.arguments or "")
                            result_msg = result_msg.model_copy(update={"content": detailed})
        return result_msg

    run_tool_with_argument_coercion._apex_tool_arg_patch = True
    agent_cls.run_tool = run_tool_with_argument_coercion


def annotate_schema_ref_types(schema: Any) -> Any:
    """Return a copy of a JSON schema with each $ref site annotated with its definition's type.

    Local refs ("#/$defs/<name>" or "#/definitions/<name>") are resolved against the
    schema's own root table of the matching spelling; when the resolved definition
    carries a "type" (following bare-ref chains, with cycles terminated) and the ref
    node has no "type" of its own, that type is copied onto the ref node. Nothing else
    is copied: a type sibling equal to the definition's type is a conjunctive no-op
    under JSON Schema 2020-12 (and ignored under draft-07), so the schema's semantics
    are provably unchanged. $refs and the $defs/definitions tables stay intact, and
    ref sites inside the tables are annotated too. Non-local or unresolvable refs pass
    through unchanged. Every dict and list in the result is rebuilt, so the input is
    never mutated and the output shares no mutable state with it. Never raises: any
    unexpected error returns the original schema.
    """
    try:
        if not isinstance(schema, dict):
            return schema
        definition_tables: dict[str, Any] = {}
        for table_key in ("$defs", "definitions"):
            table = schema.get(table_key)
            if isinstance(table, dict):
                for name, definition in table.items():
                    # RFC 6901 pointer escaping so a def named "a/b" cannot
                    # shadow the pointer path $defs -> a -> b.
                    escaped = name.replace("~", "~0").replace("/", "~1")
                    definition_tables[f"#/{table_key}/{escaped}"] = definition

        def resolve_ref_type(ref: Any) -> str | list[Any] | None:
            """Follow a ref (through bare-ref chains) to its definition's "type", or None."""
            visited: set[str] = set()
            while isinstance(ref, str) and ref in definition_tables and ref not in visited:
                visited.add(ref)
                definition = definition_tables[ref]
                if not isinstance(definition, dict):
                    return None
                if "type" in definition:
                    def_type = definition["type"]
                    return copy.deepcopy(def_type) if isinstance(def_type, (str, list)) else None
                ref = definition.get("$ref")
            return None

        # Values of these keywords are instance DATA, not schemas; a data dict
        # that happens to carry a "$ref" key must never gain a "type".
        data_keywords = {"const", "enum", "default", "examples"}

        def annotate(node: Any) -> Any:
            if isinstance(node, list):
                return [annotate(item) for item in node]
            if not isinstance(node, dict):
                return node
            rebuilt = {
                key: copy.deepcopy(value) if key in data_keywords else annotate(value) for key, value in node.items()
            }
            if "type" not in rebuilt:
                ref_type = resolve_ref_type(node.get("$ref"))
                if ref_type is not None:
                    rebuilt["type"] = ref_type
            return rebuilt

        return annotate(schema)
    except Exception:
        return schema


# GLM-family vLLM tool-call parsers reconstruct each argument's type from the
# wire schema's properties[key].type; a property site that is a bare
# {"$ref": "#/$defs/..."} carries no inline "type", so the parser
# string-encodes the whole object server-side before the agent ever sees it.
# Copying only the resolved definition's type onto the ref site removes that
# trigger without changing what the schema accepts: fully inlining a $ref and
# merging its sibling keys would alter 2020-12 conjunctive semantics when a
# sibling conflicts with the definition, while a type sibling equal to the
# definition's type is a conjunctive no-op (and ignored under draft-07).
# install_tool_argument_coercion stays as the safety net for parsers that
# string-encode regardless of schema. ChatCompletionsClient and LiteLLMClient
# bind to_openai_tools via `from stirrup.clients.utils import ...`, so
# rebinding stirrup.clients.utils alone is not enough — each client module's
# own namespace binding must be rebound too.
def install_tool_schema_type_annotation() -> None:
    """Patch stirrup's to_openai_tools so $ref sites in wire tool schemas carry an inline type."""
    import stirrup.clients.utils as stirrup_client_utils

    original_to_openai_tools = stirrup_client_utils.to_openai_tools
    if getattr(original_to_openai_tools, "_apex_schema_type_patch", False):
        return

    def to_openai_tools_with_annotated_ref_types(tools: Any) -> list[dict[str, Any]]:
        entries = original_to_openai_tools(tools)
        for entry in entries:
            function = entry.get("function") if isinstance(entry, dict) else None
            if isinstance(function, dict) and isinstance(function.get("parameters"), dict):
                function["parameters"] = annotate_schema_ref_types(function["parameters"])
        return entries

    to_openai_tools_with_annotated_ref_types._apex_schema_type_patch = True

    client_modules: list[Any] = [stirrup_client_utils]
    with suppress(Exception):
        import stirrup.clients.chat_completions_client as chat_completions_client_module

        client_modules.append(chat_completions_client_module)
    with suppress(Exception):
        import stirrup.clients.litellm_client as litellm_client_module

        client_modules.append(litellm_client_module)
    for module in client_modules:
        if hasattr(module, "to_openai_tools"):
            module.to_openai_tools = to_openai_tools_with_annotated_ref_types


# Stock Stirrup raises ContextOverflowError whenever a completion ends with
# finish_reason "length" (or "max_tokens"). That conflates two different events:
# the prompt overflowing the context window, which vLLM rejects with HTTP 400
# before generating anything, and one turn exhausting max_output_tokens
# mid-reasoning. GLM-5.2 hits the second case on long deliberations. The raise
# then unwinds the last completed turn and, when that would cross a progress
# boundary, ends the rollout with zero credit. On the full 452 x 3 benchmark,
# 45 rollouts died this way, every one with valid partial content and a usable
# history.
#
# The recovery mirrors the GDPVal Stirrup client: keep the truncated message as
# an ordinary assistant turn and, if it carried no tool call, steer the very next
# request with thinking disabled and a short notice. The notice is sent to the
# server only; it never enters the agent's history or the recorded trajectory.
LENGTH_TRUNCATION_RECOVERY_NUDGE = (
    "SYSTEM NOTICE: your previous response hit the output token limit before it "
    "produced a tool call, so none of that reasoning was saved. Do not start that "
    "analysis over. Act now on what you already know: respond with a tool call and "
    "keep any preamble to a few sentences."
)
_LENGTH_FINISH_REASONS = ("length", "max_tokens")


class _LengthRecordingCompletions:
    """Stand-in for ``openai_client.chat.completions`` that records a length finish and neutralises it.

    Stirrup checks ``finish_reason`` before parsing the message. Rewriting a
    length finish to ``stop`` ahead of that check keeps the stock parsing of
    reasoning, tool calls and token usage intact while suppressing the raise.
    """

    def __init__(self, inner_create: Any, record: dict[str, Any]) -> None:
        self._inner_create = inner_create
        self._record = record

    async def create(self, **kwargs: Any) -> Any:
        response = await self._inner_create(**kwargs)
        choices = getattr(response, "choices", None) or []
        if choices:
            choice = choices[0]
            if getattr(choice, "finish_reason", None) in _LENGTH_FINISH_REASONS:
                self._record["finish_reason"] = choice.finish_reason
                choice.finish_reason = "stop"
        return response


class _LengthRecordingClient:
    def __init__(self, inner_client: Any, record: dict[str, Any]) -> None:
        self.chat = SimpleNamespace(
            completions=_LengthRecordingCompletions(inner_client.chat.completions.create, record)
        )


def make_length_tolerant_client_class(base_cls: Any) -> Any:
    """Subclass Stirrup's ``ChatCompletionsClient`` so a length finish is a turn, not a crash.

    ``base_cls`` is passed in because stirrup is only importable inside the
    sandbox; the subclass is created at call time in ``run_stirrup_rollout``.
    """

    class LengthTolerantChatCompletionsClient(base_cls):
        def __init__(self, *args: Any, truncation_recovery: bool = True, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.truncation_recovery = truncation_recovery
            self.length_truncations = 0
            self.recovery_turns = 0
            self._recover_from_truncation = False

        async def generate(self, messages: Any, tools: Any) -> Any:
            from stirrup.core.models import UserMessage

            recovering = self._recover_from_truncation and self.truncation_recovery
            self._recover_from_truncation = False
            request_messages = list(messages)
            saved_kwargs = self._kwargs
            saved_client = self._client
            if recovering:
                self.recovery_turns += 1
                request_messages.append(UserMessage(content=LENGTH_TRUNCATION_RECOVERY_NUDGE))
                extra_body = dict(saved_kwargs.get("extra_body") or {})
                chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
                chat_template_kwargs["enable_thinking"] = False
                extra_body["chat_template_kwargs"] = chat_template_kwargs
                self._kwargs = {**saved_kwargs, "extra_body": extra_body}
            record: dict[str, Any] = {}
            self._client = _LengthRecordingClient(saved_client, record)
            try:
                message = await super().generate(request_messages, tools)
            finally:
                self._client = saved_client
                self._kwargs = saved_kwargs
            if record.get("finish_reason") is not None:
                self.length_truncations += 1
                if message.tool_calls:
                    LOGGER.warning(
                        "completion budget (%d tokens) exhausted with %d tool call(s) present; continuing [truncation #%d]",
                        self._max_tokens,
                        len(message.tool_calls),
                        self.length_truncations,
                    )
                else:
                    self._recover_from_truncation = True
                    LOGGER.warning(
                        "completion budget (%d tokens) exhausted with no tool call [truncation #%d]; %s",
                        self._max_tokens,
                        self.length_truncations,
                        "next turn runs with thinking disabled and a recovery notice"
                        if self.truncation_recovery
                        else "truncation recovery is off, next turn is unchanged",
                    )
            return message

    LengthTolerantChatCompletionsClient.__name__ = f"LengthTolerant{base_cls.__name__}"
    LengthTolerantChatCompletionsClient.__qualname__ = LengthTolerantChatCompletionsClient.__name__
    return LengthTolerantChatCompletionsClient


def replace_tool_images_for_text_only_model(
    content: Any,
    *,
    supports_vision: bool,
    image_content_type: type[Any],
) -> Any:
    """Preserve tool text while replacing images for models without vision support."""
    if supports_vision:
        return content

    if isinstance(content, image_content_type):
        return "[1 image(s) not shown: model does not support vision]"
    if not isinstance(content, list):
        return content

    image_count = sum(isinstance(block, image_content_type) for block in content)
    if image_count == 0:
        return content
    non_image_content = [block for block in content if not isinstance(block, image_content_type)]
    return [
        *non_image_content,
        f"[{image_count} image(s) not shown: model does not support vision]",
    ]


def _clear_root(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for child in root.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink(missing_ok=True)


def _safe_extract_world(world_zip: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(world_zip) as archive:
        for member in archive.infolist():
            relative = PurePosixPath(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe world archive member: {member.filename!r}")
            target = (root / Path(*relative.parts)).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"world archive member escapes extraction root: {member.filename!r}")
        archive.extractall(root)


def _copy_tree_contents(source: Path, destination: Path) -> None:
    if not source.is_dir():
        return
    destination.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        target = destination / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, target)


def populate_world(world_zip: Path, scratch_root: Path) -> None:
    """Restore a world ZIP into Archipelago's filesystem and app-state roots."""
    _clear_root(FILESYSTEM_ROOT)
    _clear_root(APPS_DATA_ROOT)
    extracted = scratch_root / "world"
    extracted.mkdir(parents=True, exist_ok=True)
    _safe_extract_world(world_zip, extracted)

    wrapped = extracted / "world_files"
    filesystem_source = next(
        (candidate for candidate in (extracted / "filesystem", wrapped / "filesystem") if candidate.is_dir()),
        None,
    )
    apps_source = next(
        (candidate for candidate in (extracted / ".apps_data", wrapped / ".apps_data") if candidate.is_dir()),
        None,
    )
    if filesystem_source is None:
        entries = list(extracted.iterdir())
        only_apps = len(entries) == 1 and entries[0].name in {".apps_data", "world_files"}
        if not only_apps:
            filesystem_source = extracted

    if filesystem_source is not None:
        _copy_tree_contents(filesystem_source, FILESYSTEM_ROOT)
    if apps_source is not None:
        _copy_tree_contents(apps_source, APPS_DATA_ROOT)
    (FILESYSTEM_ROOT / "tmp").mkdir(parents=True, exist_ok=True)


def overlay_task_files(task_files_zip: Path, scratch_root: Path) -> None:
    """Overlay task-specific input files without clearing the shared world state."""
    extracted = scratch_root / "task_files"
    extracted.mkdir(parents=True, exist_ok=True)
    _safe_extract_world(task_files_zip, extracted)

    filesystem_source = extracted / "filesystem"
    apps_source = extracted / ".apps_data"
    if not filesystem_source.is_dir() and not apps_source.is_dir():
        filesystem_source = extracted
    _copy_tree_contents(filesystem_source, FILESYSTEM_ROOT)
    _copy_tree_contents(apps_source, APPS_DATA_ROOT)


def write_snapshot(destination: Path) -> list[str]:
    """Write the local-grader ZIP shape and return its file manifest."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest: list[str] = []
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for prefix, root in (("filesystem", FILESYSTEM_ROOT), (".apps_data", APPS_DATA_ROOT)):
            if not root.exists():
                continue
            for path in sorted(root.rglob("*")):
                relative = path.relative_to(root).as_posix()
                archive_name = f"{prefix}/{relative}"
                if path.is_file():
                    archive.write(path, archive_name)
                    manifest.append(archive_name)
                elif not any(path.iterdir()):
                    archive.writestr(f"{archive_name}/", b"")
    return manifest


def _base_server_environment() -> dict[str, str]:
    return {key: value for key in ("HOME", "LANG", "LC_ALL", "PATH", "TMPDIR") if (value := os.environ.get(key))} | {
        "MCP_TRANSPORT": "stdio",
        "USE_INDIVIDUAL_TOOLS": "true",
    }


def _server_config(component: str, server_dir: str, environment: dict[str, str]) -> dict[str, Any]:
    project = MCP_ROOT / component
    server = project / "mcp_servers" / server_dir
    python = project / ".venv" / "bin" / "python3"
    if not python.is_file() or not server.is_dir():
        raise FileNotFoundError(f"Archipelago MCP server is not installed: {server}")
    return {
        "transport": "stdio",
        "command": str(python),
        "args": ["main.py"],
        "cwd": str(server),
        "env": _base_server_environment() | environment | {"VIRTUAL_ENV": str(project / ".venv")},
    }


def gateway_config(foundry_services: list[str], edgar_user_agent: str | None) -> dict[str, Any]:
    servers: dict[str, dict[str, Any]] = {}
    for name, (component, server_dir, root_key) in _STANDARD_SERVERS.items():
        if name == "code":
            env = {"APP_FS_ROOT": str(FILESYSTEM_ROOT), "SANDBOX_ROOT": str(FILESYSTEM_ROOT)}
        elif name in {"mail", "chat", "calendar"}:
            state = APPS_DATA_ROOT / name
            state.mkdir(parents=True, exist_ok=True)
            env = {
                "APP_FS_ROOT": str(FILESYSTEM_ROOT),
                "APP_APPS_DATA_ROOT": str(APPS_DATA_ROOT),
                root_key: str(state),
                "HAS_STATE": "true",
                "STATE_LOCATION": str(state),
            }
        else:
            env = {root_key: str(FILESYSTEM_ROOT)}
        servers[name] = _server_config(component, server_dir, env)

    unsupported = sorted(set(foundry_services) - {"fmp", "edgar"})
    if unsupported:
        raise ValueError(f"Archipelago image does not package requested services: {', '.join(unsupported)}")
    if "fmp" in foundry_services:
        state = APPS_DATA_ROOT / "fmp"
        state.mkdir(parents=True, exist_ok=True)
        servers["fmp"] = _server_config(
            "fmp",
            "fmp_server",
            {
                "APP_FS_ROOT": str(FILESYSTEM_ROOT),
                "APP_APPS_DATA_ROOT": str(APPS_DATA_ROOT),
                "HAS_STATE": "true",
                "STATE_LOCATION": str(state),
            },
        )
    if "edgar" in foundry_services:
        state = APPS_DATA_ROOT / "edgar"
        state.mkdir(parents=True, exist_ok=True)
        edgar_env = {
            "APP_FS_ROOT": str(FILESYSTEM_ROOT),
            "APP_APPS_DATA_ROOT": str(APPS_DATA_ROOT),
            "HAS_STATE": "true",
            "STATE_LOCATION": str(state),
            "EDGAR_OFFLINE_MODE": "true",
            "INTERNET_ENABLED": "false",
        }
        if edgar_user_agent:
            edgar_env["EDGAR_USER_AGENT"] = edgar_user_agent
        servers["edgar"] = _server_config("edgar_sec", "edgar_sec", edgar_env)
    return {"mcpServers": servers}


async def wait_for_gateway(gateway_url: str, timeout_seconds: float = 60.0) -> None:
    import httpx

    deadline = asyncio.get_running_loop().time() + timeout_seconds
    async with httpx.AsyncClient() as client:
        while True:
            try:
                response = await client.get(f"{gateway_url}/health", timeout=2.0)
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("Archipelago gateway did not become healthy")
            await asyncio.sleep(0.25)


async def configure_gateway(config: dict[str, Any], gateway_url: str) -> None:
    import httpx

    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(f"{gateway_url}/apps", json=config)
        response.raise_for_status()


def _serialize_history(history: list[list[Any]]) -> list[dict[str, Any]]:
    trajectory: list[dict[str, Any]] = []
    for group in history:
        for message in group:
            if hasattr(message, "model_dump"):
                trajectory.append(message.model_dump(mode="json"))
            else:
                trajectory.append({"content": str(message)})
    return trajectory


def _token_usage(history: list[list[Any]]) -> tuple[int, int, int]:
    input_tokens = output_tokens = reasoning_tokens = 0
    for group in history:
        for message in group:
            usage = getattr(message, "token_usage", None)
            if usage is None:
                continue
            input_tokens += int(getattr(usage, "input", 0) or 0)
            answer = int(getattr(usage, "answer", 0) or 0)
            reasoning = int(getattr(usage, "reasoning", 0) or 0)
            output_tokens += answer + reasoning
            reasoning_tokens += reasoning
    return input_tokens, output_tokens, reasoning_tokens


def partial_result_from_session(session: Any, *, completion_status: str = "running") -> dict[str, Any] | None:
    """Project Stirrup's latest completed-turn cache state into a recoverable rollout result."""
    state = getattr(session, "_current_run_state", None)
    if state is None:
        return None
    history = [*getattr(state, "full_msg_history", []), list(getattr(state, "msgs", []))]
    input_tokens, output_tokens, reasoning_tokens = _token_usage(history)
    return {
        "final_answer": "",
        "completion_status": completion_status,
        "completed": False,
        "n_input_tokens": input_tokens,
        "n_output_tokens": output_tokens,
        "n_reasoning_tokens": reasoning_tokens,
        "trajectory": _serialize_history(history),
        "tool_metadata": {},
    }


def write_partial_result_checkpoint(
    session: Any,
    destination: Path,
    *,
    completion_status: str = "running",
    error: str | None = None,
) -> bool:
    """Atomically retain the latest completed Stirrup turns for crash recovery."""
    result = partial_result_from_session(session, completion_status=completion_status)
    if result is None:
        return False
    if error is not None:
        result["checkpoint_error"] = error
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(temporary, destination)
    return True


async def _checkpoint_partial_result(session: Any, destination: Path) -> None:
    checkpointed_state: Any = None
    while True:
        current_state = getattr(session, "_current_run_state", None)
        if current_state is not None and current_state is not checkpointed_state:
            try:
                if write_partial_result_checkpoint(session, destination):
                    checkpointed_state = current_state
            except Exception:
                pass
        await asyncio.sleep(PARTIAL_RESULT_CHECKPOINT_INTERVAL_SECONDS)


async def run_stirrup_rollout(
    config: dict[str, Any],
    gateway_url: str,
    *,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    """Run one 200-turn Stirrup session against the Archipelago MCP gateway."""
    from typing import Annotated, Literal

    from pydantic import BaseModel, Field
    from stirrup import Agent
    from stirrup.clients.chat_completions_client import ChatCompletionsClient
    from stirrup.core.models import ImageContentBlock, Tool, ToolProvider, ToolResult, ToolUseCountMetadata
    from stirrup.tools.mcp import MCPConfig, MCPToolProvider, StreamableHttpServerConfig

    install_tool_argument_coercion(Agent)
    install_tool_schema_type_annotation()

    class ToolNameParams(BaseModel):
        name: Annotated[str, Field(description="Exact MCP tool name from list_tools.")]

    class ListToolsParams(BaseModel):
        query: Annotated[str | None, Field(default=None, description="Optional case-insensitive name filter.")]

    class TodoItem(BaseModel):
        id: Annotated[str, Field(description="Stable todo ID.")]
        content: Annotated[str | None, Field(default=None, description="Todo text; required for new todos.")]
        status: Literal["pending", "in_progress", "completed", "cancelled"] = "pending"

    class TodoWriteParams(BaseModel):
        mode: Literal["replace", "merge"]
        todos: list[TodoItem]

    class FinishParams(BaseModel):
        final_answer: Annotated[str, Field(description="Final answer submitted for grading.")]
        status: Literal["completed", "incomplete"]

    todo_state: dict[str, TodoItem] = {}

    async def todo_write(params: TodoWriteParams) -> ToolResult[ToolUseCountMetadata]:
        if params.mode == "replace":
            replacement: dict[str, TodoItem] = {}
            for item in params.todos:
                if not item.id.strip() or not (item.content or "").strip():
                    return ToolResult(
                        content="Every replacement todo needs a non-empty ID and content.", success=False
                    )
                if item.id in replacement:
                    return ToolResult(content=f"Duplicate todo ID: {item.id}", success=False)
                replacement[item.id] = item
            todo_state.clear()
            todo_state.update(replacement)
        else:
            for update in params.todos:
                existing = todo_state.get(update.id)
                if existing is None and not (update.content or "").strip():
                    return ToolResult(content=f"New todo {update.id!r} needs content.", success=False)
                content = update.content if update.content is not None else existing.content
                todo_state[update.id] = TodoItem(id=update.id, content=content, status=update.status)
        return ToolResult(
            content=json.dumps([item.model_dump(mode="json") for item in todo_state.values()], indent=2),
            metadata=ToolUseCountMetadata(),
        )

    async def finish(params: FinishParams) -> ToolResult[ToolUseCountMetadata]:
        unfinished = [item.id for item in todo_state.values() if item.status not in {"completed", "cancelled"}]
        if unfinished:
            return ToolResult(
                content=f"Finish rejected. Complete or cancel these todos first: {', '.join(unfinished)}",
                success=False,
                metadata=ToolUseCountMetadata(),
            )
        return ToolResult(content=params.final_answer, metadata=ToolUseCountMetadata())

    todo_tool = Tool(
        name="todo_write",
        description="Create/update the todo list by replacing it or merging updates by todo ID.",
        parameters=TodoWriteParams,
        executor=todo_write,
    )
    finish_tool = Tool(
        name="finish",
        description="Submit the final answer and completion status. This is the only submission mechanism.",
        parameters=FinishParams,
        executor=finish,
    )

    class ManagedMCPTools(ToolProvider):
        def __init__(self) -> None:
            self.agent: Any = None
            self.provider = MCPToolProvider(
                MCPConfig(
                    mcpServers={
                        "archipelago": StreamableHttpServerConfig(
                            url=f"{gateway_url}/mcp/",
                            timeout=MCP_TOOL_TIMEOUT_SECONDS,
                            sse_read_timeout=MCP_TOOL_TIMEOUT_SECONDS,
                        )
                    }
                )
            )
            self.catalog: dict[str, Any] = {}
            self.core_names = {"list_tools", "inspect_tool", "add_tool", "remove_tool", "todo_write", "finish"}

        def attach(self, agent: Any) -> None:
            self.agent = agent

        def _active(self, name: str) -> bool:
            return self.agent is not None and name in self.agent._active_tools

        async def __aenter__(self) -> list[Any]:
            generated = await self.provider.__aenter__()
            for tool in generated:
                public_name = tool.name.removeprefix("archipelago__")

                async def bounded_executor(params: BaseModel, _tool: str = public_name) -> Any:
                    try:
                        content = await asyncio.wait_for(
                            self.provider.call_tool(
                                "archipelago",
                                _tool,
                                mcp_call_arguments(params),
                            ),
                            timeout=MCP_TOOL_TIMEOUT_SECONDS,
                        )
                    except asyncio.TimeoutError:
                        return ToolResult(
                            content=f"MCP tool timed out after {MCP_TOOL_TIMEOUT_SECONDS} seconds.",
                            success=False,
                            metadata=ToolUseCountMetadata(),
                        )
                    content = replace_tool_images_for_text_only_model(
                        content,
                        supports_vision=bool(config.get("supports_vision", True)),
                        image_content_type=ImageContentBlock,
                    )
                    result = ToolResult(content=content, metadata=ToolUseCountMetadata())
                    content = result.content
                    if isinstance(content, str):
                        result.content = truncate_tool_text(content)
                    elif isinstance(content, list):
                        total_text = "\n".join(block for block in content if isinstance(block, str))
                        truncated_text = truncate_tool_text(total_text)
                        if truncated_text != total_text:
                            non_text = [block for block in content if not isinstance(block, str)]
                            result.content = [truncated_text, *non_text]
                    return result

                self.catalog[public_name] = Tool(
                    name=public_name,
                    description=tool.description,
                    parameters=tool.parameters,
                    executor=bounded_executor,
                )

            async def list_tools(params: ListToolsParams) -> ToolResult[ToolUseCountMetadata]:
                query = (params.query or "").lower()
                rows = [
                    {"name": name, "active": self._active(name)}
                    for name in sorted(self.catalog)
                    if not query or query in name.lower()
                ]
                return ToolResult(
                    content=truncate_tool_text(json.dumps(rows, indent=2)), metadata=ToolUseCountMetadata()
                )

            async def inspect_tool(params: ToolNameParams) -> ToolResult[ToolUseCountMetadata]:
                tool = self.catalog.get(params.name)
                if tool is None:
                    return ToolResult(content=f"Unknown MCP tool: {params.name}", success=False)
                detail = {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": annotate_schema_ref_types(tool.parameters.model_json_schema()),
                    "active": self._active(tool.name),
                }
                return ToolResult(
                    content=truncate_tool_text(json.dumps(detail, indent=2)), metadata=ToolUseCountMetadata()
                )

            async def add_tool(params: ToolNameParams) -> ToolResult[ToolUseCountMetadata]:
                tool = self.catalog.get(params.name)
                if tool is None:
                    return ToolResult(content=f"Unknown MCP tool: {params.name}", success=False)
                self.agent._active_tools[params.name] = tool
                return ToolResult(content=f"Added MCP tool: {params.name}", metadata=ToolUseCountMetadata())

            async def remove_tool(params: ToolNameParams) -> ToolResult[ToolUseCountMetadata]:
                if params.name in self.core_names:
                    return ToolResult(content=f"Core tool cannot be removed: {params.name}", success=False)
                if params.name not in self.catalog or not self._active(params.name):
                    return ToolResult(content=f"MCP tool is not active: {params.name}", success=False)
                self.agent._active_tools.pop(params.name, None)
                return ToolResult(content=f"Removed MCP tool: {params.name}", metadata=ToolUseCountMetadata())

            return [
                Tool(
                    name="list_tools",
                    description="List MCP tools available to add and their active state.",
                    parameters=ListToolsParams,
                    executor=list_tools,
                ),
                Tool(
                    name="inspect_tool",
                    description="Inspect one MCP tool before adding it.",
                    parameters=ToolNameParams,
                    executor=inspect_tool,
                ),
                Tool(
                    name="add_tool",
                    description="Add an MCP-backed tool to the active toolbelt.",
                    parameters=ToolNameParams,
                    executor=add_tool,
                ),
                Tool(
                    name="remove_tool",
                    description="Remove an active MCP-backed tool.",
                    parameters=ToolNameParams,
                    executor=remove_tool,
                ),
                todo_tool,
            ]

        async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            await self.provider.__aexit__(exc_type, exc_val, exc_tb)

    model_kwargs: dict[str, Any] = {
        "temperature": float(config["temperature"]),
        "top_p": float(config["top_p"]),
    }
    length_tolerant_client_class = make_length_tolerant_client_class(ChatCompletionsClient)
    client = length_tolerant_client_class(
        model=config["policy_model"],
        base_url=config["model_base_url"],
        api_key="unused",
        max_tokens=int(config["max_output_tokens"]),
        kwargs=model_kwargs,
        truncation_recovery=bool(config.get("truncation_recovery", True)),
    )
    managed_tools = ManagedMCPTools()
    agent = Agent(
        client=client,
        name="apex_stirrup_agent",
        max_turns=int(config["max_turns"]),
        system_prompt=SYSTEM_PROMPT,
        tools=[managed_tools],
        finish_tool=finish_tool,
        # Chat Completions tool messages accept text only. Stirrup preserves
        # image results by moving each image into a following user message.
        text_only_tool_responses=True,
    )
    managed_tools.attach(agent)

    async with agent.session() as session:
        checkpoint_task = (
            asyncio.create_task(_checkpoint_partial_result(session, checkpoint_path))
            if checkpoint_path is not None
            else None
        )
        try:
            finish_params, history, metadata = await session.run(config["instruction"])
        except BaseException as exc:
            if checkpoint_path is not None:
                with suppress(Exception):
                    write_partial_result_checkpoint(
                        session,
                        checkpoint_path,
                        completion_status="error",
                        error=f"{type(exc).__name__}: {exc}",
                    )
            raise
        finally:
            if checkpoint_task is not None:
                checkpoint_task.cancel()
                with suppress(asyncio.CancelledError):
                    await checkpoint_task

    input_tokens, output_tokens, reasoning_tokens = _token_usage(history)
    completion_status = getattr(finish_params, "status", None)
    result = {
        "final_answer": getattr(finish_params, "final_answer", "") if finish_params is not None else "",
        "completion_status": completion_status or "max_turns",
        "completed": completion_status == "completed",
        "n_input_tokens": input_tokens,
        "n_output_tokens": output_tokens,
        "n_reasoning_tokens": reasoning_tokens,
        "n_length_truncations": client.length_truncations,
        "n_truncation_recovery_turns": client.recovery_turns,
        "trajectory": _serialize_history(history),
        "tool_metadata": metadata,
    }
    if checkpoint_path is not None:
        with suppress(Exception):
            temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
            temporary.write_text(json.dumps(result, ensure_ascii=False, default=str), encoding="utf-8")
            os.replace(temporary, checkpoint_path)
    return result
