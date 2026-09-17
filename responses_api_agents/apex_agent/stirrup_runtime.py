# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stirrup rollout primitives for the Archipelago environment sandbox."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import os
import shutil
import time
import zipfile
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import Any, Callable, get_args, get_origin


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
RESUME_CHECKPOINT_SCHEMA_VERSION = 1
RESUME_MANIFEST_FILENAME = "manifest.json"
RESUME_INITIAL_FILENAME = "initial.zip"
RESUME_STIRRUP_CACHE_DIRNAME = "stirrup_cache"
RESUME_HEARTBEAT_FILENAME = "heartbeat.json"
RESUME_HEARTBEAT_INTERVAL_SECONDS = 30.0
# After an MCP tool call times out client-side the server may still be writing;
# hold the next checkpoint back this long so the snapshot is not torn.
RESUME_SETTLE_AFTER_TOOL_TIMEOUT_SECONDS = 120.0
DEFAULT_RESUME_CHECKPOINT_INTERVAL_SECONDS = 60.0

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


def make_checkpointing_client_class(base_cls: Any) -> Any:
    """Subclass Stirrup's ``ChatCompletionsClient`` with a hook awaited at the start of every model call.

    ``base_cls`` is passed in because stirrup is only importable inside the
    sandbox; the subclass is created at call time in ``run_stirrup_rollout``.
    The resume checkpointer hangs on the hook: the top of a turn is the one
    point where every tool call of the previous turn has landed and none of the
    new turn has started, so the world is quiescent.
    """

    class CheckpointingChatCompletionsClient(base_cls):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.on_generate_start: Any = None

        async def generate(self, messages: Any, tools: Any) -> Any:
            if self.on_generate_start is not None:
                await self.on_generate_start()
            return await super().generate(messages, tools)

    CheckpointingChatCompletionsClient.__name__ = f"Checkpointing{base_cls.__name__}"
    CheckpointingChatCompletionsClient.__qualname__ = CheckpointingChatCompletionsClient.__name__
    return CheckpointingChatCompletionsClient


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


def partial_result_from_session(
    session: Any,
    *,
    completion_status: str = "running",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Project Stirrup's latest completed-turn cache state into a recoverable rollout result."""
    state = getattr(session, "_current_run_state", None)
    if state is None:
        return None
    history = [*getattr(state, "full_msg_history", []), list(getattr(state, "msgs", []))]
    input_tokens, output_tokens, reasoning_tokens = _token_usage(history)
    return (extra or {}) | {
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
    extra: dict[str, Any] | None = None,
) -> bool:
    """Atomically retain the latest completed Stirrup turns for crash recovery."""
    result = partial_result_from_session(session, completion_status=completion_status, extra=extra)
    if result is None:
        return False
    if error is not None:
        result["checkpoint_error"] = error
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(temporary, destination)
    return True


async def _checkpoint_partial_result(
    session: Any,
    destination: Path,
    extra: dict[str, Any] | None = None,
    checkpointer: Any = None,
) -> None:
    checkpointed_state: Any = None
    next_heartbeat = 0.0
    while True:
        current_state = getattr(session, "_current_run_state", None)
        if current_state is not None and current_state is not checkpointed_state:
            try:
                if write_partial_result_checkpoint(session, destination, extra=extra):
                    checkpointed_state = current_state
            except Exception:
                pass
        if checkpointer is not None and checkpointer.checkpoints_written and time.monotonic() >= next_heartbeat:
            with suppress(Exception):
                checkpointer.heartbeat()
            next_heartbeat = time.monotonic() + RESUME_HEARTBEAT_INTERVAL_SECONDS
        await asyncio.sleep(PARTIAL_RESULT_CHECKPOINT_INTERVAL_SECONDS)


# ---------------------------------------------------------------------------
# Mid-rollout checkpoint and resume
# ---------------------------------------------------------------------------
# A cluster job is walled or preempted after a few hours while an Apex rollout
# can run far longer, and a rollout killed mid-flight restarts from turn zero.
# Stirrup already rebuilds a CacheState (messages, history groups, per-turn
# tool metadata) at the top of every turn and continues from one with
# session(resume=True); it only writes that state on Ctrl-C, keys it on the
# prompt alone, and knows nothing about the Archipelago world or the state the
# Apex runtime keeps outside Stirrup (active toolbelt, todo list, the
# length-tolerant client's counters). The checkpointer below fills those gaps
# at the start of a turn's model call, the one quiescent point: every tool
# call of the previous turn has landed and none of the new turn has started.
# It writes the Stirrup state, the Apex state, a world snapshot in the shape
# populate_world accepts, and the segment-1 initial snapshot the grader needs,
# then the manifest last (sha256 + size per file) so a torn write is never
# mistaken for a checkpoint. Files are versioned by turn and the previous
# generation is pruned only after the new manifest is in place, so a crash
# mid-write leaves the previous checkpoint intact. The directory is whatever
# the host mounted; durability across nodes is the host's job.


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replace_atomically(destination: Path, write: Callable[[Path], Any]) -> Any:
    """Write through a sibling temp file and os.replace so readers never see a torn file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        result = write(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def _count_assistant_turns(history: list[list[Any]]) -> int:
    return sum(1 for group in history for message in group if getattr(message, "role", None) == "assistant")


def _json_token_usage(trajectory: list[dict[str, Any]]) -> tuple[int, int, int]:
    input_tokens = output_tokens = reasoning_tokens = 0
    for message in trajectory:
        usage = message.get("token_usage") if isinstance(message, dict) else None
        if not isinstance(usage, dict):
            continue
        input_tokens += int(usage.get("input") or 0)
        answer = int(usage.get("answer") or 0)
        reasoning = int(usage.get("reasoning") or 0)
        output_tokens += answer + reasoning
        reasoning_tokens += reasoning
    return input_tokens, output_tokens, reasoning_tokens


def zip_manifest(archive_path: Path) -> list[str]:
    """File manifest of a snapshot ZIP, in the shape write_snapshot returns."""
    with zipfile.ZipFile(archive_path) as archive:
        return [info.filename for info in archive.infolist() if not info.is_dir()]


_RESUME_ROLES = ("stirrup_state", "apex_state", "world", "initial")


class ResumeCheckpoint:
    """A checkpoint whose manifest verified, ready to be resumed from."""

    def __init__(self, directory: Path, manifest: dict[str, Any], apex_state: dict[str, Any]) -> None:
        self.directory = Path(directory)
        self.manifest = manifest
        self.apex_state = apex_state
        self.turn = int(manifest.get("turn") or 0)
        self.segments = int(manifest.get("segments") or 1)
        self.generation = int(manifest.get("generation") or 0)
        self.elapsed_seconds = float(apex_state.get("elapsed_seconds") or 0.0)
        # The heartbeat charges time spent after the last checkpoint (the turn
        # that was interrupted) so a resumed rollout cannot outlive its budget.
        heartbeat = _read_heartbeat(self.directory)
        if heartbeat is not None and int(heartbeat.get("segments") or 0) == self.segments:
            self.elapsed_seconds = max(self.elapsed_seconds, float(heartbeat.get("elapsed_seconds") or 0.0))

    def path(self, role: str) -> Path:
        return self.directory / self.manifest["files"][role]["name"]

    @property
    def stirrup_state_path(self) -> Path:
        return self.path("stirrup_state")

    @property
    def world_zip(self) -> Path:
        return self.path("world")

    @property
    def initial_zip(self) -> Path:
        return self.path("initial")

    def trajectory(self) -> list[dict[str, Any]]:
        """Completed turns as JSON messages, for a row that must be written without another segment."""
        state = json.loads(self.stirrup_state_path.read_text(encoding="utf-8"))
        flattened = [message for group in state.get("full_msg_history") or [] for message in group]
        return [*flattened, *(state.get("msgs") or [])]


def _read_heartbeat(directory: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads((directory / RESUME_HEARTBEAT_FILENAME).read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def write_resume_heartbeat(directory: Path, *, elapsed_seconds: float, segments: int) -> None:
    """Record time spent since the last checkpoint; best effort, validated only for shape on load."""
    _replace_atomically(
        directory / RESUME_HEARTBEAT_FILENAME,
        lambda temporary: temporary.write_text(
            json.dumps({"elapsed_seconds": float(elapsed_seconds), "segments": int(segments)}), encoding="utf-8"
        ),
    )


def load_resume_checkpoint(directory: Path | str | None) -> ResumeCheckpoint | None:
    """Return the checkpoint in ``directory`` when its manifest verifies, otherwise None.

    A missing manifest means no checkpoint. A manifest that fails verification is
    logged and treated the same way; the caller decides whether to retry (a
    shared filesystem can return transient errors) before starting fresh.
    """
    if directory is None:
        return None
    directory = Path(directory)
    manifest_path = directory / RESUME_MANIFEST_FILENAME
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != RESUME_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(f"unsupported checkpoint schema {manifest.get('schema')!r}")
        if int(manifest.get("turn") or 0) < 1:
            raise ValueError("checkpoint records no completed turn")
        files = manifest["files"]
        for role in _RESUME_ROLES:
            entry = files[role]
            path = directory / str(entry["name"])
            if path.name != str(entry["name"]) or not path.is_file():
                raise ValueError(f"checkpoint file for {role} is missing")
            if path.stat().st_size != int(entry["size"]) or _sha256(path) != entry["sha256"]:
                raise ValueError(f"checkpoint file for {role} does not match its manifest")
        apex_state = json.loads((directory / str(files["apex_state"]["name"])).read_text(encoding="utf-8"))
        if not isinstance(apex_state, dict):
            raise ValueError("apex state is not an object")
    except Exception as exc:
        LOGGER.warning("Ignoring unusable resume checkpoint in %s: %s", directory, exc)
        return None
    return ResumeCheckpoint(directory, manifest, apex_state)


def partial_result_from_checkpoint(
    checkpoint: ResumeCheckpoint, *, completion_status: str = "timeout"
) -> dict[str, Any]:
    """Rollout result for a checkpoint that will not get another segment (budget already spent)."""
    trajectory = checkpoint.trajectory()
    input_tokens, output_tokens, reasoning_tokens = _json_token_usage(trajectory)
    return {
        "final_answer": "",
        "completion_status": completion_status,
        "completed": False,
        "n_input_tokens": input_tokens,
        "n_output_tokens": output_tokens,
        "n_reasoning_tokens": reasoning_tokens,
        "trajectory": trajectory,
        "tool_metadata": {},
        "resume_segments": checkpoint.segments,
        "resumed_from_turn": checkpoint.turn,
        "elapsed_seconds": checkpoint.elapsed_seconds,
    }


class ResumeCheckpointer:
    """Write a resumable checkpoint at turn boundaries, at most once per ``min_interval_seconds``.

    Filenames carry a generation counter seeded from the checkpoint being
    resumed, so a write never replaces a file the live manifest references;
    the previous generation is pruned only after the new manifest is in place.
    """

    def __init__(
        self,
        directory: Path | str,
        *,
        snapshot_world: Callable[[Path], Any],
        apex_state: Callable[[], dict[str, Any]],
        initial_snapshot: Path | str | None = None,
        min_interval_seconds: float = DEFAULT_RESUME_CHECKPOINT_INTERVAL_SECONDS,
        prior_elapsed_seconds: float = 0.0,
        prior_segments: int = 0,
        prior_generation: int = 0,
        resumed_turn: int | None = None,
        segment_started_at: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.directory = Path(directory)
        self._snapshot_world = snapshot_world
        self._apex_state = apex_state
        self._initial_snapshot = Path(initial_snapshot) if initial_snapshot is not None else None
        self.min_interval_seconds = float(min_interval_seconds)
        self.prior_elapsed_seconds = float(prior_elapsed_seconds)
        self.segments = int(prior_segments) + 1
        self.generation = int(prior_generation)
        self.resumed_turn = resumed_turn
        self._clock = clock
        self._segment_started_at = clock() if segment_started_at is None else float(segment_started_at)
        self._last_state: Any = None
        self._generate_seen_state: Any = None
        self._last_written_at: float | None = None
        self._not_before: float | None = None
        self.checkpoints_written = 0
        self.last_turn: int | None = resumed_turn
        self._remove_stale_temporaries()

    def _remove_stale_temporaries(self) -> None:
        """A hard kill mid-write leaves ``.<name>.tmp`` behind; nothing else writes here at segment start."""
        if not self.directory.is_dir():
            return
        for child in self.directory.iterdir():
            if child.is_file() and child.name.startswith(".") and child.name.endswith(".tmp"):
                child.unlink(missing_ok=True)

    def elapsed_seconds(self) -> float:
        return self.prior_elapsed_seconds + (self._clock() - self._segment_started_at)

    def defer(self, seconds: float) -> None:
        """Hold checkpoints back, e.g. after a tool call timed out client-side but may still be running."""
        self._not_before = max(self._not_before or 0.0, self._clock() + float(seconds))

    def eligible(self, state: Any) -> bool:
        """A state is checkpointed once, only before its turn's first model call, and not too often."""
        if state is None or state is self._last_state or state is self._generate_seen_state:
            return False
        turn = _count_assistant_turns([*state.full_msg_history, list(state.msgs)])
        if turn < 1:
            return False
        if self.checkpoints_written == 0 and self.resumed_turn is not None and turn <= self.resumed_turn:
            # The segment has not produced a new turn yet; the checkpoint on disk is current.
            return False
        now = self._clock()
        if self._not_before is not None and now < self._not_before:
            return False
        if self._last_written_at is not None and now - self._last_written_at < self.min_interval_seconds:
            return False
        return True

    async def on_generate_start(self, state: Any) -> None:
        """Model-client hook. A failed checkpoint is logged and the rollout goes on without it."""
        try:
            if self.eligible(state):
                await asyncio.to_thread(self.write, state)
        except Exception as exc:
            LOGGER.warning("Resume checkpoint failed; continuing without it: %s", exc)
        finally:
            if state is not None:
                self._generate_seen_state = state

    def heartbeat(self) -> None:
        write_resume_heartbeat(self.directory, elapsed_seconds=self.elapsed_seconds(), segments=self.segments)

    def write(self, state: Any) -> int:
        """Persist ``state`` plus the world and Apex state; returns the checkpointed turn."""
        history = [*state.full_msg_history, list(state.msgs)]
        turn = _count_assistant_turns(history)
        generation = self.generation + 1
        self.directory.mkdir(parents=True, exist_ok=True)
        names = {
            "stirrup_state": f"stirrup_state.g{generation}.t{turn}.json",
            "apex_state": f"apex_state.g{generation}.t{turn}.json",
            "world": f"world.g{generation}.t{turn}.zip",
            "initial": RESUME_INITIAL_FILENAME,
        }
        initial = self.directory / names["initial"]
        if not initial.is_file():
            if self._initial_snapshot is None or not self._initial_snapshot.is_file():
                raise FileNotFoundError("the segment-1 initial snapshot is required for a resume checkpoint")
            _replace_atomically(initial, lambda temporary: shutil.copy2(self._initial_snapshot, temporary))
        _replace_atomically(self.directory / names["world"], self._snapshot_world)
        apex_state = dict(self._apex_state()) | {
            "turn": turn,
            "elapsed_seconds": self.elapsed_seconds(),
            "segments": self.segments,
            "checkpoints_written": self.checkpoints_written + 1,
        }
        _replace_atomically(
            self.directory / names["stirrup_state"],
            lambda temporary: temporary.write_text(json.dumps(state.to_dict(), ensure_ascii=False), encoding="utf-8"),
        )
        _replace_atomically(
            self.directory / names["apex_state"],
            lambda temporary: temporary.write_text(json.dumps(apex_state, ensure_ascii=False), encoding="utf-8"),
        )
        manifest = {
            "schema": RESUME_CHECKPOINT_SCHEMA_VERSION,
            "turn": turn,
            "segments": self.segments,
            "generation": generation,
            "written_at": time.time(),
            "files": {
                role: {
                    "name": name,
                    "size": (self.directory / name).stat().st_size,
                    "sha256": _sha256(self.directory / name),
                }
                for role, name in names.items()
            },
        }
        _replace_atomically(
            self.directory / RESUME_MANIFEST_FILENAME,
            lambda temporary: temporary.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8"),
        )
        keep = set(names.values()) | {
            RESUME_MANIFEST_FILENAME,
            RESUME_HEARTBEAT_FILENAME,
            RESUME_STIRRUP_CACHE_DIRNAME,
        }
        for child in self.directory.iterdir():
            if child.name not in keep and child.is_file() and not child.name.startswith("."):
                child.unlink(missing_ok=True)
        self.generation = generation
        self._last_state = state
        self._last_written_at = self._clock()
        self.checkpoints_written += 1
        self.last_turn = turn
        # The interrupted turn's time is charged from here on.
        with suppress(Exception):
            self.heartbeat()
        return turn


def stage_stirrup_resume_state(checkpoint: ResumeCheckpoint, cache_dir: Path | str, instruction: str) -> Path:
    """Put the checkpointed Stirrup state where ``Agent.run`` looks when ``session(resume=True)``.

    Stirrup keys the cache on a hash of the prompt alone and constructs its
    CacheManager with the module default directory, so the default is pointed
    at ``cache_dir`` for this process. Only this rollout runs in the sandbox,
    so the prompt-only key is unambiguous here.
    """
    import stirrup.core.cache as stirrup_cache

    cache_dir = Path(cache_dir)
    destination = cache_dir / stirrup_cache.compute_task_hash(instruction) / "state.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint.stirrup_state_path, destination)
    stirrup_cache.DEFAULT_CACHE_DIR = cache_dir
    return destination


def collect_apex_state(
    *, agent: Any, catalog: dict[str, Any], todo_state: dict[str, Any], client: Any
) -> dict[str, Any]:
    """The rollout state Stirrup's cache does not cover.

    The client counters exist only on clients that track length truncations;
    on the stock client they read as zero and round-trip harmlessly.
    """
    return {
        "active_tools": sorted(name for name in getattr(agent, "_active_tools", {}) if name in catalog),
        "todos": [item.model_dump(mode="json") for item in todo_state.values()],
        "length_truncations": int(getattr(client, "length_truncations", 0) or 0),
        "recovery_turns": int(getattr(client, "recovery_turns", 0) or 0),
        "recover_from_truncation": bool(getattr(client, "_recover_from_truncation", False)),
    }


def restore_apex_state(
    *,
    agent: Any,
    catalog: dict[str, Any],
    todo_state: dict[str, Any],
    todo_item_cls: Any,
    client: Any,
    apex_state: dict[str, Any],
) -> dict[str, int]:
    """Inverse of collect_apex_state, applied after the session's tools exist and before run()."""
    restored_tools = 0
    for name in apex_state.get("active_tools") or []:
        tool = catalog.get(name)
        if tool is not None:
            agent._active_tools[name] = tool
            restored_tools += 1
    todo_state.clear()
    for item in apex_state.get("todos") or []:
        todo = todo_item_cls.model_validate(item)
        todo_state[todo.id] = todo
    client.length_truncations = int(apex_state.get("length_truncations") or 0)
    client.recovery_turns = int(apex_state.get("recovery_turns") or 0)
    client._recover_from_truncation = bool(apex_state.get("recover_from_truncation"))
    return {"active_tools": restored_tools, "todos": len(todo_state)}


async def run_stirrup_rollout(
    config: dict[str, Any],
    gateway_url: str,
    *,
    checkpoint_path: Path | None = None,
    resume_checkpoint: ResumeCheckpoint | None = None,
    resume_checkpoint_dir: Path | None = None,
    initial_snapshot_path: Path | None = None,
    segment_started_at: float | None = None,
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
            self.checkpointer: Any = None

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
                        # The server side may still be writing; keep the next
                        # world snapshot away from a half-written file.
                        if self.checkpointer is not None:
                            self.checkpointer.defer(RESUME_SETTLE_AFTER_TOOL_TIMEOUT_SECONDS)
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
    checkpointing_client_class = make_checkpointing_client_class(ChatCompletionsClient)
    client = checkpointing_client_class(
        model=config["policy_model"],
        base_url=config["model_base_url"],
        api_key="unused",
        max_tokens=int(config["max_output_tokens"]),
        kwargs=model_kwargs,
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

    checkpointer: ResumeCheckpointer | None = None
    if resume_checkpoint_dir is not None:
        checkpointer = ResumeCheckpointer(
            resume_checkpoint_dir,
            snapshot_world=write_snapshot,
            apex_state=lambda: collect_apex_state(
                agent=agent, catalog=managed_tools.catalog, todo_state=todo_state, client=client
            ),
            initial_snapshot=initial_snapshot_path,
            min_interval_seconds=float(
                config.get("resume_checkpoint_interval_seconds", DEFAULT_RESUME_CHECKPOINT_INTERVAL_SECONDS)
            ),
            prior_elapsed_seconds=resume_checkpoint.elapsed_seconds if resume_checkpoint is not None else 0.0,
            prior_segments=resume_checkpoint.segments if resume_checkpoint is not None else 0,
            prior_generation=resume_checkpoint.generation if resume_checkpoint is not None else 0,
            resumed_turn=resume_checkpoint.turn if resume_checkpoint is not None else None,
            segment_started_at=segment_started_at,
        )
        client.on_generate_start = lambda: checkpointer.on_generate_start(getattr(agent, "_current_run_state", None))
        managed_tools.checkpointer = checkpointer
    resumed_from_turn: int | None = None
    if resume_checkpoint is not None:
        stage_stirrup_resume_state(
            resume_checkpoint,
            resume_checkpoint.directory / RESUME_STIRRUP_CACHE_DIRNAME,
            config["instruction"],
        )
        resumed_from_turn = resume_checkpoint.turn
    segment_fields = {
        "resume_segments": checkpointer.segments if checkpointer is not None else 1,
        "resumed_from_turn": resumed_from_turn,
    }

    async with agent.session(resume=resume_checkpoint is not None) as session:
        if resume_checkpoint is not None:
            restored = restore_apex_state(
                agent=agent,
                catalog=managed_tools.catalog,
                todo_state=todo_state,
                todo_item_cls=TodoItem,
                client=client,
                apex_state=resume_checkpoint.apex_state,
            )
            LOGGER.warning(
                "resuming from turn %d as segment %d with %.0fs already spent; restored %d tool(s), %d todo(s)",
                resume_checkpoint.turn,
                resume_checkpoint.segments + 1,
                resume_checkpoint.elapsed_seconds,
                restored["active_tools"],
                restored["todos"],
            )
        checkpoint_task = (
            asyncio.create_task(_checkpoint_partial_result(session, checkpoint_path, segment_fields, checkpointer))
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
                        extra=segment_fields,
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
        "trajectory": _serialize_history(history),
        "tool_metadata": metadata,
        **segment_fields,
        "n_resume_checkpoints": checkpointer.checkpoints_written if checkpointer is not None else 0,
        "elapsed_seconds": checkpointer.elapsed_seconds() if checkpointer is not None else None,
    }
    if checkpoint_path is not None:
        with suppress(Exception):
            temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
            temporary.write_text(json.dumps(result, ensure_ascii=False, default=str), encoding="utf-8")
            os.replace(temporary, checkpoint_path)
    return result
