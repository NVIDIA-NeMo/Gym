# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stirrup rollout primitives for the Archipelago environment sandbox."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import zipfile
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import Any


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
                    "parameters": tool.parameters.model_json_schema(),
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
    client = ChatCompletionsClient(
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
        "trajectory": _serialize_history(history),
        "tool_metadata": metadata,
    }
    if checkpoint_path is not None:
        with suppress(Exception):
            temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
            temporary.write_text(json.dumps(result, ensure_ascii=False, default=str), encoding="utf-8")
            os.replace(temporary, checkpoint_path)
    return result
