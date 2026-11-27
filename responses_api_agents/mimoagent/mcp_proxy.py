# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MCP tools for the DefaultAgent path.

The DefaultAgent runs on the trainer (a Ray actor, outside the task pod), so it
cannot open a socket to the pod-local MCP servers directly. Instead each MCP
operation is tunnelled through ``env.execute()`` into the pod's main container,
which runs the one-shot ``mcp_bridge.py`` (``--list`` / ``--call``) and dials
the sidecar over 127.0.0.1. This mirrors how codex/claude-code reach MCP
servers, just with the exec channel as transport (the agent is out-of-pod).

Each remote MCP tool becomes one :class:`McpProxyTool` (a normal ``BaseTool``)
named ``mcp__<server>__<fn>``, registered into the agent's ``ToolRegistry`` — so
the policy model sees it in the tool catalogue and dispatch flows through the
unchanged ``_parse_tool_call`` / ``execute_action`` path.

The endpoint URL only ever appears on the exec command line the framework
builds; the policy model sees just the tool definitions and the tool results.
"""

from __future__ import annotations

import json
import logging
import shlex
from typing import Any

from mimoagent.tools.base import BaseTool, ToolException, ToolOutput


logger = logging.getLogger(__name__)

ONESHOT_MARKER = "__MCP_ONESHOT__"
"""Must match mcp_bridge.py: the one-shot result is a single line
``__MCP_ONESHOT__<compact json>`` so we can recover it from a merged
stdout+stderr exec stream."""

DEFAULT_CALL_TIMEOUT = 120
DEFAULT_LIST_TIMEOUT = 120


def _extract_oneshot(output: str) -> dict:
    """Pull the marker-prefixed JSON payload out of an exec output blob.

    Raises ToolException if no marker line is present (the bridge failed before
    it could emit, e.g. python/import error) — surfaced to the model as a tool
    error rather than a silent empty result.
    """
    payload_line = None
    for line in (output or "").splitlines():
        idx = line.find(ONESHOT_MARKER)
        if idx != -1:
            payload_line = line[idx + len(ONESHOT_MARKER) :]  # last match wins
    if payload_line is None:
        raise ToolException(f"MCP bridge produced no result marker; raw output tail:\n{(output or '')[-800:]}")
    try:
        return json.loads(payload_line)
    except json.JSONDecodeError as e:
        raise ToolException(f"MCP bridge emitted invalid JSON: {e}; line={payload_line[:400]!r}") from e


def _bridge_cmd(bridge_python: str, bridge_script: str, url: str, server: str, *extra: str) -> str:
    argv = [bridge_python, bridge_script, "--url", url, "--name", server, *extra]
    return " ".join(shlex.quote(a) for a in argv)


class McpProxyTool(BaseTool):
    """One remote MCP tool, invoked one-shot via the in-pod bridge over env.execute."""

    def __init__(
        self,
        *,
        server: str,
        fn: str,
        description: str,
        input_schema: dict,
        url: str,
        bridge_python: str,
        bridge_script: str,
        timeout: int = DEFAULT_CALL_TIMEOUT,
    ):
        super().__init__({})
        self._server = server
        self._fn = fn
        self._full_name = f"mcp__{server}__{fn}"
        self._description = description or f"MCP tool {fn} on server {server}."
        self._input_schema = input_schema or {"type": "object", "properties": {}}
        self._url = url
        self._bridge_python = bridge_python
        self._bridge_script = bridge_script
        self._timeout = timeout

    @property
    def name(self) -> str:
        return self._full_name

    @property
    def description(self) -> str:
        return self._description

    def get_function_parameters(self) -> dict[str, Any]:
        return self._input_schema

    def execute(self, params: Any, context: dict[str, Any] | None = None) -> ToolOutput:
        context = context or {}
        env = context.get("env")
        if env is None:
            raise ToolException(f"{self._full_name}: no env in execution context")
        args_json = json.dumps(params if isinstance(params, dict) else {}, ensure_ascii=False)
        cmd = _bridge_cmd(
            self._bridge_python,
            self._bridge_script,
            self._url,
            self._server,
            "--call",
            self._fn,
            "--args-json",
            args_json,
        )
        res = env.execute(cmd, timeout=self._timeout)
        payload = _extract_oneshot(res.get("output", ""))
        if not payload.get("ok"):
            return ToolOutput(output=f"[mcp error] {payload.get('error', 'unknown')}", success=False)
        text = payload.get("content", "")
        if payload.get("structured") is not None and not text:
            text = json.dumps(payload["structured"], ensure_ascii=False)
        return ToolOutput(output=text, success=not payload.get("is_error", False))


def discover_mcp_tools(
    env,
    servers: dict[str, dict],
    bridge_python: str,
    bridge_script: str,
    *,
    call_timeout: int = DEFAULT_CALL_TIMEOUT,
    list_timeout: int = DEFAULT_LIST_TIMEOUT,
) -> list[McpProxyTool]:
    """tools/list every server (via the in-pod bridge) → McpProxyTool instances.

    A server that fails to list is logged and skipped (its tools are absent
    from the catalogue) rather than crashing the whole agent — but if EVERY
    server fails, that's raised, since a general_agent task with no MCP tools
    cannot proceed.
    """
    tools: list[McpProxyTool] = []
    failures: list[str] = []
    for server, spec in servers.items():
        url = spec.get("url") if isinstance(spec, dict) else None
        if not url:
            failures.append(f"{server}: no url in {spec!r}")
            continue
        cmd = _bridge_cmd(bridge_python, bridge_script, url, server, "--list")
        try:
            res = env.execute(cmd, timeout=list_timeout)
            payload = _extract_oneshot(res.get("output", ""))
            if not payload.get("ok"):
                failures.append(f"{server}: {payload.get('error', 'list failed')}")
                continue
            for t in payload.get("tools", []):
                tools.append(
                    McpProxyTool(
                        server=server,
                        fn=t["name"],
                        description=t.get("description", ""),
                        input_schema=t.get("inputSchema") or {"type": "object", "properties": {}},
                        url=url,
                        bridge_python=bridge_python,
                        bridge_script=bridge_script,
                        timeout=call_timeout,
                    )
                )
        except Exception as e:  # noqa: BLE001 — collect, decide after all servers
            failures.append(f"{server}: {type(e).__name__}: {e}")
    if failures:
        logger.warning("MCP tools/list issues: %s", "; ".join(failures))
    if not tools and failures:
        raise ToolException(
            f"no MCP tools discovered from any server ({len(failures)} failed): " + "; ".join(failures)
        )
    return tools
