# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How the golden fixtures in ``fixtures/`` were made: a one-off generator, never run by the tests.

The fixtures record what the Python model gateway and hook staging did at commit 00be1884
(``environment_servers/agentic_if/model_gateway.py``, ``model_boundary.py``, ``model_streaming.py`` and
``drivers/opencode_hooks.py``, all removed since). The committed fixtures are the reference; the tests read only them.

- ``fixtures/<case>.json`` from the phase-0 spike (``example-row-*``, ``prepare-*``, ``gateway-*``) were recorded by
  the spike's own generator at 00be1884; ``index.json`` lists which are needed and which exercise dropped features.
- ``fixtures/derived-*.json``: the needed paths without the dropped features (aliases, split user turns, the
  output-token cap), recorded here by driving ``ModelGateway.forward`` with the 00be1884 test double
  ``tests.support.ModelServer``.
- ``fixtures/profile-goldens.json``: for each spec in ``specs.py``, the profile configuration the environment-server
  driver staged (``compile_hooks`` plus ``stage_hooks``' continuation keys, as ``json.dumps`` wrote it), its planted
  events, and the permission denies of that driver's ``_configuration``.

Since recording, the fixtures store OpenCode's request as ``opencode_request`` (this script writes that key; the
spike's recordings use ``SPIKE_REQUEST``) and some ``source`` descriptions of the spike's copies were reworded. Four
expected refusals carry the boundary plugin's wording, which names OpenCode's user prompts, permission configuration
and history; this script records the gateway's wording, so the three refusals it derives differ there.

Run with the Python of a checkout at 00be1884 (it must contain ``environment_servers/agentic_if``)::

    <checkout>/.venv/bin/python derive_fixtures.py --checkout <checkout> --spike-fixtures <phase0-spike/fixtures>

``--spike-fixtures`` supplies the real OpenCode 1.17.11 requests of ``spike-boundary-cross-check.json``. On 2026-10-22
the spike's own fixtures were first regenerated with this code and found identical.
"""

import argparse
import asyncio
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace


HERE = Path(__file__).parent
OUT = HERE / "fixtures"
CODE = "the Python gateway at commit 00be1884 (environment_servers/agentic_if/model_gateway.py)"
# The phase-0 spike's recordings keep OpenCode's request under this key.
SPIKE_REQUEST = "native_request"


def sse_wire(events: list[dict]) -> str:
    """The upstream bytes the 00be1884 test model server streamed for a /chat/completions call."""
    return "".join("data: " + json.dumps(event) + "\r\n\r\n" for event in events) + "data: [DONE]\r\n\r\n"


def sse_events(message: dict, *, finish: str, call_id: str = "call_1") -> list[dict]:
    """Upstream SSE events for one assistant message: content, reasoning, then tool calls in fragments."""
    chunk = {"id": "chatcmpl-1", "object": "chat.completion.chunk"}
    events = [
        chunk
        | {
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": message.get("content") or ""},
                    "finish_reason": None,
                }
            ]
        }
    ]
    if message.get("reasoning_content"):
        events.append(
            chunk
            | {
                "choices": [
                    {"index": 0, "delta": {"reasoning_content": message["reasoning_content"]}, "finish_reason": None}
                ]
            }
        )
    for position, call in enumerate(message.get("tool_calls") or []):
        name, identity = call["function"]["name"], call.get("id", call_id)
        events.append(
            chunk
            | {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": position, "id": identity, "type": "function", "function": {"name": name}}
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        )
        arguments = call["function"]["arguments"]
        step = max(1, len(arguments) // 3)
        for start in range(0, len(arguments), step):
            events.append(
                chunk
                | {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {"index": position, "function": {"arguments": arguments[start : start + step]}}
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                }
            )
    events.append(chunk | {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
    events.append(
        chunk | {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}}
    )
    return events


def tool_call_message(call_id: str, name: str, arguments: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}],
    }


def headers(session: str = "root", assistant: str | None = None, kind: str | None = None) -> dict:
    result = {"x-session-id": session}
    if assistant is not None:
        result["x-opencode-assistant-message-id"] = assistant
    if kind is not None:
        result["x-opencode-request-kind"] = kind
    return result


ROOT = headers("root", "a1")


def nested_tool(name: str) -> dict:
    """A tool whose schema has nested subschemas, $defs, array items and anyOf, for JSON-pointer targets."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Native " + name,
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "native command"},
                    "options": {
                        "type": "object",
                        "properties": {
                            "timeout": {"type": "integer"},
                            "env": {"type": "array", "items": {"$ref": "#/$defs/pair"}},
                        },
                    },
                    "mode": {"anyOf": [{"type": "string", "enum": ["fast", "slow"]}, {"type": "null"}]},
                    "paths": {"type": "array", "items": [{"type": "string"}, {"type": "string"}]},
                },
                "required": ["command"],
                "$defs": {
                    "pair": {"type": "object", "properties": {"key": {"type": "string"}}, "default": {"type": "x"}}
                },
                "additionalProperties": False,
            },
        },
    }


class EnvServerCheckout:
    """The 00be1884 modules, imported from the given checkout."""

    def __init__(self, checkout: Path) -> None:
        sys.path.insert(0, str(checkout))
        from aiohttp import ClientResponseError, RequestInfo
        from multidict import CIMultiDict
        from yarl import URL

        from environment_servers.agentic_if.drivers import opencode as driver
        from environment_servers.agentic_if.drivers import opencode_hooks as hooks
        from environment_servers.agentic_if.drivers import opencode_profile as profile
        from environment_servers.agentic_if.model_gateway import ModelGateway
        from environment_servers.agentic_if.tests import support
        from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
        from resources_servers.agentic_if.task_variants.runtime import RuntimeSpec

        self.gateway, self.support, self.driver, self.hooks, self.profile = (
            ModelGateway,
            support,
            driver,
            hooks,
            profile,
        )
        self.params_type, self.spec_type = NeMoGymResponseCreateParamsNonStreaming, RuntimeSpec
        model_server = support.ModelServer

        class RejectingServer(model_server):
            """The 00be1884 test model server, but the first calls answer with an HTTP error, as aiohttp reports it."""

            def __init__(self, result: object, *, status: int = 429, rejections: int = 1) -> None:
                super().__init__(result)
                self.status, self.rejections = status, rejections

            async def post(self, **request: object) -> object:
                response = await super().post(**request)
                if self.rejections <= 0:
                    response.ok = True
                    return response
                self.rejections -= 1
                status, url = self.status, URL("http://model.test/v1/chat/completions")

                def fail() -> None:
                    raise ClientResponseError(RequestInfo(url, "POST", CIMultiDict(), url), (), status=status)

                async def read() -> bytes:
                    return b'{"error": "rejected"}'

                response.ok, response.status, response.raise_for_status, response.content.read = (
                    False,
                    status,
                    fail,
                    read,
                )
                return response

        self.rejecting_server = RejectingServer

    def spec(self, **tools: object) -> object:
        return self.spec_type(harness="opencode", task_backend="swe", tools=tools)

    def user(self, text: str = "Fix it", system: str | None = None) -> object:
        messages = [{"role": "user", "content": text}]
        if system is not None:
            messages.insert(0, {"role": "system", "content": system})
        return self.params_type(input=messages)

    async def run(
        self,
        case: str,
        *,
        spec: object,
        params: object,
        calls: list[dict],
        source: str,
        notes: str = "",
        server: object = None,
    ) -> dict:
        """Drive ModelGateway.forward over a call sequence; record every input and output (the spike's format)."""
        server = server or self.support.ModelServer({})
        route = self.gateway(
            spec=spec, params=params, client=server, model_server="actor", run_id="attempt-a", timeout_s=30
        )
        records = []
        for call in calls:
            server.result = deepcopy(call["upstream"])
            server.stream = bool(call["body"].get("stream"))
            body = deepcopy(call["body"])
            entry = {"headers": call["headers"], "opencode_request": deepcopy(body)}
            if server.stream:
                entry["upstream_reply"] = {"sse_events": call["upstream"], "sse_wire": sse_wire(call["upstream"])}
            else:
                entry["upstream_reply"] = {"json": call["upstream"]}
            if getattr(server, "rejections", 0) > 0:
                entry["upstream_reply"] |= {"status": server.status, "error_body": '{"error": "rejected"}'}
            before = len(server.requests)
            try:
                response = await route.forward(body, headers=call["headers"])
                if hasattr(response, "body_iterator"):
                    parts = [part.encode() if isinstance(part, str) else part async for part in response.body_iterator]
                    entry["returned"] = {"sse_text": b"".join(parts).decode()}
                else:
                    entry["returned"] = {"json": json.loads(response.body)}
                entry["error"] = None
            except Exception as error:  # recorded: some fixtures pin the gateway's refusals
                entry["returned"], entry["error"] = None, f"{type(error).__name__}: {error}"
            entry["outgoing_request"] = server.requests[before]["json"] if len(server.requests) > before else None
            entry["opencode_request_unchanged"] = body == call["body"]
            records.append(entry)
        return {
            "case": case,
            "kind": "gateway",
            "source": source,
            "code": CODE,
            "notes": notes,
            "spec": spec.model_dump(mode="json", exclude_none=True),
            "params": params.model_dump(mode="json", exclude_none=True),
            "calls": records,
            "observations": route.observations,
            "issues": route.issues,
        }

    def staged_profile(self, spec: object) -> dict:
        """What the environment-server driver staged: hook-config.json, hook-events.json and the permission denies."""
        config = self.hooks.compile_hooks(spec)
        if spec.continuation is not None:
            config["prefix_receipts"] = self.hooks.prefix_receipts(spec.continuation)
            config["prefix_mode"] = spec.continuation.mode
        config["profile_sha256"] = self.profile.profile_sha256(config)
        staged = json.dumps(config)
        driver = self.driver.OpenCodeDriver(
            version="1.17.11", binary_sha256="0" * 64, context_limit=131072, max_turns=None
        )
        context = SimpleNamespace(
            spec=spec,
            params=self.user(),
            model=SimpleNamespace(base_url="http://x/v1", api_key="k"),
            sandbox=SimpleNamespace(timeout_s=60),
        )
        permission = driver._configuration(context).get("permission", {})
        return {
            "profile": json.loads(staged),
            "staged_sha256": hashlib.sha256(staged.encode()).hexdigest(),
            "plants": json.loads(
                json.dumps([event.model_dump(mode="json", exclude_none=True) for event in spec.events])
            ),
            "deny": [tool for tool, rule in permission.items() if rule == "deny"],
        }


async def derived(env_server: EnvServerCheckout, spike_fixtures: Path) -> list[dict]:
    """The derived gateway fixtures (see the module docstring)."""
    spec, user, run = env_server.spec, env_server.user, env_server.run
    definition = env_server.support.definition
    out = []
    # The spike's real OpenCode 1.17.11 requests, alias-free: literal system prompt, every tool selected, a description
    # and a JSON-pointer parameter description on bash, and the raw tool-call arguments restored in history.
    spike = json.loads((spike_fixtures / "spike-boundary-cross-check.json").read_text())
    names = [tool["function"]["name"] for tool in spike["calls"][0][SPIKE_REQUEST]["tools"]]
    calls = []
    for index, call in enumerate(spike["calls"]):
        events = json.loads(json.dumps(call["upstream_reply"]["sse_events"]).replace('"run_shell"', '"bash"'))
        calls.append(
            {
                "body": deepcopy(call[SPIKE_REQUEST]),
                "headers": headers("ses_spike", f"msg_{index}"),
                "upstream": events,
            }
        )
    out.append(
        await run(
            "derived-opencode-literal-system-and-history",
            spec=spec(
                tool_names={name: name for name in names},
                tool_descriptions={"bash": "SPIKE DESCRIPTION: run one shell command."},
                tool_parameter_descriptions={
                    "bash": {"/properties/command": "SPIKE PARAMETER DESCRIPTION: the command line."}
                },
            ),
            params=user(
                spike["calls"][0][SPIKE_REQUEST]["messages"][1]["content"], system="SPIKE LITERAL SYSTEM PROMPT."
            ),
            calls=calls,
            source="phase0-spike fixture spike-boundary-cross-check (real OpenCode 1.17.11 requests), alias-free spec;"
            " "
            "single user message equal to OpenCode's joined user text",
        )
    )
    # Descriptions and nested JSON-pointer parameter descriptions; tool-message names removed; history restored.
    tools = [nested_tool("bash"), definition("read")]
    pointers = spec(
        tool_names={"bash": "bash", "read": "read"},
        tool_descriptions={"read": "Read {{literal}} exactly", "bash": "Run one command."},
        tool_parameter_descriptions={
            "read": {"path": "Path exactly"},
            "bash": {
                "/properties/command": "The command line",
                "/properties/options/properties/env/items": "One pair",
                "/properties/mode/anyOf/0": "Speed",
                "/properties/paths/items/1": "Second path",
                "/$defs/pair/properties/key": "Pair key",
                "/properties/options": "Options object",
            },
        },
    )
    first = {
        "stream": True,
        "messages": [{"role": "system", "content": "Native"}, {"role": "user", "content": "Fix it"}],
        "tools": deepcopy(tools),
        "tool_choice": {"type": "function", "function": {"name": "bash"}},
    }
    raw_args = '{ "command" : "ls -la",  "options": {"timeout": 5} }'
    second = deepcopy(first)
    second.pop("tool_choice")
    second["messages"] += [
        tool_call_message("c-1", "bash", '{"command":"ls -la","options":{"timeout":5}}'),
        {"role": "tool", "tool_call_id": "c-1", "name": "bash", "content": "a.py\n"},
    ]
    out.append(
        await run(
            "derived-descriptions-pointers-history",
            spec=pointers,
            params=user(system="Literal system é ✓"),
            calls=[
                {
                    "body": first,
                    "headers": ROOT,
                    "upstream": sse_events(
                        {
                            "content": "Looking",
                            "tool_calls": [{"id": "c-1", "function": {"name": "bash", "arguments": raw_args}}],
                        },
                        finish="tool_calls",
                    ),
                },
                {
                    "body": second,
                    "headers": headers("root", "a2"),
                    "upstream": sse_events({"content": "Done"}, finish="stop"),
                },
            ],
            source="derived: model_boundary describe_parameters (pointers into properties, items, anyOf, $defs, tuple"
            " items), "
            "tool_choice, tool-message name removal, history restore",
        )
    )
    # OpenCode's lowercase repair without aliases: the model calls "Read", execution_tools records it, and the history
    # shows the model's call.
    lower_tools = [definition("read"), definition("bash")]
    out.append(
        await run(
            "derived-lowercase-repair",
            spec=spec(tool_names={"read": "read", "bash": "bash"}),
            params=user(),
            calls=[
                {
                    "body": {"messages": [{"role": "user", "content": "Fix it"}], "tools": deepcopy(lower_tools)},
                    "headers": ROOT,
                    "upstream": {
                        "id": "r1",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "id": "c-lower",
                                            "type": "function",
                                            "function": {"name": "Read", "arguments": '{"path": "/app/a.py"}'},
                                        }
                                    ],
                                },
                                "finish_reason": "tool_calls",
                            }
                        ],
                    },
                },
                {
                    "body": {
                        "messages": [
                            {"role": "user", "content": "Fix it"},
                            tool_call_message("c-lower", "read", '{"path":"/app/a.py"}'),
                            {"role": "tool", "tool_call_id": "c-lower", "content": "x = 1"},
                        ],
                        "tools": deepcopy(lower_tools),
                    },
                    "headers": headers("root", "a2"),
                    "upstream": {
                        "id": "r2",
                        "choices": [
                            {"index": 0, "message": {"role": "assistant", "content": "Done"}, "finish_reason": "stop"}
                        ],
                    },
                },
            ],
            source="derived: model_boundary reply mapping for OpenCode, lowercase branch, no aliases",
        )
    )
    # A streamed tool name split across fragments (held until its arguments start); an unavailable name passes through.
    split_events = [
        {
            "id": "c1",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "Inspect é",
                        "tool_calls": [{"index": 0, "id": "call-a", "type": "function", "function": {"name": "Run"}}],
                    },
                }
            ],
        },
        {
            "id": "c1",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"name": "Command", "arguments": '{"command":'}}]
                    },
                }
            ],
        },
        {
            "id": "c1",
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"pwd"}'}}]},
                    "finish_reason": "tool_calls",
                }
            ],
        },
    ]
    out.append(
        await run(
            "derived-stream-split-name",
            spec=spec(tool_names={"bash": "bash"}),
            params=user(),
            calls=[
                {
                    "body": {
                        "stream": True,
                        "messages": [{"role": "user", "content": "Fix it"}],
                        "tools": [
                            {"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}}
                        ],
                    },
                    "headers": ROOT,
                    "upstream": split_events,
                }
            ],
            source="derived from test_model_gateway.py::"
            "test_chat_stream_preserves_fragments_and_observes_model_visible_calls, "
            "alias-free",
        )
    )
    # Whole-JSON reply; history restores the model's raw arguments.
    first = {
        "messages": [{"role": "user", "content": "Fix it"}],
        "tools": [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}],
    }
    second = deepcopy(first)
    second["messages"] += [
        tool_call_message("r1", "read", '{"filePath":"/app/a.py"}'),
        {"role": "tool", "tool_call_id": "r1", "name": "read", "content": "contents"},
    ]
    out.append(
        await run(
            "derived-json-history-restore",
            spec=spec(tool_names={"read": "read"}),
            params=user(),
            calls=[
                {
                    "body": first,
                    "headers": ROOT,
                    "upstream": {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "id": "r1",
                                            "type": "function",
                                            "function": {"name": "read", "arguments": '{ "filePath" : "/app/a.py" }'},
                                        }
                                    ],
                                },
                                "finish_reason": "tool_calls",
                            }
                        ]
                    },
                },
                {
                    "body": second,
                    "headers": headers("root", "a2"),
                    "upstream": {
                        "choices": [{"message": {"role": "assistant", "content": "Done"}, "finish_reason": "stop"}]
                    },
                },
            ],
            source="derived from test_model_gateway.py::"
            "test_successful_invalid_alias_keeps_its_result_in_current_and_imported_history, "
            "alias-free",
        )
    )
    # Invalid-tool recovery presentation, root and child (the offered tools of that call).
    for child in (False, True):
        base = {
            "messages": [{"role": "user", "content": "Fix it"}],
            "tools": [
                {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
                for name in ("read", "task")
            ],
        }
        later = deepcopy(base)
        if child:
            later["tools"] = later["tools"][:1]
        third = deepcopy(later)
        third["messages"] += [
            tool_call_message(
                "unknown", "invalid", json.dumps({"tool": "Missing", "error": "Available tools: read, task"})
            ),
            {"role": "tool", "tool_call_id": "unknown", "content": "Available tools: read, task"},
        ]
        session = "child" if child else "root"
        out.append(
            await run(
                f"derived-invalid-tool-recovery-{session}",
                spec=spec(tool_names={"read": "read", "task": "task"}),
                params=user(),
                calls=[
                    {
                        "body": base,
                        "headers": headers("root"),
                        "upstream": {"choices": [{"message": {"role": "assistant", "content": "Continue"}}]},
                    },
                    {
                        "body": later,
                        "headers": headers(session),
                        "upstream": {
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "tool_calls": [
                                            {
                                                "id": "unknown",
                                                "type": "function",
                                                "function": {"name": "Missing", "arguments": '{ "path": "source" }'},
                                            }
                                        ],
                                    }
                                }
                            ]
                        },
                    },
                    {
                        "body": third,
                        "headers": headers(session),
                        "upstream": {"choices": [{"message": {"role": "assistant", "content": "Done"}}]},
                    },
                ],
                source="derived from test_model_gateway.py::"
                "test_invalid_tool_recovery_lists_only_the_tools_offered_for_that_call, alias-free",
            )
        )
    # OpenCode's repair of invalid JSON arguments: a selecting row presents the model's own call, an omitted
    # selection keeps OpenCode's repair.
    for explicit in (True, False):
        request = {
            "messages": [{"role": "user", "content": "Fix it"}],
            "tools": [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}],
        }
        repaired = {"name": "invalid", "arguments": json.dumps({"tool": "read", "error": "invalid JSON"})}
        later = deepcopy(request)
        later["messages"] += [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "bad-json", "type": "function", "function": repaired}],
            },
            {"role": "tool", "tool_call_id": "bad-json", "content": "Invalid arguments"},
        ]
        out.append(
            await run(
                f"derived-invalid-json-repair-{'selected' if explicit else 'unrestricted'}",
                spec=spec(tool_names={"read": "read"}) if explicit else spec(),
                params=user(),
                calls=[
                    {
                        "body": request,
                        "headers": headers("root"),
                        "upstream": {
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": None,
                                        "tool_calls": [
                                            {
                                                "id": "bad-json",
                                                "type": "function",
                                                "function": {"name": "read", "arguments": "{"},
                                            }
                                        ],
                                    }
                                }
                            ]
                        },
                    },
                    {
                        "body": later,
                        "headers": headers("root"),
                        "upstream": {"choices": [{"message": {"role": "assistant", "content": "Done"}}]},
                    },
                ],
                source="derived from test_model_gateway.py::"
                "test_selected_tool_json_repair_preserves_observed_arguments_without_changing_defaults",
            )
        )
    # A historical rejected call restored from a continuation prefix (root), and refused in a child session.
    receipt = {"function": {"name": "Missing", "arguments": '{ "path": "source" }'}, "available_names": ["read"]}
    prefix = {
        "info": {"id": "source"},
        "messages": [
            {"info": {"id": "u", "role": "user"}, "parts": [{"type": "text", "text": "Fix it"}]},
            {
                "info": {"id": "a", "role": "assistant"},
                "parts": [
                    {
                        "type": "tool",
                        "tool": "invalid",
                        "callID": "rejected",
                        "state": {
                            "status": "completed",
                            "input": {"tool": "Missing", "error": "Available tools: read"},
                            "output": "Available tools: read",
                            "metadata": {"nemo_gym_model_call": receipt},
                        },
                    }
                ],
            },
        ],
    }
    continuation = env_server.spec_type.model_validate(
        spec(tool_names={"read": "read"}).model_dump(mode="json")
        | {
            "continuation": {
                "mode": "restore",
                "prefix_export": prefix,
                "workspace_diff": "",
                "source_task_sha256": "a" * 64,
                "source_tools_sha256": "b" * 64,
            }
        }
    )
    history = {
        "messages": [
            {"role": "user", "content": "Fix it"},
            tool_call_message(
                "rejected", "invalid", json.dumps({"tool": "Missing", "error": "Available tools: read"})
            ),
            {"role": "tool", "tool_call_id": "rejected", "content": "Available tools: read"},
            {"role": "user", "content": "Continue"},
        ],
        "tools": [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}],
    }
    out.append(
        await run(
            "derived-historical-rejected-call",
            spec=continuation,
            params=user("Continue"),
            calls=[
                {
                    "body": history,
                    "headers": headers("new-root", "a1"),
                    "upstream": {
                        "choices": [{"message": {"role": "assistant", "content": "Done"}, "finish_reason": "stop"}]
                    },
                },
                {
                    "body": deepcopy(history),
                    "headers": headers("child", "c1"),
                    "upstream": {
                        "choices": [{"message": {"role": "assistant", "content": "never"}, "finish_reason": "stop"}]
                    },
                },
            ],
            source="derived from test_invalid_replay.py::"
            "test_captured_rejected_call_survives_export_and_continuation (invalid case)",
        )
    )
    # Compaction (JSON and stream) bypasses presentation and is recorded as such, with a literal system prompt.
    for stream in (False, True):
        message = {"role": "assistant", "content": "Native summary"}
        upstream = (
            [
                {"id": "summary", "choices": [{"index": 0, "delta": message}]},
                {"id": "summary", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ]
            if stream
            else {"choices": [{"message": message}]}
        )
        out.append(
            await run(
                f"derived-compaction-{'stream' if stream else 'json'}",
                spec=spec(tool_names={"read": "read"}),
                params=user(system="Literal system"),
                calls=[
                    {
                        "body": {
                            "messages": [
                                {"role": "system", "content": "Summarize natively"},
                                {"role": "user", "content": "History without the original task"},
                            ],
                            "tools": [],
                            **({"stream": True} if stream else {}),
                        },
                        "headers": headers("root", "summary", "compaction"),
                        "upstream": upstream,
                    }
                ],
                source="derived: OpenCode's compaction call bypasses actor presentation but is recorded",
            )
        )
    # A title-style auxiliary call (no assistant, no tools) is forwarded untouched and not recorded.
    out.append(
        await run(
            "derived-auxiliary-title",
            spec=spec(tool_names={"bash": "bash"}),
            params=user(system="Literal"),
            calls=[
                {
                    "body": {
                        "messages": [
                            {"role": "system", "content": "Generate a title"},
                            {"role": "user", "content": "Fix it"},
                        ]
                    },
                    "headers": headers("root"),
                    "upstream": {
                        "id": "title",
                        "choices": [{"message": {"role": "assistant", "content": "Title"}, "finish_reason": "stop"}],
                    },
                }
            ],
            source="derived from test_model_gateway.py::"
            "test_open_code_auxiliary_calls_are_not_actor_evidence_or_prompt_overrides",
        )
    )
    # An HTTP rejection kept for audit; OpenCode's retry succeeds (the Python gateway raised; the plugin passes the
    # reply on).
    payload = {"messages": [{"role": "user", "content": "Fix it"}]}
    out.append(
        await run(
            "derived-http-rejection",
            spec=spec(),
            params=user(),
            server=env_server.rejecting_server({}, status=429),
            calls=[
                {
                    "body": payload,
                    "headers": headers("root", "a"),
                    "upstream": {"choices": [{"message": {"role": "assistant", "content": "Done"}}]},
                },
                {
                    "body": payload,
                    "headers": headers("root", "a"),
                    "upstream": {"choices": [{"message": {"role": "assistant", "content": "Done"}}]},
                },
            ],
            source="derived: an HTTP rejection is retained without poisoning OpenCode's retry",
            notes="returned differs by design: the boundary plugin passes the upstream 429 through instead of raising",
        )
    )
    # Refusals and stream failures recorded as issues.
    read_tool = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}]
    selected_read = spec(tool_names={"read": "read"})
    never = {"choices": [{"message": {"role": "assistant", "content": "never"}}]}
    out.append(
        await run(
            "derived-missing-call-identity",
            spec=selected_read,
            params=user(),
            calls=[
                {
                    "body": {"messages": [{"role": "user", "content": "Fix it"}], "tools": deepcopy(read_tool)},
                    "headers": ROOT,
                    "upstream": {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "tool_calls": [
                                        {"type": "function", "function": {"name": "read", "arguments": "{}"}}
                                    ],
                                }
                            }
                        ]
                    },
                }
            ],
            source="derived: ModelGateway._record_dispatch refuses a tool call without an ID",
        )
    )
    out.append(
        await run(
            "derived-duplicate-call-identity",
            spec=selected_read,
            params=user(),
            calls=[
                {
                    "body": {
                        "stream": True,
                        "messages": [{"role": "user", "content": "Fix it"}],
                        "tools": deepcopy(read_tool),
                    },
                    "headers": ROOT,
                    "upstream": sse_events(
                        {
                            "tool_calls": [
                                {"id": "same", "function": {"name": "read", "arguments": "{}"}},
                                {"id": "same", "function": {"name": "read", "arguments": "{}"}},
                            ]
                        },
                        finish="tool_calls",
                    ),
                }
            ],
            source="derived: ModelGateway._record_dispatch refuses a duplicate tool call ID (stream)",
        )
    )
    out.append(
        await run(
            "derived-stream-without-terminal-choice",
            spec=spec(),
            params=user(),
            calls=[
                {
                    "body": {"stream": True, "messages": [{"role": "user", "content": "Fix it"}]},
                    "headers": ROOT,
                    "upstream": [
                        {"id": "c1", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Partial"}}]}
                    ],
                }
            ],
            source="derived from test_model_gateway.py::test_truncated_stream_is_not_a_completed_observation",
        )
    )
    out.append(
        await run(
            "derived-name-after-arguments",
            spec=spec(),
            params=user(),
            calls=[
                {
                    "body": {"stream": True, "messages": [{"role": "user", "content": "Fix it"}]},
                    "headers": ROOT,
                    "upstream": [
                        {
                            "id": "c1",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "id": "x",
                                                "type": "function",
                                                "function": {"name": "bash", "arguments": "{"},
                                            }
                                        ]
                                    },
                                }
                            ],
                        },
                        {
                            "id": "c1",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [{"index": 0, "function": {"name": "more", "arguments": "}"}}]
                                    },
                                }
                            ],
                        },
                    ],
                }
            ],
            source="derived: model_streaming refuses a function name continued after its arguments started",
        )
    )
    out.append(
        await run(
            "derived-unexpected-tools",
            spec=selected_read,
            params=user(),
            calls=[
                {
                    "body": {
                        "messages": [{"role": "user", "content": "Fix it"}],
                        "tools": [definition("read"), definition("bash"), definition("webfetch")],
                    },
                    "headers": ROOT,
                    "upstream": never,
                }
            ],
            source="derived: model_boundary inventory check (OpenCode exposed unexpected tools)",
        )
    )
    out.append(
        await run(
            "derived-root-missing-selected-tool",
            spec=spec(tool_names={"read": "read", "task": "task"}),
            params=user(),
            calls=[
                {
                    "body": {"messages": [{"role": "user", "content": "Fix it"}], "tools": [definition("read")]},
                    "headers": ROOT,
                    "upstream": never,
                }
            ],
            source="derived from test_model_boundary.py::"
            "test_child_permissions_are_intersected_not_expanded (root), alias-free",
        )
    )
    out.append(
        await run(
            "derived-child-intersection",
            spec=spec(tool_names={"read": "read", "task": "task"}),
            params=user(system="Literal system for every session"),
            calls=[
                {
                    "body": {
                        "messages": [{"role": "user", "content": "Fix it"}],
                        "tools": [definition("read"), definition("task")],
                    },
                    "headers": ROOT,
                    "upstream": {"choices": [{"message": {"role": "assistant", "content": "Delegating"}}]},
                },
                {
                    "body": {"messages": [{"role": "user", "content": "Child goal"}], "tools": [definition("read")]},
                    "headers": headers("child", "c1"),
                    "upstream": {"choices": [{"message": {"role": "assistant", "content": "Child done"}}]},
                },
            ],
            source="derived from test_model_boundary.py::"
            "test_child_permissions_are_intersected_not_expanded (child), alias-free",
        )
    )
    out.append(
        await run(
            "derived-tool-choice-unselected",
            spec=selected_read,
            params=user(),
            calls=[
                {
                    "body": {
                        "messages": [{"role": "user", "content": "Fix it"}],
                        "tools": [definition("read")],
                        "tool_choice": {"type": "function", "function": {"name": "bash"}},
                    },
                    "headers": ROOT,
                    "upstream": never,
                }
            ],
            source="derived: model_boundary refuses a tool_choice that names an unselected tool",
        )
    )
    out.append(
        await run(
            "derived-non-text-user-content",
            spec=spec(),
            params=user(),
            calls=[
                {
                    "body": {
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Fix it"},
                                    {"type": "image_url", "image_url": {"url": "data:,"}},
                                ],
                            }
                        ]
                    },
                    "headers": ROOT,
                    "upstream": never,
                }
            ],
            source="derived: model_boundary refuses non-text user content in the root session",
        )
    )
    out.append(
        await run(
            "derived-unknown-parameter-target",
            spec=spec(tool_names={"read": "read"}, tool_parameter_descriptions={"read": {"/properties/missing": "x"}}),
            params=user(),
            calls=[
                {
                    "body": {"messages": [{"role": "user", "content": "Fix it"}], "tools": [definition("read")]},
                    "headers": ROOT,
                    "upstream": never,
                }
            ],
            source="derived: model_boundary describe_parameters refuses a pointer to a missing property",
        )
    )
    # Serialization fidelity: floats, big integers, unicode, escapes, null tool_calls.
    text = 'é ✓ \U0001f600 \u2028 "quoted" \\ \t\u0001'
    numbers = {
        "id": "n1",
        "object": "chat.completion.chunk",
        "created": 1729600000,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": text, "tool_calls": None},
                "logprobs": {
                    "content": [
                        {
                            "token": "x",
                            "logprob": -0.00001,
                            "top": [1.0, 0.1, 1e16, 1e-07, 123456789.125, -0.0, 2.5e-300, 1e22],
                        }
                    ]
                },
                "finish_reason": None,
            }
        ],
        "big": 12345678901234567890123,
        "neg": -7,
    }
    tail = {
        "id": "n1",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop", "stop_reason": None}],
    }
    out.append(
        await run(
            "derived-stream-number-and-text-fidelity",
            spec=selected_read,
            params=user(),
            calls=[
                {
                    "body": {
                        "stream": True,
                        "messages": [{"role": "user", "content": "Fix it"}],
                        "tools": deepcopy(read_tool),
                    },
                    "headers": ROOT,
                    "upstream": [numbers, tail],
                }
            ],
            source="derived: model_streaming re-serializes every data event with json.dumps(ensure_ascii=False)",
        )
    )
    out.append(
        await run(
            "derived-json-number-and-text-fidelity",
            spec=spec(),
            params=user(),
            calls=[
                {
                    "body": {"messages": [{"role": "user", "content": "Fix it"}]},
                    "headers": ROOT,
                    "upstream": {
                        "id": "j1",
                        "created": 1.5,
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": 'é ✓ \U0001f600 \u2028 "q" \\',
                                    "tool_calls": None,
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 3, "ratio": 0.30000000000000004, "big": 2**70},
                    },
                }
            ],
            source="derived: the JSON reply as Starlette's JSONResponse rendered it",
        )
    )
    return out


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkout", type=Path, required=True, help="a checkout at 00be1884")
    parser.add_argument("--spike-fixtures", type=Path, required=True, help="the phase-0 spike's fixtures directory")
    arguments = parser.parse_args()
    env_server = EnvServerCheckout(arguments.checkout.resolve())
    sys.path.insert(0, str(HERE))
    from specs import SPECS  # noqa: PLC0415  (the shared specs, built with the checkout's RuntimeSpec)

    OUT.mkdir(exist_ok=True)
    written = []
    for fixture in await derived(env_server, arguments.spike_fixtures):
        (OUT / f"{fixture['case']}.json").write_text(json.dumps(fixture, indent=1, ensure_ascii=False) + "\n")
        written.append(fixture["case"])
    goldens = {
        "source": "the profile, plants and denies the environment-server driver staged at commit 00be1884 "
        "(environment_servers/agentic_if/drivers/opencode_hooks.py, drivers/opencode.py)",
        "specs": {name: env_server.staged_profile(build()) for name, build in SPECS.items()},
    }
    (OUT / "profile-goldens.json").write_text(json.dumps(goldens, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"written": written + ["profile-goldens"]}, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
