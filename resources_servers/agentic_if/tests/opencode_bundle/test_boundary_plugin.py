# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``if-boundary.mjs`` as OpenCode drives it: receipts, per-row denies, identity, and failing closed."""

import hashlib
import json
from pathlib import Path

import pytest

from resources_servers.agentic_if.opencode_bundle.boundary import compile_boundary
from resources_servers.agentic_if.opencode_bundle.layout import BOUNDARY_PLUGIN, SOURCE_DIR

from .harness import BASE_URL, call, make_bundle, needs_node, run_boundary, sse_wire, upstream
from .specs import SELECTED, private_like_spec, unrestricted_spec


pytestmark = needs_node

SYSTEM = "Literal row system prompt: keep {{cwd}} literal. é"
OK = upstream(
    json.dumps({"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}).encode()
)


def opencode_tools(names: list[str]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"built-in {name}",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "command": {"type": "string"}},
                },
            },
        }
        for name in names
    ]


def request(names: list[str] = SELECTED, **extra: object) -> dict:
    return {
        "model": "dummy_model",
        "messages": [
            {"role": "system", "content": "OpenCode's own system prompt"},
            {"role": "developer", "content": "Environment and AGENTS.md text"},
            {"role": "user", "content": "Fix it"},
        ],
        "tools": opencode_tools(names),
        "tool_choice": "auto",
        **extra,
    }


def test_load_and_configuration_receipts_and_per_row_denies(tmp_path: Path) -> None:
    boundary = compile_boundary(private_like_spec())
    permission = {"*": "allow", "webfetch": "allow", "edit": {"*.lock": "deny"}, "bash": {"*": "allow"}}
    run = run_boundary(tmp_path, boundary=boundary, calls=[call(request(), OK)], system=SYSTEM, permission=permission)
    loaded, configured = run.rows[0], run.rows[1]
    assert loaded == {
        "kind": "loaded",
        "plugin": "if-boundary",
        "boundary_version": 1,
        "config_sha256": hashlib.sha256((run.bundle / "boundary.json").read_bytes()).hexdigest(),
        "plugin_sha256": hashlib.sha256((SOURCE_DIR / BOUNDARY_PLUGIN).read_bytes()).hexdigest(),
        "pid": loaded["pid"],
    }
    assert configured == {
        "kind": "configured",
        "provider": "nemo_gym",
        "base_url": BASE_URL,
        "denied": boundary["deny"],
        "instructions": [str(tmp_path / "agent-session" / "instructions.md")],
        "system_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
    }
    # The deployment's rules stay first; each unselected tool is denied last, so its rule wins.
    assert list(run.permission) == ["*", "edit", "bash", *boundary["deny"]]
    assert run.permission["edit"] == {"*.lock": "deny"}
    assert all(run.permission[tool] == "deny" for tool in boundary["deny"])


def test_presentation_on_the_wire_and_identity_headers_removed(tmp_path: Path) -> None:
    run = run_boundary(
        tmp_path, boundary=compile_boundary(private_like_spec()), calls=[call(request(), OK)], system=SYSTEM
    )
    sent = json.loads(run.calls[0]["sent"])
    assert sent["messages"] == [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Fix it"}]
    tools = {tool["function"]["name"]: tool["function"] for tool in sent["tools"]}
    assert list(tools) == SELECTED
    assert all(tools[name]["description"] == f"Row description of {name}." for name in SELECTED)
    assert tools["read"]["parameters"]["properties"]["path"]["description"] == "Absolute path of the file."
    assert tools["bash"]["parameters"]["properties"]["command"]["description"] == "One shell command."
    assert "description" not in tools["glob"]["parameters"]["properties"]["path"]
    # OpenCode's and the agent's headers pass; the boundary's identity headers never leave the process.
    assert run.calls[0]["sent_headers"] == {"Content-Type": "application/json", "X-Session-Id": "activation:root"}
    observation = run.records.observations[0]
    assert observation["source"] == "model" and observation["session_id"] == "root"
    assert (
        observation["assistant_message_id"] == "msg-1"
        and observation["response"]["choices"][0]["message"]["content"] == "ok"
    )


def test_without_an_instructions_file_the_opencode_system_prompt_stays(tmp_path: Path) -> None:
    run = run_boundary(
        tmp_path,
        boundary=compile_boundary(unrestricted_spec()),
        calls=[call(request(["bash", "read", "skill"]), OK)],
        instructions=["/abs/AGENTS.md", "instructions.md"],
    )
    assert json.loads(run.calls[0]["sent"]) == request(["bash", "read", "skill"])
    assert run.rows[1]["instructions"] == [] and run.rows[1]["system_sha256"] is None


def test_an_empty_instructions_file_is_an_empty_literal_system_prompt(tmp_path: Path) -> None:
    run = run_boundary(
        tmp_path, boundary=compile_boundary(unrestricted_spec()), calls=[call(request(), OK)], system=""
    )
    assert json.loads(run.calls[0]["sent"])["messages"] == [
        {"role": "system", "content": ""},
        {"role": "user", "content": "Fix it"},
    ]


def test_compaction_is_recorded_as_such_and_bypasses_presentation(tmp_path: Path) -> None:
    body = {
        "messages": [{"role": "system", "content": "Summarize"}, {"role": "user", "content": "History"}],
        "stream": True,
    }
    events = [
        {"id": "s", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Summary"}}]},
        {"id": "s", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    run = run_boundary(
        tmp_path,
        boundary=compile_boundary(private_like_spec()),
        calls=[
            call(
                body,
                upstream(sse_wire(events), content_type="text/event-stream"),
                agent="compaction",
                assistant="msg-s",
            )
        ],
        system=SYSTEM,
    )
    assert json.loads(run.calls[0]["sent"]) == body
    assert run.records.observations == [
        {
            "source": "compaction",
            "session_id": "root",
            "assistant_message_id": "msg-s",
            "request": body,
            "execution_tools": {},
            "response": {
                "id": "s",
                "object": "chat.completion",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "Summary"}, "finish_reason": "stop"}
                ],
            },
        }
    ]


def test_a_call_without_one_saved_assistant_and_without_tools_is_forwarded_unrecorded(tmp_path: Path) -> None:
    body = {"messages": [{"role": "user", "content": "Title this"}]}
    entry = call(body, OK, assistant="msg-t")
    entry["duplicate_assistant"] = True
    run = run_boundary(tmp_path, boundary=compile_boundary(private_like_spec()), calls=[entry], system=SYSTEM)
    assert json.loads(run.calls[0]["sent"]) == body
    assert run.records.observations == [] and run.records.issues == []


def test_upstream_http_errors_reach_opencode_unchanged(tmp_path: Path) -> None:
    rejected = upstream(b'{"error": {"message": "maximum context length exceeded"}}', status=400)
    actor = call(request(), rejected)
    title = call({"messages": [{"role": "user", "content": "Title this"}]}, rejected, assistant=None)
    run = run_boundary(tmp_path, boundary=compile_boundary(private_like_spec()), calls=[actor, title], system=SYSTEM)
    assert [entry["status"] for entry in run.calls] == [400, 400]
    assert run.returned(0) == b'{"error": {"message": "maximum context length exceeded"}}'
    assert run.records.observations[0]["transport_error"] == {"phase": "http_rejected", "status": 400}
    assert "response" not in run.records.observations[0]
    # As the 00be1884 gateway: an unobserved call's HTTP error is an issue, an observed one is a transport error.
    assert run.records.issues == ["ClientResponseError"]


# The shape a hosted endpoint sent under load in a real run: HTTP 200, keep-alive comments, then an error event.
RATE_LIMITED = {
    "error": {
        "message": "Priority-based rate limit exceeded ... Rate limit type: tokens",
        "type": "rate_limit_error",
        "code": "rate_limit_exceeded",
    },
    "status": 429,
}
STREAM_ERROR = b": keep-alive\n\n: keep-alive\n\nevent: error\ndata: " + json.dumps(RATE_LIMITED).encode() + b"\n\n"


def stream_reply(body: bytes) -> dict:
    return upstream(body, content_type="text/event-stream")


def chunk_events(*texts: str) -> list[dict]:
    return [{"id": "c", "choices": [{"index": 0, "delta": {"content": text}}]} for text in texts]


def test_an_error_event_inside_a_200_stream_is_a_transport_error_and_reaches_opencode_unchanged(
    tmp_path: Path,
) -> None:
    retried = stream_reply(
        sse_wire([*chunk_events("Done"), {"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}])
    )
    calls = [call(request(stream=True), stream_reply(STREAM_ERROR)), call(request(stream=True), retried)]
    run = run_boundary(tmp_path, boundary=compile_boundary(private_like_spec()), calls=calls, system=SYSTEM)
    # OpenCode reads the whole stream, as the model server sent it, and handles the error event itself.
    assert run.calls[0]["error"] is None and run.calls[0]["status"] == 200
    assert run.returned(0) == STREAM_ERROR
    first, second = run.records.observations
    assert first["transport_error"] == {
        "phase": "stream_error",
        "status": 429,
        "message": "Priority-based rate limit exceeded ... Rate limit type: tokens",
        "chunks": 0,
    }
    assert "response" not in first
    # OpenCode's retry is the same assistant turn; it completes as usual.
    assert (second["session_id"], second["assistant_message_id"]) == (
        first["session_id"],
        first["assistant_message_id"],
    )
    assert second["response"]["choices"][0]["message"]["content"] == "Done"
    assert run.records.issues == []


@pytest.mark.parametrize(
    "body, status, text",
    [
        (b'data: {"object": "error", "message": "Rate limited", "code": 429}\n\n', 429, "Rate limited"),
        (b"event: error\ndata: upstream overloaded\n\n", None, "upstream overloaded"),
        (b'event:error\ndata: {"error": "busy", "code": "503"}\n\ndata: [DONE]\n\n', 503, "busy"),
        (b"event: error\n\n", None, ""),
    ],
    ids=["error-object-without-event-field", "text-data", "string-code-then-done", "no-data"],
)
def test_error_events_without_the_usual_shape(tmp_path: Path, body: bytes, status: int | None, text: str) -> None:
    run = run_boundary(
        tmp_path,
        boundary=compile_boundary(private_like_spec()),
        calls=[call(request(stream=True), stream_reply(body))],
        system=SYSTEM,
    )
    assert run.calls[0]["error"] is None and run.returned(0) == body
    assert run.records.observations[0]["transport_error"] == {
        "phase": "stream_error",
        "status": status,
        "message": text,
        "chunks": 0,
    }
    assert run.records.issues == []


def test_chunks_before_an_error_event_are_rewritten_as_usual_and_counted(tmp_path: Path) -> None:
    before = chunk_events("Partial ", "answer")
    after = b'event: error\ndata: {"error": {"message": "Internal error", "code": 500}}\n\ndata: [DONE]\n\n'
    run = run_boundary(
        tmp_path,
        boundary=compile_boundary(private_like_spec()),
        calls=[call(request(stream=True), stream_reply(sse_wire(before)[: -len(b"data: [DONE]\r\n\r\n")] + after))],
        system=SYSTEM,
    )
    rewritten = "".join("data: " + json.dumps(event, ensure_ascii=False) + "\n\n" for event in before).encode()
    assert run.calls[0]["error"] is None and run.returned(0) == rewritten + after
    error = run.records.observations[0]["transport_error"]
    assert error == {"phase": "stream_error", "status": 500, "message": "Internal error", "chunks": 2}
    assert run.records.issues == []


def test_an_error_event_after_the_terminal_event_is_an_ordinary_event(tmp_path: Path) -> None:
    events = [*chunk_events("Done"), {"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}]
    late = b'event: error\ndata: {"error": {"message": "late"}}\n\n'
    run = run_boundary(
        tmp_path,
        boundary=compile_boundary(private_like_spec()),
        calls=[call(request(stream=True), stream_reply(sse_wire(events) + late))],
        system=SYSTEM,
    )
    assert run.returned(0).endswith(b'event: error\ndata: {"error": {"message": "late"}}\n\n')
    observation = run.records.observations[0]
    assert "transport_error" not in observation and observation["response"]["choices"][0]["finish_reason"] == "stop"
    assert run.records.issues == []


def test_an_unobserved_call_with_an_error_event_is_an_issue(tmp_path: Path) -> None:
    title = call(
        {"messages": [{"role": "user", "content": "Title this"}], "stream": True},
        stream_reply(STREAM_ERROR),
        assistant=None,
    )
    run = run_boundary(tmp_path, boundary=compile_boundary(private_like_spec()), calls=[title], system=SYSTEM)
    assert run.returned(0) == STREAM_ERROR
    assert run.records.observations == [] and run.records.issues == ["ClientResponseError"]


def test_a_stream_that_ends_without_a_terminal_or_error_event_is_still_an_issue(tmp_path: Path) -> None:
    truncated = sse_wire(chunk_events("Partial"))[: -len(b"data: [DONE]\r\n\r\n")]
    run = run_boundary(
        tmp_path,
        boundary=compile_boundary(private_like_spec()),
        calls=[call(request(stream=True), stream_reply(truncated))],
        system=SYSTEM,
    )
    assert run.calls[0]["error"] == {"name": "ValueError", "message": "model stream has no complete terminal event"}
    assert run.records.issues == ["ValueError"]
    assert "response" not in run.records.observations[0] and "transport_error" not in run.records.observations[0]


def test_a_cancelled_stream_is_an_issue_and_leaves_no_reply(tmp_path: Path) -> None:
    events = [
        {"id": "c", "choices": [{"index": 0, "delta": {"role": "assistant", "content": f"part {n}"}}]}
        for n in range(50)
    ]
    entry = call(
        request(stream=True), upstream(sse_wire(events), content_type="text/event-stream", chunk=64), cancel=True
    )
    run = run_boundary(tmp_path, boundary=compile_boundary(private_like_spec()), calls=[entry], system=SYSTEM)
    assert run.returned(0).startswith(b"data: {")
    assert run.records.issues == ["CancelledError"]
    assert "response" not in run.records.observations[0]


@pytest.mark.parametrize(
    "boundary, reason",
    [
        (None, "ENOENT"),
        (b"{not json", "JSON"),
        ({"boundary_version": 2}, "unsupported boundary_version 2"),
    ],
    ids=["missing", "unparseable", "wrong-version"],
)
def test_a_bad_configuration_refuses_every_call(tmp_path: Path, boundary: object, reason: str) -> None:
    run = run_boundary(tmp_path, boundary=boundary, calls=[call(request(), OK)], system=SYSTEM, global_probe=True)
    assert run.rows[0]["kind"] == "load_error" and reason in run.rows[0]["message"]
    assert run.calls[0]["sent"] is None
    assert run.calls[0]["error"]["name"] == "AgenticIfBoundaryError"
    assert "the boundary did not load" in run.calls[0]["error"]["message"]
    assert run.records.issues[0] == "RuntimeError"
    assert run.global_probe["refused"] is True


def test_calls_are_refused_while_the_profile_plugin_is_not_loaded(tmp_path: Path) -> None:
    run = run_boundary(
        tmp_path, boundary=compile_boundary(private_like_spec()), calls=[call(request(), OK)], register_profile=False
    )
    assert run.calls[0]["sent"] is None
    assert "the if-profile plugin did not load" in run.calls[0]["error"]["message"]


def test_ambiguous_instructions_are_a_configuration_error(tmp_path: Path) -> None:
    other = tmp_path / "other" / "instructions.md"
    other.parent.mkdir()
    other.write_text("Another system prompt")
    run = run_boundary(
        tmp_path,
        boundary=compile_boundary(private_like_spec()),
        calls=[call(request(), OK)],
        system=SYSTEM,
        instructions=[str(other)],
    )
    assert run.records.configurations[0]["kind"] == "config_error"
    assert "ambiguous instructions files" in run.records.configurations[0]["message"]
    assert run.calls[0]["sent"] is None and "was not configured" in run.calls[0]["error"]["message"]


def test_a_missing_provider_installs_no_wrapper_and_the_process_guard_refuses_bypassing_calls(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path, compile_boundary(private_like_spec()))
    case = {
        "plugin": str(bundle / BOUNDARY_PLUGIN),
        "base_url": BASE_URL,
        "instructions": [],
        "register_profile": True,
        "provider_id": "other",
        "global_probe": True,
        "calls": [],
    }
    import subprocess

    from .harness import NODE, RUNNER

    (tmp_path / "case.json").write_text(json.dumps(case))
    finished = subprocess.run(
        [NODE, str(RUNNER), str(tmp_path / "case.json"), str(tmp_path / "result.json")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert finished.returncode == 0, finished.stderr
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["provider_fetch_installed"] is False
    assert result["global_probe"]["refused"] is True
    rows = [json.loads(line) for line in (bundle / "receipts" / "model-calls.jsonl").read_text().splitlines()]
    assert rows[1] == {"kind": "config_error", "message": "provider nemo_gym is not configured"}
    assert rows[2]["kind"] == "issue" and rows[2]["error"] == "ValueError" and rows[2]["call"] is None


def test_the_process_guard_refuses_a_bypassing_call_even_when_configured(tmp_path: Path) -> None:
    run = run_boundary(
        tmp_path, boundary=compile_boundary(private_like_spec()), calls=[call(request(), OK)], global_probe=True
    )
    assert run.calls[0]["error"] is None
    assert run.global_probe == {
        "refused": True,
        "name": "AgenticIfBoundaryError",
        "message": "agentic-if boundary refused the model call: "
        "ValueError: model call bypassed the agentic-if boundary",
    }


def test_unwritable_receipts_refuse_every_call(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path, compile_boundary(private_like_spec()))
    (bundle / "receipts").rmdir()
    run = run_boundary(tmp_path, boundary=None, calls=[call(request(), OK)], bundle=bundle)
    assert run.rows == []
    assert run.calls[0]["sent"] is None and "cannot write the boundary receipts" in run.calls[0]["error"]["message"]


@pytest.mark.parametrize(
    "entry, message",
    [
        (call(request(), OK, url=f"{BASE_URL}/completions"), "unsupported model endpoint: POST"),
        ({**call(request(), OK), "skip_headers": True}, "without an OpenCode session identity"),
        (call("{not json", OK), "JSONDecodeError"),
        (call([1, 2], OK), "AttributeError"),
    ],
    ids=["endpoint", "identity", "json", "not-an-object"],
)
def test_malformed_calls_are_refused(tmp_path: Path, entry: dict, message: str) -> None:
    run = run_boundary(tmp_path, boundary=compile_boundary(private_like_spec()), calls=[entry], system=SYSTEM)
    assert run.calls[0]["sent"] is None and message in run.calls[0]["error"]["message"]
    assert len(run.records.issues) == 1 and run.records.observations == []
