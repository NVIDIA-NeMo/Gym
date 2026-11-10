# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Codex transcript records become schema 2.0 tool attempts with explicit missing evidence."""

import asyncio
import json
import random
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

from nemo_gym.config import ModelServerRef
from nemo_gym.rollouts import enrich_run_result
from nemo_gym.trajectory import AgentInvocation, ToolCallObservation, TrajectoryRecord
from responses_api_agents.codex_agent.observability import parse_codex_observations
from responses_api_agents.codex_agent.tests.test_app import _make_agent


def _call(call_id, name="exec_command", arguments='{"cmd":"true"}'):
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": arguments}


def _result(call_id, output):
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def _exit(code):
    return f"Chunk ID: a\nProcess exited with code {code}\nOutput:\nProcess exited with code 0"


def _transcript(*records):
    lines = []
    for record in records:
        if isinstance(record, tuple):
            second, payload = record
            event = {"type": "response_item", "payload": payload}
            if second is not None:
                event["timestamp"] = second if isinstance(second, str) else f"2026-11-10T10:00:{second:06.3f}Z"
        else:
            event = record
        lines.append(json.dumps(event))
    return "\n".join(lines)


# Two responses reuse one native ID; one response requests two tools; a patch fails; a command
# returns a live session that a separate write_stdin call later polls; one call is interrupted.
TRANSCRIPT = _transcript(
    (0, {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "task"}]}),
    (1, _call("reused")),
    (1, _call("parallel", arguments='{"cmd":"exit 3"}')),
    (2.25, _result("reused", _exit(0))),
    (3.5, _result("parallel", _exit(3))),
    (4, _call("reused", arguments='{"cmd":"sleep 30"}')),
    (5, _result("reused", "Chunk ID: b\nProcess running with session ID 7\nOutput:\n")),
    (6, _call("poll", name="write_stdin", arguments='{"session_id":7,"chars":""}')),
    (8, _result("poll", _exit(0))),
    (9, {"type": "custom_tool_call", "call_id": "patch", "name": "apply_patch", "input": "*** Begin Patch"}),
    (None, {"type": "custom_tool_call_output", "call_id": "patch", "output": "patch rejected"}),
    {"type": "event_msg", "payload": {"type": "patch_apply_end", "call_id": "patch", "success": False}},
    (11, _call("interrupted")),
)


def _parse(transcript=TRANSCRIPT, **options):
    return parse_codex_observations(
        transcript,
        "0-0",
        returncode=None,
        timed_out=True,
        agent_instance="codex-agent:a",
        clock_domain="codex-wall:a",
        observation_schema_version="2.0",
        **options,
    )


def _tools(bundle):
    return [record for record in bundle.records if isinstance(record, ToolCallObservation)]


def test_transcript_records_are_distinct_attempts_with_honest_outcomes():
    bundle = _parse()
    assert bundle.schema_version == "2.0"
    tools = _tools(bundle)
    assert len({tool.span_id for tool in tools}) == len(tools) == 6
    # A reused native ID is qualified by its record position; the native ID is retained.
    assert [tool.tool_call_id for tool in tools] == [
        "reused",
        "parallel",
        "reused:record-5",
        "poll",
        "patch",
        "interrupted",
    ]
    first, parallel, second, poll, patch_tool, interrupted = tools
    assert first.native_tool_call_id == second.native_tool_call_id == "reused"
    assert (first.status, parallel.status, parallel.error_type) == ("completed", "failed", "nonzero_exit")
    # A live session is not a process exit; its later poll is a separate operation.
    assert second.status == poll.status == "unknown"
    assert (patch_tool.status, patch_tool.error_type) == ("failed", "tool_error")
    assert interrupted.status == "incomplete"
    for tool in tools:
        (timing,) = tool.timings
        assert (timing.boundary, timing.timing_source, timing.clock_domain) == (
            "transcript",
            "codex.response_item",
            "codex-wall:a",
        )
        assert tool.model_call_id is None and tool.model_call_ref is None
        assert tool.agent_instance == "codex-agent:a"
    assert first.timings[0].duration_ms == 1250 and first.timings[0].duration_source == "wall_clock"
    assert parallel.timings[0].started_at == first.timings[0].started_at
    # Unknown result time and an unfinished call keep the known start without a duration.
    for tool in (patch_tool, interrupted):
        assert tool.timings[0].completed_at is None and tool.timings[0].duration_ms is None
    invocation = next(record for record in bundle.records if isinstance(record, AgentInvocation))
    assert invocation.agent_instance == "codex-agent:a"
    requests = [item.call_id for item in invocation.conversation if item.type in {"function_call", "custom_tool_call"}]
    results = [item.call_id for item in invocation.conversation if item.type.endswith("call_output")]
    assert requests == [tool.tool_call_id for tool in tools]
    assert results == [tool.tool_call_id for tool in tools[:-1]]
    assert invocation.conversation[6].output.startswith("Chunk ID: b")
    gaps = {(gap.code, gap.span_id, gap.detail) for gap in bundle.gaps}
    assert ("tool_completion_unavailable", patch_tool.span_id, None) in gaps
    assert ("tool_completion_unavailable", interrupted.span_id, None) in gaps
    assert {
        ("tool_outcome_unavailable", second.span_id, "reused"),
        ("tool_outcome_unavailable", poll.span_id, "poll"),
    } <= gaps
    assert all(
        code != "tool_completion_unavailable" or span in {patch_tool.span_id, interrupted.span_id}
        for code, span, _ in gaps
    )
    assert _tools(_parse()) == tools


def test_reversed_unknown_and_unattributable_records_become_gaps_without_clamping():
    bundle = _parse(
        _transcript(
            (5, _call("late")),
            (4, _result("late", _exit(0))),
            ("2026-11-10T10:00:06", _call("zone-less")),
            (7, _result("orphan", _exit(1))),
        )
    )
    late, zoneless, orphan = _tools(bundle)
    assert late.status == "completed"
    assert late.timings[0].started_at is not None
    assert late.timings[0].completed_at is None and late.timings[0].duration_ms is None
    assert zoneless.timings == []
    # A result without its call keeps its completion and does not borrow a start.
    assert orphan.timings[0].started_at is None and orphan.timings[0].completed_at is not None
    assert orphan.status == "unknown"
    gaps = {(gap.code, gap.span_id) for gap in bundle.gaps}
    assert ("tool_clock_discontinuity", late.span_id) in gaps
    # Codex reports no duration, so reversed endpoints do not produce an invalid one.
    assert ("tool_timing_invalid", late.span_id) not in gaps
    assert ("tool_timing_unavailable", zoneless.span_id) in gaps
    assert ("tool_start_unavailable", orphan.span_id) in gaps


def test_results_for_concurrent_reused_ids_close_their_group_without_attribution():
    bundle = _parse(
        _transcript(
            (1, _call("t")),
            (1, _call("t")),
            (2, _result("t", _exit(0))),
            (2, _result("t", _exit(0))),
            (5, _call("t")),
            (6, _result("t", _exit(1))),
        )
    )
    twin, twin_again, later = _tools(bundle)
    # Each concurrent call has a result, but record order cannot say which.
    for tool in (twin, twin_again):
        assert tool.status == "unknown" and tool.timings[0].completed_at is None
    # The answered group closes, so a later request reusing the ID pairs normally.
    assert (later.status, later.error_type, later.timings[0].duration_ms) == ("failed", "nonzero_exit", 1000)
    invocation = bundle.records[0]
    assert [item.call_id for item in invocation.conversation] == [
        "t",
        "t:record-1",
        "t:record-2",
        "t:record-3",
        "t:record-4",
        "t:record-4",
    ]
    gaps = {(gap.code, gap.span_id) for gap in bundle.gaps}
    for tool in (twin, twin_again):
        assert {("tool_output_ambiguous", tool.span_id), ("tool_outcome_unavailable", tool.span_id)} <= gaps
    assert not any(span == later.span_id for code, span in gaps if code != "tool_clock_domain_unavailable")


def test_reused_web_search_and_patch_ids_stay_separate_and_conservative():
    search = {"type": "web_search_call", "id": "ws", "status": "completed", "action": {"type": "search", "query": "q"}}
    patch_call = {"type": "custom_tool_call", "call_id": "p", "name": "apply_patch", "input": "*** Begin Patch"}
    patch_result = {"type": "custom_tool_call_output", "call_id": "p", "output": "done"}
    patch_end = {"type": "event_msg", "payload": {"type": "patch_apply_end", "call_id": "p", "success": True}}
    bundle = _parse(
        _transcript(
            (1, search),
            (2, {**search, "status": "in_progress"}),
            (3, patch_call),
            (4, patch_result),
            patch_end,
            (5, patch_call),
            (6, patch_result),
            patch_end,
        )
    )
    first_search, second_search, first_patch, second_patch = _tools(bundle)
    # Searches keep their item ID, matching the conversation, and remain separate attempts.
    assert first_search.tool_call_id == second_search.tool_call_id == first_search.native_tool_call_id == "ws"
    assert first_search.span_id != second_search.span_id
    assert first_search.status == "completed" and first_search.timings[0].started_at is None
    assert second_search.status == "incomplete" and second_search.timings == []
    invocation = bundle.records[0]
    assert [item.id for item in invocation.conversation[:2]] == ["ws", "ws"]
    # Patch events carry only the native ID, so a reused ID cannot classify either attempt.
    assert first_patch.status == second_patch.status == "unknown"


@pytest.mark.parametrize(("copies", "status"), [([False, False], "failed"), ([True, False], "unknown")])
def test_forked_patch_event_copies_classify_only_when_they_agree(copies, status):
    events = [
        {"type": "event_msg", "payload": {"type": "patch_apply_end", "call_id": "p", "success": success}}
        for success in copies
    ]
    patch_call = {"type": "custom_tool_call", "call_id": "p", "name": "apply_patch", "input": "*** Begin Patch"}
    bundle = _parse(
        _transcript((1, patch_call), (2, {"type": "custom_tool_call_output", "call_id": "p", "output": "x"}), *events)
    )
    assert _tools(bundle)[0].status == status


def _captures():
    # Captured responses that did request these native IDs. Codex persists no response ID, so
    # the native ID alone must not select the requesting model call.
    outputs = [[_call("reused"), _call("parallel")], [_call("reused")], [_call("poll", name="write_stdin")]]
    return [
        {
            "model_call_id": f"model-{index}",
            "client_session_id": "0-0",
            "status_code": 200,
            "response": {"id": f"resp-{index}", "output": output},
        }
        for index, output in enumerate(outputs)
    ]


async def test_v2_trajectory_reports_model_join_gaps_and_is_replay_stable():
    bundle = _parse()
    raw = _tools(bundle)
    result = {
        "ng_agent_observations": bundle.model_dump(mode="json"),
        "ng_model_call_capture": {"calls": _captures()},
        "reward": 1.0,
    }
    row = {"_ng_task_index": 0, "_ng_rollout_index": 0}
    first = None
    for _ in range(3):
        await enrich_run_result(
            row, result, capture_dirs=[], observability_enabled=True, trajectory_schema_version="2.0"
        )
        saved = TrajectoryRecord.model_validate(result["ng_trajectory"])
        tools = saved.tool_calls
        assert len(tools) == len(raw) and len({tool.span_id for tool in tools}) == len(raw)
        assert all(tool.model_call_id is None for tool in tools)
        for tool in tools:
            assert any(gap.code == "tool_model_call_unavailable" and gap.span_id == tool.span_id for gap in saved.gaps)
        outputs = {tool.tool_call_id: tool.output for tool in tools}
        assert outputs["reused"] == _exit(0)
        assert outputs["reused:record-5"].startswith("Chunk ID: b")
        assert outputs["interrupted"] is None
        assert [tool.status for tool in tools] == [tool.status for tool in raw]
        assert all(tool.timings[0].boundary == "transcript" for tool in tools)
        if first is not None:
            assert saved.tool_calls == first.tool_calls and saved.gaps == first.gaps
        first = saved
    assert result["reward"] == 1.0


async def test_default_v1_output_restores_native_ids_and_is_replay_stable():
    search = {"type": "web_search_call", "id": "ws", "status": "completed", "action": {"type": "search", "query": "q"}}
    # Through a Gym model server, Codex persists the request's item ID equal to its call ID.
    request = {**_call("native"), "id": "native"}
    transcript = _transcript((1, request), (1, search), (3, _result("native", _exit(2))), (4, _call("open")))
    legacy = parse_codex_observations(transcript, "0-0", returncode=1, timed_out=False)
    result = {"ng_agent_observations": _parse(transcript).model_dump(mode="json")}
    row = {"_ng_task_index": 0, "_ng_rollout_index": 0}
    first = None
    for _ in range(2):
        await enrich_run_result(row, result, capture_dirs=[], observability_enabled=True)
        trajectory, observations = result["ng_trajectory"], result["ng_agent_observations"]
        assert trajectory["schema_version"] == "1.0" and "schema_version" not in observations
        tools = trajectory["tool_calls"]
        assert all("span_id" not in tool and "timings" not in tool for tool in tools)
        assert [(tool["tool_call_id"], tool["status"], tool["timing_source"]) for tool in tools] == [
            ("native", "failed", "artifact"),
            ("ws", "completed", "artifact"),
            ("open", "incomplete", "artifact"),
        ]
        assert tools[0]["duration_ms"] == 2000 == legacy.records[1].duration_ms
        for envelope in (trajectory["invocations"][0], observations["records"][0]):
            conversation = envelope["conversation"]
            assert [item.get("call_id", item.get("id")) for item in conversation] == ["native", "ws", "native", "open"]
            assert conversation == legacy.records[0].model_dump(mode="json")["conversation"]
        # Re-enriching v1 output re-derives unscoped trajectory gaps in the shared reader (as for
        # the v1 parser), so replay stability is checked on tools, conversations and the bundle.
        stable = (tools, trajectory["invocations"], observations)
        if first is not None:
            assert stable == first
        first = deepcopy(stable)


async def test_default_v1_output_reports_reused_ids_it_cannot_restore():
    result = {"ng_agent_observations": _parse().model_dump(mode="json")}
    await enrich_run_result(
        {"_ng_task_index": 0, "_ng_rollout_index": 0}, result, capture_dirs=[], observability_enabled=True
    )
    gaps = result["ng_trajectory"]["gaps"]
    assert "tool_native_identity_unrepresentable" in {gap["code"] for gap in gaps}
    # The outcome gap keeps its tool identity after v1 drops span IDs.
    assert {"code": "tool_outcome_unavailable", "invocation_id": "0-0", "detail": "poll"}.items() <= next(
        gap for gap in gaps if gap["code"] == "tool_outcome_unavailable" and gap.get("detail") == "poll"
    ).items()


def test_local_cli_path_emits_v2_bundle_with_run_identity(tmp_path: Path):
    agent = _make_agent(model_server=ModelServerRef(type="responses_api_models", name="policy_model"))

    @asynccontextmanager
    async def proxy(base_url, invocation_id):
        yield base_url

    class FakeProc:
        returncode = 0

        async def communicate(self):
            return b'{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}\n', b""

    async def fake_exec(*cmd, **kwargs):
        sessions = Path(kwargs["env"]["CODEX_HOME"]) / "sessions"
        sessions.mkdir()
        (sessions / "rollout.jsonl").write_text(_transcript((1, _call("c")), (2, _result("c", _exit(0)))))
        return FakeProc()

    with (
        # Local execution installs Codex lazily; this test must never install on the host.
        patch("responses_api_agents.codex_agent.app.ensure_codex", return_value=None),
        patch("responses_api_agents.codex_agent.app.Path.home", return_value=tmp_path),
        patch.object(type(agent), "resolve_model_base_url", return_value="http://model:9000/v1"),
        patch("responses_api_agents.codex_agent.app.asyncio.create_subprocess_exec", fake_exec),
        patch("responses_api_agents.codex_agent.app.rate_limit_proxy", proxy),
    ):
        _, _, bundle, _ = asyncio.run(agent._run_codex("hi", rollout_id="0-0"))
    assert bundle.schema_version == "2.0"
    invocation, tool = bundle.records
    assert invocation.agent_instance == tool.agent_instance and tool.agent_instance.startswith("codex-agent:")
    assert tool.timings[0].clock_domain.startswith("codex-wall:") and tool.status == "completed"


def test_detached_results_never_take_a_persisted_native_id():
    bundle = _parse(
        _transcript(
            (1, _call("t:record-3")),
            (2, _call("t")),
            (2, _call("t")),
            (3, _result("t", "ambiguous output")),
            (4, _call("t:record-4")),
            (5, _result("t", "second ambiguous output")),
            (6, _call("t")),
            (7, _call("t:record-6")),
        )
    )
    unrelated, twin, twin_again, other, reused, later_native = _tools(bundle)
    ids = [tool.tool_call_id for tool in _tools(bundle)]
    assert len(set(ids)) == len(ids)
    assert (unrelated.tool_call_id, other.tool_call_id, later_native.tool_call_id) == (
        "t:record-3",
        "t:record-4",
        "t:record-6",
    )
    # Allocated IDs avoid every persisted ID, including one first seen later in the stream.
    assert twin_again.tool_call_id == "t:record-2" and reused.tool_call_id == "t:record-6.1"
    results = [item.call_id for item in bundle.records[0].conversation if item.type == "function_call_output"]
    assert results == ["t:record-3.1", "t:record-5"]
    assert not set(results) & set(ids)


def _generated_stream(rng):
    natives = ["a", "b", "a:record-1", "a:record-2", "a:record-3", "a:record-2.1", "b:record-4"]
    records, payloads = [], []
    for index in range(rng.randint(1, 12)):
        native = rng.choice(natives)
        kind = rng.choice(["call", "call", "result", "search"])
        if kind == "call":
            payload = _call(native)
        elif kind == "result":
            payload = _result(native, f"out|{native}|{index}")
        else:
            payload = {"type": "web_search_call", "id": native, "status": "completed"}
        records.append((index, payload))
        payloads.append(payload)
    return _transcript(*records), payloads


async def test_generated_streams_attach_outputs_only_to_their_native_request():
    rng = random.Random(5935)
    row = {"_ng_task_index": 0, "_ng_rollout_index": 0}
    for _ in range(60):
        transcript, payloads = _generated_stream(rng)
        bundle = _parse(transcript)
        raw = _tools(bundle)
        call_ids = [tool.tool_call_id for tool in raw if tool.tool_name != "web_search"]
        assert len(set(call_ids)) == len(call_ids)
        result = {"ng_agent_observations": bundle.model_dump(mode="json")}
        await enrich_run_result(
            row, result, capture_dirs=[], observability_enabled=True, trajectory_schema_version="2.0"
        )
        for tool in result["ng_trajectory"]["tool_calls"]:
            if tool["output"] is not None:
                assert tool["output"].split("|")[1] == tool["native_tool_call_id"], (transcript, tool)
        legacy = {"ng_agent_observations": bundle.model_dump(mode="json")}
        await enrich_run_result(row, legacy, capture_dirs=[], observability_enabled=True)
        natives = [tool.native_tool_call_id for tool in raw]
        v1_ids = [tool["tool_call_id"] for tool in legacy["ng_trajectory"]["tool_calls"]]
        for native, v1_id in zip(natives, v1_ids):
            if natives.count(native) == 1:
                assert v1_id == native, (transcript, v1_ids)
