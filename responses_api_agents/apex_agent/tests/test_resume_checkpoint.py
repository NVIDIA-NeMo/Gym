# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A rollout killed mid-flight continues from its last turn boundary instead of turn zero."""

from __future__ import annotations

import ast
import inspect
import json
import textwrap
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from responses_api_agents.apex_agent import stirrup_runtime
from responses_api_agents.apex_agent.stirrup_runtime import (
    RESUME_INITIAL_FILENAME,
    RESUME_MANIFEST_FILENAME,
    ResumeCheckpointer,
    collect_apex_state,
    load_resume_checkpoint,
    partial_result_from_checkpoint,
    restore_apex_state,
    zip_manifest,
)


class _Message:
    def __init__(self, role: str, content: str, *, input_tokens: int = 0, answer_tokens: int = 0) -> None:
        self.role = role
        self.content = content
        self.token_usage = SimpleNamespace(input=input_tokens, answer=answer_tokens, reasoning=0)

    def model_dump(self, *, mode: str) -> dict:
        return {"role": self.role, "content": self.content, "token_usage": vars(self.token_usage)}


class _State:
    """Stand-in for Stirrup's CacheState: history groups, live messages, and to_dict()."""

    def __init__(self, *messages: _Message, history: list[list[_Message]] | None = None) -> None:
        self.full_msg_history = history or []
        self.msgs = list(messages)

    def to_dict(self) -> dict:
        return {
            "msgs": [message.model_dump(mode="json") for message in self.msgs],
            "full_msg_history": [
                [message.model_dump(mode="json") for message in group] for group in self.full_msg_history
            ],
            "run_metadata_by_turn": {},
            "task_hash": "abc",
            "agent_name": "test",
        }


def _turn_state(turns: int) -> _State:
    messages = [_Message("system", "sys"), _Message("user", "task")]
    for index in range(turns):
        messages.append(_Message("assistant", f"turn {index + 1}", input_tokens=10, answer_tokens=5))
        messages.append(_Message("tool", f"result {index + 1}"))
    return _State(*messages)


class _World:
    """A fake world whose snapshot content the test controls."""

    def __init__(self) -> None:
        self.content = "v1"
        self.snapshots = 0

    def snapshot(self, destination: Path) -> list[str]:
        self.snapshots += 1
        with zipfile.ZipFile(destination, "w") as archive:
            archive.writestr("filesystem/report.txt", self.content)
        return ["filesystem/report.txt"]


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _checkpointer(tmp_path: Path, world: _World | None = None, **kwargs) -> tuple[ResumeCheckpointer, _World, _Clock]:
    world = world or _World()
    clock = _Clock()
    initial = tmp_path / "initial-src.zip"
    with zipfile.ZipFile(initial, "w") as archive:
        archive.writestr("filesystem/report.txt", "pristine")
    apex_state = kwargs.pop("apex_state", lambda: {"active_tools": ["read_file"], "todos": []})
    checkpointer = ResumeCheckpointer(
        tmp_path / "ckpt",
        snapshot_world=world.snapshot,
        apex_state=apex_state,
        initial_snapshot=initial,
        clock=clock,
        **kwargs,
    )
    return checkpointer, world, clock


def test_checkpoint_round_trip_restores_turn_world_and_apex_state(tmp_path: Path) -> None:
    checkpointer, world, clock = _checkpointer(tmp_path, prior_elapsed_seconds=100.0, prior_segments=1)
    clock.now += 30.0

    turn = checkpointer.write(_turn_state(2))
    loaded = load_resume_checkpoint(checkpointer.directory)

    assert turn == 2
    assert loaded is not None
    assert loaded.turn == 2
    assert loaded.segments == 2
    assert loaded.elapsed_seconds == pytest.approx(130.0)
    assert loaded.apex_state["active_tools"] == ["read_file"]
    assert loaded.apex_state["checkpoints_written"] == 1
    assert json.loads(loaded.stirrup_state_path.read_text())["task_hash"] == "abc"
    assert zip_manifest(loaded.world_zip) == ["filesystem/report.txt"]
    with zipfile.ZipFile(loaded.initial_zip) as archive:
        assert archive.read("filesystem/report.txt") == b"pristine"
    assert [message["role"] for message in loaded.trajectory()][-1] == "tool"


def test_new_generation_replaces_the_old_one_but_keeps_the_initial_snapshot(tmp_path: Path) -> None:
    checkpointer, world, clock = _checkpointer(tmp_path)
    checkpointer.write(_turn_state(1))
    world.content = "v2"
    clock.now += 120.0

    checkpointer.write(_turn_state(3))

    names = sorted(path.name for path in checkpointer.directory.iterdir())
    assert names == sorted(
        [
            RESUME_MANIFEST_FILENAME,
            RESUME_INITIAL_FILENAME,
            "heartbeat.json",
            "apex_state.g2.t3.json",
            "stirrup_state.g2.t3.json",
            "world.g2.t3.zip",
        ]
    )
    loaded = load_resume_checkpoint(checkpointer.directory)
    assert loaded is not None and loaded.turn == 3
    with zipfile.ZipFile(loaded.world_zip) as archive:
        assert archive.read("filesystem/report.txt") == b"v2"
    with zipfile.ZipFile(loaded.initial_zip) as archive:
        assert archive.read("filesystem/report.txt") == b"pristine"
    assert checkpointer.checkpoints_written == 2


def test_a_crash_while_writing_the_next_checkpoint_keeps_the_previous_one(tmp_path: Path) -> None:
    checkpointer, world, clock = _checkpointer(tmp_path)
    checkpointer.write(_turn_state(1))
    clock.now += 120.0

    def exploding_snapshot(destination: Path) -> list[str]:
        destination.write_bytes(b"half a zip")
        raise OSError("disk full")

    checkpointer._snapshot_world = exploding_snapshot
    with pytest.raises(OSError):
        checkpointer.write(_turn_state(2))

    loaded = load_resume_checkpoint(checkpointer.directory)
    assert loaded is not None and loaded.turn == 1
    assert not any(path.name.startswith(".") for path in checkpointer.directory.iterdir())


def test_rewriting_the_same_turn_never_touches_files_the_live_manifest_references(tmp_path: Path) -> None:
    """A resumed segment (or a context-overflow unwind) can legitimately checkpoint turn N again."""
    checkpointer, world, _clock = _checkpointer(tmp_path)
    checkpointer.write(_turn_state(2))
    before = load_resume_checkpoint(checkpointer.directory)
    assert before is not None

    resumed, _world, _clock2 = _checkpointer(tmp_path, world=world, prior_generation=before.generation)
    world.content = "v2"

    def exploding_snapshot(destination: Path) -> list[str]:
        destination.write_bytes(b"half a zip")
        raise OSError("killed mid-write")

    resumed._snapshot_world = exploding_snapshot
    with pytest.raises(OSError):
        resumed.write(_turn_state(2))
    still = load_resume_checkpoint(checkpointer.directory)
    assert still is not None and still.turn == 2 and still.generation == before.generation

    resumed._snapshot_world = world.snapshot
    resumed.write(_turn_state(2))
    after = load_resume_checkpoint(checkpointer.directory)
    assert after is not None and after.generation == before.generation + 1
    assert after.world_zip.name == "world.g2.t2.zip"
    assert not (checkpointer.directory / before.world_zip.name).exists(), "the old generation is pruned"


def test_a_resumed_segment_does_not_rewrite_the_checkpoint_it_started_from(tmp_path: Path) -> None:
    checkpointer, _world, clock = _checkpointer(tmp_path, resumed_turn=3, min_interval_seconds=0.0)

    assert not checkpointer.eligible(_turn_state(3)), "Stirrup rebuilds the restored state at the same turn"
    assert not checkpointer.eligible(_turn_state(2)), "a context-overflow unwind before any new turn changes nothing"
    assert checkpointer.eligible(_turn_state(4))
    checkpointer.write(_turn_state(4))
    clock.now += 1.0
    # Once the segment has made progress, a fresh state at an already-seen turn (an
    # unwind) is checkpointed again; generation-unique filenames make that safe.
    assert checkpointer.eligible(_turn_state(4))


def test_tool_timeout_defers_the_next_checkpoint_until_the_server_settles(tmp_path: Path) -> None:
    checkpointer, _world, clock = _checkpointer(tmp_path, min_interval_seconds=0.0)

    checkpointer.defer(120.0)
    assert not checkpointer.eligible(_turn_state(1))
    clock.now += 119.0
    assert not checkpointer.eligible(_turn_state(1))
    clock.now += 1.0
    assert checkpointer.eligible(_turn_state(1))


def test_heartbeat_charges_time_spent_after_the_last_checkpoint(tmp_path: Path) -> None:
    checkpointer, _world, clock = _checkpointer(tmp_path, prior_elapsed_seconds=100.0)
    clock.now += 10.0
    checkpointer.write(_turn_state(1))
    clock.now += 500.0
    checkpointer.heartbeat()

    loaded = load_resume_checkpoint(checkpointer.directory)
    assert loaded is not None
    assert loaded.elapsed_seconds == pytest.approx(610.0)

    # A heartbeat from another segment is ignored.
    stirrup_runtime.write_resume_heartbeat(checkpointer.directory, elapsed_seconds=9999.0, segments=7)
    loaded = load_resume_checkpoint(checkpointer.directory)
    assert loaded is not None and loaded.elapsed_seconds == pytest.approx(110.0)


def test_stale_temporaries_from_a_hard_kill_are_removed_at_segment_start(tmp_path: Path) -> None:
    checkpointer, _world, _clock = _checkpointer(tmp_path)
    checkpointer.write(_turn_state(1))
    (checkpointer.directory / ".world.g2.t2.zip.tmp").write_bytes(b"torn")

    again, _world2, _clock2 = _checkpointer(tmp_path)
    assert again.directory == checkpointer.directory
    assert not (checkpointer.directory / ".world.g2.t2.zip.tmp").exists()
    assert load_resume_checkpoint(checkpointer.directory) is not None


def test_manifest_verification_rejects_tampered_or_incomplete_checkpoints(tmp_path: Path) -> None:
    assert load_resume_checkpoint(None) is None
    assert load_resume_checkpoint(tmp_path / "missing") is None

    checkpointer, _world, _clock = _checkpointer(tmp_path)
    checkpointer.write(_turn_state(1))
    directory = checkpointer.directory
    manifest_path = directory / RESUME_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())

    (directory / manifest["files"]["world"]["name"]).write_bytes(b"corrupted")
    assert load_resume_checkpoint(directory) is None

    # Rebuild a valid checkpoint (bypassing the write throttle) to test the manifest checks next.
    checkpointer._last_written_at = None
    checkpointer._last_state = None
    checkpointer.write(_turn_state(1))
    assert load_resume_checkpoint(directory) is not None

    tampered = json.loads(manifest_path.read_text())
    tampered["schema"] = 99
    manifest_path.write_text(json.dumps(tampered))
    assert load_resume_checkpoint(directory) is None

    tampered["schema"] = 1
    tampered["turn"] = 0
    manifest_path.write_text(json.dumps(tampered))
    assert load_resume_checkpoint(directory) is None


def test_eligibility_gates_on_state_identity_first_model_call_and_interval(tmp_path: Path) -> None:
    checkpointer, _world, clock = _checkpointer(tmp_path, min_interval_seconds=60.0)

    assert not checkpointer.eligible(None)
    assert not checkpointer.eligible(_turn_state(0)), "nothing to save before the first completed turn"

    first = _turn_state(1)
    assert checkpointer.eligible(first)
    checkpointer.write(first)
    assert not checkpointer.eligible(first), "a state is checkpointed once"

    second = _turn_state(2)
    assert not checkpointer.eligible(second), "throttled inside the interval"
    clock.now += 60.0
    assert checkpointer.eligible(second)

    checkpointer._generate_seen_state = second
    assert not checkpointer.eligible(second), "after its model call started the world may be mid-turn"


@pytest.mark.asyncio
async def test_generate_hook_writes_once_per_turn_and_survives_failures(tmp_path: Path) -> None:
    checkpointer, world, clock = _checkpointer(tmp_path, min_interval_seconds=0.0)

    state = _turn_state(1)
    await checkpointer.on_generate_start(state)
    await checkpointer.on_generate_start(state)  # context summarization calls generate again mid-turn
    assert checkpointer.checkpoints_written == 1
    assert world.snapshots == 1

    def exploding_snapshot(destination: Path) -> list[str]:
        raise OSError("disk full")

    checkpointer._snapshot_world = exploding_snapshot
    later = _turn_state(2)
    await checkpointer.on_generate_start(later)  # logged, never raised into the model call
    assert checkpointer.checkpoints_written == 1
    assert not checkpointer.eligible(later), "the failed turn is not retried mid-turn"


def test_partial_result_from_checkpoint_carries_trajectory_usage_and_segments(tmp_path: Path) -> None:
    checkpointer, _world, _clock = _checkpointer(tmp_path, prior_elapsed_seconds=50.0, prior_segments=2)
    checkpointer.write(_turn_state(2))
    checkpoint = load_resume_checkpoint(checkpointer.directory)
    assert checkpoint is not None

    result = partial_result_from_checkpoint(checkpoint)

    assert result["completed"] is False
    assert result["completion_status"] == "timeout"
    assert result["n_input_tokens"] == 20
    assert result["n_output_tokens"] == 10
    assert result["resume_segments"] == 3
    assert result["resumed_from_turn"] == 2
    assert [message["content"] for message in result["trajectory"] if message["role"] == "assistant"] == [
        "turn 1",
        "turn 2",
    ]


def test_apex_state_round_trip_restores_toolbelt_todos_and_client_counters() -> None:
    class TodoItem(BaseModel):
        id: str
        content: str | None = None
        status: str = "pending"

    catalog = {"read_file": object(), "send_mail": object()}
    agent = SimpleNamespace(_active_tools={"finish": object(), "read_file": catalog["read_file"], "stale": object()})
    todo_state = {"t1": TodoItem(id="t1", content="draft", status="in_progress")}
    client = SimpleNamespace(length_truncations=2, recovery_turns=1, _recover_from_truncation=True)

    state = collect_apex_state(agent=agent, catalog=catalog, todo_state=todo_state, client=client)
    assert state == {
        "active_tools": ["read_file"],
        "todos": [{"id": "t1", "content": "draft", "status": "in_progress"}],
        "length_truncations": 2,
        "recovery_turns": 1,
        "recover_from_truncation": True,
    }

    fresh_agent = SimpleNamespace(_active_tools={"finish": object()})
    fresh_todos: dict = {}
    fresh_client = SimpleNamespace(length_truncations=0, recovery_turns=0, _recover_from_truncation=False)
    restored = restore_apex_state(
        agent=fresh_agent,
        catalog=catalog,
        todo_state=fresh_todos,
        todo_item_cls=TodoItem,
        client=fresh_client,
        apex_state=json.loads(json.dumps(state)) | {"active_tools": ["read_file", "vanished_tool"]},
    )

    assert restored == {"active_tools": 1, "todos": 1}
    assert fresh_agent._active_tools["read_file"] is catalog["read_file"]
    assert "vanished_tool" not in fresh_agent._active_tools
    assert fresh_todos["t1"].status == "in_progress"
    assert (fresh_client.length_truncations, fresh_client.recovery_turns, fresh_client._recover_from_truncation) == (
        2,
        1,
        True,
    )


def test_run_stirrup_rollout_wires_the_checkpointer_and_resume_flag() -> None:
    source = textwrap.dedent(inspect.getsource(stirrup_runtime.run_stirrup_rollout))
    tree = ast.parse(source)

    session_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "session"
    ]
    assert len(session_calls) == 1
    assert [keyword.arg for keyword in session_calls[0].keywords] == ["resume"]
    assert "client.on_generate_start = " in source
    assert "stage_stirrup_resume_state(" in source
    assert "restore_apex_state(" in source

    result_keys = {
        key.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key in node.keys
        if isinstance(key, ast.Constant)
    }
    assert {"resume_segments", "resumed_from_turn", "n_resume_checkpoints", "elapsed_seconds"} <= result_keys
    assert "prior_generation=resume_checkpoint.generation" in source
    assert "resumed_turn=resume_checkpoint.turn" in source
    assert "managed_tools.checkpointer = checkpointer" in source
    assert "self.checkpointer.defer(RESUME_SETTLE_AFTER_TOOL_TIMEOUT_SECONDS)" in source


# ---------------------------------------------------------------------------
# Real Stirrup: checkpoint at a turn boundary, then a fresh Agent resumes from it.
# ---------------------------------------------------------------------------

stirrup = pytest.importorskip("stirrup")


def _response(content: str = "", tool_calls=None):
    response = MagicMock()
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message = MagicMock()
    choice.message.content = content
    choice.message.tool_calls = tool_calls or []
    choice.message.reasoning_content = None
    response.choices = [choice]
    response.usage = MagicMock()
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 7
    response.usage.completion_tokens_details = MagicMock(reasoning_tokens=3)
    return response


def _finish_call():
    tool_call = MagicMock()
    tool_call.id = "call_finish"
    tool_call.function = MagicMock()
    tool_call.function.name = "finish"
    tool_call.function.arguments = '{"reason": "done", "paths": []}'
    return tool_call


def _stirrup_client():
    from stirrup.clients.chat_completions_client import ChatCompletionsClient

    client_class = stirrup_runtime.make_checkpointing_client_class(ChatCompletionsClient)
    client = client_class(model="m", base_url="http://test", api_key="k", max_tokens=4096, kwargs={})
    client._client = MagicMock()
    return client


def _assistant_contents(history) -> list[str]:
    return [
        getattr(message, "content", "")
        for turn in history
        for message in turn
        if type(message).__name__ == "AssistantMessage"
    ]


@pytest.mark.asyncio
async def test_stirrup_agent_resumes_from_a_checkpoint_written_at_a_turn_boundary(tmp_path: Path, monkeypatch) -> None:
    import stirrup.core.cache as stirrup_cache
    from stirrup import Agent

    monkeypatch.setattr(stirrup_cache, "DEFAULT_CACHE_DIR", stirrup_cache.DEFAULT_CACHE_DIR)
    world = _World()
    initial = tmp_path / "initial.zip"
    with zipfile.ZipFile(initial, "w") as archive:
        archive.writestr("filesystem/report.txt", "pristine")

    # Segment 1: one text-only turn, then the process dies during the second model call.
    client = _stirrup_client()
    client._client.chat.completions.create = AsyncMock(
        side_effect=[_response("thinking out loud"), RuntimeError("node preempted")]
    )
    agent = Agent(client=client, name="apex_test", max_turns=6, system_prompt="sys")
    checkpointer = ResumeCheckpointer(
        tmp_path / "ckpt",
        snapshot_world=world.snapshot,
        apex_state=lambda: {"active_tools": [], "todos": [], "length_truncations": 0, "recovery_turns": 0},
        initial_snapshot=initial,
        min_interval_seconds=0.0,
    )
    client.on_generate_start = lambda: checkpointer.on_generate_start(getattr(agent, "_current_run_state", None))
    with pytest.raises(RuntimeError, match="node preempted"):
        async with agent.session(cache_on_interrupt=False) as session:
            await session.run("do the task")

    checkpoint = load_resume_checkpoint(checkpointer.directory)
    assert checkpoint is not None and checkpoint.turn == 1
    assert checkpointer.checkpoints_written == 1

    # Segment 2: a fresh process stages the state where Stirrup looks and resumes.
    stirrup_runtime.stage_stirrup_resume_state(checkpoint, checkpoint.directory / "stirrup_cache", "do the task")
    resumed_client = _stirrup_client()
    resumed_create = AsyncMock(return_value=_response(tool_calls=[_finish_call()]))
    resumed_client._client.chat.completions.create = resumed_create
    resumed_agent = Agent(client=resumed_client, name="apex_test", max_turns=6, system_prompt="sys")

    async with resumed_agent.session(resume=True, cache_on_interrupt=False) as session:
        finish_params, history, _metadata = await session.run("do the task")

    assert finish_params is not None and finish_params.reason == "done"
    assert resumed_create.await_count == 1, "turn 1 came from the checkpoint; only turn 2 was generated"
    assert _assistant_contents(history)[0] == "thinking out loud"
    sent = resumed_create.await_args.kwargs["messages"]
    assert any("thinking out loud" in json.dumps(message) for message in sent), "resumed context includes turn 1"
