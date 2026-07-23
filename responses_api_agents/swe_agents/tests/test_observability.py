# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from nemo_gym.config_types import ModelServerRef
from nemo_gym.rollout_observability import AgentInvocation
from responses_api_agents.swe_agents.observability import (
    build_swe_observations,
    sandbox_observations_from_metrics,
)


MODEL_REF = ModelServerRef(type="responses_api_models", name="policy_model")


def _invocations(bundle) -> list[AgentInvocation]:
    return [record for record in bundle.records if isinstance(record, AgentInvocation)]


def _completion(
    path: Path,
    *,
    response_id: str | None,
    content: str,
    turn: int,
    session_id: str | None = None,
    parent_session_id: str | None = None,
    tool_call_id: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    response = {
        "choices": [{"message": {"role": "assistant", "content": content}}],
    }
    if response_id is not None:
        response["id"] = response_id
    messages = [{"role": "user", "content": "Fix the bug"}]
    if tool_call_id is not None:
        response["choices"][0]["message"]["tool_calls"] = [
            {
                "id": tool_call_id,
                "type": "function",
                "function": {"name": "shell", "arguments": '{"cmd":"pytest"}'},
            }
        ]
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": "passed",
            }
        )
    path.write_text(
        json.dumps(
            {
                "messages": messages,
                "response": response,
                "kwargs": {"tools": []},
                "session_id": session_id,
                "parent_session_id": parent_session_id,
                "turn": turn,
            }
        )
    )


def test_sandbox_peak_memory_is_marked_as_sampled_process_tree_usage() -> None:
    [sandbox] = sandbox_observations_from_metrics(
        {
            "openhands_run_time": 12.5,
            "agent_peak_rss_mb": 2048,
        }
    )

    assert sandbox.wall_time_s == 12.5
    assert sandbox.cpu_time_s is None
    assert sandbox.peak_memory_mib == 2048
    assert sandbox.resource_usage_source == "proc_tree_watchdog"


def test_opencode_preserves_tree_and_all_response_ids(tmp_path: Path) -> None:
    _completion(
        tmp_path / "main-0.json",
        response_id="resp-main-0",
        content="first",
        turn=0,
        session_id="main",
    )
    _completion(
        tmp_path / "main-1.json",
        response_id="resp-main-1",
        content="latest",
        turn=1,
        session_id="main",
    )
    _completion(
        tmp_path / "child.json",
        response_id="resp-child",
        content="child",
        turn=0,
        session_id="child",
        parent_session_id="main",
        tool_call_id="call-child",
    )

    bundle = build_swe_observations(tmp_path.glob("*.json"), framework="opencode", model_ref=MODEL_REF)

    records = _invocations(bundle)
    invocations = {item.invocation_id: item for item in records}
    assert invocations["main"].parent_invocation_id is None
    assert invocations["child"].parent_invocation_id == "main"
    assert [ref.response_id for ref in invocations["main"].model_calls] == [
        "resp-main-0",
        "resp-main-1",
    ]
    assert [ref.response_id for ref in invocations["child"].model_calls] == ["resp-child"]
    assert all(ref.model_ref == MODEL_REF for invocation in records for ref in invocation.model_calls)
    assert "latest" in json.dumps([item.model_dump() for item in invocations["main"].conversation])
    assert "subagent_hierarchy_unavailable" not in {gap.code for gap in bundle.gaps}
    assert "subagent_spawn_tool_unavailable" in {gap.code for gap in bundle.gaps}
    assert {(gap.code, gap.invocation_id) for gap in bundle.gaps} >= {
        ("tool_timing_unavailable", "child"),
        ("tool_outcome_unavailable", "child"),
    }


def test_openhands_uses_one_root_and_latest_cumulative_conversation(tmp_path: Path) -> None:
    old = tmp_path / "old.json"
    latest = tmp_path / "latest.json"
    _completion(old, response_id="resp-0", content="old", turn=0)
    _completion(latest, response_id="resp-1", content="latest", turn=1)
    bundle = build_swe_observations((old, latest), framework="openhands", model_ref=MODEL_REF)

    [root] = _invocations(bundle)
    assert root.invocation_id == "root"
    assert root.parent_invocation_id is None
    assert [ref.response_id for ref in root.model_calls] == ["resp-0", "resp-1"]
    assert "latest" in json.dumps([item.model_dump() for item in root.conversation])
    assert {gap.code for gap in bundle.gaps} == {
        "subagent_hierarchy_unavailable",
        "context_compaction_unavailable",
    }


def test_missing_response_id_is_reported_without_a_synthetic_ref(tmp_path: Path) -> None:
    _completion(
        tmp_path / "missing.json",
        response_id=None,
        content="answer",
        turn=0,
        session_id="main",
    )

    bundle = build_swe_observations(tmp_path.glob("*.json"), framework="opencode", model_ref=MODEL_REF)

    [invocation] = _invocations(bundle)
    assert invocation.model_calls == []
    assert "model_response_id_missing" in {gap.code for gap in bundle.gaps}


def test_empty_openhands_capture_still_has_one_root_and_an_exact_gap() -> None:
    bundle = build_swe_observations((), framework="openhands", model_ref=MODEL_REF)

    assert [invocation.invocation_id for invocation in _invocations(bundle)] == ["root"]
    assert "agent_artifact_unavailable" in {gap.code for gap in bundle.gaps}


def test_untagged_opencode_artifact_does_not_fabricate_an_invocation(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "legacy.json"
    _completion(
        artifact,
        response_id="resp-legacy",
        content="legacy",
        turn=0,
    )

    bundle = build_swe_observations((artifact,), framework="opencode", model_ref=MODEL_REF)

    assert _invocations(bundle) == []
    assert "agent_session_id_missing" in {gap.code for gap in bundle.gaps}


def test_duplicate_response_id_is_not_assigned_to_two_invocations(tmp_path: Path) -> None:
    _completion(tmp_path / "root.json", response_id="resp-1", content="root", turn=0, session_id="root")
    _completion(
        tmp_path / "child.json",
        response_id="resp-1",
        content="child",
        turn=0,
        session_id="child",
        parent_session_id="root",
    )

    bundle = build_swe_observations(tmp_path.glob("*.json"), framework="opencode", model_ref=MODEL_REF)

    refs = [ref for invocation in _invocations(bundle) for ref in invocation.model_calls]
    assert len(refs) == 1
    assert "model_response_owner_conflict" in {gap.code for gap in bundle.gaps}
