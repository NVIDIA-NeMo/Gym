# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from environment_servers.swe_together.diff_protocol import _strip_junk, truncate_diff_for_user_sim
from environment_servers.swe_together.evaluation import derive_score, interaction_metrics, stage_judge
from environment_servers.swe_together.simulator import UserAgent


@pytest.mark.asyncio
async def test_exact_simulator_messages_and_cursor():
    llm = SimpleNamespace(
        call=AsyncMock(
            side_effect=[
                SimpleNamespace(
                    content="Wait for tests",
                    tool_calls=[{"function": {"name": "question", "arguments": '{"content":"did the tests pass?"}'}}],
                ),
                SimpleNamespace(content="", tool_calls=[{"function": {"name": "no-op", "arguments": "{}"}}]),
            ]
        )
    )
    sim = UserAgent(llm, original_user_messages=["one", "two"], session_analysis="Analysis")
    await sim.process(
        "Task",
        "[1] tool_call(bash)",
        "[1] agent: done",
        None,
        1,
        True,
        elapsed_sec=65,
        turn_duration_sec=10,
        code_changes_diff="@@ diff",
    )
    first = "## Turn 1\n**Timing:** Elapsed: 1min 5s, this turn took 10s\n** The agent is signaling completion.\n\n## Task\nTask\n\n## Agent activity (this turn)\n[1] tool_call(bash)\n\n## Agent output\n[1] agent: done\n\n## Code changes (this turn)\n```diff\n@@ diff\n```\n\nPick ONE tool. Default to no-op unless you have a clear, new reason to speak."
    assert sim.last_messages_sent == [{"role": "system", "content": sim._sys}, {"role": "user", "content": first}]
    sim.advance_original_index()
    await sim.process("Task", "activity", "report", None, 2, True)
    assert sim.last_messages_sent[1:3] == [
        {"role": "user", "content": first},
        {"role": "assistant", "content": "Wait for tests\n\n→ question: did the tests pass?"},
    ]
    assert "\n## Task\n" not in sim.last_turn_content
    assert sim.get_stats()["ground_truth_consumed"] == 1
    assert sim.message_count == 1 and sim.wait_count == 1


@pytest.mark.asyncio
async def test_simulator_failure_is_not_noop():
    sim = UserAgent(SimpleNamespace(call=AsyncMock(side_effect=RuntimeError("model unavailable"))))
    with pytest.raises(RuntimeError, match="User simulator request failed"):
        await sim.process("task", "activity", "report", None, 1, True)
    assert sim.wait_count == 0 and sim._messages == []


def test_diff_line_boundary_and_junk():
    diff = "diff --git a/node_modules/x b/node_modules/x\n+discard\ndiff --git a/main.py b/main.py\n+keep\n "
    assert _strip_junk(diff) == "diff --git a/main.py b/main.py\n+keep\n "
    assert (
        truncate_diff_for_user_sim("aaa\nbbb\nccc", limit=6)
        == "aaa\n[... diff truncated for the user simulator: 11 chars across 0 files; showing the first 3 chars ...]"
    )


def test_frozen_weights_preserve_upstream_gameable_override_bug():
    rubric = {"completeness_goals": [{"id": "a", "weight": 0.845}, {"id": "b", "weight": 0.155}]}
    verdict = {
        "judge_score": 0,
        "verdict": "gameable",
        "goal_results": [{"id": "a", "met": True}, {"id": "b", "met": False}],
    }
    result = derive_score(rubric, verdict)
    assert result["judge_score"] == 0.84 and result["verdict"] == "partial"
    assert result["judge_reported_verdict"] == "gameable"
    with pytest.raises(ValueError, match="omitted"):
        derive_score(rubric, {"goal_results": [{"id": "a", "met": True}]})


@pytest.mark.asyncio
async def test_judge_user_message_names_required_verdict_fields(tmp_path):
    (tmp_path / "canonical_goals.json").write_text('{"completeness_goals":[]}')
    (tmp_path / "user_simulation_prompt.md").write_text("task context")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test.sh").write_text("#!/bin/sh\n")
    sandbox = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0, error_type=None)), upload=AsyncMock()
    )
    prompt = await stage_judge(sandbox, tmp_path, "", "/workspace")
    # Keep the parser's contract explicit at the final task instruction boundary.
    for field in ("judge_score", "verdict", "rubric_source", "goal_results", "judge_notes"):
        assert field in prompt
    assert "completeness_goals" in prompt
    assert "/tmp/judge_inputs/verdict.json" in prompt


@pytest.mark.asyncio
async def test_zero_messages_and_missing_tags_are_distinct(tmp_path):
    (tmp_path / "oracle_intents.json").write_text(json.dumps({"intents": []}))
    model = SimpleNamespace(call=AsyncMock(return_value=SimpleNamespace(content='{"results":[]}')))
    empty = await interaction_metrics(model, task_dir=tmp_path, messages=[], artifact_dir=tmp_path)
    assert empty["user_correction"] is None
    missing = await interaction_metrics(
        model,
        task_dir=tmp_path,
        messages=[{"trial_idx": 1, "turn": 1, "text": "fix it", "action": "redirect"}],
        artifact_dir=tmp_path,
    )
    assert missing["user_correction"] is None and missing["tagger_error"]


def test_task_assets_and_image_match_official_definition(tmp_path, monkeypatch):
    import hashlib

    from environment_servers.swe_together import task

    toml = tmp_path / "task.toml"
    toml.write_text(
        '[environment]\ndocker_image = "ghcr.io/official/task:pin"\nbuild_timeout_sec = 600\n'
        '[agent.kwargs]\ndisallowed_tools = "WebFetch,WebSearch"\n'
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"tasks": [{"task_id": "fixture", "files": {"task.toml": hashlib.sha256(toml.read_bytes()).hexdigest()}}]}
        )
    )
    monkeypatch.setattr(task, "MANIFEST", manifest)
    data = task.TaskData(task_id="fixture", image="ghcr.io/official/task:pin", image_digest="sha256:" + "a" * 64)
    assert task.load_task(tmp_path, data)["judge_timeout"] == 1200
    assert task.load_task(tmp_path, data)["agent_kwargs"] == {"disallowed_tools": "WebFetch,WebSearch"}
    with pytest.raises(ValueError, match="Task image"):
        task.load_task(tmp_path, data.model_copy(update={"image": "unrelated:latest"}))
    toml.write_text(toml.read_text() + "# changed\n")
    with pytest.raises(ValueError, match="missing or changed"):
        task.load_task(tmp_path, data)


@pytest.mark.asyncio
async def test_optional_reward_prefers_canonical_script_location(tmp_path):
    import asyncio

    from environment_servers.swe_together.evaluation import read_test_reward

    canonical = tmp_path / "canonical.txt"
    staged = tmp_path / "staged.txt"
    canonical.write_text("0.875\n")
    staged.write_text("0.0\n")

    class FilesystemSandbox:
        async def exec(self, command, *, timeout_s):
            command = command.replace("/logs/verifier/reward.txt", str(canonical)).replace(
                "/tmp/judge_inputs/logs/reward.txt", str(staged)
            )
            process = await asyncio.create_subprocess_shell(command, stdout=asyncio.subprocess.PIPE)
            stdout, _ = await process.communicate()
            return SimpleNamespace(stdout=stdout.decode(), return_code=process.returncode, error_type=None)

    sandbox = FilesystemSandbox()
    result = await read_test_reward(sandbox)
    assert result["test_reward_raw"] == 0.875 and result["test_reward_path"] == str(canonical)
    assert result["test_reward_source"] == "judge_optional_tests"
    canonical.unlink()
    assert (await read_test_reward(sandbox))["test_reward_raw"] == 0.0
    staged.unlink()
    result = await read_test_reward(sandbox)
    assert result["test_reward_raw"] is None and "test_reward_error" not in result
    canonical.write_text("nan\n")
    result = await read_test_reward(sandbox)
    assert result["test_reward_raw"] is None and "not finite" in result["test_reward_error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("tags", [None, [], "request", ["unsupported"]])
async def test_invalid_tagger_output_is_missing_measurement(tmp_path, tags):
    (tmp_path / "oracle_intents.json").write_text(json.dumps({"intents": []}))
    raw = json.dumps({"results": [{"trial_idx": 1, "tags": tags}]})
    model = SimpleNamespace(call=AsyncMock(return_value=SimpleNamespace(content=raw)))
    result = await interaction_metrics(
        model,
        task_dir=tmp_path,
        messages=[{"trial_idx": 1, "turn": 1, "text": "fix it", "action": "redirect"}],
        artifact_dir=tmp_path,
    )
    assert result["user_correction"] is None and result["tagger_error"]
    assert result["tagger_response"] == raw


@pytest.mark.asyncio
@pytest.mark.parametrize("has_message", [False, True])
async def test_empty_oracle_intents_score_only_accepted_feedback(tmp_path, has_message):
    (tmp_path / "oracle_intents.json").write_text(json.dumps({"intents": []}))
    (tmp_path / "instruction.md").write_text("This initial instruction is not simulator feedback.")
    messages = [{"trial_idx": 1, "turn": 1, "text": "fix it", "action": "redirect"}] if has_message else []
    raw = json.dumps({"results": [{"trial_idx": 1, "tags": ["correction", "nudge"]}]})
    model = SimpleNamespace(call=AsyncMock(return_value=SimpleNamespace(content=raw)))
    result = await interaction_metrics(model, task_dir=tmp_path, messages=messages, artifact_dir=tmp_path)
    assert result["intent_coverage"] == {
        "coverage_rate": 1.0,
        "weighted_coverage": 1.0,
        "scope_precision": 0.0 if has_message else 1.0,
        "overall_score": 0.7 if has_message else 1.0,
    }
    assert result["n_simulator_messages"] == len(messages)
    assert model.call.await_count == int(has_message)
    assert result["user_correction"] == (1.2 if has_message else None)
    if has_message:
        assert result["message_tags"] == [
            {"trial_idx": 1, "tags": ["correction", "nudge", "request"], "frustration": 0}
        ]
    assert json.loads((tmp_path / "interaction.json").read_text()) == result


@pytest.mark.asyncio
async def test_duplicate_tagger_rows_do_not_overwrite_each_other(tmp_path):
    (tmp_path / "oracle_intents.json").write_text(json.dumps({"intents": []}))
    raw = json.dumps({"results": [{"trial_idx": 1, "tags": ["correction"]}, {"trial_idx": 1, "tags": ["request"]}]})
    model = SimpleNamespace(call=AsyncMock(return_value=SimpleNamespace(content=raw)))
    result = await interaction_metrics(
        model,
        task_dir=tmp_path,
        messages=[{"trial_idx": 1, "turn": 1, "text": "fix it", "action": "redirect"}],
        artifact_dir=tmp_path,
    )
    assert result["user_correction"] is None and "duplicate" in result["tagger_error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("valid", [False, True])
async def test_coverage_missing_rows_are_distinct_from_explicit_no_match(tmp_path, valid):
    intent = {"intent_id": 0, "intent_kind": "request", "text": "fix tests", "verbatim_excerpt": "fix tests"}
    (tmp_path / "oracle_intents.json").write_text(json.dumps({"intents": [intent]}))
    (tmp_path / "instruction.md").write_text("task")
    table = {"per_intent": [{"intent_id": 0, "matched_trial_idx": None, "match_confidence": 0}] if valid else []}
    raw = json.dumps(table)
    model = SimpleNamespace(call=AsyncMock(return_value=SimpleNamespace(content=raw)))
    result = await interaction_metrics(model, task_dir=tmp_path, messages=[], artifact_dir=tmp_path)
    if valid:
        assert result["intent_coverage"]["overall_score"] == 0
        assert "coverage_error" not in result
    else:
        assert result["intent_coverage"] is None and result["coverage_error"]
        diagnostic = result["intent_coverage_diagnostics"]
        assert diagnostic["raw_response"] == raw and diagnostic["schema_warnings"]
        assert diagnostic["match_table"]["per_intent"][0]["matched_trial_idx"] is None


@pytest.mark.parametrize("confidence", [True, float("nan"), -0.1, 1.1])
def test_repaired_confidence_is_reported(confidence):
    from environment_servers.swe_together.coverage import normalize_match_table

    _, warnings = normalize_match_table(
        {"per_intent": [{"intent_id": 0, "matched_trial_idx": 0, "match_confidence": confidence}]}, 1, 1
    )
    assert warnings


@pytest.mark.asyncio
@pytest.mark.parametrize("code,error", [(1, None), (None, "timeout")])
async def test_judge_setup_failure_keeps_provider_diagnostics(tmp_path, code, error):
    sandbox = SimpleNamespace(
        exec=AsyncMock(
            return_value=SimpleNamespace(
                return_code=code, error_type=error, stdout="merged diagnostic", stderr="status"
            )
        ),
        upload=AsyncMock(),
    )
    with pytest.raises(RuntimeError, match="merged diagnostic"):
        await stage_judge(sandbox, tmp_path, "", "/workspace")
    sandbox.upload.assert_not_awaited()


def test_interaction_aggregation_keeps_missing_metrics_separate_from_zero():
    from environment_servers.swe_together.metrics import aggregate

    report = aggregate(
        [
            {"task_id": "first", "judge_score": 0, "user_correction": 0, "intent_coverage": {"overall_score": 0}},
            {
                "task_id": "second",
                "judge_score": None,
                "mask_sample": True,
                "user_correction": None,
                "intent_coverage": None,
            },
        ],
        planned=3,
        repeats=1,
    )
    assert report["graded"] == 1 and report["missing"] == 1 and report["masked"] == 1
    assert report["intent_coverage"] == report["user_correction"] == 0
    assert report["coverage_episodes"] == report["tagged_episodes"] == 1
