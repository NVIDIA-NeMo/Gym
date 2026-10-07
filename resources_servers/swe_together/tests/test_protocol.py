# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_gym.interactive_agent_types import AgentActivationObservation
from resources_servers.swe_together.context import project_turn
from resources_servers.swe_together.diff_protocol import _strip_junk, truncate_diff_for_user_sim
from resources_servers.swe_together.evaluation import derive_score, interaction_metrics
from resources_servers.swe_together.patch_normalize import apply_candidates
from resources_servers.swe_together.simulator import UserAgent


def test_projection_excludes_private_reasoning_and_tool_payloads():
    observation = AgentActivationObservation(
        events=[
            {"sequence": 0, "kind": "step_start"},
            {"sequence": 1, "kind": "reasoning", "text": "PRIVATE"},
            {"sequence": 2, "kind": "text", "text": "A" * 301},
            {"sequence": 3, "kind": "tool_use", "name": "bash", "arguments": "SECRET_ARG", "result": "SECRET_RESULT"},
            {"sequence": 4, "kind": "text", "text": "B" * 3001},
            {"sequence": 5, "kind": "step_finish"},
            {"sequence": 6, "kind": "text", "text": "OUTSIDE_STEP"},
        ]
    )
    activity, report = project_turn(observation, raw_history=[])
    assert activity == "[1] thinking: " + "A" * 300 + "…\n[1] tool_call(bash)"
    assert report == "[1] agent: " + "B" * 3000
    assert "PRIVATE" not in activity + report


def test_projection_fallbacks_and_tail():
    empty = AgentActivationObservation()
    assert project_turn(empty, raw_history=[]) == ("(nothing yet)", "(nothing yet)")
    assert project_turn(empty, raw_history=["a" * 600, "b" * 600], context_chars=500) == ("b" * 500, "b" * 500)
    tool = AgentActivationObservation(
        events=[
            {"sequence": 0, "kind": "step_start"},
            {"sequence": 1, "kind": "tool_use", "name": "read", "result": "R" * 510},
        ]
    )
    assert project_turn(tool, raw_history=[])[1] == "[1] result: " + "R" * 500


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


def test_patch_normalization_selects_main_repo_and_repairs_context():
    patch = "=== /tmp/scratch (cumulative vs harbor-base) ===\ndiff --git a/a b/a\n--- a/a\n+++ b/a\n@@ -1 +1 @@\n-old\n+new\n=== /workspace (cumulative vs harbor-base) ===\ndiff --git a/b b/b\n--- a/b\n+++ b/b\n@@ -1,2 +1,2 @@\n-old\n+new"
    candidates = apply_candidates(patch)
    assert len(candidates) == 2 and "a/a" not in candidates[0]
    assert "@@ -1,1 +1,1 @@" in candidates[0]
    assert candidates[1].endswith("+new\n \n")


@pytest.mark.asyncio
async def test_zero_messages_and_missing_tags_are_distinct(tmp_path):
    (tmp_path / "oracle_intents.json").write_text(json.dumps({"intents": []}))
    model = SimpleNamespace(call=AsyncMock(return_value=SimpleNamespace(content='{"results":[]}')))
    empty = await interaction_metrics(model, task_dir=tmp_path, messages=[], artifact_dir=tmp_path)
    assert empty["user_correction"] == 0.0
    missing = await interaction_metrics(
        model,
        task_dir=tmp_path,
        messages=[{"trial_idx": 1, "turn": 1, "text": "fix it", "action": "redirect"}],
        artifact_dir=tmp_path,
    )
    assert missing["user_correction"] is None and missing["tagger_error"]


@pytest.mark.asyncio
async def test_postprocessing_verifier_fixture():
    from nemo_gym.verifier_fixture import exercise_verifier_fixture
    from resources_servers.swe_together.app import VERIFIER_FIXTURE

    cases = await exercise_verifier_fixture(
        VERIFIER_FIXTURE, reward_range=(0.0, 1.0), higher_is_better=True, determinism="stochastic"
    )
    assert len(cases) == 4


def test_task_assets_and_image_match_official_definition(tmp_path, monkeypatch):
    import hashlib

    from resources_servers.swe_together import task

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

    from resources_servers.swe_together.evaluation import read_test_reward

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


def test_importer_emits_catalog_envelope_without_hidden_prompts(tmp_path, monkeypatch):
    import hashlib

    from benchmarks.swe_together import prepare
    from nemo_gym.base_resources_server import BaseRunRequest
    from resources_servers.swe_together import task

    directory = tmp_path / "source/tasks/fixture"
    directory.mkdir(parents=True)
    toml = directory / "task.toml"
    toml.write_text('[environment]\ndocker_image = "ghcr.io/official/task:pin"\n')
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"tasks": [{"task_id": "fixture", "files": {"task.toml": hashlib.sha256(toml.read_bytes()).hexdigest()}}]}
        )
    )
    monkeypatch.setattr(task, "MANIFEST", manifest)
    monkeypatch.setattr(prepare, "MANIFEST", manifest)
    images = tmp_path / "images.json"
    images.write_text(
        json.dumps(
            {
                "fixture": {
                    "image": "ghcr.io/official/task:pin",
                    "image_digest": "sha256:" + "a" * 64,
                    "workdir": "/workspace",
                }
            }
        )
    )
    output = prepare.prepare(tmp_path / "source", images, tmp_path / "tasks.jsonl")
    row = json.loads(output.read_text())
    request = BaseRunRequest.model_validate(row)
    assert request.responses_create_params.input == []
    assert row["task_input"]["task_data"]["task_id"] == row["task_id"]["task_id"] == "fixture"
