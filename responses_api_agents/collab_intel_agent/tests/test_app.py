# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from nemo_gym.config import ModelServerRef
from nemo_gym.rollouts.trajectory_builder import _build_trajectory_record
from resources_servers.collab_intel.task_data import TaskData
from responses_api_agents.collab_intel_agent import app as app_module
from responses_api_agents.collab_intel_agent.app import CollabIntelAgent, CollabIntelRunRequest, split_reasoning
from responses_api_agents.collab_intel_agent.boxed_exec import SandboxExecError


DIMENSIONS = [f"d{i}" for i in range(8)]


def _row(dimension: str, test_pass, mask: bool = False, alone_pass: bool = True, unfinished: bool = False) -> dict:
    return {
        "reward": float(bool(test_pass)),
        "dimension": dimension,
        "test_pass": test_pass,
        "mask_sample": mask,
        "alone_pass": alone_pass,
        "unfinished": unfinished,
    }


def test_score_weights_dimensions_equally():
    # one passed row in d0 and three failed rows in each other dimension: 1/8, not 1/22
    tasks = [[_row("d0", True)]] + [[_row(d, False)] for d in DIMENSIONS[1:] for _ in range(3)]
    assert CollabIntelAgent._score(tasks)["collab_intel/score"] == pytest.approx(12.5)
    # duplicating rows within a dimension, or adding excluded rows, leaves the score unchanged
    more = tasks + [[_row("d0", True)]] + [[_row("d1", None, mask=True)], [_row("d2", False, mask=True)]]
    # gate-failed rows are unmasked but carry alone_pass=false, and the score leaves them out
    more += [[_row("d3", True, alone_pass=False)], [_row("d4", None, alone_pass=False)]]
    assert CollabIntelAgent._score(more)["collab_intel/score"] == pytest.approx(12.5)


def test_score_without_scored_rows_reports_no_score():
    metrics = CollabIntelAgent._score([[_row("d0", None, mask=True, alone_pass=False, unfinished=True)]])
    assert "collab_intel/score" not in CollabIntelAgent._score([[_row("d0", None, alone_pass=False)]])
    assert "collab_intel/score" not in metrics and metrics["collab_intel/scored_rows"] == 0
    assert "collab_intel/alone_pass_rate" not in metrics and metrics["collab_intel/unfinished"] == 1


def test_score_reports_gate_pass_rate_and_unfinished_rows():
    tasks = [
        [_row("d0", True)],
        [_row("d0", None, alone_pass=False)],
        [_row("d1", None, alone_pass=False)],
        [_row("d1", None, mask=True, unfinished=True)],
    ]
    metrics = CollabIntelAgent._score(tasks)
    assert metrics["collab_intel/alone_pass_rate"] == pytest.approx(100 / 3)
    assert metrics["collab_intel/unfinished"] == 1
    assert set(CollabIntelAgent.get_key_metrics(None, metrics)) == {
        "collab_intel/score",
        "collab_intel/scored_rows",
        "collab_intel/alone_pass_rate",
        "collab_intel/unfinished",
    }


@pytest.mark.parametrize(
    "content, reasoning, text",
    [
        ("<think>plan</think>\n\nhello", "plan", "hello"),
        ("<think>cut off by max tokens", "cut off by max tokens", ""),
        ("hello", None, "hello"),
    ],
)
def test_split_reasoning(content, reasoning, text):
    msg = {"content": content}
    split_reasoning(msg)
    assert (msg.get("reasoning"), msg["content"]) == (reasoning, text)


def test_run_request_matches_task_data():
    row = {"responses_create_params": {"input": []}, "dimension": "d0", "scenario_path": "a", "gate_path": "a"}
    request = CollabIntelRunRequest.model_validate(row | {"extra": 1})
    assert (request.gate_seed, request.seed, request.model_extra["extra"]) == (0, 0, 1)
    for model in (TaskData, CollabIntelRunRequest):
        with pytest.raises(ValidationError):
            model.model_validate({"responses_create_params": {"input": []}, "dimension": "d0", "gate_path": "a"})
        with pytest.raises(ValidationError):
            model.model_validate(row | {"seed": "x"})
    assert {k: (f.annotation, f.default, f.is_required()) for k, f in TaskData.model_fields.items()} == {
        k: (f.annotation, f.default, f.is_required())
        for k, f in CollabIntelRunRequest.model_fields.items()
        if k in TaskData.model_fields
    }


def _agent_with_fake_cli(
    tmp_path: Path, transcript: dict | None, exit_code: int, monkeypatch, log: str = "", timeout_s: float = 60
) -> CollabIntelAgent:
    """An agent whose harness interpreter is a script that writes \`transcript\` (or nothing), prints \`log\` and exits
    with \`exit_code\`."""
    script = tmp_path / "fake_harness"
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, sys\n"
        "arg = lambda name: sys.argv[sys.argv.index(name) + 1]\n"
        f"t = {transcript!r}\n"
        "if t is not None:\n"
        "    out = pathlib.Path(arg('--out-dir')) / 'ci-policy'\n"
        "    out.mkdir(parents=True)\n"
        "    (out / f\"{arg('--version')}_seed{arg('--seed')}.json\").write_text(json.dumps(t))\n"
        f"print({log!r})\n"
        f"sys.exit({exit_code})\n"
    )
    script.chmod(0o755)
    monkeypatch.setattr(CollabIntelAgent, "model_post_init", lambda self, context: None)  # no uv sync, no server
    agent = CollabIntelAgent.model_construct(config=SimpleNamespace(episode_timeout_s=timeout_s))
    agent._python, agent._root, agent._boxes = str(script), tmp_path, None
    agent._env = lambda home, phase_key: {}
    return agent


@pytest.mark.parametrize(
    "transcript, exit_code, log, failure_kind, passed",
    [
        ({"score": {"pass": True, "outcome": "ok"}, "messages": [{"seat": "a", "text": "hi"}]}, 0, "", None, True),
        ({"score": {"pass": False, "outcome": "wrong"}}, 0, "", None, False),
        ({"messages": []}, 0, "", "judge_failed", False),  # no score
        ({"score": {"pass": None}}, 0, "", "judge_failed", False),  # empty verdict
        ({"judge_failed": "Timeout", "score": {"pass": False}}, 0, "", "judge_failed", False),
        (None, 1, "", "agent_run_error", False),  # failed process, no output
        (
            None,
            1,
            "BadRequestError: Error code: 400 - {'code': 'context_length_exceeded'}",
            "model_context_limit",
            False,
        ),
    ],
)
def test_run_types_unfinished_results(tmp_path, monkeypatch, transcript, exit_code, log, failure_kind, passed):
    agent = _agent_with_fake_cli(tmp_path, transcript, exit_code, monkeypatch, log=log)
    result = asyncio.run(agent._run(tmp_path / "scenario", "solo", 0, tmp_path / "out"))
    assert (result["pass"], result.get("failure_kind")) == (passed, failure_kind)
    if failure_kind is None:
        assert result["messages"] == transcript.get("messages", [])
    else:
        assert result["failure_reason"]


def test_timeout_kills_the_harness_process_group(tmp_path, monkeypatch):
    # the fake harness starts a child that outlives it unless the whole group is killed
    pid_file = tmp_path / "child.pid"
    agent = _agent_with_fake_cli(tmp_path, None, 0, monkeypatch, timeout_s=1)
    script = Path(agent._python)
    script.write_text(
        f"#!{sys.executable}\n"
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
        "time.sleep(60)\n"
    )
    result = asyncio.run(agent._run(tmp_path / "scenario", "solo", 0, tmp_path / "out"))
    assert (result["pass"], result["failure_kind"]) == (False, "agent_timeout")
    child = int(pid_file.read_text())
    for _ in range(50):  # the killed child is reaped by init shortly after
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail("the harness's child survived the timeout")


def test_policy_chat_passes_a_model_rejection_through(monkeypatch):
    # a 4xx (e.g. context_length_exceeded) reaches the harness as is, not as a 500 it would retry
    body = b'{"error": {"message": "too long", "code": "context_length_exceeded"}}'
    reply = SimpleNamespace(status=400, read=AsyncMock(return_value=body))
    monkeypatch.setattr(CollabIntelAgent, "model_post_init", lambda self, context: None)
    agent = CollabIntelAgent.model_construct(
        config=SimpleNamespace(model_server=SimpleNamespace(name="policy_model")),
        server_client=SimpleNamespace(post=AsyncMock(return_value=reply)),
    )
    request = SimpleNamespace(
        json=AsyncMock(return_value={"messages": []}), path_params={}, url=SimpleNamespace(path="")
    )
    response = asyncio.run(agent.policy_chat(request))
    assert (response.status_code, response.body) == (400, body)


def test_policy_chat_records_the_response_id_for_its_phase(monkeypatch):
    reply = SimpleNamespace(status=200)
    monkeypatch.setattr(app_module, "raise_for_status", AsyncMock())
    monkeypatch.setattr(app_module, "get_response_json", AsyncMock(return_value={"id": "chatcmpl-1", "choices": []}))
    monkeypatch.setattr(CollabIntelAgent, "model_post_init", lambda self, context: None)
    agent = CollabIntelAgent.model_construct(
        config=SimpleNamespace(model_server=SimpleNamespace(name="policy_model")),
        server_client=SimpleNamespace(post=AsyncMock(return_value=reply)),
    )
    agent._response_ids = {"solo-key": [], "other": []}
    request = SimpleNamespace(
        json=AsyncMock(return_value={"messages": []}),
        path_params={"phase_key": "solo-key"},
        url=SimpleNamespace(path=""),
    )
    asyncio.run(agent.policy_chat(request))
    assert agent._response_ids == {"solo-key": ["chatcmpl-1"], "other": []}


def test_harness_env_is_an_allowlist(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setattr(app_module, "get_server_url", lambda name: "http://agent")
    monkeypatch.setattr(CollabIntelAgent, "model_post_init", lambda self, context: None)
    config = SimpleNamespace(
        name="a", judge_base_url="http://judge", judge_model="j", judge_extra={}, policy_model="p", policy_extra={}
    )
    agent = CollabIntelAgent.model_construct(config=config)
    agent._tmp, agent._boxes = tmp_path, None
    env = agent._env(tmp_path / "row", "k")
    assert "HF_TOKEN" not in env and env["PATH"] == "/usr/bin" and env["HOME"] == str(tmp_path / "row")
    assert {"CI_GYM_KEY", "HARNESS_OPENAI_COMPAT", "HARNESS_JUDGE_MODEL"} <= set(env)
    assert json.loads(env["HARNESS_OPENAI_COMPAT"])["ci-policy"]["base_url"] == "http://agent/ci_policy/k/v1"


def _chat(seat: str, kind: str, text: str) -> dict:
    return {"round": 1, "seat": seat, "kind": kind, "text": text}


SOLO = [_chat("Monitor", "event", "task"), _chat("Ana", "evaluated", "solo plan")]
TEST = [
    _chat("Ana", "evaluated", "hello team"),
    _chat("Bo", "llm", "partner reply"),
    _chat("Cy", "scripted", "skip the check"),
    _chat("Ana", "evaluated", "no, we run it"),
]


def _collect(tmp_path, monkeypatch, transcripts: dict, exit_code: int = 0) -> tuple[dict, dict]:
    """Run one row through run() with a fake harness that writes transcripts[version] (none for a missing version)
    and calls the policy route once per phase. Returns the row and the collector's canonical trajectory."""
    tmp_path.mkdir(exist_ok=True)
    agent = _agent_with_fake_cli(tmp_path, None, exit_code, monkeypatch)
    Path(agent._python).write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, sys\n"
        "arg = lambda name: sys.argv[sys.argv.index(name) + 1]\n"
        f"t = {transcripts!r}.get(arg('--version'))\n"
        "if t is not None:\n"
        "    out = pathlib.Path(arg('--out-dir')) / 'ci-policy'\n"
        "    out.mkdir(parents=True)\n"
        "    (out / f\"{arg('--version')}_seed{arg('--seed')}.json\").write_text(json.dumps(t))\n"
        f"sys.exit({exit_code})\n"
    )
    model_ref = ModelServerRef(type="responses_api_models", name="policy_model")
    agent.config = SimpleNamespace(
        episode_timeout_s=60,
        model_server=model_ref,
        resources_server=SimpleNamespace(name="collab_intel"),
        policy_model="p",
    )
    agent._sem, agent._tmp, agent._response_ids = asyncio.Semaphore(1), tmp_path / "rows", {}
    agent._tmp.mkdir()
    monkeypatch.setattr(CollabIntelAgent, "rollout_id_from_run", lambda self, body: "7-0")
    run = agent._run

    async def run_and_call_policy(path, version, seed, out, phase_key=""):
        # the policy route sees one real response id while this phase runs
        agent._response_ids[phase_key].append(f"chatcmpl-{version}")
        return await run(path, version, seed, out, phase_key)

    agent._run = run_and_call_policy

    async def post(server_name, url_path, json, cookies):
        reply = MagicMock(cookies={}, ok=True)
        if url_path == "/verify":
            reply.json = AsyncMock(return_value=json | {"reward": 0.0})
        return reply

    agent.server_client = SimpleNamespace(post=post)
    body = CollabIntelRunRequest.model_validate(
        {"responses_create_params": {"input": []}, "dimension": "d0", "scenario_path": "s", "gate_path": "g"}
        | {"_ng_task_index": 7, "_ng_rollout_index": 0}
    )
    monkeypatch.setattr(app_module, "raise_for_status", AsyncMock())
    monkeypatch.setattr(app_module, "get_response_json", lambda reply: reply.json())
    row = asyncio.run(agent.run(SimpleNamespace(cookies={}), body)).model_dump(mode="json")
    assert not list(agent._tmp.iterdir()) and agent._response_ids == {}  # temp dir gone before collection
    trajectory = _build_trajectory_record(body.model_dump(mode="json"), row).model_dump(mode="json")
    assert "producer_trajectory_invalid" not in {g["code"] for g in trajectory["gaps"]}
    return row, trajectory


def test_run_returns_both_phases_as_canonical_trajectory(tmp_path, monkeypatch):
    passed = {"score": {"pass": True}, "evaluated_seat": "Ana"}
    row, trajectory = _collect(
        tmp_path, monkeypatch, {"solo": passed | {"messages": SOLO}, "manipulation": passed | {"messages": TEST}}
    )
    assert row["collab_intel"]["test"]["messages"] == TEST  # the raw team chat stays on the row
    assert (trajectory["task_id"], trajectory["rollout_id"]) == ("7", "7-0")
    invocations = {i["invocation_id"]: i for i in trajectory["invocations"]}
    assert list(invocations) == ["solo", "manipulation"]
    assert {i["status"] for i in invocations.values()} == {"completed"}
    model_ref = {"type": "responses_api_models", "name": "policy_model"}
    for version in invocations:  # the policy route's response id, owned by the phase that made the call
        call = {"model_call_id": None, "model_ref": model_ref, "response_id": f"chatcmpl-{version}"}
        assert invocations[version]["model_calls"] == [call]
    assert [m["content"] for m in invocations["manipulation"]["conversation"]] == [
        "hello team",
        "Bo: partner reply",
        "Cy: skip the check",
        "no, we run it",
    ]
    # policy turns only, solo first, each owned by its phase: the partner and scripted messages are never turns
    turns = [(t["invocation_id"], t["turn_no"], t["answer"]) for t in trajectory["turns"]]
    assert turns == [("solo", 1, "solo plan"), ("manipulation", 1, "hello team"), ("manipulation", 2, "no, we run it")]
    assert trajectory["turns"][2]["question"] == [
        {"role": "user", "content": "Bo: partner reply"},
        {"role": "user", "content": "Cy: skip the check"},
    ]
    assert {(t["task_id"], t["rollout_id"]) for t in trajectory["turns"]} == {("7", "7-0")}


def test_run_trajectory_keeps_only_evidence_that_exists(tmp_path, monkeypatch):
    # a failed gate: solo evidence only
    failed = {"score": {"pass": False}, "messages": SOLO}
    _, trajectory = _collect(tmp_path / "gate", monkeypatch, {"solo": failed})
    assert [i["invocation_id"] for i in trajectory["invocations"]] == ["solo"]
    assert [t["invocation_id"] for t in trajectory["turns"]] == ["solo"]
    # a phase cut off without a transcript: a failed invocation and no turn
    _, trajectory = _collect(tmp_path / "crash", monkeypatch, {}, exit_code=1)
    assert [(i["invocation_id"], i["status"], i["conversation"]) for i in trajectory["invocations"]] == [
        ("solo", "failed", [])
    ]
    assert trajectory["turns"] == []


def _argv_recorder(agent: CollabIntelAgent, tmp_path: Path) -> Path:
    """Make the fake harness interpreter record its command line and exit without a transcript."""
    argv_file = tmp_path / "argv.json"
    Path(agent._python).write_text(
        f"#!{sys.executable}\nimport json, sys\nopen({str(argv_file)!r}, 'w').write(json.dumps(sys.argv[1:]))\nsys.exit(1)\n"
    )
    return argv_file


def test_the_harness_starts_through_boxed_cli_only_in_sandbox_mode(tmp_path, monkeypatch):
    agent = _agent_with_fake_cli(tmp_path, None, 1, monkeypatch)
    argv_file = _argv_recorder(agent, tmp_path)
    asyncio.run(agent._run(tmp_path / "scenario", "solo", 0, tmp_path / "out"))
    assert json.loads(argv_file.read_text())[:2] == ["-m", "collab_intel.harness.cli"]
    agent._boxes = object()
    asyncio.run(agent._run(tmp_path / "scenario", "solo", 0, tmp_path / "out"))
    argv = json.loads(argv_file.read_text())
    assert argv[0] == str(app_module.BOXED_CLI) and "-m" not in argv and argv[1] == str(tmp_path / "scenario")


def test_the_sandbox_url_is_in_the_harness_env_only_in_sandbox_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "get_server_url", lambda name: "http://agent")
    monkeypatch.setattr(CollabIntelAgent, "model_post_init", lambda self, context: None)
    config = SimpleNamespace(
        name="a", judge_base_url="http://judge", judge_model="j", judge_extra={}, policy_model="p", policy_extra={}
    )
    agent = CollabIntelAgent.model_construct(config=config)
    agent._tmp, agent._boxes = tmp_path, None
    assert "CI_EXEC_URL" not in agent._env(tmp_path / "row", "k")
    agent._boxes = object()
    assert agent._env(tmp_path / "row", "k")["CI_EXEC_URL"] == "http://agent/ci_exec/k/run"


def test_each_phase_releases_its_sandbox(tmp_path, monkeypatch):
    agent = _agent_with_fake_cli(tmp_path, None, 1, monkeypatch)
    agent._response_ids, agent._boxes = {}, SimpleNamespace(release=AsyncMock())
    asyncio.run(agent._phase(tmp_path / "scenario", "solo", 0, tmp_path / "out"))
    agent._boxes.release.assert_awaited_once()
    assert isinstance(agent._boxes.release.await_args.args[0], str)


def test_the_exec_route_runs_a_command_and_fails_closed(monkeypatch):
    monkeypatch.setattr(CollabIntelAgent, "model_post_init", lambda self, context: None)
    agent = CollabIntelAgent.model_construct(config=SimpleNamespace())
    request = SimpleNamespace(path_params={"phase_key": "k"}, json=AsyncMock(return_value={"argv": []}))
    agent._boxes = SimpleNamespace(run=AsyncMock(return_value={"rc": 0}))
    assert asyncio.run(agent.exec_in_sandbox(request)) == {"rc": 0}
    agent._boxes.run.assert_awaited_once_with("k", {"argv": []})
    agent._boxes = SimpleNamespace(run=AsyncMock(side_effect=SandboxExecError("the sandbox has network egress")))
    refused = asyncio.run(agent.exec_in_sandbox(request))
    assert (refused.status_code, json.loads(refused.body)) == (502, {"error": "the sandbox has network egress"})
    agent._boxes = None
    assert asyncio.run(agent.exec_in_sandbox(request)).status_code == 404  # host mode has no sandbox to run in


def test_sandbox_mode_is_the_default_and_builds_the_spec_from_the_config(monkeypatch):
    fields = app_module.CollabIntelAgentConfig.model_fields
    assert (fields["code_execution"].default, fields["sandbox_provider"].default) == ("sandbox", "sandbox")
    with pytest.raises(ValidationError):
        app_module.SandboxSpecConfig(unknown=1)
    built = {}
    monkeypatch.setattr(
        app_module, "BoxedExec", lambda provider, spec, **kw: built.update(provider=provider, spec=spec, **kw)
    )
    monkeypatch.setattr(CollabIntelAgent, "model_post_init", lambda self, context: None)
    spec = app_module.SandboxSpecConfig(
        image="img:1", provider_options={"network_policy": {"defaultAction": "deny"}}, metadata={"k": "v"}
    )
    config = SimpleNamespace(name="ci", sandbox_provider={"opensandbox": {"connection": {}}}, sandbox_spec=spec)
    CollabIntelAgent.model_construct(config=config)._start_boxes()
    assert built["provider"] == {"opensandbox": {"connection": {}}}
    assert (built["spec"].image, built["spec"].workdir) == ("img:1", "/tmp")
    assert built["spec"].provider_options == {"network_policy": {"defaultAction": "deny"}}
    assert built["require_no_egress"] is True
    assert built["spec"].metadata == {"k": "v", "nemo_gym_server": "ci"}
