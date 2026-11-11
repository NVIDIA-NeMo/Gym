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
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from resources_servers.collab_intel.task_data import TaskData
from responses_api_agents.collab_intel_agent import app as app_module
from responses_api_agents.collab_intel_agent.app import CollabIntelAgent, CollabIntelRunRequest, split_reasoning


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
    agent._python, agent._root = str(script), tmp_path
    agent._env = lambda home: {}
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


def test_harness_env_is_an_allowlist(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setattr(app_module, "get_server_url", lambda name: "http://agent")
    monkeypatch.setattr(CollabIntelAgent, "model_post_init", lambda self, context: None)
    config = SimpleNamespace(
        name="a", judge_base_url="http://judge", judge_model="j", judge_extra={}, policy_model="p", policy_extra={}
    )
    agent = CollabIntelAgent.model_construct(config=config)
    agent._tmp = tmp_path
    env = agent._env(tmp_path / "row")
    assert "HF_TOKEN" not in env and env["PATH"] == "/usr/bin" and env["HOME"] == str(tmp_path / "row")
    assert {"CI_GYM_KEY", "HARNESS_OPENAI_COMPAT", "HARNESS_JUDGE_MODEL"} <= set(env)
