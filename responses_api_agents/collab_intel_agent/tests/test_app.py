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
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from resources_servers.collab_intel.task_data import TaskData
from responses_api_agents.collab_intel_agent.app import CollabIntelAgent, CollabIntelRunRequest


DIMENSIONS = [f"d{i}" for i in range(8)]


def _row(dimension: str, test_pass, mask: bool = False) -> dict:
    return {"reward": float(bool(test_pass)), "dimension": dimension, "test_pass": test_pass, "mask_sample": mask}


def test_score_weights_dimensions_equally():
    # one passed row in d0 and three failed rows in each other dimension: 1/8, not 1/22
    tasks = [[_row("d0", True)]] + [[_row(d, False)] for d in DIMENSIONS[1:] for _ in range(3)]
    assert CollabIntelAgent._score(tasks)["collab_intel/score"] == pytest.approx(12.5)
    # duplicating rows within a dimension, or adding excluded rows, leaves the score unchanged
    more = tasks + [[_row("d0", True)]] + [[_row("d1", None, mask=True)], [_row("d2", False, mask=True)]]
    assert CollabIntelAgent._score(more)["collab_intel/score"] == pytest.approx(12.5)


def test_score_without_scored_rows_reports_no_score():
    metrics = CollabIntelAgent._score([[_row("d0", None, mask=True)]])
    assert "collab_intel/score" not in metrics and metrics["collab_intel/scored_rows"] == 0


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


def _agent_with_fake_cli(tmp_path: Path, transcript: dict | None, exit_code: int, monkeypatch) -> CollabIntelAgent:
    """An agent whose harness interpreter is a script that writes \`transcript\` (or nothing) and exits with \`exit_code\`."""
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
        f"sys.exit({exit_code})\n"
    )
    script.chmod(0o755)
    monkeypatch.setattr(CollabIntelAgent, "model_post_init", lambda self, context: None)  # no uv sync, no server
    agent = CollabIntelAgent.model_construct(config=SimpleNamespace(episode_timeout_s=60))
    agent._python, agent._root = str(script), tmp_path
    agent._env = lambda: {}
    return agent


@pytest.mark.parametrize(
    "transcript, exit_code, measured, passed",
    [
        ({"score": {"pass": True, "outcome": "ok"}, "messages": [{"seat": "a", "text": "hi"}]}, 0, True, True),
        ({"score": {"pass": False, "outcome": "wrong"}}, 0, True, False),
        ({"messages": []}, 0, False, False),  # no score
        ({"score": {"pass": None}}, 0, False, False),  # empty verdict
        ({"judge_failed": "Timeout", "score": {"pass": False, "outcome": "judge_failed"}}, 0, False, False),
        (None, 1, False, False),  # failed process, no output
    ],
)
def test_run_keeps_unmeasured_results_as_errors(tmp_path, monkeypatch, transcript, exit_code, measured, passed):
    agent = _agent_with_fake_cli(tmp_path, transcript, exit_code, monkeypatch)
    result = asyncio.run(agent._run(tmp_path / "scenario", "solo", 0, tmp_path / "out"))
    assert result["pass"] is passed
    assert bool(result.get("error")) is not measured
    if measured:
        assert result["messages"] == transcript.get("messages", [])
