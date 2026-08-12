# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Where the sandbox may write ``timeoutSeconds`` in ``openclaw.json``.

OpenClaw validates ``agents.list`` entries with a strict schema, so an unknown
``timeoutSeconds`` there fails the whole document and the agent never starts.
Only the provider and ``agents.defaults`` accept it, and ``agents.defaults``
already applies to every list entry that does not override it.

These run the config script the sandbox wrapper carries and assert on the JSON
it produces, so they fail if either write moves back onto a list entry.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from responses_api_agents.pinchbench.tests.test_app import make_agent


_SEEDED_AGENT_ID = "bench-seeded"


def _generate_openclaw_config(tmp_path: Path, **agent_kwargs) -> dict:
    agent = make_agent(**agent_kwargs)
    script = (
        agent._write_direct_exec_wrapper(tmp_path)
        .read_text()
        .split("python3 - <<'PYCFG'\n", 1)[1]
        .split("\nPYCFG", 1)[0]
    )

    work_base = tmp_path / "sandbox"
    config_path = work_base / "home" / ".openclaw" / "openclaw.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps({"agents": {"list": [{"id": _SEEDED_AGENT_ID, "model": "custom/seeded"}]}}))

    subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, **agent._task_env("task_x"), "PINCHBENCH_WORK_BASE": str(work_base)},
        check=True,
    )
    return json.loads(config_path.read_text())


def _seeded_agent(config: dict) -> dict:
    return next(entry for entry in config["agents"]["list"] if entry["id"] == _SEEDED_AGENT_ID)


def test_agents_list_entries_never_carry_a_timeout(tmp_path):
    config = _generate_openclaw_config(tmp_path, openclaw_provider_timeout_seconds=14400)

    assert "timeoutSeconds" not in _seeded_agent(config)


def test_the_ceiling_lands_on_agent_defaults(tmp_path):
    config = _generate_openclaw_config(tmp_path, openclaw_provider_timeout_seconds=14400)

    assert config["agents"]["defaults"]["timeoutSeconds"] == 14400


def test_the_watchdog_lands_on_the_provider(tmp_path):
    config = _generate_openclaw_config(tmp_path, openclaw_provider_timeout_seconds=14400)

    assert config["models"]["providers"]["custom"]["timeoutSeconds"] == 14400


def test_no_timeout_is_written_when_it_is_unset(tmp_path):
    config = _generate_openclaw_config(tmp_path)

    assert "timeoutSeconds" not in _seeded_agent(config)
    assert "timeoutSeconds" not in config["agents"]["defaults"]
    assert "timeoutSeconds" not in config["models"]["providers"]["custom"]
