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
"""The sandbox-API backend against a fake sandbox that runs the real sandbox server locally."""

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import nemo_gym.sandbox as gym_sandbox
from responses_api_agents.arc_code_agent import arc_sandbox_api
from responses_api_agents.arc_code_agent.arc_prompt import task_header


class FakeSandbox:
    """Maps the sandbox's /tmp onto a local directory and runs commands there with this interpreter."""

    created = 0
    fail_next_exec = False

    def __init__(self, provider, spec):
        self.spec = spec
        self.root = Path(tempfile.mkdtemp(prefix="fake_sandbox_"))

    def _local(self, text: str) -> str:
        return re.sub(r"/tmp(?=/|\s|$)", str(self.root), text).replace("python3 ", f"{sys.executable} ")

    async def start(self):
        FakeSandbox.created += 1
        for path, content in self.spec.files.items():
            Path(self._local(path)).write_text(content)
        return self

    async def upload(self, local_path, remote_path):
        shutil.copy(local_path, self._local(remote_path))

    async def exec(self, command, timeout_s=None, **kwargs):
        if FakeSandbox.fail_next_exec and "arc-req-" in command:
            FakeSandbox.fail_next_exec = False
            raise ConnectionError("sandbox lost")
        done = subprocess.run(["bash", "-c", self._local(command)], capture_output=True, text=True, timeout=timeout_s)
        return SimpleNamespace(stdout=done.stdout, stderr=done.stderr, return_code=done.returncode)

    async def stop(self):
        shutil.rmtree(self.root, ignore_errors=True)


@pytest.fixture
def pool(monkeypatch):
    monkeypatch.setattr(gym_sandbox, "AsyncSandbox", FakeSandbox)
    FakeSandbox.created, FakeSandbox.fail_next_exec = 0, False
    spec = {"image": "python:3.12-slim", "ttl_s": 3600, "resources": {"cpu": 1, "memory_mib": 4096}}
    return arc_sandbox_api.SandboxApiPool(
        provider={"opensandbox": {}}, sandbox_spec=spec, setup_command=None, num_workers=2, timeout_s=20
    )


EXAMPLES = [{"input": [[1, 0]], "output": [[0, 1]]}]
HEADER = task_header(EXAMPLES, [[2, 0]])


@pytest.mark.asyncio
async def test_runs_replay_earlier_cells_in_a_remote_sandbox(pool):
    reply = await pool.run(HEADER, ["x = 20"], "print(x + 1, examples[0]['output'].tolist())")
    assert reply["error"] == "" and reply["output"].strip() == "21 [[0, 1]]"
    # Predictions go through the same path.
    grid = await pool.predict_answer(HEADER, [], "transform")
    assert grid is None  # no `transform` defined: no answer, no exception
    # Sandboxes are created once per pool member and reused across runs.
    for _ in range(4):
        await pool.run(HEADER, [], "print(0)")
    assert FakeSandbox.created == pool.num_workers
    await pool.close()


@pytest.mark.asyncio
async def test_a_lost_sandbox_is_an_error_reply_and_is_recreated(pool):
    await pool.run(HEADER, [], "print(1)")
    FakeSandbox.fail_next_exec = True
    reply = await pool.run(HEADER, [], "print(2)")
    assert reply["output"] == "" and reply["error"] == arc_sandbox_api.RESTARTED_REPLY
    assert "sandbox lost" not in reply["error"]  # provider details stay in the log, out of the trajectory
    reply = await pool.run(HEADER, [], "print(3)")
    reply = await pool.run(HEADER, [], "print(3)")  # both pool members have run again
    assert reply["output"].strip() == "3" and FakeSandbox.created == pool.num_workers + 1
    await pool.close()


def test_spec_carries_the_sandbox_files_and_rejects_unknown_keys():
    spec = arc_sandbox_api.build_spec({"image": "python:3.12-slim", "metadata": {"a": "b"}})
    assert "/tmp/sandbox_server.py" in spec.files and "/tmp/arc_solve_tools.py" in spec.files
    with pytest.raises(ValueError, match="Unknown sandbox_spec keys: imgae"):
        arc_sandbox_api.build_spec({"imgae": "x"})


def test_reply_is_the_last_json_line_after_the_readiness_line():
    out = '{"ready": true}\nnoise\n{"id": 0, "output": "ok", "timed_out": false, "error": ""}\n'
    assert arc_sandbox_api.parse_reply(out)["output"] == "ok"
    assert arc_sandbox_api.parse_reply('{"ready": true}\n') is None
