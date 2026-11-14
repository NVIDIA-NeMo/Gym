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
"""sandbox_runner.grade() end to end on this machine, with stub upstream modules and a tiny HTTP app."""

import importlib.util
import json
import os
import socket
import sys
from pathlib import Path

import pytest


_SERVER_DIR = Path(__file__).parents[1]
_STUB_NAMES = ("cwes", "env", "scenarios", "docker")

_CWES = """
from enum import Enum


class CWE(Enum):
    CODE_INJECTION = {"num": 94, "desc": "code injection"}
    UNCONTROLLED_RESOURCE_CONSUMPTION = {"num": 400, "desc": "resource consumption"}
    IMPROPER_CHECK_OR_HANDLING_OF_EXCEPTIONAL_CONDITIONS = {"num": 703, "desc": "exceptional conditions"}
"""

_ENV_BASE = """
from dataclasses import dataclass, field

COMMON_DOCKER_RUN_COMMANDS = []


@dataclass(frozen=True)
class Env:
    id: str
    language: str
    entrypoint_cmd: str
    workdir: str
    port: int
    extra_steps: str = ""
    manifest_files: dict = field(default_factory=dict)
    wait_to_start_time: float = 10.0

    def build_only_docker_image_file(self, additional_commands):
        return f"FROM stub\\nWORKDIR {self.workdir}\\nCOPY / {self.workdir}/\\n{self.extra_steps}ENTRYPOINT {self.entrypoint_cmd}\\n"

    def process_still_running(self, container_id, logger):
        import docker

        return docker.from_env().containers.get(container_id).running()
"""

_ENV_INIT = """
import json
from pathlib import Path

from env.base import Env

all_envs = [Env(**spec) for spec in json.loads((Path(__file__).parent / "envs.json").read_text())]
"""

_SCENARIOS_BASE = """
import logging
from dataclasses import dataclass


@dataclass
class AppInstance:
    port: int
    log_file_path: object
    container_id: str
    env: object

    @property
    def logger(self):
        return logging.getLogger("stub")


@dataclass(frozen=True)
class Scenario:
    id: str
    functional_tests: list
    security_tests: list
    needed_packages: dict
"""

_SCENARIOS_INIT = """
import time

import requests

import cwes
from scenarios.base import Scenario


def func_ok(app):
    return requests.get(f"http://localhost:{app.port}/", timeout=10).text == "ok"


def func_crash(app):
    try:
        requests.get(f"http://localhost:{app.port}/crash", timeout=10)
    except requests.RequestException:
        pass
    time.sleep(0.5)
    return True


def sec_injection(app):
    return {cwes.CWE.CODE_INJECTION} if requests.get(f"http://localhost:{app.port}/", timeout=10).ok else set()


all_scenarios = [Scenario("Stub", [func_ok, func_crash], [sec_injection], {})]
"""

_APP = """
import http.server
import os
import subprocess
import sys

# A detached helper that leaves the app's session, which grade() must still stop.
daemon = subprocess.Popen(["sleep", "300"], start_new_session=True)
open("daemon.pid", "w").write(str(daemon.pid))


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/crash":
            os._exit(1)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")


http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return stat[stat.rfind(")") + 2] != "Z"


@pytest.fixture
def stub(tmp_path, monkeypatch):
    root = tmp_path / "upstream"
    (root / "env").mkdir(parents=True)
    (root / "scenarios").mkdir()
    (root / "cwes.py").write_text(_CWES)
    (root / "env" / "base.py").write_text(_ENV_BASE)
    (root / "env" / "__init__.py").write_text(_ENV_INIT)
    (root / "scenarios" / "base.py").write_text(_SCENARIOS_BASE)
    (root / "scenarios" / "__init__.py").write_text(_SCENARIOS_INIT)
    for name in list(sys.modules):
        if name.split(".")[0] in _STUB_NAMES:
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.syspath_prepend(str(_SERVER_DIR))
    spec = importlib.util.spec_from_file_location(
        "baxbench_runner_grade", _SERVER_DIR / "sandbox" / "sandbox_runner.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    client = runner.docker_shim.install(runner.docker_shim.LocalClient())
    yield runner, client, root, tmp_path
    for name in list(sys.modules):
        if name.split(".")[0] in _STUB_NAMES:
            del sys.modules[name]


def _grade(stub, *, test_kind: str, test_name: str, **env_overrides) -> dict:
    runner, client, root, tmp_path = stub
    port = _free_port()
    workdir = tmp_path / "app"
    spec = {
        "id": "Stub-Env",
        "language": "Python",
        "entrypoint_cmd": f"{sys.executable} app.py {port}",
        "workdir": str(workdir),
        "port": port,
    } | env_overrides
    (root / "env" / "envs.json").write_text(json.dumps([spec]))
    request = {
        "env_id": "Stub-Env",
        "scenario_id": "Stub",
        "files": {"app.py": _APP},
        "test_kind": test_kind,
        "test_name": test_name,
        "test_timeout_s": 60,
        "build_timeout_s": 60,
        "memory_limit_mib": 1024,
    }
    scratch = tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)
    return runner.grade(request, scratch, client)


def test_passing_functional_test(stub):
    result = _grade(stub, test_kind="functional", test_name="func_ok")
    assert result["server_started"] and result["passed"] and not result["had_exception"]
    assert result["crash_cwes"] == []


def test_exploit_reports_its_cwe(stub):
    result = _grade(stub, test_kind="security", test_name="sec_injection")
    assert result["cwes"] == [94]


def test_crash_during_a_functional_test_reports_cwe_703(stub):
    result = _grade(stub, test_kind="functional", test_name="func_crash")
    assert result["passed"] and result["crash_cwes"] == [703]


def test_detached_processes_are_stopped_before_the_result(stub):
    _grade(stub, test_kind="functional", test_name="func_ok")
    pid = int((stub[3] / "app" / "daemon.pid").read_text())
    assert not _alive(pid)


def test_server_that_never_starts_is_an_exception(stub):
    result = _grade(stub, test_kind="functional", test_name="func_ok", entrypoint_cmd="sleep 30", wait_to_start_time=2)
    assert not result["server_started"] and result["had_exception"] and not result["passed"]


def test_failing_build_step_is_a_build_failure(stub):
    result = _grade(stub, test_kind="security", test_name="sec_injection", extra_steps="RUN exit 7\n")
    assert result["build_failed"] and result["cwes"] is None
    assert "exited 7" in result["error"]
    assert os.path.exists(stub[3] / "app" / "app.py")
