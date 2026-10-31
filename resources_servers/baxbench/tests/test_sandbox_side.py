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
"""The code that runs in the grading sandbox: Dockerfile split, per-sample build and the Docker shim."""

import importlib.util
import io
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest

from resources_servers.baxbench import docker_shim
from resources_servers.baxbench.dockerfile_split import parse_env, split_dockerfile


_SERVER_DIR = Path(__file__).parents[1]

_DOCKERFILE = """
# setup base
FROM golang:1.23-bullseye
RUN apt-get update
ENV A=1 B="two"
WORKDIR /app
COPY go.mod build_helper.go ./
RUN go build . \\
    && echo cached
# build the generated code
COPY * ./
RUN go mod tidy || echo "go mod tidy failed"
ENV APP_SECRET supers3cret
ENTRYPOINT ./myapp
"""


def _load_runner():
    # The runner imports its neighbours as top-level modules, as it does in the sandbox.
    sys.path.insert(0, str(_SERVER_DIR))
    spec = importlib.util.spec_from_file_location("baxbench_runner", _SERVER_DIR / "sandbox" / "sandbox_runner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_split_puts_everything_before_the_last_copy_in_the_image():
    split = split_dockerfile(_DOCKERFILE)
    assert split.image_lines[-1].startswith("# build the generated code")
    assert "RUN go build . \\\n    && echo cached" in split.image_lines
    assert split.workdir == "/app" and split.env == {"A": "1", "B": "two"}
    assert [(s.keyword, s.argument) for s in split.sample_steps] == [
        ("COPY", "* ./"),
        ("RUN", 'go mod tidy || echo "go mod tidy failed"'),
        ("ENV", "APP_SECRET supers3cret"),
        ("ENTRYPOINT", "./myapp"),
    ]


def test_parse_env_accepts_both_forms():
    assert parse_env("APP_SECRET=supers3cret RUST_BACKTRACE=1") == {"APP_SECRET": "supers3cret", "RUST_BACKTRACE": "1"}
    assert parse_env("APP_SECRET supers3cret") == {"APP_SECRET": "supers3cret"}


def test_split_without_copy_is_rejected():
    with pytest.raises(ValueError):
        split_dockerfile("FROM scratch\nRUN true\n")


def test_build_sample_copies_like_docker_and_runs_steps(tmp_path):
    runner = _load_runner()
    workdir = tmp_path / "app"
    workdir.mkdir()
    steps = split_dockerfile(
        f"FROM x\nWORKDIR {workdir}\nCOPY / {workdir}/\nRUN echo built > out.txt\nENV FLAG=on\nENTRYPOINT run\n"
    ).sample_steps
    final_workdir, env = runner.build_sample(
        steps,
        context={"Dockerfile": "FROM x", "app.py": "print(1)", "pkg/mod.py": "x = 1"},
        workdir=str(workdir),
        env={"PATH": os.environ["PATH"]},
        scratch=tmp_path / "scratch",
        build_log=tmp_path / "build.log",
        timeout_s=60,
    )
    assert final_workdir == str(workdir) and env["FLAG"] == "on"
    assert (workdir / "app.py").read_text() == "print(1)"
    assert (workdir / "pkg" / "mod.py").exists() and (workdir / "Dockerfile").exists()
    assert (workdir / "out.txt").read_text().strip() == "built"


def test_build_sample_copies_a_single_file_to_its_destination(tmp_path):
    runner = _load_runner()
    steps = split_dockerfile(f"FROM x\nWORKDIR {tmp_path}\nCOPY main.rs src/main.rs\n").sample_steps
    runner.build_sample(
        steps,
        context={"main.rs": "fn main() {}"},
        workdir=str(tmp_path),
        env={},
        scratch=tmp_path / "scratch",
        build_log=tmp_path / "build.log",
        timeout_s=60,
    )
    assert (tmp_path / "src" / "main.rs").read_text() == "fn main() {}"


def test_failing_run_step_is_a_build_failure(tmp_path):
    runner = _load_runner()
    steps = split_dockerfile(f"FROM x\nCOPY * {tmp_path}/\nRUN exit 3\n").sample_steps
    with pytest.raises(runner.BuildFailed, match="exited 3"):
        runner.build_sample(
            steps,
            context={"main.go": "package main"},
            workdir=str(tmp_path),
            env={"PATH": os.environ["PATH"]},
            scratch=tmp_path / "scratch",
            build_log=tmp_path / "build.log",
            timeout_s=60,
        )


def test_local_container_archives_round_trip(tmp_path):
    container = docker_shim.LocalContainer("c1", app_sid=-1, workdir=str(tmp_path), env={}, log_path=tmp_path / "log")
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as tar:
        # Upstream's place_file_on_docker uses the absolute path as the member name.
        info = tarfile.TarInfo(name=str(tmp_path / "secret.txt"))
        info.size = 6
        tar.addfile(info, io.BytesIO(b"SECRET"))
    container.put_archive("/", payload.getvalue())
    stream, _ = container.get_archive(str(tmp_path / "secret.txt"))
    with tarfile.open(fileobj=io.BytesIO(b"".join(stream))) as tar:
        assert tar.extractfile("secret.txt").read() == b"SECRET"
    with pytest.raises(docker_shim.NotFound):
        container.get_archive(str(tmp_path / "missing.txt"))


def test_local_container_exec_and_stats_follow_the_app_process(tmp_path):
    app = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        container = docker_shim.LocalContainer(
            "c2", app_sid=app.pid, workdir=str(tmp_path), env=dict(os.environ), log_path=tmp_path / "log"
        )
        assert container.exec_run("echo hi").output.strip() == b"hi"
        assert container.exec_run("no-such-binary-baxbench").exit_code == 127
        assert container.stats()["memory_stats"]["usage"] > 0
        container.remove(force=True)
        app.wait(timeout=10)
        time.sleep(0.1)
        # A stopped app reports no memory usage, which upstream reads as a crash.
        assert container.stats() == {"memory_stats": {}}
    finally:
        app.kill()


def test_install_makes_upstream_docker_imports_resolve():
    client = docker_shim.LocalClient()
    docker_shim.install(client)
    import docker
    from docker.models.containers import Container, ExecResult

    assert docker.from_env() is client
    assert Container is docker_shim.LocalContainer and ExecResult(0, b"").exit_code == 0


def test_kill_marked_stops_detached_processes_and_spares_others(tmp_path):
    marked = subprocess.Popen(
        ["sleep", "60"], start_new_session=True, env=os.environ | {docker_shim.MARKER_VARIABLE: "tok"}
    )
    unmarked = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        docker_shim.kill_marked("tok")
        assert marked.wait(timeout=10) == -9
        assert unmarked.poll() is None
    finally:
        marked.kill()
        unmarked.kill()


def test_generated_file_that_collides_with_an_image_directory_is_a_build_failure(tmp_path):
    runner = _load_runner()
    workdir = tmp_path / "app"
    (workdir / "src").mkdir(parents=True)
    steps = split_dockerfile(f"FROM x\nWORKDIR {workdir}\nCOPY / {workdir}/\n").sample_steps
    with pytest.raises(runner.BuildFailed, match="COPY"):
        runner.build_sample(
            steps,
            context={"src": "not a directory"},
            workdir=str(workdir),
            env={},
            scratch=tmp_path / "scratch",
            build_log=tmp_path / "build.log",
            timeout_s=60,
        )


def test_kill_new_processes_stops_everything_outside_the_baseline():
    # An unmarked daemon (its environment cleared) is still new, so it is stopped.
    child = subprocess.Popen(["sleep", "60"], start_new_session=True, env={})
    baseline = {1, 2}
    try:
        docker_shim.kill_new_processes(
            baseline, list_pids=lambda: baseline | ({child.pid} if child.poll() is None else set())
        )
        assert child.wait(timeout=10) == -9
    finally:
        child.kill()


def test_log_tail_reads_only_the_end(tmp_path, monkeypatch):
    runner = _load_runner()
    log = tmp_path / "app.log"
    log.write_bytes(b"x" * 100_000 + b"END")
    reads = []
    real_open = Path.open

    class Recording:
        def __init__(self, handle):
            self._handle = handle

        def read(self, size=-1):
            reads.append(size)
            return self._handle.read(size)

        def __getattr__(self, name):
            return getattr(self._handle, name)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._handle.close()

    monkeypatch.setattr(Path, "open", lambda self, *args, **kwargs: Recording(real_open(self, *args, **kwargs)))
    tail = runner._tail(log)
    assert len(tail) == runner._LOG_TAIL_BYTES and tail.endswith("END")
    # A read without a size (or a larger one) would load the whole log into memory.
    assert reads and all(0 <= size <= runner._LOG_TAIL_BYTES for size in reads)
    assert runner._tail(tmp_path / "missing.log") == ""
