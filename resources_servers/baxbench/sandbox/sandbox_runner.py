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
"""Runs inside a BaxBench framework sandbox: build one sample, start it, run one upstream test.

Usage: ``python sandbox_runner.py REQUEST_JSON RESULT_JSON``

The sandbox image is upstream's framework Dockerfile up to the COPY of the generated code, plus the
upstream source at ``/opt/baxbench/upstream`` (see ``sandbox/images``). This runner does what upstream's
``docker build`` and ``ContainerRunner`` do for one test: copy the files, run the remaining Dockerfile
steps, start the entrypoint, wait for the port, then run the upstream test function unchanged through
``docker_shim`` (one sandbox per test, as upstream uses one container per test). A failing sample is a
normal result; the runner exits nonzero only when grading itself could not run.
"""

import json
import logging
import multiprocessing
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import docker_shim
from dockerfile_split import Instruction, parse_env, split_dockerfile


UPSTREAM_ROOT = Path("/opt/baxbench/upstream")
# The image's environment at the end of its build (``printenv -0``), the base for every step and the app.
IMAGE_ENV_FILE = Path("/opt/baxbench/image.env0")
# Build-time variables that describe the build container, not the image.
_BUILD_ONLY_ENV = {"HOSTNAME", "PWD", "OLDPWD", "SHLVL", "_"}
_LOG_TAIL_BYTES = 4000
_MEMORY_POLL_S = 0.05

LOG = logging.getLogger("baxbench.runner")


class BuildFailed(Exception):
    pass


def _image_env() -> dict[str, str]:
    if not IMAGE_ENV_FILE.exists():
        return dict(os.environ)
    pairs = (item.split("=", 1) for item in IMAGE_ENV_FILE.read_text().split("\0") if "=" in item)
    return {key: value for key, value in pairs if key not in _BUILD_ONLY_ENV}


def _write_context(context: dict[str, str], root: Path) -> None:
    for relative, content in context.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)


def _copy_file(source: Path | str, target: Path | str) -> None:
    # docker build refuses to replace a directory with a file; copy2 would quietly copy into it.
    if Path(target).is_dir():
        raise IsADirectoryError(f"{target} is a directory in the image")
    shutil.copy2(source, target)


def _copy_tree_contents(source: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for entry in source.iterdir():
        if entry.is_dir():
            shutil.copytree(entry, dest / entry.name, dirs_exist_ok=True, copy_function=_copy_file)
        else:
            _copy_file(entry, dest / entry.name)


def _copy(argument: str, context_dir: Path, workdir: str) -> None:
    """Docker ``COPY`` from the build context, for the source forms upstream Dockerfiles use."""
    *sources, dest_arg = argument.split()
    dest = Path(dest_arg) if dest_arg.startswith("/") else Path(workdir) / dest_arg
    for source in sources:
        if source == "/":
            _copy_tree_contents(context_dir, dest)
        elif source == "*":
            # A directory matched by the glob contributes its contents, not itself.
            for entry in context_dir.iterdir():
                if entry.is_dir():
                    _copy_tree_contents(entry, dest)
                else:
                    dest.mkdir(parents=True, exist_ok=True)
                    _copy_file(entry, dest / entry.name)
        else:
            path = context_dir / source
            if not path.exists():
                raise BuildFailed(f"COPY {source}: not in the build context")
            if dest_arg.endswith("/") or dest.is_dir() or len(sources) > 1:
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest / path.name)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)


def build_sample(
    steps: list[Instruction],
    *,
    context: dict[str, str],
    workdir: str,
    env: dict[str, str],
    scratch: Path,
    build_log: Path,
    timeout_s: float,
) -> tuple[str, dict[str, str]]:
    """Run the code COPY and the Dockerfile steps after it; return the final WORKDIR and ENV."""
    context_dir = scratch / "context"
    try:
        _write_context(context, context_dir)
    except OSError as e:
        # The generated file layout cannot exist on disk (a path that is also a directory): the answer's fault.
        raise BuildFailed(f"cannot write the generated files: {e}") from e
    deadline = time.monotonic() + timeout_s
    env = dict(env)
    with build_log.open("ab") as log:
        for step in steps:
            if step.keyword == "COPY":
                try:
                    _copy(step.argument, context_dir, workdir)
                except (OSError, shutil.Error) as e:
                    # A generated path that collides with the image's files, as docker build would refuse it.
                    raise BuildFailed(f"COPY {step.argument}: {e}") from e
            elif step.keyword == "WORKDIR":
                workdir = step.argument if step.argument.startswith("/") else str(Path(workdir) / step.argument)
                Path(workdir).mkdir(parents=True, exist_ok=True)
            elif step.keyword == "ENV":
                env.update(parse_env(step.argument))
            elif step.keyword == "RUN":
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise BuildFailed("build timed out")
                log.write(f"\n$ {step.argument}\n".encode())
                log.flush()
                try:
                    proc = subprocess.run(
                        ["/bin/sh", "-c", step.argument],
                        cwd=workdir,
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=remaining,
                    )
                except subprocess.TimeoutExpired as e:
                    raise BuildFailed("build timed out") from e
                if proc.returncode != 0:
                    raise BuildFailed(f"RUN exited {proc.returncode}: {step.argument}")
            elif step.keyword != "ENTRYPOINT":
                raise RuntimeError(f"unsupported Dockerfile step after the code COPY: {step.keyword}")
    return workdir, env


def _memory_watchdog(sid: int, limit_bytes: int, stop: threading.Event, killed: list[bool]) -> None:
    # Upstream starts the container with mem_limit=1 GiB, so the kernel kills an app that grows past it.
    while not stop.wait(_MEMORY_POLL_S):
        if docker_shim.session_rss_bytes(sid) > limit_bytes:
            docker_shim.kill_session(sid)
            killed.append(True)
            return


def _wait_for_server(port: int, wait_s: float) -> None:
    import requests

    start = time.monotonic()
    while True:
        try:
            requests.get(f"http://localhost:{port}", timeout=wait_s)
            return
        except (requests.ConnectionError, requests.Timeout):
            pass
        if time.monotonic() - start > wait_s:
            raise TimeoutError("Server did not start in time")
        time.sleep(1.0)


def _stop_sample_processes(token: str, baseline: set[int] | None) -> None:
    docker_shim.kill_marked(token)
    if baseline is not None:
        docker_shim.kill_new_processes(baseline)


def run_test_with_timeout(test: Any, app_instance: Any, timeout: float) -> Any:
    """Upstream ``tasks.run_test_with_timeout`` (``tasks.py`` imports the model clients, so it is not imported)."""
    with multiprocessing.Pool(processes=1) as pool:
        async_result = pool.apply_async(test, [app_instance])
        try:
            return async_result.get(timeout=timeout)
        except multiprocessing.TimeoutError:
            pool.terminate()
            raise TimeoutError("Functional test timed out")


def _tail(path: Path) -> str:
    # A generated app can write gigabytes of log; read only the end.
    try:
        with path.open("rb") as log:
            log.seek(max(0, path.stat().st_size - _LOG_TAIL_BYTES))
            data = log.read(_LOG_TAIL_BYTES)
    except OSError:
        return ""
    return data.decode(errors="replace")


def grade(
    request: dict[str, Any], scratch: Path, client: docker_shim.LocalClient, *, baseline: set[int] | None = None
) -> dict[str, Any]:
    """Grade one test. ``baseline`` (the PIDs alive when the runner started) is set only in a single-use
    grading sandbox: then every later process is killed before the result returns, marked or not."""
    import cwes
    import env as bx_env
    import scenarios as bx_scenarios
    from env.base import COMMON_DOCKER_RUN_COMMANDS
    from scenarios.base import AppInstance

    environment = next(e for e in bx_env.all_envs if e.id == request["env_id"])
    scenario = next(s for s in bx_scenarios.all_scenarios if s.id == request["scenario_id"])
    is_functional = request["test_kind"] == "functional"
    tests = scenario.functional_tests if is_functional else scenario.security_tests
    test = next(t for t in tests if t.__name__ == request["test_name"])

    additional = (
        COMMON_DOCKER_RUN_COMMANDS
        + scenario.needed_packages.get("_all_", [])
        + scenario.needed_packages.get(environment.language, [])
    )
    dockerfile = environment.build_only_docker_image_file(additional)
    split = split_dockerfile(dockerfile)
    # Upstream's build context: the Dockerfile, the generated files, then the framework's manifest files.
    context = {"Dockerfile": dockerfile, **request["files"], **environment.manifest_files}
    build_log = scratch / "build.log"
    app_log = scratch / "app.log"
    test_log = scratch / "test.log"
    result: dict[str, Any] = {"build_failed": False, "server_started": False, "passed": False, "had_exception": False}
    result["cwes"] = None
    result["crash_cwes"] = []
    # The generated code runs as root next to this runner: every process it starts carries this token and is
    # killed before the result is written, so nothing it left behind can rewrite the result file.
    token = uuid.uuid4().hex
    try:
        workdir, app_env = build_sample(
            split.sample_steps,
            context=context,
            workdir=split.workdir,
            env=_image_env() | split.env | {docker_shim.MARKER_VARIABLE: token},
            scratch=scratch,
            build_log=build_log,
            timeout_s=request["build_timeout_s"],
        )
    except BuildFailed as e:
        _stop_sample_processes(token, baseline)
        result.update(build_failed=True, error=str(e), build_log_tail=_tail(build_log))
        return result

    container_id = f"baxbench-{uuid.uuid4()}"
    with app_log.open("wb") as log:
        app = subprocess.Popen(
            ["/bin/sh", "-c", environment.entrypoint_cmd],
            cwd=workdir,
            env=app_env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    client.registry[container_id] = docker_shim.LocalContainer(
        container_id, app_sid=app.pid, workdir=workdir, env=app_env, log_path=app_log
    )
    stop, oom_killed = threading.Event(), []
    watchdog = threading.Thread(
        target=_memory_watchdog, args=(app.pid, request["memory_limit_mib"] * 2**20, stop, oom_killed), daemon=True
    )
    watchdog.start()
    logger = logging.getLogger(container_id)
    crash_cwe = cwes.CWE.IMPROPER_CHECK_OR_HANDLING_OF_EXCEPTIONAL_CONDITIONS
    try:
        _wait_for_server(environment.port, environment.wait_to_start_time)
        result["server_started"] = True
        server_ran_before = environment.process_still_running(container_id, logger)
        outcome = run_test_with_timeout(
            test,
            AppInstance(port=environment.port, log_file_path=test_log, container_id=container_id, env=environment),
            request["test_timeout_s"],
        )
        crashed = server_ran_before and not environment.process_still_running(container_id, logger)
        if is_functional:
            result["passed"] = bool(outcome)
            if crashed:
                result["crash_cwes"] = [crash_cwe.value["num"]]
        else:
            found = set(outcome) if outcome is not None else None
            if crashed:
                found = found if found is not None else set()
                if cwes.CWE.UNCONTROLLED_RESOURCE_CONSUMPTION not in found:
                    found.add(crash_cwe)
            result["cwes"] = None if found is None else sorted(c.value["num"] for c in found)
    except Exception as e:
        # Upstream counts any exception around a test as a failed functional test or an inconclusive exploit.
        result["had_exception"] = True
        result["error"] = f"{type(e).__name__}: {e}"
    finally:
        stop.set()
        docker_shim.kill_session(app.pid)
        app.wait(timeout=30)
        _stop_sample_processes(token, baseline)
    result["oom_killed"] = bool(oom_killed)
    result["app_log_tail"] = _tail(app_log)
    result["test_log_tail"] = _tail(test_log)
    return result


def main(request_path: Path, result_path: Path) -> None:
    baseline = docker_shim.live_pids()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request_path.unlink()
    image_sha = (UPSTREAM_ROOT / "SHA").read_text().strip()
    if image_sha != request["upstream_sha"]:
        raise RuntimeError(f"image has upstream {image_sha}, the task needs {request['upstream_sha']}")
    # Upstream modules call docker.from_env() at import time, so the shim goes in first.
    client = docker_shim.install(docker_shim.LocalClient())
    sys.path.insert(0, str(UPSTREAM_ROOT / "src"))
    scratch = request_path.parent / "scratch"
    scratch.mkdir(exist_ok=True)
    payload = grade(request, scratch, client, baseline=baseline if request.get("isolated_sandbox") else None)
    pending = result_path.with_name(result_path.name + ".pending")
    pending.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(pending, result_path)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
