# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the terminal_bench_4 server against a fake sandbox that honours the `: ng-tb4-<step>;` labels."""

from __future__ import annotations

import fnmatch
import io
import json
import re
import shlex
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, Optional
from unittest.mock import MagicMock

import pytest

import resources_servers.terminal_bench_4.app as app_module
from nemo_gym.sandbox import SandboxExecResult, SandboxResources
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.terminal_bench_4.app import (
    REMOTE_TARBALL,
    RewardParseError,
    TerminalBench4ResourcesServer,
    TerminalBench4ResourcesServerConfig,
    TerminalBench4SeedSessionRequest,
    TerminalBench4VerifyRequest,
    build_pack_command,
    build_prepare_targets_command,
    build_probe_command,
    clamp_resource_requests,
    derive_resources,
    parse_pack_output,
    parse_probe_output,
    parse_reward_payload,
    parse_verifier_files_probe,
    translate_excludes,
)
from resources_servers.terminal_bench_4.task_manifest import (
    CONVENTION_ARTIFACTS_DIR,
    ArtifactEntry,
    load_task,
    parse_artifact,
    with_convention_entry,
)


EMPTY_RESPONSE = {
    "id": "fixture_response",
    "created_at": 0,
    "model": "fixture",
    "object": "response",
    "output": [],
    "parallel_tool_calls": False,
    "tool_choice": "none",
    "tools": [],
}
STEP_RE = re.compile(r"^: ng-tb4-([a-z-]+); ?(.*)$", re.DOTALL)


# ------------------------------------------------------------------------------------------------
# Fake sandbox
# ------------------------------------------------------------------------------------------------


class FakeSandbox:
    """In-memory filesystem plus a dispatcher for the labelled commands the server issues."""

    def __init__(
        self,
        *,
        name: str,
        files: Optional[Dict[str, bytes]] = None,
        dirs: Optional[set] = None,
        on_run_tests: Optional[Callable[["FakeSandbox"], int]] = None,
        on_run_solution: Optional[Callable[["FakeSandbox"], int]] = None,
        fail_steps: Optional[Dict[str, int]] = None,
        pack_rc: int = 0,
        pack_warning: str = "",
        stop_raises: bool = False,
    ) -> None:
        self.files: Dict[str, bytes] = dict(files or {})
        self.dirs: set = set(dirs or set())
        self.commands: list = []
        self.calls: list = []
        self.stopped = False
        self.on_run_tests = on_run_tests
        self.on_run_solution = on_run_solution
        self.fail_steps = fail_steps or {}
        self.pack_rc = pack_rc
        self.pack_warning = pack_warning
        self.stop_raises = stop_raises
        self.excludes_seen: list = []
        self._handle = SimpleNamespace(sandbox_id=f"sb-{name}", provider_name="fake")

    # filesystem helpers
    def kind(self, path: str) -> str:
        path = path.rstrip("/") or "/"
        if path == "/":
            return "dir"
        if path in self.dirs or any(f.startswith(path + "/") for f in self.files):
            return "dir"
        if path in self.files:
            return "file"
        return "missing"

    @staticmethod
    def _excluded(name: str, excludes) -> bool:
        parts = name.split("/")
        for pattern in excludes:
            if "/" in pattern:
                if name == pattern or name.startswith(pattern + "/"):
                    return True
            elif any(fnmatch.fnmatchcase(part, pattern) for part in parts):
                return True
        return False

    def _pack(self, members, excludes=()) -> bytes:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            for member in members:
                absolute = "/" + member
                if self.kind(absolute) == "file":
                    info = tarfile.TarInfo(member)
                    info.size = len(self.files[absolute])
                    archive.addfile(info, io.BytesIO(self.files[absolute]))
                elif self.kind(absolute) == "dir":
                    for path, data in sorted(self.files.items()):
                        if path.startswith(absolute + "/") and not self._excluded(path.lstrip("/"), excludes):
                            info = tarfile.TarInfo(path.lstrip("/"))
                            info.size = len(data)
                            archive.addfile(info, io.BytesIO(data))
        return buffer.getvalue()

    def _extract(self, blob: bytes) -> None:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
            for member in archive.getmembers():
                if member.isfile():
                    self.files["/" + member.name] = archive.extractfile(member).read()

    # sandbox API
    async def exec(self, command, *, cwd=None, env=None, timeout_s=180, user=None) -> SandboxExecResult:
        self.commands.append(command)
        self.calls.append({"command": command, "cwd": cwd, "env": env, "timeout_s": timeout_s, "user": user})
        match = STEP_RE.match(command)
        assert match, f"unlabelled command: {command!r}"
        step, body = match.group(1), match.group(2)
        if step in self.fail_steps:
            return SandboxExecResult(stdout="", stderr=f"forced failure of {step}", return_code=self.fail_steps[step])
        if step == "probe":
            paths = shlex.split(body.split("for p in ", 1)[1].split("; do", 1)[0])
            return SandboxExecResult(stdout="".join(f"{self.kind(p)}\t{p}\n" for p in paths), stderr="", return_code=0)
        if step == "pack":
            tar_segment = body.split("then tar ", 1)[1].split("; rc=$?", 1)[0]
            tokens = shlex.split(tar_segment)
            excludes = [tok[len("--exclude=") :] for tok in tokens if tok.startswith("--exclude=")]
            members = tokens[tokens.index("--") + 1 :]
            members = [m for m in members if not m.startswith("2>")]
            self.excludes_seen = excludes
            blob = self._pack(members, excludes)
            self.files[REMOTE_TARBALL] = blob
            stdout = f"NG_TB4_SIZE={len(blob)}\nNG_TB4_TAR_RC={self.pack_rc}\n{self.pack_warning}"
            return SandboxExecResult(stdout=stdout, stderr="", return_code=0)
        if step == "prepare-targets":
            for target in re.findall(r"find (\S+) -mindepth", body):
                target = shlex.split(target)[0]
                for path in [p for p in self.files if p.startswith(target.rstrip("/") + "/")]:
                    del self.files[path]
                self.dirs.add(target)
            for parent in re.findall(r"mkdir -p (\S+) && chmod 777", body):
                self.dirs.add(shlex.split(parent)[0])
            return SandboxExecResult(stdout="", stderr="", return_code=0)
        if step == "extract":
            self._extract(self.files.pop(REMOTE_TARBALL))
            return SandboxExecResult(stdout="", stderr="", return_code=0)
        if step == "check-tests":
            present = "/tests/test.sh" in self.files
            return SandboxExecResult(stdout="", stderr="" if present else "missing", return_code=0 if present else 1)
        if step == "run-tests":
            return_code = self.on_run_tests(self) if self.on_run_tests else 0
            return SandboxExecResult(stdout="", stderr="", return_code=return_code)
        if step == "probe-reward":
            lines = [
                f"{len(data)}\t{path.rsplit('/', 1)[1]}\n"
                for path, data in self.files.items()
                if path.startswith("/logs/verifier/")
                and path.rsplit("/", 1)[1] in ("reward.json", "reward.txt", "ctrf.json", "test-stdout.txt")
            ]
            return SandboxExecResult(stdout="".join(lines), stderr="", return_code=0)
        if step in ("solution-mkdir", "solution-chmod", "collect-hook"):
            return SandboxExecResult(stdout="", stderr="", return_code=0)
        if step == "run-solution":
            return_code = self.on_run_solution(self) if self.on_run_solution else 0
            return SandboxExecResult(stdout="solved\n", stderr="", return_code=return_code)
        raise AssertionError(f"unexpected step {step!r}")

    async def upload(self, local_path, remote_path) -> None:
        self.files[remote_path] = Path(local_path).read_bytes()

    async def download(self, remote_path, local_path) -> None:
        if remote_path not in self.files:
            raise FileNotFoundError(remote_path)
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        Path(local_path).write_bytes(self.files[remote_path])

    async def stop(self) -> None:
        if self.stop_raises:
            raise RuntimeError("stop failed")
        self.stopped = True


# ------------------------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------------------------


def make_task_dir(
    tmp_path: Path,
    *,
    name: str = "terminal-bench/demo",
    artifacts: str = '["/app/out.step", "/app/evalbench/"]',
    extra_toml: str = "",
    compose: bool = False,
    solution: bool = True,
) -> Path:
    task_dir = tmp_path / "tasks" / name.rsplit("/", 1)[1]
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "instruction.md").write_text("Do the task.\n")
    (task_dir / "task.toml").write_text(
        f"""
schema_version = "1.1"
artifacts = {artifacts}

[task]
name = "{name}"

[verifier]
timeout_sec = 240.0
environment_mode = "separate"

[agent]
timeout_sec = 28800.0

[environment]
build_timeout_sec = 600.0
cpus = 2
memory_mb = 8192
storage_mb = 10240
gpus = 0
{extra_toml}
"""
    )
    (task_dir / "tests" / "test.sh").write_text("#!/bin/bash\necho hi\n")
    (task_dir / "tests" / "Dockerfile").write_text("FROM python:3.12-slim\nCOPY . /tests/\n")
    (task_dir / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\nWORKDIR /app\n")
    if compose:
        (task_dir / "environment" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    build: .\n  api:\n    image: nginx\n"
        )
    if solution:
        (task_dir / "solution").mkdir()
        (task_dir / "solution" / "solve.sh").write_text("#!/bin/bash\npython /solution/solve.py\n")
        (task_dir / "solution" / "solve.py").write_text("print('x')\n")
    return task_dir


def make_server(tmp_path: Path, **overrides) -> TerminalBench4ResourcesServer:
    config = TerminalBench4ResourcesServerConfig(
        sandbox_provider="",
        sandbox_config={"resources": {"cpu": 1, "memory_mib": 1024, "disk_gib": 10}},
        host="",
        port=0,
        entrypoint="",
        name="tb4_test",
        logs_dir=tmp_path / "logs",
        **overrides,
    )
    return TerminalBench4ResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def wire_sandboxes(server: TerminalBench4ResourcesServer, agent: Optional[FakeSandbox], verifier: FakeSandbox):
    created = []

    async def fake_create(*, image, resources, role, task):
        created.append({"image": image, "resources": resources, "role": role, "task": task.task_name})
        if role == "agent":
            assert agent is not None, "agent sandbox creation was not expected"
            return agent
        return verifier

    server._create_sandbox = fake_create  # type: ignore[method-assign]
    return created


def fake_request(session_id: Optional[str] = "sess-1") -> SimpleNamespace:
    return SimpleNamespace(session={SESSION_ID_KEY: session_id} if session_id else {}, cookies={})


def verifier_with_tests(**kwargs) -> FakeSandbox:
    files = {"/tests/test.sh": b"#!/bin/bash\n"} | kwargs.pop("files", {})
    return FakeSandbox(name="verifier", files=files, **kwargs)


def write_reward(value: str, *, as_json: bool = False, also_txt: Optional[str] = None):
    def _run(sandbox: FakeSandbox) -> int:
        sandbox.files["/logs/verifier/test-stdout.txt"] = b"pytest ... 3 passed\n"
        if as_json:
            sandbox.files["/logs/verifier/reward.json"] = value.encode()
        else:
            sandbox.files["/logs/verifier/reward.txt"] = value.encode()
        if also_txt is not None:
            sandbox.files["/logs/verifier/reward.txt"] = also_txt.encode()
        return 0

    return _run


def row(task_dir: Path, name: str = "terminal-bench/demo") -> dict:
    return {
        "task_name": name,
        "docker_image": "repo:demo-environment-v4.0.0",
        "verifier_docker_image": "repo:demo-verifier-v4.0.0",
        "task_folder": str(task_dir),
    }


async def seed_and_verify(server, task_dir, *, session_id="sess-1", name="terminal-bench/demo"):
    request = fake_request(session_id)
    await server.seed_session(request, TerminalBench4SeedSessionRequest(**row(task_dir, name)))
    body = TerminalBench4VerifyRequest(
        **row(task_dir, name), responses_create_params={"input": "x"}, response=EMPTY_RESPONSE
    )
    return await server.verify(request, body)


# ------------------------------------------------------------------------------------------------
# Pure helpers and manifest parsing
# ------------------------------------------------------------------------------------------------


class TestManifest:
    def test_artifact_forms(self) -> None:
        assert parse_artifact("/app/out.step") == ArtifactEntry(source="/app/out.step")
        sidecar = parse_artifact("/shared/x.json@api")
        assert sidecar.service == "api" and not sidecar.is_main
        table = parse_artifact({"source": "/app/", "service": "main", "exclude": ["*.pyc"]})
        assert table.normalized_source == "/app" and table.relative_to_root == "app" and table.exclude == ("*.pyc",)
        with pytest.raises(ValueError):
            parse_artifact("@api")

    def test_convention_entry_is_prepended_once(self) -> None:
        entries = with_convention_entry([ArtifactEntry("/app/x")])
        assert [e.source for e in entries] == [CONVENTION_ARTIFACTS_DIR, "/app/x"]
        again = with_convention_entry(list(entries) + [ArtifactEntry("/logs/artifacts/")])
        assert (
            sum(1 for e in again if e.normalized_source == CONVENTION_ARTIFACTS_DIR) == 2
        )  # explicit one kept, none added

    def test_load_task_fields(self, tmp_path: Path) -> None:
        task_dir = make_task_dir(
            tmp_path,
            artifacts='["/app/out.step", "/shared/x@api", {source = "/app/dir/", service = "main"}]',
            extra_toml=(
                "[verifier.environment]\ncpus = 1\nmemory_mb = 4096\n\n"
                '[[verifier.collect]]\ncommand = "echo main"\n\n'
                '[[verifier.collect]]\ncommand = "echo side"\nservice = "api"\ntimeout_sec = 12\n'
            ),
        )
        task = load_task(task_dir)
        assert task.task_name == "terminal-bench/demo"
        assert [a.source for a in task.main_artifacts] == [CONVENTION_ARTIFACTS_DIR, "/app/out.step", "/app/dir/"]
        assert [a.source for a in task.sidecar_artifacts] == ["/shared/x"]
        assert task.verifier_timeout_sec == 240.0 and task.agent_timeout_sec == 28800.0
        assert task.effective_verifier_environment() == {"cpus": 1, "memory_mb": 4096}
        assert [h.command for h in task.main_collect_hooks] == ["echo main"]
        assert task.sidecar_collect_hooks[0].timeout_sec == 12
        assert not task.is_compose and not task.requires_gpu

    def test_verifier_environment_falls_back_to_a_copy_of_environment(self, tmp_path: Path) -> None:
        task = load_task(make_task_dir(tmp_path))
        effective = task.effective_verifier_environment()
        assert effective == task.environment
        effective["cpus"] = 99
        assert task.environment["cpus"] == 2

    def test_compose_and_gpu_detection(self, tmp_path: Path) -> None:
        compose_task = load_task(make_task_dir(tmp_path / "a", compose=True))
        assert compose_task.is_compose and set(compose_task.compose_services) == {"main", "api"}
        gpu_task = load_task(make_task_dir(tmp_path / "b", extra_toml="[verifier.environment]\ngpus = 1\n"))
        assert gpu_task.requires_gpu


class TestHelpers:
    def test_probe_roundtrip(self) -> None:
        command = build_probe_command(["/app/out.step", "/app/with space"])
        assert command.startswith(": ng-tb4-probe;") and "'/app/with space'" in command
        assert parse_probe_output("dir\t/app/x\nfile\t/app/y\nmissing\t/app/z\nnoise\n") == {
            "/app/x": "dir",
            "/app/y": "file",
            "/app/z": "missing",
        }

    def test_pack_and_prepare_commands(self) -> None:
        command = build_pack_command(["app/out.step", "app/with space"], excludes=["app/x/skip me", "*.pyc"])
        assert "tar -czf" in command and "--ignore-failed-read" in command and "'app/with space'" in command
        assert "--exclude='app/x/skip me' --exclude='*.pyc' --" in command
        assert '[ "$rc" -le 1 ] || exit "$rc"' in command and f"NG_TB4_SIZE=$(stat -c %s {REMOTE_TARBALL})" in command
        with pytest.raises(ValueError):
            build_pack_command([])
        assert parse_pack_output("NG_TB4_SIZE=12\nNG_TB4_TAR_RC=1\ntar: app: file changed as we read it\n") == (
            12,
            1,
            "tar: app: file changed as we read it",
        )
        with pytest.raises(ValueError):
            parse_pack_output("108\n")
        assert translate_excludes("app/nemo", ["./megatron_parallel.py", "__pycache__", "*.pyc", "sub/dir/", ""]) == [
            "app/nemo/megatron_parallel.py",
            "__pycache__",
            "*.pyc",
            "app/nemo/sub/dir",
        ]
        prepare = build_prepare_targets_command(["/app/evalbench"], ["/app/out.step", "/app/results/a.npz"])
        assert "find /app/evalbench -mindepth 1 -delete && chmod 777 /app/evalbench" in prepare
        assert (
            "if [ -L /app/evalbench ] || { [ -e /app/evalbench ] && [ ! -d /app/evalbench ]; }; then unlink /app/evalbench; fi"
            in prepare
        )
        assert (
            "mkdir -p /app && chmod 777 /app" in prepare
            and "mkdir -p /app/results && chmod 777 /app/results" in prepare
        )
        assert "mkdir -p /logs/verifier /logs/artifacts && chmod 777 /logs/verifier /logs/artifacts" in prepare

    def test_reward_parsing(self) -> None:
        assert parse_reward_payload("reward.txt", b"1\n") == (1.0, {"reward": 1.0})
        assert parse_reward_payload("reward.json", b'{"reward": 0.5, "n_passed": 3}') == (
            0.5,
            {"reward": 0.5, "n_passed": 3.0},
        )
        assert parse_reward_payload("reward.json", b'{"score": 1}') == (1.0, {"score": 1.0})
        assert parse_reward_payload("reward.json", b"0") == (0.0, {"reward": 0.0})
        for source, payload in [
            ("reward.txt", b""),
            ("reward.txt", b"one"),
            ("reward.json", b"{bad"),
            ("reward.json", b'{"a": 1, "b": 2}'),
            ("reward.json", b"true"),
            ("reward.json", b"[1]"),
        ]:
            with pytest.raises(RewardParseError):
                parse_reward_payload(source, payload)

    def test_verifier_files_probe_parsing(self) -> None:
        assert parse_verifier_files_probe("2\treward.txt\n0\treward.json\nx\ty\n") == {
            "reward.txt": 2,
            "reward.json": 0,
        }

    def test_derive_resources(self) -> None:
        base = {"cpu": 1, "memory_mib": 1024, "disk_gib": 10}
        common = dict(cpu_multiplier=1.5, memory_multiplier=2.0, min_cpu=1, min_memory_mib=2048, min_disk_gib=20)
        derived = derive_resources(
            {"cpus": 2, "memory_mb": 4096, "storage_mb": 51200}, base=base, use_task_resources=True, **common
        )
        assert derived == SandboxResources(cpu=3.0, memory_mib=8192, disk_gib=50)
        floored = derive_resources(
            {"cpus": 0.25, "memory_mb": 512, "storage_mb": 1024}, base=base, use_task_resources=True, **common
        )
        assert floored == SandboxResources(cpu=1.0, memory_mib=2048, disk_gib=20)
        untouched = derive_resources({"cpus": 8}, base=base, use_task_resources=False, **common)
        assert untouched == SandboxResources(cpu=1.0, memory_mib=1024, disk_gib=10)


# ------------------------------------------------------------------------------------------------
# Server behaviour
# ------------------------------------------------------------------------------------------------


class TestApp:
    def test_sanity(self, tmp_path: Path) -> None:
        make_server(tmp_path)

    @pytest.mark.asyncio
    async def test_happy_path_file_and_dir_artifacts(self, tmp_path: Path) -> None:
        task_dir = make_task_dir(tmp_path)
        server = make_server(tmp_path)
        agent = FakeSandbox(
            name="agent",
            files={"/app/out.step": b"STEP", "/app/evalbench/a.py": b"print(1)\n", "/app/evalbench/sub/b.py": b"2"},
        )
        verifier = verifier_with_tests(files={"/app/evalbench/stale.py": b"old"}, on_run_tests=write_reward("1\n"))
        created = wire_sandboxes(server, agent, verifier)

        response = await seed_and_verify(server, task_dir)

        assert response.reward == 1.0 and response.evaluation_completed and response.failure_reason is None
        assert response.reward_source == "reward.txt" and response.rewards == {"reward": 1.0}
        assert response.verifier_exit_code == 0 and "3 passed" in response.test_output
        assert [c["role"] for c in created] == ["agent", "verifier"]
        assert (
            created[0]["image"] == "repo:demo-environment-v4.0.0"
            and created[1]["image"] == "repo:demo-verifier-v4.0.0"
        )
        assert created[0]["resources"] == SandboxResources(cpu=2.0, memory_mib=8192, disk_gib=10)
        statuses = {m["source"]: m["status"] for m in response.artifact_manifest}
        assert statuses == {CONVENTION_ARTIFACTS_DIR: "missing", "/app/out.step": "ok", "/app/evalbench/": "ok"}
        assert response.artifacts_packed == 2 and response.artifact_tarball_bytes > 0
        # Re-materialized at the original paths; the verifier image's own directory content was emptied first.
        assert verifier.files["/app/out.step"] == b"STEP" and verifier.files["/app/evalbench/sub/b.py"] == b"2"
        assert "/app/evalbench/stale.py" not in verifier.files and REMOTE_TARBALL not in verifier.files
        assert agent.stopped and verifier.stopped and response.agent_sandbox_stopped
        run_tests_call = next(c for c in verifier.calls if c["command"].startswith(": ng-tb4-run-tests;"))
        assert (
            run_tests_call["timeout_s"] == 240.0
            and "(/tests/test.sh) > /logs/verifier/test-stdout.txt 2>&1" in run_tests_call["command"]
        )
        assert response.verifier_sandbox_observation.outcome == "completed"
        assert response.verifier_sandbox_observation.sandbox_id == "sb-verifier"
        log_dir = Path(response.log_dir)
        assert (log_dir / "verify_summary.json").exists() and (log_dir / "test-stdout.txt").exists()
        assert json.loads((log_dir / "artifact_manifest.json").read_text())["entries"][1]["kind"] == "file"

    @pytest.mark.asyncio
    async def test_reward_json_takes_precedence(self, tmp_path: Path) -> None:
        task_dir = make_task_dir(tmp_path)
        server = make_server(tmp_path)
        agent = FakeSandbox(name="agent", files={"/app/out.step": b"x"})
        verifier = verifier_with_tests(on_run_tests=write_reward('{"reward": 0}', as_json=True, also_txt="1\n"))
        wire_sandboxes(server, agent, verifier)
        response = await seed_and_verify(server, task_dir)
        assert response.reward == 0.0 and response.evaluation_completed and response.reward_source == "reward.json"

    @pytest.mark.asyncio
    async def test_missing_reward_fails_closed(self, tmp_path: Path) -> None:
        task_dir = make_task_dir(tmp_path)
        server = make_server(tmp_path)
        agent = FakeSandbox(name="agent", files={"/app/out.step": b"x"})

        def no_reward(sandbox: FakeSandbox) -> int:
            sandbox.files["/logs/verifier/test-stdout.txt"] = b"boom\n"
            return 2

        verifier = verifier_with_tests(on_run_tests=no_reward)
        wire_sandboxes(server, agent, verifier)
        response = await seed_and_verify(server, task_dir)
        assert response.reward == 0.0 and not response.evaluation_completed
        assert "neither" in response.failure_reason and response.verifier_exit_code == 2
        assert response.test_output == "boom\n" and verifier.stopped

    @pytest.mark.asyncio
    async def test_unparseable_reward_fails_closed(self, tmp_path: Path) -> None:
        task_dir = make_task_dir(tmp_path)
        server = make_server(tmp_path)
        agent = FakeSandbox(name="agent", files={"/app/out.step": b"x"})
        verifier = verifier_with_tests(on_run_tests=write_reward("PASS\n"))
        wire_sandboxes(server, agent, verifier)
        response = await seed_and_verify(server, task_dir)
        assert (
            response.reward == 0.0 and not response.evaluation_completed and "not a number" in response.failure_reason
        )

    @pytest.mark.asyncio
    async def test_verifier_timeout_fails_closed(self, tmp_path: Path) -> None:
        task_dir = make_task_dir(tmp_path)
        server = make_server(tmp_path, verifier_timeout_multiplier=0.5, verifier_timeout_floor_s=1)

        def hang(sandbox: FakeSandbox) -> int:
            raise TimeoutError("command timed out")

        agent = FakeSandbox(name="agent", files={"/app/out.step": b"x"})
        verifier = verifier_with_tests(on_run_tests=hang)
        wire_sandboxes(server, agent, verifier)
        response = await seed_and_verify(server, task_dir)
        assert response.reward == 0.0 and not response.evaluation_completed
        assert "TimeoutError" in response.failure_reason and "120s" in response.failure_reason
        assert response.verifier_sandbox_observation.outcome == "timeout" and verifier.stopped

    @pytest.mark.asyncio
    async def test_missing_tests_script_is_an_infrastructure_error(self, tmp_path: Path) -> None:
        task_dir = make_task_dir(tmp_path)
        server = make_server(tmp_path)
        agent = FakeSandbox(name="agent", files={"/app/out.step": b"x"})
        verifier = FakeSandbox(name="verifier")  # no /tests/test.sh baked in
        wire_sandboxes(server, agent, verifier)
        with pytest.raises(RuntimeError, match="no executable /tests/test.sh"):
            await seed_and_verify(server, task_dir)
        assert agent.stopped and verifier.stopped

    @pytest.mark.asyncio
    async def test_artifact_probe_failure_stops_agent_sandbox_and_raises(self, tmp_path: Path) -> None:
        task_dir = make_task_dir(tmp_path)
        server = make_server(tmp_path)
        agent = FakeSandbox(name="agent", files={"/app/out.step": b"x"}, fail_steps={"probe": 1})
        verifier = verifier_with_tests()
        created = wire_sandboxes(server, agent, verifier)
        with pytest.raises(RuntimeError, match="artifact probe failed"):
            await seed_and_verify(server, task_dir)
        assert agent.stopped and [c["role"] for c in created] == ["agent"]

    @pytest.mark.asyncio
    async def test_oversized_tarball_raises(self, tmp_path: Path) -> None:
        task_dir = make_task_dir(tmp_path)
        server = make_server(tmp_path, artifact_max_bytes=1)
        agent = FakeSandbox(name="agent", files={"/app/out.step": b"x" * 100})
        verifier = verifier_with_tests()
        wire_sandboxes(server, agent, verifier)
        with pytest.raises(RuntimeError, match="above artifact_max_bytes"):
            await seed_and_verify(server, task_dir)
        assert agent.stopped

    @pytest.mark.asyncio
    async def test_no_artifacts_present_still_grades(self, tmp_path: Path) -> None:
        task_dir = make_task_dir(tmp_path)
        server = make_server(tmp_path)
        agent = FakeSandbox(name="agent")
        verifier = verifier_with_tests(on_run_tests=write_reward("0\n"))
        wire_sandboxes(server, agent, verifier)
        response = await seed_and_verify(server, task_dir)
        assert response.evaluation_completed and response.reward == 0.0 and response.artifacts_packed == 0
        assert not any(c["command"].startswith(": ng-tb4-pack;") for c in agent.calls)
        assert not any(c["command"].startswith(": ng-tb4-extract;") for c in verifier.calls)

    @pytest.mark.asyncio
    async def test_sidecar_artifacts_and_hooks_are_skipped_when_compose_allowed(self, tmp_path: Path) -> None:
        task_dir = make_task_dir(
            tmp_path,
            artifacts='["/app/out.step", "/shared/x.json@api"]',
            extra_toml='[[verifier.collect]]\ncommand = "echo main"\n\n[[verifier.collect]]\ncommand = "echo api"\nservice = "api"\n',
            compose=True,
        )
        server = make_server(tmp_path, allow_compose_tasks=True)
        agent = FakeSandbox(name="agent", files={"/app/out.step": b"x"})
        verifier = verifier_with_tests(on_run_tests=write_reward("1\n"))
        wire_sandboxes(server, agent, verifier)
        response = await seed_and_verify(server, task_dir)
        assert response.evaluation_completed
        assert {m["source"]: m["status"] for m in response.artifact_manifest}["/shared/x.json"] == "skipped_sidecar"
        assert [h["status"] for h in response.collect_hook_results] == ["skipped_sidecar", "ok"]
        assert any(c["command"] == ": ng-tb4-collect-hook; echo main" for c in agent.calls)

    @pytest.mark.asyncio
    async def test_seed_session_refuses_compose_and_gpu_tasks(self, tmp_path: Path) -> None:
        server = make_server(tmp_path)
        wire_sandboxes(server, FakeSandbox(name="agent"), verifier_with_tests())
        compose_dir = make_task_dir(tmp_path / "c", compose=True)
        with pytest.raises(ValueError, match="compose"):
            await server.seed_session(fake_request(), TerminalBench4SeedSessionRequest(**row(compose_dir)))
        gpu_dir = make_task_dir(tmp_path / "g", extra_toml="[verifier.environment]\ngpus = 1\n")
        with pytest.raises(ValueError, match="gpus"):
            await server.seed_session(fake_request(), TerminalBench4SeedSessionRequest(**row(gpu_dir)))

    @pytest.mark.asyncio
    async def test_verify_without_seed_is_an_error_outside_golden_mode(self, tmp_path: Path) -> None:
        task_dir = make_task_dir(tmp_path)
        server = make_server(tmp_path)
        wire_sandboxes(server, FakeSandbox(name="agent"), verifier_with_tests())
        body = TerminalBench4VerifyRequest(
            **row(task_dir), responses_create_params={"input": "x"}, response=EMPTY_RESPONSE
        )
        with pytest.raises(RuntimeError, match="seed_session must precede verify"):
            await server.verify(fake_request(), body)

    @pytest.mark.asyncio
    async def test_golden_mode_uploads_solution_runs_it_and_grades(self, tmp_path: Path) -> None:
        task_dir = make_task_dir(tmp_path)
        server = make_server(tmp_path, is_verifying_golden_patch=True)

        def solve(sandbox: FakeSandbox) -> int:
            assert (
                sandbox.files["/solution/solve.sh"].startswith(b"#!/bin/bash")
                and "/solution/solve.py" in sandbox.files
            )
            sandbox.files["/app/out.step"] = b"STEP"
            return 0

        agent = FakeSandbox(name="agent", on_run_solution=solve)
        verifier = verifier_with_tests(on_run_tests=write_reward("1\n"))
        created = wire_sandboxes(server, agent, verifier)
        body = TerminalBench4VerifyRequest(
            **row(task_dir), responses_create_params={"input": "x"}, response=EMPTY_RESPONSE
        )
        response = await server.verify(fake_request(session_id=None), body)
        assert response.reward == 1.0 and response.evaluation_completed
        assert response.golden_patch_exit_code == 0 and "solved" in response.golden_patch_output
        assert [c["role"] for c in created] == ["agent", "verifier"]
        solution_run = next(c for c in agent.calls if c["command"].startswith(": ng-tb4-run-solution;"))
        assert solution_run["timeout_s"] == 3600.0 and "bash /solution/solve.sh" in solution_run["command"]
        assert verifier.files["/app/out.step"] == b"STEP" and agent.stopped and verifier.stopped


class TestLeakOnLocalFailure:
    @pytest.mark.asyncio
    async def test_temp_dir_failure_stops_agent_sandbox_and_raises(self, tmp_path: Path, monkeypatch) -> None:
        """Observed live 2026-09-09: node-local /tmp filled up, TemporaryDirectory raised OSError(28) and the agent
        sandbox leaked because the directory was created outside the try block."""
        import tempfile as _tempfile

        task_dir = make_task_dir(tmp_path)
        server = make_server(tmp_path)
        agent = FakeSandbox(name="agent", files={"/app/out.step": b"x"})
        verifier = verifier_with_tests()
        created = wire_sandboxes(server, agent, verifier)

        def boom(*args, **kwargs):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(_tempfile, "TemporaryDirectory", boom)
        with pytest.raises(OSError, match="No space left"):
            await seed_and_verify(server, task_dir)
        assert agent.stopped and [c["role"] for c in created] == ["agent"]


class TestResourceRequests:
    def test_requests_are_clamped_to_limits(self) -> None:
        limits = SandboxResources(cpu=2.0, memory_mib=4096, disk_gib=10)
        options = clamp_resource_requests(
            {"resource_requests": {"cpu": 0.5, "memory_mib": 512, "disk_gib": 30}, "other": {"x": 1}}, limits
        )
        assert options["resource_requests"] == {"cpu": 0.5, "memory_mib": 512, "disk_gib": 10}
        assert options["other"] == {"x": 1}
        assert clamp_resource_requests({}, limits) == {}
        untouched = clamp_resource_requests({"resource_requests": {"disk_gib": 30}}, SandboxResources())
        assert untouched["resource_requests"] == {"disk_gib": 30}


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def now(self) -> float:
        return self.t


def write_task_toml(task_dir: Path, text: str) -> None:
    (task_dir / "task.toml").write_text(text)


class TestReviewFindings:
    """Regression tests for the 2026-09-09 adversarial review (workflow wf_a9da1516-1d2)."""

    @pytest.mark.asyncio
    async def test_empty_reward_file_is_data_not_infrastructure(self, tmp_path: Path) -> None:
        task_dir = make_task_dir(tmp_path)
        server = make_server(tmp_path)
        agent = FakeSandbox(name="agent", files={"/app/out.step": b"x"})
        verifier = verifier_with_tests(on_run_tests=write_reward(""))
        wire_sandboxes(server, agent, verifier)
        response = await seed_and_verify(server, task_dir)
        assert response.reward == 0.0 and not response.evaluation_completed
        assert "is empty" in response.failure_reason and response.reward_source is None and verifier.stopped

    @pytest.mark.asyncio
    async def test_empty_reward_json_beside_valid_txt_is_not_scored(self, tmp_path: Path) -> None:
        task_dir = make_task_dir(tmp_path)
        server = make_server(tmp_path)
        agent = FakeSandbox(name="agent", files={"/app/out.step": b"x"})
        verifier = verifier_with_tests(on_run_tests=write_reward("", as_json=True, also_txt="1\n"))
        wire_sandboxes(server, agent, verifier)
        response = await seed_and_verify(server, task_dir)
        assert not response.evaluation_completed and "reward.json is empty" in response.failure_reason

    @pytest.mark.asyncio
    async def test_server_enforced_timeout_with_prewritten_reward_is_not_scored(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        clock = FakeClock()
        monkeypatch.setattr(app_module, "monotonic", clock.now)
        task_dir = make_task_dir(tmp_path)
        server = make_server(tmp_path)

        def killed_after_budget(sandbox: FakeSandbox) -> int:
            sandbox.files["/logs/verifier/reward.txt"] = b"0\n"
            clock.t += 240.0  # execd enforced the 240 s budget and returned an ordinary exit code
            return 137

        agent = FakeSandbox(name="agent", files={"/app/out.step": b"x"})
        verifier = verifier_with_tests(on_run_tests=killed_after_budget)
        wire_sandboxes(server, agent, verifier)
        response = await seed_and_verify(server, task_dir)
        assert not response.evaluation_completed and response.reward == 0.0 and response.reward_source is None
        assert "budget" in response.failure_reason and response.verifier_sandbox_observation.outcome == "timeout"
        assert response.verifier_exit_code == 137 and response.verifier_wall_time_s >= 240.0

    @pytest.mark.asyncio
    async def test_run_tests_error_type_is_not_scored(self, tmp_path: Path) -> None:
        task_dir = make_task_dir(tmp_path)
        server = make_server(tmp_path)

        class SandboxErrorFake(FakeSandbox):
            async def exec(self, command, **kwargs):
                if command.startswith(": ng-tb4-run-tests;"):
                    self.files["/logs/verifier/reward.txt"] = b"1\n"
                    return SandboxExecResult(stdout="", stderr="execd died", return_code=125, error_type="sandbox")
                return await super().exec(command, **kwargs)

        agent = FakeSandbox(name="agent", files={"/app/out.step": b"x"})
        verifier = SandboxErrorFake(name="verifier", files={"/tests/test.sh": b"#!/bin/bash\n"})
        wire_sandboxes(server, agent, verifier)
        response = await seed_and_verify(server, task_dir)
        assert not response.evaluation_completed and response.reward_source is None
        assert (
            response.verifier_sandbox_observation.outcome == "sandbox_error"
            and "execd died" in response.verifier_stderr_tail
        )

    @pytest.mark.asyncio
    async def test_non_timeout_exception_from_run_tests_is_infrastructure(self, tmp_path: Path) -> None:
        task_dir = make_task_dir(tmp_path)
        server = make_server(tmp_path)

        def backend_dead(sandbox: FakeSandbox) -> int:
            raise ConnectionError("Sandbox backend unreachable")

        agent = FakeSandbox(name="agent", files={"/app/out.step": b"x"})
        verifier = verifier_with_tests(on_run_tests=backend_dead)
        wire_sandboxes(server, agent, verifier)
        with pytest.raises(ConnectionError):
            await seed_and_verify(server, task_dir)
        assert agent.stopped and verifier.stopped

    @pytest.mark.asyncio
    async def test_reward_probe_failure_is_infrastructure(self, tmp_path: Path) -> None:
        task_dir = make_task_dir(tmp_path)
        server = make_server(tmp_path)
        agent = FakeSandbox(name="agent", files={"/app/out.step": b"x"})
        verifier = verifier_with_tests(on_run_tests=write_reward("1\n"), fail_steps={"probe-reward": 2})
        wire_sandboxes(server, agent, verifier)
        with pytest.raises(RuntimeError, match="probing /logs/verifier failed"):
            await seed_and_verify(server, task_dir)
        assert verifier.stopped

    @pytest.mark.asyncio
    async def test_collect_hook_failure_is_recorded_and_grading_continues(self, tmp_path: Path) -> None:
        task_dir = make_task_dir(tmp_path, extra_toml='[[verifier.collect]]\ncommand = "echo main"\n')
        server = make_server(tmp_path)
        agent = FakeSandbox(name="agent", files={"/app/out.step": b"x"}, fail_steps={"collect-hook": 3})
        verifier = verifier_with_tests(on_run_tests=write_reward("1\n"))
        wire_sandboxes(server, agent, verifier)
        response = await seed_and_verify(server, task_dir)
        assert response.evaluation_completed and response.reward == 1.0
        assert response.collect_hook_results == [
            {
                "service": "main",
                "command": "echo main",
                "status": "failed",
                "return_code": 3,
                "error_type": None,
                "output_tail": "forced failure of collect-hook",
            }
        ]

    @pytest.mark.asyncio
    async def test_tar_exit_1_is_tolerated_and_recorded(self, tmp_path: Path) -> None:
        task_dir = make_task_dir(tmp_path)
        server = make_server(tmp_path)
        agent = FakeSandbox(
            name="agent", files={"/app/out.step": b"x"}, pack_rc=1, pack_warning="tar: app: file changed as we read it"
        )
        verifier = verifier_with_tests(on_run_tests=write_reward("1\n"))
        wire_sandboxes(server, agent, verifier)
        response = await seed_and_verify(server, task_dir)
        assert response.evaluation_completed and response.reward == 1.0
        assert response.artifact_tar_return_code == 1 and "file changed" in response.artifact_tar_warnings
        manifest = json.loads((Path(response.log_dir) / "artifact_manifest.json").read_text())
        assert manifest["tar_return_code"] == 1 and verifier.files["/app/out.step"] == b"x"

    @pytest.mark.asyncio
    async def test_excludes_are_applied_when_packing(self, tmp_path: Path) -> None:
        task_dir = make_task_dir(
            tmp_path,
            artifacts='[{source = "/app/evalbench/", exclude = ["./notes.txt", "__pycache__", "*.pyc", "cache/deep"]}]',
        )
        server = make_server(tmp_path)
        agent = FakeSandbox(
            name="agent",
            files={
                "/app/evalbench/a.py": b"1",
                "/app/evalbench/notes.txt": b"n",
                "/app/evalbench/__pycache__/a.cpython-311.pyc": b"p",
                "/app/evalbench/sub/b.pyc": b"p",
                "/app/evalbench/cache/deep/x": b"c",
                "/app/evalbench/cache/keep": b"k",
            },
        )
        verifier = verifier_with_tests(on_run_tests=write_reward("1\n"))
        wire_sandboxes(server, agent, verifier)
        response = await seed_and_verify(server, task_dir)
        assert response.evaluation_completed
        assert response.artifact_excludes == [
            "app/evalbench/notes.txt",
            "__pycache__",
            "*.pyc",
            "app/evalbench/cache/deep",
        ]
        assert agent.excludes_seen == response.artifact_excludes
        shipped = {p for p in verifier.files if p.startswith("/app/evalbench/")}
        assert shipped == {"/app/evalbench/a.py", "/app/evalbench/cache/keep"}

    def test_root_and_parent_artifacts_are_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="filesystem root"):
            parse_artifact("/")
        with pytest.raises(ValueError, match="'..'"):
            parse_artifact("/app/../etc")
        task_dir = make_task_dir(tmp_path, artifacts='["/"]')
        with pytest.raises(ValueError, match="filesystem root"):
            load_task(task_dir)

    @pytest.mark.asyncio
    async def test_row_extras_named_like_response_fields_do_not_break_the_response(self, tmp_path: Path) -> None:
        task_dir = make_task_dir(tmp_path)
        server = make_server(tmp_path)
        agent = FakeSandbox(name="agent", files={"/app/out.step": b"x"})
        verifier = verifier_with_tests(on_run_tests=write_reward("1\n"))
        wire_sandboxes(server, agent, verifier)
        request = fake_request()
        extras = {"reward": 0.25, "log_dir": "/stale", "test_output": "stale"}
        await server.seed_session(request, TerminalBench4SeedSessionRequest(**row(task_dir), **extras))
        body = TerminalBench4VerifyRequest(
            **row(task_dir), **extras, responses_create_params={"input": "x"}, response=EMPTY_RESPONSE
        )
        response = await server.verify(request, body)
        assert response.reward == 1.0 and response.test_output != "stale" and response.log_dir != "/stale"

    @pytest.mark.asyncio
    async def test_bad_task_folder_at_verify_stops_the_seeded_sandbox(self, tmp_path: Path) -> None:
        task_dir = make_task_dir(tmp_path)
        server = make_server(tmp_path)
        agent = FakeSandbox(name="agent", files={"/app/out.step": b"x"})
        wire_sandboxes(server, agent, verifier_with_tests())
        request = fake_request()
        await server.seed_session(request, TerminalBench4SeedSessionRequest(**row(task_dir)))
        (task_dir / "task.toml").write_text("this is = not [ toml")
        body = TerminalBench4VerifyRequest(
            **row(task_dir), responses_create_params={"input": "x"}, response=EMPTY_RESPONSE
        )
        with pytest.raises(Exception):
            await server.verify(request, body)
        assert agent.stopped and request.session[SESSION_ID_KEY] not in server._sessions

    @pytest.mark.asyncio
    async def test_reseed_stops_the_previous_sandbox_of_the_same_session(self, tmp_path: Path) -> None:
        task_dir = make_task_dir(tmp_path)
        server = make_server(tmp_path)
        first = FakeSandbox(name="first")
        second = FakeSandbox(name="second")
        created = []

        async def fake_create(*, image, resources, role, task):
            created.append(role)
            return first if len(created) == 1 else second

        server._create_sandbox = fake_create  # type: ignore[method-assign]
        request = fake_request()
        await server.seed_session(request, TerminalBench4SeedSessionRequest(**row(task_dir)))
        await server.seed_session(request, TerminalBench4SeedSessionRequest(**row(task_dir)))
        assert first.stopped and not second.stopped and server._sessions["sess-1"].sandbox is second

    @pytest.mark.asyncio
    async def test_golden_mode_reuses_a_seeded_sandbox(self, tmp_path: Path) -> None:
        task_dir = make_task_dir(tmp_path)
        server = make_server(tmp_path, is_verifying_golden_patch=True)

        def solve(sandbox: FakeSandbox) -> int:
            sandbox.files["/app/out.step"] = b"STEP"
            return 0

        agent = FakeSandbox(name="agent", on_run_solution=solve)
        verifier = verifier_with_tests(on_run_tests=write_reward("1\n"))
        created = wire_sandboxes(server, agent, verifier)
        response = await seed_and_verify(server, task_dir)
        assert response.reward == 1.0 and [c["role"] for c in created] == ["agent", "verifier"]
        solution_run = next(c for c in agent.calls if c["command"].startswith(": ng-tb4-run-solution;"))
        assert solution_run["env"] == {"DEBIAN_FRONTEND": "noninteractive"}

    @pytest.mark.asyncio
    async def test_solution_env_from_task_toml_reaches_the_oracle(self, tmp_path: Path) -> None:
        task_dir = make_task_dir(tmp_path, extra_toml='[solution.env]\nFOO = "bar"\n')
        server = make_server(tmp_path, is_verifying_golden_patch=True)
        agent = FakeSandbox(name="agent")
        verifier = verifier_with_tests(on_run_tests=write_reward("0\n"))
        wire_sandboxes(server, agent, verifier)
        await seed_and_verify(server, task_dir)
        solution_run = next(c for c in agent.calls if c["command"].startswith(": ng-tb4-run-solution;"))
        assert solution_run["env"] == {"DEBIAN_FRONTEND": "noninteractive", "FOO": "bar"}

    @pytest.mark.asyncio
    async def test_oracle_on_non_root_image_is_a_typed_error(self, tmp_path: Path) -> None:
        from resources_servers.terminal_bench_4.app import OracleUnsupportedError

        task_dir = make_task_dir(tmp_path)
        server = make_server(tmp_path, is_verifying_golden_patch=True)
        agent = FakeSandbox(name="agent", fail_steps={"solution-mkdir": 1})
        wire_sandboxes(server, agent, verifier_with_tests())
        body = TerminalBench4VerifyRequest(
            **row(task_dir), responses_create_params={"input": "x"}, response=EMPTY_RESPONSE
        )
        with pytest.raises(OracleUnsupportedError, match="non-root agent images"):
            await server.verify(fake_request(session_id=None), body)
        assert agent.stopped

    @pytest.mark.asyncio
    async def test_agent_stop_failure_does_not_affect_grading(self, tmp_path: Path) -> None:
        task_dir = make_task_dir(tmp_path)
        server = make_server(tmp_path)
        agent = FakeSandbox(name="agent", files={"/app/out.step": b"x"}, stop_raises=True)
        verifier = verifier_with_tests(on_run_tests=write_reward("1\n"))
        wire_sandboxes(server, agent, verifier)
        response = await seed_and_verify(server, task_dir)
        assert response.reward == 1.0 and response.evaluation_completed
        assert response.agent_sandbox_stopped is False and verifier.stopped

    @pytest.mark.asyncio
    async def test_verifier_user_and_env_reach_only_the_test_run(self, tmp_path: Path) -> None:
        task_dir = make_task_dir(tmp_path)
        write_task_toml(
            task_dir,
            (task_dir / "task.toml")
            .read_text()
            .replace(
                "[verifier]\ntimeout_sec = 240.0",
                '[verifier]\ntimeout_sec = 240.0\nuser = "grader"\nenv = { FOO = "bar" }',
            ),
        )
        server = make_server(tmp_path)
        agent = FakeSandbox(name="agent", files={"/app/out.step": b"x"})
        verifier = verifier_with_tests(on_run_tests=write_reward("1\n"))
        wire_sandboxes(server, agent, verifier)
        response = await seed_and_verify(server, task_dir)
        assert response.evaluation_completed
        run_tests = [c for c in verifier.calls if c["command"].startswith(": ng-tb4-run-tests;")]
        others = [c for c in verifier.calls if not c["command"].startswith(": ng-tb4-run-tests;")]
        assert run_tests[0]["user"] == "grader" and run_tests[0]["env"] == {"FOO": "bar"}
        assert all(c["user"] is None and c["env"] is None for c in others)

    @pytest.mark.asyncio
    async def test_truncated_tarball_download_raises_before_the_verifier_starts(self, tmp_path: Path) -> None:
        task_dir = make_task_dir(tmp_path)
        server = make_server(tmp_path)

        class Truncating(FakeSandbox):
            async def download(self, remote_path, local_path) -> None:
                Path(local_path).parent.mkdir(parents=True, exist_ok=True)
                Path(local_path).write_bytes(self.files[remote_path][:-1])

        agent = Truncating(name="agent", files={"/app/out.step": b"x" * 50})
        created = wire_sandboxes(server, agent, verifier_with_tests())
        with pytest.raises(RuntimeError, match=r"downloaded tarball is \d+ bytes, sandbox reported \d+"):
            await seed_and_verify(server, task_dir)
        assert agent.stopped and [c["role"] for c in created] == ["agent"]

    def test_shipped_config_requests_never_exceed_task_limits(self) -> None:
        import yaml

        config_path = Path(__file__).resolve().parents[1] / "configs" / "terminal_bench_4.yaml"
        server_cfg = yaml.safe_load(config_path.read_text())["terminal_bench_4_resources_server"]["resources_servers"][
            "terminal_bench_4"
        ]
        sandbox_config = server_cfg["sandbox_config"]
        for storage_mb in (4096, 10240, 51200):
            limits = derive_resources(
                {"cpus": 1, "memory_mb": 2048, "storage_mb": storage_mb},
                base=dict(sandbox_config["resources"]),
                use_task_resources=True,
                cpu_multiplier=server_cfg["cpu_multiplier"],
                memory_multiplier=server_cfg["memory_multiplier"],
                min_cpu=server_cfg["min_cpu"],
                min_memory_mib=server_cfg["min_memory_mib"],
                min_disk_gib=server_cfg["min_disk_gib"],
            )
            requests = clamp_resource_requests(dict(sandbox_config["provider_options"]), limits)["resource_requests"]
            assert requests["cpu"] <= limits.cpu and requests["memory_mib"] <= limits.memory_mib
            assert requests["disk_gib"] <= limits.disk_gib

    def test_verifier_sandbox_does_not_get_cpu_cap_env_by_default(self, tmp_path: Path) -> None:
        server = make_server(tmp_path)
        assert server.config.verifier_derive_cpu_env is False


class TestPrepare:
    def test_unset_tasks_dir_raises_and_leaves_existing_output_intact(self, tmp_path: Path, monkeypatch) -> None:
        from benchmarks.terminal_bench_4.prepare import prepare

        monkeypatch.delenv("TB4_TASKS_DIR", raising=False)
        output = tmp_path / "benchmark.jsonl"
        output.write_text("keep me\n")
        with pytest.raises(FileNotFoundError, match="TB4_TASKS_DIR"):
            prepare(output_path=str(output))
        assert output.read_text() == "keep me\n"

    def test_prepare_builds_rows_and_skips_compose(self, tmp_path: Path) -> None:
        from benchmarks.terminal_bench_4.prepare import prepare

        tasks_root = tmp_path / "wl"
        make_task_dir(tmp_path / "a", name="terminal-bench/alpha")
        make_task_dir(tmp_path / "b", name="terminal-bench/beta", compose=True)
        tasks_root.mkdir()
        (tmp_path / "a" / "tasks" / "alpha").rename(tasks_root / "alpha")
        (tmp_path / "b" / "tasks" / "beta").rename(tasks_root / "beta")
        output = tmp_path / "out.jsonl"
        result = prepare(tasks_dir=str(tasks_root), output_path=str(output), release_tag="v9")
        rows = [json.loads(line) for line in result.read_text().splitlines()]
        assert [r["task_name"] for r in rows] == ["terminal-bench/alpha"]
        assert rows[0]["docker_image"] == "harborframework/terminal-bench:alpha-environment-v9"
        assert rows[0]["verifier_docker_image"] == "harborframework/terminal-bench:alpha-verifier-v9"
        assert rows[0]["task_folder"] == str((tasks_root / "alpha").resolve())
        assert rows[0]["agent_timeout_sec"] == 28800.0
        assert not (tmp_path / "out.jsonl.tmp").exists()
