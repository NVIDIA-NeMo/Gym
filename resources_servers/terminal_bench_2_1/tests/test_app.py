# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from pydantic import TypeAdapter

import resources_servers.terminal_bench_2_1.app as terminal_bench_app
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.terminal_bench_2_1.app import (
    _BULLSEYE_SECURITY_SNAPSHOT_SETUP,
    TerminalBench21ResourcesServer,
    TerminalBench21ResourcesServerConfig,
    TerminalBench21RunRequest,
    TerminalBench21SeedSessionRequest,
    TerminalBench21SessionVerifyRequest,
    TerminalBench21VerifyRequest,
)


class TestApp:
    def test_sanity(self) -> None:
        config = TerminalBench21ResourcesServerConfig(
            sandbox_provider="",
            sandbox_config=dict(),
            host="",
            port=0,
            entrypoint="",
            name="",
        )
        TerminalBench21ResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    async def test_create_sandbox_uses_start_with_setup(self, monkeypatch, tmp_path: Path) -> None:
        sandbox = AsyncMock()
        sandbox.start_with_setup = AsyncMock(return_value=sandbox)
        monkeypatch.setattr(terminal_bench_app, "AsyncSandbox", lambda _provider: sandbox)
        monkeypatch.setattr(terminal_bench_app, "get_global_config_dict", lambda: {})
        monkeypatch.setattr(terminal_bench_app, "resolve_provider_config", lambda *_: MagicMock())
        monkeypatch.setattr(terminal_bench_app, "resolve_provider_metadata", lambda *_: {})
        server = TerminalBench21ResourcesServer(
            config=TerminalBench21ResourcesServerConfig(
                sandbox_provider="test",
                sandbox_config={},
                evaluation_timeout=30,
                host="",
                port=0,
                entrypoint="",
                name="terminal_bench_2_1_resources_server",
            ),
            server_client=MagicMock(spec=ServerClient),
        )

        result = await server._create_sandbox(
            TerminalBench21SeedSessionRequest(
                task_name="terminal-bench/test-task",
                docker_image="terminal-bench/test-task:latest",
                task_folder=str(tmp_path),
            )
        )

        assert result is sandbox
        sandbox.start_with_setup.assert_awaited_once()
        spec, setup = sandbox.start_with_setup.call_args.args
        assert spec is not None
        sandbox.exec.return_value = MagicMock(return_code=0)
        await setup(sandbox)
        assert sandbox.exec.await_count == 2
        assert "/etc/apt/sources.list" in sandbox.exec.await_args_list[0].args[0]
        assert sandbox.exec.await_args_list[1].args[0] == "apt-get update"
        sandbox.exec.reset_mock()
        sandbox.exec.return_value = MagicMock(return_code=1)
        with pytest.raises(RuntimeError, match="Failed to prepare TerminalBench package sources"):
            await setup(sandbox)
        sandbox.exec.assert_awaited_once()

    async def test_create_sandbox_setup_failure_propagates(self, monkeypatch, tmp_path: Path) -> None:
        sandbox = AsyncMock()
        sandbox.start_with_setup = AsyncMock(side_effect=RuntimeError("setup command failed"))
        monkeypatch.setattr(terminal_bench_app, "AsyncSandbox", lambda _provider: sandbox)
        monkeypatch.setattr(terminal_bench_app, "get_global_config_dict", lambda: {})
        monkeypatch.setattr(terminal_bench_app, "resolve_provider_config", lambda *_: MagicMock())
        monkeypatch.setattr(terminal_bench_app, "resolve_provider_metadata", lambda *_: {})
        server = TerminalBench21ResourcesServer(
            config=TerminalBench21ResourcesServerConfig(
                sandbox_provider="test",
                sandbox_config={},
                evaluation_timeout=30,
                host="",
                port=0,
                entrypoint="",
                name="terminal_bench_2_1_resources_server",
            ),
            server_client=MagicMock(spec=ServerClient),
        )

        with pytest.raises(RuntimeError, match="setup command failed"):
            await server._create_sandbox(
                TerminalBench21SeedSessionRequest(
                    task_name="terminal-bench/test-task",
                    docker_image="terminal-bench/test-task:latest",
                    task_folder=str(tmp_path),
                )
            )


@pytest.mark.parametrize("distro", ["debian:bullseye", "debian:bookworm", "ubuntu:noble"])
def test_security_snapshot_preserves_other_repositories(tmp_path: Path, distro: str) -> None:
    distro_id, codename = distro.split(":")
    os_release = tmp_path / "os-release"
    os_release.write_text(f"ID={distro_id}\nVERSION_CODENAME={codename}\n")
    sources = tmp_path / "sources.list"
    live = "deb http://deb.debian.org/debian-security bullseye-security main"
    unrelated = "# " + live + "\ndeb http://deb.debian.org/debian bullseye main\n"
    sources.write_text(unrelated + live + "\n")
    command = ["bash", "-c", _BULLSEYE_SECURITY_SNAPSHOT_SETUP, "--", str(os_release), str(sources)]
    subprocess.run(command, check=True)
    expected = live
    if distro == "debian:bullseye":
        expected = (
            "deb [check-valid-until=no] "
            "https://snapshot.debian.org/archive/debian-security/20260831T235959Z/ bullseye-security main"
        )
    assert sources.read_text() == unrelated + expected + "\n"

    subprocess.run(command, check=True)
    assert sources.read_text() == unrelated + expected + "\n"


@pytest.fixture
def session_server(monkeypatch, tmp_path):
    server = TerminalBench21ResourcesServer(
        config=TerminalBench21ResourcesServerConfig(
            sandbox_provider="sandbox",
            sandbox_config={},
            host="",
            port=0,
            entrypoint="",
            name="tb2",
            session_records_dir=tmp_path / "records",
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    sandbox = SimpleNamespace(
        _handle=SimpleNamespace(sandbox_id="original-sandbox"),
        serialize=AsyncMock(return_value={"sandbox_id": "original-sandbox", "workdir": "/app"}),
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0, stdout="tests passed", stderr="")),
        stop=AsyncMock(),
    )

    async def download(remote_path, local_path):
        assert remote_path == "/logs/verifier/reward.txt"
        Path(local_path).write_text("1\n")

    sandbox.download = AsyncMock(side_effect=download)
    create = AsyncMock(return_value=sandbox)
    upload = AsyncMock()
    monkeypatch.setattr(TerminalBench21ResourcesServer, "_create_sandbox", create)
    monkeypatch.setattr(TerminalBench21ResourcesServer, "_upload_folder", upload)
    monkeypatch.setattr(terminal_bench_app, "get_global_config_dict", lambda: {"sandbox": {"local": {}}})
    request = SimpleNamespace(session={SESSION_ID_KEY: "cookie-session"})
    task = dict(task_name="synthetic", docker_image="original-image", task_folder=str(tmp_path))
    return SimpleNamespace(
        server=server, sandbox=sandbox, create=create, upload=upload, request=request, task=task, tmp_path=tmp_path
    )


def _response():
    return dict(
        id="resp_test",
        created_at=0,
        model="model",
        object="response",
        output=[],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


@pytest.mark.parametrize("content", ["Write the answer", [{"type": "input_text", "text": "Write the answer"}]])
@pytest.mark.parametrize("workflow", ["legacy", "session", "session_timeout"])
async def test_seed_and_verify_both_contracts_use_original_sandbox(session_server, content, workflow):
    f = session_server
    params = {"input": [{"role": "user", "content": content}]}
    seed = await f.server.seed_session(
        f.request, TerminalBench21RunRequest(**f.task, responses_create_params=params, agent_timeout_sec=60)
    )
    assert seed.sandbox_handle == seed.sandbox_descriptor["sandbox_id"] == "original-sandbox"
    assert seed.session_id == "cookie-session"
    assert seed.instruction == "Write the answer"
    assert seed.sandbox_provider == {"local": {}}
    assert seed.agent_timeout_sec == 60
    payload = dict(
        responses_create_params=params,
        response=dict(
            id="resp_test",
            created_at=0,
            model="model",
            object="response",
            output=[],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        ),
    )
    if workflow == "legacy":
        payload.update(f.task)
    else:
        payload.update(
            session_id=seed.session_id,
            termination={"reason": "timeout" if workflow == "session_timeout" else "completed"},
            agent_started=True,
            harness_metadata={"harness_version": "2.4.6", "reward": -999, "mask_sample": True},
        )
    body = TypeAdapter(TerminalBench21VerifyRequest | TerminalBench21SessionVerifyRequest).validate_python(payload)
    result = await f.server.verify(f.request, body)
    assert result.reward == 1 and result.evaluation_completed and not result.mask_sample
    assert result.task_name == "synthetic"
    assert f.upload.await_args.args == (
        f.sandbox,
        Path(f.task["task_folder"]) / "tests",
        "/tests",
        terminal_bench_app.TEST_SH_PATCHES,
        "synthetic",
    )
    f.sandbox.exec.assert_awaited_once_with("bash /tests/test.sh", timeout_s=None, env=None)
    f.create.assert_awaited_once()  # Verification must use the seeded sandbox, never create another one.
    f.sandbox.stop.assert_awaited_once()
    assert not f.server._session_id_to_sandbox and not f.server._session_id_to_task
    if workflow != "legacy":
        assert result.termination == payload["termination"]
        assert result.agent_started and result.harness_version == "2.4.6"


async def test_seed_accepts_existing_task_only_request(session_server):
    f = session_server
    seed = await f.server.seed_session(f.request, TerminalBench21RunRequest(**f.task))
    assert seed.sandbox_handle == "original-sandbox" and seed.instruction == ""
    assert seed.agent_timeout_sec == 28800


@pytest.mark.parametrize("session_id, status", [("other-session", 409), ("cookie-session", 404)])
async def test_session_verify_requires_matching_cookie_and_seeded_task(session_server, session_id, status):
    f = session_server
    body = TerminalBench21SessionVerifyRequest(
        session_id=session_id,
        responses_create_params={"input": []},
        response=dict(
            id="resp_test",
            created_at=0,
            model="model",
            object="response",
            output=[],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        ),
        termination={"reason": "completed"},
    )
    with pytest.raises(HTTPException) as error:
        await f.server.verify(f.request, body)
    assert error.value.status_code == status
    f.sandbox.exec.assert_not_awaited()
    f.sandbox.stop.assert_not_awaited()


async def test_session_verify_records_and_graded_outcome(session_server):
    f = session_server
    params = {"input": [{"role": "user", "content": "Write the answer"}]}
    seed = await f.server.seed_session(
        f.request,
        TerminalBench21RunRequest(
            **f.task, responses_create_params=params, agent_timeout_sec=60, rollout_id="set/task/h-00"
        ),
    )
    record = json.loads((f.tmp_path / "records" / "cookie-session.json").read_text())
    assert record["phase"] == "open" and record["request"]["rollout_id"] == "set/task/h-00"
    assert record["sandbox_id"] == "original-sandbox" and "responses_create_params" not in record["request"]
    body = TerminalBench21SessionVerifyRequest(
        session_id=seed.session_id,
        responses_create_params=params,
        response=_response(),
        termination={"reason": "nonzero_exit", "detail": "ContextWindowExceeded"},
        agent_started=True,
    )
    result = await f.server.verify(f.request, body)
    assert result.reward == 1 and result.evaluation_completed and not result.mask_sample
    assert result.failure_kind is None and result.infrastructure_error is None
    record = json.loads((f.tmp_path / "records" / "cookie-session.json").read_text())
    assert record["phase"] == "closed" and record["termination"]["detail"] == "ContextWindowExceeded"
    assert record["verified_response"]["reward"] == 1 and record["verified_response"]["evaluation_completed"]


@pytest.mark.parametrize(
    "termination, agent_started, kind",
    [
        ({"reason": "infrastructure_error", "detail": "Sandbox lacks setsid"}, False, "agent_run_error"),
        ({"reason": "infrastructure_error", "detail": "runner crashed"}, True, "agent_run_error"),
        ({"reason": "cancelled"}, True, "cancelled"),
        ({"reason": "timeout", "detail": "setup"}, False, "agent_run_error"),
    ],
)
async def test_agent_failures_release_the_sandbox_without_grading(session_server, termination, agent_started, kind):
    f = session_server
    params = {"input": [{"role": "user", "content": "Write the answer"}]}
    seed = await f.server.seed_session(f.request, TerminalBench21RunRequest(**f.task, responses_create_params=params))
    body = TerminalBench21SessionVerifyRequest(
        session_id=seed.session_id,
        responses_create_params=params,
        response=_response(),
        termination=termination,
        agent_started=agent_started,
    )
    result = await f.server.verify(f.request, body)
    assert result.reward == 0 and not result.evaluation_completed and result.mask_sample
    assert result.failure_kind == kind
    assert result.infrastructure_error == (termination.get("detail") or termination["reason"])
    assert result.failure_reason == result.infrastructure_error
    f.upload.assert_not_awaited()  # the tests were never uploaded ...
    f.sandbox.exec.assert_not_awaited()  # ... nor run
    f.sandbox.stop.assert_awaited_once()  # but the sandbox is released
    assert not f.server._session_id_to_sandbox and not f.server._session_id_to_task
    record = json.loads((f.tmp_path / "records" / "cookie-session.json").read_text())
    assert record["phase"] == "closed" and record["verified_response"]["infrastructure_error"]


async def test_row_verifier_timeout_and_env_reach_the_test_run(session_server):
    f = session_server
    params = {"input": [{"role": "user", "content": "Write the answer"}]}
    seed = await f.server.seed_session(
        f.request,
        TerminalBench21RunRequest(
            **f.task,
            responses_create_params=params,
            verifier_timeout_sec=3600,
            verifier_env={"VERIFIER_WALL_SEC": "3600"},
            agent_user="cam",
        ),
    )
    assert seed.user == "cam"
    body = TerminalBench21SessionVerifyRequest(
        session_id=seed.session_id,
        responses_create_params=params,
        response=_response(),
        termination={"reason": "completed"},
        agent_started=True,
    )
    result = await f.server.verify(f.request, body)
    assert result.reward == 1
    f.sandbox.exec.assert_awaited_once_with("bash /tests/test.sh", timeout_s=3600, env={"VERIFIER_WALL_SEC": "3600"})


async def test_missing_reward_is_masked_as_a_verifier_error(session_server):
    f = session_server
    f.sandbox.download = AsyncMock(side_effect=FileNotFoundError("no reward"))
    params = {"input": [{"role": "user", "content": "Write the answer"}]}
    seed = await f.server.seed_session(f.request, TerminalBench21RunRequest(**f.task, responses_create_params=params))
    body = TerminalBench21SessionVerifyRequest(
        session_id=seed.session_id,
        responses_create_params=params,
        response=_response(),
        termination={"reason": "completed"},
        agent_started=True,
    )
    result = await f.server.verify(f.request, body)
    assert result.reward == 0 and not result.evaluation_completed and result.mask_sample
    assert result.failure_kind == "verifier_error" and result.infrastructure_error is None
    assert result.failure_reason.startswith("MissingOfficialReward")
    f.sandbox.exec.assert_awaited_once()
    f.sandbox.stop.assert_awaited_once()


async def test_create_sandbox_can_skip_apt_preparation(monkeypatch, tmp_path: Path) -> None:
    sandbox = AsyncMock()
    monkeypatch.setattr(terminal_bench_app, "AsyncSandbox", lambda _provider: sandbox)
    monkeypatch.setattr(terminal_bench_app, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(terminal_bench_app, "resolve_provider_config", lambda *_: MagicMock())
    monkeypatch.setattr(terminal_bench_app, "resolve_provider_metadata", lambda *_: {})
    server = TerminalBench21ResourcesServer(
        config=TerminalBench21ResourcesServerConfig(
            sandbox_provider="test",
            sandbox_config={"provider_options": {"network_policy": {"defaultAction": "deny", "egress": []}}},
            prepare_apt_sources=False,
            host="",
            port=0,
            entrypoint="",
            name="tb2",
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    result = await server._create_sandbox(
        TerminalBench21SeedSessionRequest(task_name="t", docker_image="img@sha256:abc", task_folder=str(tmp_path))
    )
    assert result is sandbox
    sandbox.start.assert_awaited_once()
    sandbox.start_with_setup.assert_not_awaited()
    spec = sandbox.start.call_args.args[0]
    assert spec.provider_options["network_policy"] == {"defaultAction": "deny", "egress": []}
    assert spec.image == "img@sha256:abc"


async def test_upload_folder_ships_one_archive(tmp_path: Path) -> None:
    import io
    import tarfile

    server = TerminalBench21ResourcesServer(
        config=TerminalBench21ResourcesServerConfig(
            sandbox_provider="sandbox", sandbox_config={}, host="", port=0, entrypoint="", name="tb2"
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    tests = tmp_path / "tests"
    (tests / "hidden" / "arena-01").mkdir(parents=True)
    (tests / "test.sh").write_text("#!/bin/bash\necho -w torch==2.7.1\n")
    (tests / "test.sh").chmod(0o755)
    (tests / "hidden" / "arena-01" / "case.json").write_text("{}")
    commands = []
    uploads = {}

    async def exec_(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    async def upload(local_path, remote_path):
        uploads[remote_path] = Path(local_path).read_bytes()

    sandbox = SimpleNamespace(exec=exec_, upload=upload)
    await server._upload_folder(
        sandbox, tests, "/tests", terminal_bench_app.TEST_SH_PATCHES, "terminal-bench/pytorch-model-recovery"
    )
    assert commands[0] == 'mkdir -p "/tests"'
    assert commands[1].startswith('tar -xzf "/tests.upload.tar.gz" -C "/tests"') and len(commands) == 2
    with tarfile.open(fileobj=io.BytesIO(uploads["/tests.upload.tar.gz"]), mode="r:gz") as tar:
        names = sorted(tar.getnames())
        assert names == ["hidden/arena-01/case.json", "test.sh"]
        member = tar.getmember("test.sh")
        assert member.mode & 0o111  # execute bit preserved
        content = tar.extractfile(member).read().decode()
    assert "--index https://download.pytorch.org/whl/cpu" in content  # the task patch was applied inside the archive
