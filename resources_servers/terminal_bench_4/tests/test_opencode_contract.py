# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The resources-side adaptation to the unmodified OpenCode sandboxed agent (harness: opencode)."""

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.terminal_bench_4 import app as app_module
from resources_servers.terminal_bench_4 import opencode
from resources_servers.terminal_bench_4.app import TerminalBench4Config, TerminalBench4ResourcesServer
from resources_servers.terminal_bench_4.models import SandboxedVerifyRequest, TerminalBench4RunRequest
from resources_servers.terminal_bench_4.task import content_hash
from resources_servers.terminal_bench_4.tests.test_environment import environment_config, make_environment
from resources_servers.terminal_bench_4.tests.test_task import package


INSTRUCTION = "Do the task.\n"
TEACHER = "\n\n## Additional instructions\n\nBe careful.\n"
RESPONSE = {
    "id": "resp_1",
    "created_at": 1,
    "model": "policy",
    "object": "response",
    "output": [
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "done", "annotations": []}],
        }
    ],
    "tool_choice": "auto",
    "tools": [],
    "parallel_tool_calls": True,
    "usage": {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    },
}


class FakeRequest:
    def __init__(self, session=None):
        self.session = {SESSION_ID_KEY: uuid4().hex} if session is None else session
        self.cookies = {}


def config(tmp_path, manifest, **overrides):
    return TerminalBench4Config(
        host="localhost",
        port=1,
        name="tb4",
        entrypoint="app.py",
        manifest_path=manifest,
        artifacts_dir=tmp_path / "results",
        environment=environment_config(),
        local_task_packages=True,
        **overrides,
    )


def local_manifest(tmp_path):
    source = package(tmp_path / "source")
    ref = "sha256:" + content_hash(source)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "gym-tb4-local-v1",
                "ref": "sha256:" + "d" * 64,
                "tasks": [{"name": "terminal-bench/test", "ref": ref, "path": str(source)}],
            }
        )
    )
    return manifest, ref


def row(ref, prompt=INSTRUCTION + TEACHER, **extra):
    return {
        "task_name": "terminal-bench/test",
        "task_ref": ref,
        "dataset_ref": "sha256:" + "d" * 64,
        "rollout_id": "terminal-bench/test/attempt-0",
        "task_id": "terminal-bench/test",
        "agent_user": "task-user",
        "responses_create_params": {"input": [{"role": "user", "content": prompt}]},
        **extra,
    }


def make_server(tmp_path, manifest, monkeypatch, **overrides):
    server = TerminalBench4ResourcesServer(
        config=config(tmp_path, manifest, harness="opencode", **overrides),
        server_client=MagicMock(spec=ServerClient),
    )
    box = MagicMock()
    box._handle = SimpleNamespace(sandbox_id="owned-box")
    box.serialize = AsyncMock(return_value={"sandbox_id": "owned-box"})
    box.exec = AsyncMock(return_value=SimpleNamespace(return_code=0, stdout="", stderr=""))
    box.upload = AsyncMock()
    prepared = []

    async def fake_prepare(session, loader):
        prepared.append(session)
        assert session.task is not None, "the seed check must preload the package"
        session.result = {"runtime": "gym-tb4-native"}
        session.environment = SimpleNamespace(
            main=box,
            role_user="task-user",
            bootstrap_uid=0,
            provider_config={"opensandbox": {}},
            closed=False,
            session_id=session.session_id,
            resources=[],
            cleanup_errors=[],
            resource_identities=lambda: [{"service": "main", "sandbox_id": "owned-box"}],
        )

    staged = []

    async def fake_stage(sandbox, cfg, **kwargs):
        staged.append((sandbox, cfg, kwargs))
        return {"stage_dir": cfg.stage_dir, "agent_user": kwargs["agent_user"]}

    monkeypatch.setattr(app_module.lifecycle, "prepare_session", fake_prepare)
    monkeypatch.setattr(app_module.opencode_harness, "stage_launcher", fake_stage)
    return server, box, prepared, staged


async def finish(server):
    for session in server._sessions.values():
        if session.expiry_task is not None:
            session.expiry_task.cancel()
    await asyncio.sleep(0)


# --- configuration -------------------------------------------------------------------------------------------------


def test_opencode_harness_has_no_oracle_mode(tmp_path):
    manifest, _ = local_manifest(tmp_path)
    with pytest.raises(ValidationError, match="oracle"):
        config(tmp_path, manifest, harness="opencode", execution_mode="oracle")
    assert config(tmp_path, manifest, harness="miniswe", execution_mode="oracle").execution_mode == "oracle"


@pytest.mark.parametrize("gateway", ["10.0.0.1:1", "http://10.0.0.1/v1", "http://h?x=1", "http://h:1/ 'q'"])
def test_gateway_must_be_a_plain_origin(gateway):
    with pytest.raises(ValidationError):
        opencode.OpenCodeHarnessConfig(model_gateway=gateway)


def test_gateway_origin_normalizes_and_paths_are_absolute():
    cfg = opencode.OpenCodeHarnessConfig(model_gateway="http://10.109.22.242:24401/")
    assert cfg.model_gateway == "http://10.109.22.242:24401"
    for bad in ("relative/dir", "/a/../b", "/trailing/"):
        with pytest.raises(ValidationError):
            opencode.OpenCodeHarnessConfig(stage_dir=bad)


# --- launcher rendering ---------------------------------------------------------------------------------------------


def test_scripts_render_stage_dir_and_env_is_quoted():
    cfg = opencode.OpenCodeHarnessConfig(stage_dir="/tmp/stage", model_gateway="http://gw:1")
    scripts = opencode.render_scripts(cfg)
    assert scripts["install.sh"].startswith("#!/usr/bin/env bash") and 'stage="/tmp/stage"' in scripts["install.sh"]
    launcher = scripts["opencode-launcher"]
    assert launcher.startswith("#!/bin/sh") and 'TB4_STAGE_DIR="/tmp/stage"' in launcher
    assert "__PY_REWRITE__" not in launcher and "__RECORD_DIR__" not in launcher
    assert "setpriv --reuid" in launcher and "setsid --wait" in launcher and "$TB4_PIDS" in launcher
    assert "-run-exit.json" in launcher and "XDG_DATA_HOME=" in launcher and 'cd "$TB4_WORKDIR" || exit 95' in launcher
    # Every privilege switch names a groups policy: util-linux setpriv rejects a bare --reuid.
    for line in launcher.splitlines():
        if "setpriv --reuid" in line:
            assert '"$groups_flag"' in line, line
    assert "-f /etc/alpine-release" not in scripts["install.sh"] and "ld-musl-" in scripts["install.sh"]
    env = opencode.launcher_env(
        session_id="tb4-1", agent_user="cam", gateway=cfg.model_gateway, workdir="/home/cam/job"
    )
    assert (
        env == "TB4_SESSION_ID=tb4-1\nTB4_AGENT_USER=cam\nTB4_MODEL_GATEWAY=http://gw:1\nTB4_WORKDIR=/home/cam/job\n"
    )
    assert "TB4_AGENT_USER=1000\n" in opencode.launcher_env(session_id="s", agent_user=1000, gateway=None)
    assert "TB4_AGENT_USER=''\n" in opencode.launcher_env(session_id="s", agent_user=None, gateway=None)
    with pytest.raises(ValueError, match="cannot carry"):
        opencode.launcher_env(session_id="s", agent_user="a b", gateway=None)


def test_gateway_rewrite_changes_only_the_origin():
    config_json = json.dumps(
        {
            "provider": {
                "nemo_gym": {"options": {"baseURL": "http://127.0.0.1:24503/ng-rollout/r.1/training-token-capture/v1"}}
            }
        }
    )
    rewritten = json.loads(opencode.rewrite_gateway(config_json, "http://10.109.22.242:24401"))
    assert (
        rewritten["provider"]["nemo_gym"]["options"]["baseURL"]
        == "http://10.109.22.242:24401/ng-rollout/r.1/training-token-capture/v1"
    )


# --- fail-closed checks ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "params,ok",
    [
        ({"input": [{"role": "user", "content": INSTRUCTION + TEACHER}]}, True),
        ({"input": [{"role": "user", "content": [{"type": "input_text", "text": INSTRUCTION}]}]}, True),
        ({"input": [{"role": "user", "content": TEACHER}]}, False),
        ({"input": [{"role": "user", "content": INSTRUCTION}, {"role": "user", "content": "again"}]}, False),
        ({"input": [{"role": "system", "content": INSTRUCTION}]}, False),
        (
            {
                "input": [
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": "a"}, {"type": "input_text", "text": "b"}],
                    }
                ]
            },
            False,
        ),
        ({"input": []}, False),
    ],
)
def test_prompt_check_mirrors_the_agent(params, ok):
    if ok:
        opencode.check_prompt(params, INSTRUCTION, require_instruction=True)
    else:
        with pytest.raises(ValueError):
            opencode.check_prompt(params, INSTRUCTION, require_instruction=True)


def test_prompt_check_can_skip_instruction_containment():
    opencode.check_prompt(
        {"input": [{"role": "user", "content": "free prompt"}]}, INSTRUCTION, require_instruction=False
    )


@pytest.mark.parametrize(
    "user,uid,resolved", [("cam", 1000, None), ("cam", 1000, 1001), ("cam", None, None), (1001, 1000, None)]
)
def test_identity_gate_fails_closed(user, uid, resolved):
    with pytest.raises(RuntimeError):
        opencode.check_identity(agent_user=user, bootstrap_uid=uid, resolved_uid=resolved)


@pytest.mark.parametrize(
    "user,uid,resolved",
    [
        (None, 1000, None),
        ("root", 1000, None),
        (0, 1000, None),
        ("cam", 0, None),
        (1000, 0, None),
        (1000, 1000, None),
        ("agent", 1000, 1000),
    ],
)
def test_identity_gate_accepts(user, uid, resolved):
    opencode.check_identity(agent_user=user, bootstrap_uid=uid, resolved_uid=resolved)


async def test_quiesce_agent_user_only_for_a_distinct_root_started_identity():
    box = MagicMock()
    box.exec = AsyncMock(return_value=SimpleNamespace(return_code=0, stdout="", stderr=""))
    await opencode.quiesce_agent_user(box, "cam", 0)
    assert box.exec.await_args.kwargs["user"] == "cam" and "kill -TERM -- -1" in box.exec.await_args.args[0]
    box.exec.reset_mock()
    for user, uid in ((None, 0), ("root", 0), ("agent", 1000)):
        await opencode.quiesce_agent_user(box, user, uid)
    box.exec.assert_not_awaited()


async def test_stage_launcher_uploads_and_locks_down_as_root(tmp_path):
    box = MagicMock()
    box.exec = AsyncMock(return_value=SimpleNamespace(return_code=0, stdout="", stderr=""))
    uploads = {}

    async def upload(local, remote):
        uploads[remote] = Path(local).read_text()

    box.upload = AsyncMock(side_effect=upload)
    cfg = opencode.OpenCodeHarnessConfig(stage_dir="/tmp/stage", model_gateway="http://gw:1")
    record = await opencode.stage_launcher(
        box, cfg, session_id="tb4-1", agent_user="cam", bootstrap_uid=0, scratch=tmp_path / "scratch"
    )
    assert set(uploads) == {"/tmp/stage/install.sh", "/tmp/stage/opencode-launcher", "/tmp/stage/launcher.env"}
    assert (
        uploads["/tmp/stage/launcher.env"]
        == "TB4_SESSION_ID=tb4-1\nTB4_AGENT_USER=cam\nTB4_MODEL_GATEWAY=http://gw:1\nTB4_WORKDIR=''\n"
    )
    first, last = box.exec.await_args_list[0].args[0], box.exec.await_args_list[-1].args[0]
    assert "test ! -L /tmp/stage" in first and "mkdir -p /tmp/stage/bin /tmp/stage/records" in first
    assert "chmod 0700 /tmp/stage/records" in first
    assert (
        "chmod 0755 /tmp/stage/install.sh /tmp/stage/opencode-launcher" in last and "chown -R 0:0 /tmp/stage" in last
    )
    assert record == {
        "stage_dir": "/tmp/stage",
        "install_script": "/tmp/stage/install.sh",
        "agent_user": "cam",
        "bootstrap_uid": 0,
        "resolved_uid": None,
        "workdir": None,
        "model_gateway": "http://gw:1",
    }
    # A non-root-started sandbox keeps its own ownership; a named task user is fine when it IS the default user.
    box.exec.reset_mock()
    await opencode.stage_launcher(
        box, cfg, session_id="tb4-2", agent_user=None, bootstrap_uid=1000, scratch=tmp_path / "s2"
    )
    assert "chown" not in box.exec.await_args_list[-1].args[0]
    box.exec = AsyncMock(return_value=SimpleNamespace(return_code=0, stdout="1000\n", stderr=""))
    same = await opencode.stage_launcher(
        box, cfg, session_id="tb4-4", agent_user="agent", bootstrap_uid=1000, scratch=tmp_path / "s4"
    )
    assert same["resolved_uid"] == 1000 and box.exec.await_args_list[0].args[0] == "id -u -- agent"
    box.exec = AsyncMock(return_value=SimpleNamespace(return_code=0, stdout="1001\n", stderr=""))
    with pytest.raises(RuntimeError, match="root-started"):
        await opencode.stage_launcher(
            box, cfg, session_id="tb4-3", agent_user="cam", bootstrap_uid=1000, scratch=tmp_path
        )


# --- network ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("verifier", [False, True])
def test_agent_egress_allow_applies_to_the_agent_role_only(tmp_path, monkeypatch, verifier):
    env, *_ = make_environment(
        tmp_path,
        monkeypatch,
        verifier=verifier,
        config={"agent_egress_allow": ["10.109.22.242"]},
        task_config={
            "environment": {"network_mode": "no-network"},
            "verifier": {"environment": {"docker_image": "public/verifier", "network_mode": "no-network"}},
        },
    )
    policy = env.build_spec().provider_options["network_policy"]
    expected = [] if verifier else [{"action": "allow", "target": "10.109.22.242"}]
    assert policy == {"defaultAction": "deny", "egress": expected}


def test_public_tasks_keep_the_provider_default(tmp_path, monkeypatch):
    env, *_ = make_environment(tmp_path, monkeypatch, config={"agent_egress_allow": ["10.109.22.242"]})
    assert "network_policy" not in env.build_spec().provider_options


# --- seed ------------------------------------------------------------------------------------------------------------


async def test_seed_requires_rollout_id_for_both_harnesses(tmp_path, monkeypatch):
    manifest, ref = local_manifest(tmp_path)
    for harness in ("miniswe", "opencode"):
        server = TerminalBench4ResourcesServer(
            config=config(tmp_path, manifest, harness=harness), server_client=MagicMock(spec=ServerClient)
        )
        body = TerminalBench4RunRequest.model_validate({k: v for k, v in row(ref).items() if k != "rollout_id"})
        with pytest.raises(HTTPException) as error:
            await server.seed_session(FakeRequest(), body)
        assert error.value.status_code == 422 and "rollout_id" in error.value.detail


async def test_opencode_seed_returns_handle_and_a_cookieless_retry_joins_the_episode(tmp_path, monkeypatch):
    manifest, ref = local_manifest(tmp_path)
    server, box, prepared, staged = make_server(
        tmp_path, manifest, monkeypatch, opencode={"model_gateway": "http://gw:1"}
    )
    first = FakeRequest()
    seed = await server.seed_session(first, TerminalBench4RunRequest.model_validate(row(ref)))
    assert seed.sandbox_handle == "owned-box" and seed.user == "task-user" and seed.termination is None
    session = server._sessions[seed.session_id]
    assert session.request.rollout_id == "terminal-bench/test/attempt-0"
    expected_owner = hashlib.sha256(f"sha256:{'d' * 64}:{ref}:terminal-bench/test/attempt-0".encode()).hexdigest()
    assert first.session["tb4_client_session_id"] == expected_owner
    assert session.owner == hashlib.sha256(expected_owner.encode()).hexdigest()
    assert session.result["harness"] == "opencode" and session.result["opencode_launcher"]["agent_user"] == "task-user"
    assert staged[0][2] == {
        "session_id": seed.session_id,
        "agent_user": "task-user",
        "bootstrap_uid": 0,
        "scratch": session.directory / "sandbox" / "opencode",
        "workdir": None,
    }
    assert "agent_ready_at" in session.deadlines
    # The agent's HTTP client retries a lost first request without any resources cookie: same episode, no second box.
    again = await server.seed_session(FakeRequest(), TerminalBench4RunRequest.model_validate(row(ref)))
    assert again.session_id == seed.session_id and len(prepared) == 1 and len(server._sessions) == 1
    # A different attempt of the same task is a different episode.
    other = await server.seed_session(
        FakeRequest(), TerminalBench4RunRequest.model_validate(row(ref, rollout_id="terminal-bench/test/attempt-1"))
    )
    assert other.session_id != seed.session_id and len(prepared) == 2
    await finish(server)


async def test_opencode_seed_rejects_bad_rows_before_provisioning(tmp_path, monkeypatch):
    manifest, ref = local_manifest(tmp_path)
    server, box, prepared, staged = make_server(tmp_path, manifest, monkeypatch)
    with pytest.raises(HTTPException) as error:
        await server.seed_session(FakeRequest(), TerminalBench4RunRequest.model_validate(row(ref, prompt=TEACHER)))
    assert error.value.status_code == 422 and "instruction" in error.value.detail
    assert prepared == [] and server._sessions == {}
    relaxed, *_ = make_server(tmp_path, manifest, monkeypatch, opencode={"require_instruction_in_prompt": False})
    seed = await relaxed.seed_session(FakeRequest(), TerminalBench4RunRequest.model_validate(row(ref, prompt="free")))
    assert seed.sandbox_handle == "owned-box"
    await finish(relaxed)


async def test_opencode_seed_fails_closed_on_provisioning_errors_and_replays(tmp_path, monkeypatch):
    manifest, ref = local_manifest(tmp_path)
    server, box, prepared, staged = make_server(tmp_path, manifest, monkeypatch)

    async def broken(session, loader):
        raise RuntimeError("image pull failed")

    monkeypatch.setattr(app_module.lifecycle, "prepare_session", broken)
    monkeypatch.setattr(app_module.lifecycle, "cleanup", AsyncMock())
    with pytest.raises(HTTPException) as error:
        await server.seed_session(FakeRequest(), TerminalBench4RunRequest.model_validate(row(ref)))
    assert error.value.status_code == 500 and "image pull failed" in error.value.detail
    # The mini-SWE contract still receives the termination in a 200 seed response.
    miniswe = TerminalBench4ResourcesServer(
        config=config(tmp_path / "m", manifest, harness="miniswe"), server_client=MagicMock(spec=ServerClient)
    )
    seed = await miniswe.seed_session(
        FakeRequest(), TerminalBench4RunRequest.model_validate(row(ref, client_session_id="c"))
    )
    assert seed.termination is not None and seed.termination.reason == "infrastructure_error"


# --- verify ----------------------------------------------------------------------------------------------------------


RECORDS_OK = {
    "100-7-run.json": {"subcommand": "run", "uid": "1000"},
    "100-7-run-exit.json": {"exit_code": 0, "wall_s": 12},
}


async def run_episode(tmp_path, monkeypatch, verified_reward=1.0, records=RECORDS_OK):
    manifest, ref = local_manifest(tmp_path)
    server, box, prepared, staged = make_server(tmp_path, manifest, monkeypatch)
    request = FakeRequest()
    seed = await server.seed_session(request, TerminalBench4RunRequest.model_validate(row(ref)))
    finalized = []

    async def fake_finalize(session, *, grade):
        finalized.append(grade)
        if grade:
            session.result["verifier_result"] = {"rewards": {"reward": verified_reward}}
        session.phase = "closed"

    async def fake_observe(sandbox, cfg):
        assert sandbox is box
        return records

    monkeypatch.setattr(app_module.lifecycle, "finalize_session", fake_finalize)
    monkeypatch.setattr(app_module.opencode_harness, "observe_launch", fake_observe)
    payload = row(ref) | {"response": RESPONSE}
    return server, seed, request, payload, finalized


async def test_opencode_verify_binds_the_session_and_synthesizes_agent_fields(tmp_path, monkeypatch):
    server, seed, request, payload, finalized = await run_episode(tmp_path, monkeypatch)
    body = SandboxedVerifyRequest.model_validate(payload)
    assert (
        body.session_id is None and body.termination is None and body.model_extra["task_name"] == "terminal-bench/test"
    )
    # The agent presents the seed cookie, i.e. the same resources session (owner) and no session_id.
    response = await server.verify(request, body)
    assert finalized == [True]
    assert response.session_id == seed.session_id and response.reward == 1.0 and response.evaluation_completed
    assert response.termination.reason == "completed" and response.termination.exit_code == 0
    assert response.infrastructure_error is None and response.model_extra["task_name"] == "terminal-bench/test"
    assert (
        response.model_extra["rollout_id"] == "terminal-bench/test/attempt-0"
        and response.task_id == "terminal-bench/test"
    )
    assert response.responses_create_params.input[0].role == "user" and response.response.output[0].type == "message"
    session = server._sessions[seed.session_id]
    assert session.verify_body.agent_started is True
    assert session.verify_body.harness_metadata == {
        "harness": "opencode",
        "contract": "sandbox_handle",
        "output_items": 1,
        "usage": RESPONSE["usage"],
        "launch_records": RECORDS_OK,
    }
    assert session.verify_body.agent_timings["agent_execution"]["started_at"] == session.deadlines["agent_ready_at"]
    # An identical retry (the agent's client re-posting) replays the same result instead of a 409.
    assert await server.verify(request, SandboxedVerifyRequest.model_validate(payload)) == response
    # A different payload on the same session is still a conflict.
    with pytest.raises(HTTPException) as error:
        await server.verify(request, SandboxedVerifyRequest.model_validate(payload | {"agent_user": "other"}))
    assert error.value.status_code == 409
    await finish(server)


async def test_opencode_verify_without_a_run_record_is_an_unmarked_infrastructure_error(tmp_path, monkeypatch):
    server, seed, request, payload, finalized = await run_episode(tmp_path, monkeypatch, records={})
    response = await server.verify(request, SandboxedVerifyRequest.model_validate(payload))
    assert finalized == [False]  # never graded
    assert response.termination.reason == "infrastructure_error" and "never launched" in response.termination.detail
    assert response.reward == 0.0 and not response.evaluation_completed
    assert response.failure_reason and response.model_extra["_ng_failure_class"] == "infrastructure_error"
    await finish(server)


async def test_opencode_verify_without_an_exit_record_is_a_graded_timeout(tmp_path, monkeypatch):
    records = {"100-7-run.json": {"subcommand": "run"}}
    server, seed, request, payload, finalized = await run_episode(tmp_path, monkeypatch, records=records)
    response = await server.verify(request, SandboxedVerifyRequest.model_validate(payload))
    assert finalized == [True] and response.termination.reason == "timeout" and response.reward == 1.0
    await finish(server)


async def test_opencode_verify_records_a_nonzero_exit(tmp_path, monkeypatch):
    records = {"100-7-run.json": {}, "100-7-run-exit.json": {"exit_code": 3, "wall_s": 5}}
    server, seed, request, payload, finalized = await run_episode(tmp_path, monkeypatch, records=records)
    response = await server.verify(request, SandboxedVerifyRequest.model_validate(payload))
    assert (
        finalized == [True] and response.termination.reason == "nonzero_exit" and response.termination.exit_code == 3
    )
    await finish(server)


async def test_opencode_verify_unreadable_records_is_an_infrastructure_error_that_still_grades(tmp_path, monkeypatch):
    server, seed, request, payload, finalized = await run_episode(tmp_path, monkeypatch)

    async def broken(sandbox, cfg):
        raise RuntimeError("sandbox gone")

    monkeypatch.setattr(app_module.opencode_harness, "observe_launch", broken)
    response = await server.verify(request, SandboxedVerifyRequest.model_validate(payload))
    assert finalized == [True] and response.termination.reason == "infrastructure_error"
    assert "sandbox gone" in response.failure_reason
    await finish(server)


def test_derive_termination_uses_the_latest_run_record():
    records = {
        "90-3-run.json": {},
        "90-3-run-exit.json": {"exit_code": 0},
        "100-7-run.json": {},
        "100-7-run-exit.json": {"exit_code": 7},
        "100-8-session.json": {},
    }
    termination, started = opencode.derive_termination(records)
    assert (termination.reason, termination.exit_code, started) == ("nonzero_exit", 7, True)
    termination, started = opencode.derive_termination({"100-7-run.json": {}, "100-7-run-exit.json": {"raw": "x"}})
    assert (termination.reason, started) == ("timeout", True)
    termination, started = opencode.derive_termination({"100-8-session.json": {}})
    assert (termination.reason, started) == ("infrastructure_error", False)


async def test_observe_launch_parses_one_record_per_line():
    box = MagicMock()
    stdout = '100-7-run.json\t{"subcommand": "run"}\n100-7-run-exit.json\t{"exit_code": 0}\nbroken.json\tnot json\n'
    box.exec = AsyncMock(return_value=SimpleNamespace(return_code=0, stdout=stdout, stderr=""))
    records = await opencode.observe_launch(box, opencode.OpenCodeHarnessConfig())
    assert records == {
        "100-7-run.json": {"subcommand": "run"},
        "100-7-run-exit.json": {"exit_code": 0},
        "broken.json": {"raw": "not json"},
    }
    assert "/tmp/tb4-opencode/records" in box.exec.await_args.args[0]
    box.exec = AsyncMock(return_value=SimpleNamespace(return_code=1, stdout="", stderr="denied"))
    with pytest.raises(RuntimeError, match="denied"):
        await opencode.observe_launch(box, opencode.OpenCodeHarnessConfig())


async def test_opencode_verify_needs_the_seed_cookie_and_rollout_id(tmp_path, monkeypatch):
    server, seed, request, payload, finalized = await run_episode(tmp_path, monkeypatch)
    with pytest.raises(HTTPException) as error:
        await server.verify(FakeRequest(), SandboxedVerifyRequest.model_validate(payload))
    assert error.value.status_code == 404
    with pytest.raises(HTTPException) as error:
        await server.verify(
            request, SandboxedVerifyRequest.model_validate({k: v for k, v in payload.items() if k != "rollout_id"})
        )
    assert error.value.status_code == 422
    assert finalized == []
    await finish(server)


async def test_miniswe_verify_still_requires_session_and_termination(tmp_path):
    manifest, ref = local_manifest(tmp_path)
    server = TerminalBench4ResourcesServer(
        config=config(tmp_path, manifest, harness="miniswe"), server_client=MagicMock(spec=ServerClient)
    )
    with pytest.raises(HTTPException) as error:
        await server.verify(FakeRequest(), SandboxedVerifyRequest.model_validate(row(ref) | {"response": RESPONSE}))
    assert error.value.status_code == 422


def test_row_extras_never_override_resources_owned_response_fields(tmp_path, monkeypatch):
    manifest, ref = local_manifest(tmp_path)
    server = TerminalBench4ResourcesServer(
        config=config(tmp_path, manifest, harness="opencode"), server_client=MagicMock(spec=ServerClient)
    )
    body = TerminalBench4RunRequest.model_validate(row(ref))
    session = server._new_session("identity", "owner", body, "tb4-x")
    session.result = {"verifier_result": {"rewards": {"reward": 1.0}}}
    session.termination = app_module.AgentTermination(reason="completed")
    session.verify_body = SandboxedVerifyRequest.model_validate(
        row(ref, reward=0.0, artifacts="bogus", session_id="tb4-other")
        | {"response": RESPONSE, "termination": {"reason": "completed"}, "agent_started": True}
    )
    response = server._verified_response(session, {})
    assert (
        response.reward == 1.0
        and response.session_id == "tb4-x"
        and response.artifacts == {"trial": str(session.directory)}
    )
