# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from resources_servers.terminal_bench_4 import verifier as verifier_module
from resources_servers.terminal_bench_4.environment import execution_user
from resources_servers.terminal_bench_4.tests.test_environment import make_environment


@pytest.mark.parametrize(
    "value,expected", [(None, None), ("root", "root"), ("agent", "agent"), ("0", 0), ("1000", 1000), (1001, 1001)]
)
def test_task_users_normalize_without_losing_image_default_or_uid_zero(value, expected):
    assert execution_user(value) == expected


@pytest.mark.parametrize("value", ["", "agent:staff", "1000:2000", "-1", -1, 2**32, True, "a b", "--root"])
def test_invalid_or_group_qualified_users_fail_closed(value):
    with pytest.raises(ValueError, match="account name or unsigned UID"):
        execution_user(value)


@pytest.mark.parametrize("verifier", [False, True])
@pytest.mark.parametrize("user", [None, "declared-user", "root", 0, 1002, "1002"])
async def test_task_identity_and_current_image_default_are_independent(tmp_path, monkeypatch, verifier, user):
    env, box, *_ = make_environment(
        tmp_path,
        monkeypatch,
        verifier=verifier,
        task_config={"verifier" if verifier else "agent": {"user": user}},
    )
    assert env.role_user == execution_user(user)
    env.main = box
    await env.exec("image-default command")
    assert box.exec.await_args.kwargs["user"] is None
    await env.exec("task command", user=user)
    assert box.exec.await_args.kwargs["user"] == execution_user(user)
    await env.exec("trusted staging", user="root")
    assert box.exec.await_args.kwargs["user"] == "root"
    if not verifier:
        await env.agent_workdir()
        assert box.exec.await_args.kwargs["user"] == env.role_user
        await env.quiesce_agent("session")
        assert box.exec.await_args.kwargs["user"] == env.role_user


def test_removed_original_user_option_is_rejected_not_silently_ignored(tmp_path, monkeypatch):
    with pytest.raises(ValidationError, match="root_bootstrap_image_users"):
        make_environment(tmp_path, monkeypatch, config={"root_bootstrap_image_users": {"public/agent": "agent"}})


@pytest.mark.parametrize("verifier", [False, True])
def test_startup_metadata_does_not_override_preprocessed_identity(tmp_path, monkeypatch, verifier):
    images = tmp_path / "images.json"
    images.write_text(
        json.dumps(
            {
                image: {
                    "image": image,
                    "os": "linux",
                    "architecture": "amd64",
                    "config": {"Entrypoint": ["/start-service"], "Cmd": [image], "User": "metadata-user"},
                }
                for image in ("public/agent", "public/verifier")
            }
        )
    )
    env, *_ = make_environment(
        tmp_path,
        monkeypatch,
        verifier=verifier,
        config={"compose_image_configs": images},
        task_config={"agent": {"user": "preprocessed-agent"}, "verifier": {"user": "root"}},
    )
    spec = env.build_spec()
    assert spec.image == ("public/verifier" if verifier else "public/agent")
    assert spec.entrypoint == ["/start-service", "sh", "-c", "sleep infinity"]
    assert env.role_user == ("root" if verifier else "preprocessed-agent")


@pytest.mark.parametrize("verifier", [False, True])
@pytest.mark.parametrize(
    "user,uid,gid", [(None, 0, 0), ("root", 0, 0), (0, 0, 0), ("agent", 1000, 1001), (1002, 1002, 1003)]
)
async def test_root_start_prepares_only_harness_logs_for_task_role(tmp_path, monkeypatch, verifier, user, uid, gid):
    env, box, *_ = make_environment(
        tmp_path,
        monkeypatch,
        verifier=verifier,
        task_config={"agent": {"user": user}, "verifier": {"user": user}},
    )
    box.exec.side_effect = [
        SimpleNamespace(return_code=0, stdout="0\n", stderr=""),
        SimpleNamespace(return_code=0, stdout="", stderr=""),
        SimpleNamespace(return_code=0, stdout=f"{uid}\n{gid}\n" + (f"{uid}\n" if user == "agent" else ""), stderr=""),
        SimpleNamespace(return_code=0, stdout="", stderr=""),
    ]
    await env.start()
    calls = box.exec.await_args_list
    assert calls[0].args == ("id -u",) and "user" not in calls[0].kwargs
    assert calls[1].kwargs["user"] is None  # Already proved image-default root.
    assert calls[2].kwargs["user"] == user
    assert calls[3].kwargs["user"] == "root"
    setup = calls[3].args[0]
    assert f"chown {uid}:{gid} /logs /logs/agent /logs/verifier /logs/artifacts" in setup
    assert "test ! -L" in setup
    assert "chown -R" not in setup and "/home/" not in setup and "/app" not in setup
    record = json.loads((env.directory / "sandbox/session.identity.json").read_text())
    assert record["bootstrap_uid"] == 0 and record["execution_user"] == user
    assert record["execution_uid"] == uid and record["execution_gid"] == gid
    assert "original_image_user" not in record


async def test_nonroot_image_retains_setup_without_root_escalation(tmp_path, monkeypatch):
    env, box, *_ = make_environment(tmp_path, monkeypatch, task_config={"agent": {"user": None}})
    box.exec.side_effect = [
        SimpleNamespace(return_code=0, stdout="1000\n", stderr=""),
        SimpleNamespace(return_code=0, stdout="", stderr=""),
    ]
    await env.start()
    assert env.bootstrap_uid == 1000 and env.role_user is None
    assert box.exec.await_count == 2
    assert all(call.kwargs.get("user") is None for call in box.exec.await_args_list)
    assert not (env.directory / "sandbox/session.identity.json").exists()


@pytest.mark.parametrize("return_code,stdout", [(1, "0\n"), (0, ""), (0, "-1"), (0, str(2**32)), (0, "not a uid")])
async def test_unreadable_default_identity_fails_before_log_setup(tmp_path, monkeypatch, return_code, stdout):
    env, box, *_ = make_environment(tmp_path, monkeypatch)
    box.exec.return_value = SimpleNamespace(return_code=return_code, stdout=stdout, stderr="probe failure")
    with pytest.raises(RuntimeError, match="Unable to determine sandbox default execution UID"):
        await env.start()
    box.exec.assert_awaited_once_with("id -u", timeout_s=30)


@pytest.mark.parametrize("return_code,stdout", [(1, ""), (0, "0\n0\n1000\n")])
async def test_failed_or_ignored_role_switch_never_falls_back_to_root(tmp_path, monkeypatch, return_code, stdout):
    env, box, *_ = make_environment(tmp_path, monkeypatch)
    box.exec.side_effect = [
        SimpleNamespace(return_code=0, stdout="0\n", stderr=""),
        SimpleNamespace(return_code=0, stdout="", stderr=""),
        SimpleNamespace(return_code=return_code, stdout=stdout, stderr="user switch failure"),
    ]
    with pytest.raises(RuntimeError, match="Unable to execute as agent user"):
        await env.start()
    assert box.exec.await_count == 3
    assert box.exec.await_args.kwargs["user"] == "task-user"
    assert not (env.directory / "sandbox/session.identity.json").exists()


@pytest.mark.parametrize("user", [None, "grader", "root", 0, "1002"])
async def test_verifier_setup_root_but_test_execution_uses_task_user(tmp_path, monkeypatch, user):
    env, box, *_ = make_environment(tmp_path, monkeypatch, verifier=True, task_config={"verifier": {"user": user}})
    env.main = box
    env.task.stage_tests = True
    staging = AsyncMock()
    monkeypatch.setattr(verifier_module, "stage_trusted_directory", staging)
    monkeypatch.setattr(verifier_module, "download_dir", AsyncMock())
    monkeypatch.setattr(verifier_module, "parse_reward", lambda _: {"reward": 1})
    await verifier_module.run_verifier(env, tmp_path, [])
    staging.assert_awaited_once_with(box, env.task.path / "tests", "/tests")
    assert box.exec.await_args_list[0].kwargs["user"] == "root"
    assert box.exec.await_args_list[1].kwargs["user"] == execution_user(user)
