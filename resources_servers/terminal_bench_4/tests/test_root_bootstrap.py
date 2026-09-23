# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from resources_servers.terminal_bench_4 import verifier as verifier_module
from resources_servers.terminal_bench_4.environment import execution_user
from resources_servers.terminal_bench_4.tests.test_environment import make_environment


ORIGINAL_USERS = {"public/agent": "image-agent", "public/verifier": "1200"}


@pytest.mark.parametrize(
    "value,expected", [("", "root"), ("root", "root"), ("agent", "agent"), ("0", 0), ("1000", 1000), (1001, 1001)]
)
def test_original_image_users_normalize_without_losing_uid_zero(value, expected):
    assert execution_user(value, image_default=True) == expected


@pytest.mark.parametrize("value", ["", "agent:staff", "1000:2000", "-1", -1, 2**32, True, "a b", "--root"])
def test_invalid_or_group_qualified_users_fail_closed(value):
    with pytest.raises(ValueError, match="account name or unsigned UID"):
        execution_user(value)


@pytest.mark.parametrize("verifier", [False, True])
@pytest.mark.parametrize("user", [None, "declared-user", "root", 0, 1002])
async def test_role_identity_uses_declared_user_or_its_own_original_image(tmp_path, monkeypatch, verifier, user):
    env, box, *_ = make_environment(
        tmp_path,
        monkeypatch,
        verifier=verifier,
        config={"root_bootstrap_image_users": ORIGINAL_USERS},
        task_config={"verifier" if verifier else "agent": {"user": user}},
    )
    default = 1200 if verifier else "image-agent"
    assert env.role_user == (default if user is None else user)
    env.main = box
    await env.exec("task-owned command")
    assert box.exec.await_args.kwargs["user"] == default
    await env.exec("trusted staging", user="root")
    assert box.exec.await_args.kwargs["user"] == "root"
    if not verifier:
        await env.agent_workdir()
        assert box.exec.await_args.kwargs["user"] == env.role_user
        await env.quiesce_agent("session")
        assert box.exec.await_args.kwargs["user"] == env.role_user


@pytest.mark.parametrize("verifier", [False, True])
def test_missing_original_image_user_fails_before_allocation(tmp_path, monkeypatch, verifier):
    with pytest.raises(ValueError, match="requires the original image user"):
        make_environment(tmp_path, monkeypatch, verifier=verifier, config={"root_bootstrap_image_users": {}})


def test_original_user_is_bound_to_effective_rewritten_image(tmp_path, monkeypatch):
    env, *_ = make_environment(
        tmp_path,
        monkeypatch,
        config={
            "image_rewrites": [{"from": "public/", "to": "mirror/"}],
            "root_bootstrap_image_users": {"mirror/agent": "before-rewrite"},
        },
        task_config={"agent": {"user": None}},
    )
    assert env.role_user == "before-rewrite"


@pytest.mark.parametrize("verifier", [False, True])
def test_root_bootstrap_preserves_entrypoint_fix_and_role_image(tmp_path, monkeypatch, verifier):
    images = tmp_path / "images.json"
    images.write_text(
        json.dumps(
            {
                image: {
                    "image": image,
                    "os": "linux",
                    "architecture": "amd64",
                    "config": {"Entrypoint": ["/start-service"], "Cmd": [image], "User": "root"},
                }
                for image in ORIGINAL_USERS
            }
        )
    )
    env, *_ = make_environment(
        tmp_path,
        monkeypatch,
        verifier=verifier,
        config={"root_bootstrap_image_users": ORIGINAL_USERS, "single_container_image_configs": images},
        task_config={"agent": {"user": None}},
    )
    image = "public/verifier" if verifier else "public/agent"
    spec = env.build_spec()
    assert spec.image == image
    assert spec.entrypoint == ["/start-service", image]
    assert env.role_user == (1200 if verifier else "image-agent")


def test_compose_is_rejected_instead_of_silently_changing_sidecar_users(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="single-container"):
        make_environment(tmp_path, monkeypatch, compose=True, config={"root_bootstrap_image_users": ORIGINAL_USERS})


@pytest.mark.parametrize("return_code,stdout", [(0, "1000\n"), (1, "0\n"), (0, "")])
async def test_non_root_bootstrap_is_rejected_before_any_preparation(tmp_path, monkeypatch, return_code, stdout):
    env, box, *_ = make_environment(tmp_path, monkeypatch, config={"root_bootstrap_image_users": ORIGINAL_USERS})
    box.exec.return_value = SimpleNamespace(return_code=return_code, stdout=stdout, stderr="")
    with pytest.raises(RuntimeError, match="default execution user is root"):
        await env.start()
    box.exec.assert_awaited_once_with("id -u", timeout_s=30)
    await env.stop()
    box.stop.assert_awaited_once()


@pytest.mark.parametrize("verifier", [False, True])
async def test_root_start_checks_role_and_prepares_only_harness_logs(tmp_path, monkeypatch, verifier):
    env, box, *_ = make_environment(
        tmp_path,
        monkeypatch,
        verifier=verifier,
        config={"root_bootstrap_image_users": ORIGINAL_USERS},
        task_config={"agent": {"user": None}},
    )
    expected = 1200 if verifier else "image-agent"
    box.exec.side_effect = [
        SimpleNamespace(return_code=0, stdout="0\n", stderr=""),
        SimpleNamespace(return_code=0, stdout="", stderr=""),
        SimpleNamespace(return_code=0, stdout="1200\n1300\n" + ("" if verifier else "1200\n"), stderr=""),
        SimpleNamespace(return_code=0, stdout="", stderr=""),
    ]
    await env.start()
    calls = box.exec.await_args_list
    assert calls[0].args == ("id -u",) and "user" not in calls[0].kwargs
    assert calls[1].kwargs["user"] == "root"
    assert calls[2].kwargs["user"] == expected
    assert calls[3].kwargs["user"] == "root"
    setup = calls[3].args[0]
    assert "chown 1200:1300 /logs /logs/agent /logs/verifier /logs/artifacts" in setup
    assert "test ! -L" in setup
    assert "chown -R" not in setup and "/home/" not in setup and "/app" not in setup
    record = json.loads((env.directory / "sandbox/session.identity.json").read_text())
    assert record["bootstrap_uid"] == 0
    assert record["execution_user"] == expected
    assert record["original_image_user"] == ORIGINAL_USERS["public/verifier" if verifier else "public/agent"]
    assert record["execution_uid"] == 1200 and record["execution_gid"] == 1300


async def test_role_switch_failure_never_falls_back_to_root(tmp_path, monkeypatch):
    env, box, *_ = make_environment(tmp_path, monkeypatch, config={"root_bootstrap_image_users": ORIGINAL_USERS})
    box.exec.side_effect = [
        SimpleNamespace(return_code=0, stdout="0\n", stderr=""),
        SimpleNamespace(return_code=0, stdout="", stderr=""),
        SimpleNamespace(return_code=1, stdout="", stderr="user missing"),
    ]
    with pytest.raises(RuntimeError, match="Unable to execute as agent user"):
        await env.start()
    assert box.exec.await_count == 3
    assert box.exec.await_args.kwargs["user"] == "task-user"
    assert not (env.directory / "sandbox/session.identity.json").exists()


async def test_ignored_named_identity_is_not_accepted_as_root_execution(tmp_path, monkeypatch):
    env, box, *_ = make_environment(tmp_path, monkeypatch, config={"root_bootstrap_image_users": ORIGINAL_USERS})
    box.exec.side_effect = [
        SimpleNamespace(return_code=0, stdout="0\n", stderr=""),
        SimpleNamespace(return_code=0, stdout="", stderr=""),
        SimpleNamespace(return_code=0, stdout="0\n0\n1000\n", stderr=""),
    ]
    with pytest.raises(RuntimeError, match="Unable to execute as agent user"):
        await env.start()
    assert box.exec.await_count == 3


@pytest.mark.parametrize("user,expected", [(None, 1200), ("grader", "grader"), ("root", "root"), (0, 0)])
async def test_verifier_setup_root_but_test_execution_uses_role_user(tmp_path, monkeypatch, user, expected):
    env, box, *_ = make_environment(
        tmp_path,
        monkeypatch,
        verifier=True,
        config={"root_bootstrap_image_users": ORIGINAL_USERS},
        task_config={"verifier": {"user": user}},
    )
    env.main = box
    env.task.stage_tests = True
    staging = AsyncMock()
    monkeypatch.setattr(verifier_module, "stage_trusted_directory", staging)
    monkeypatch.setattr(verifier_module, "download_dir", AsyncMock())
    monkeypatch.setattr(verifier_module, "parse_reward", lambda _: {"reward": 1})
    await verifier_module.run_verifier(env, tmp_path, [])
    staging.assert_awaited_once_with(box, env.task.path / "tests", "/tests")
    assert box.exec.await_args_list[0].kwargs["user"] == "root"
    assert box.exec.await_args_list[1].kwargs["user"] == expected
