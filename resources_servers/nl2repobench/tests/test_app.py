# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import stat
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import MonkeyPatch

import resources_servers.nl2repobench.app as app_module
from nemo_gym.config_types import ModelServerRef
from nemo_gym.core import PROVIDER_UNAVAILABLE, EpisodeId, NgFailureEvent, TaskId
from nemo_gym.sandbox.providers.base import SandboxHandle
from nemo_gym.server_runtime import (
    ResourcesCloseSessionRequest,
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    ResourcesSessionConflictError,
)
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.nl2repobench.app import (
    NL2RepoBenchResourcesServer,
    NL2RepoBenchResourcesServerConfig,
    NL2RepoBenchSeedSessionRequest,
    NL2RepoBenchVerifyRequest,
    VerifierResult,
    _bounded_output,
    _compute_reward,
    _parse_pytest_summary,
    _resolve_task,
    _resolve_task_id,
)
from resources_servers.nl2repobench.task_store import task_id, task_image


UPSTREAM_IMAGE = "ghcr.io/multimodal-art-projection/nl2repobench/example-task:1.0"


def _config(tasks_dir: Path) -> NL2RepoBenchResourcesServerConfig:
    return NL2RepoBenchResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="nl2repobench_resources_server",
        tasks_dir=tasks_dir,
        expected_task_count=1,
        sandbox_provider="test",
        sandbox_config={},
        # The shipped preset's open-egress official mode; the network tests below set their own mode.
        enforce_agent_no_network=False,
        sandbox_model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
    )


def _restricted_config(tasks_dir: Path) -> NL2RepoBenchResourcesServerConfig:
    config = _config(tasks_dir)
    config.enforce_agent_no_network = True
    return config


def _global_config(model_host: str = "model.internal", provider: str = "opensandbox") -> dict:
    return {
        "test": {provider: {}},
        "policy_model": {"responses_api_agents": {"model": {"host": model_host, "port": 8000}}},
    }


def _request() -> dict:
    return {
        "task_id": "example-task",
        "image": UPSTREAM_IMAGE,
        "verifier_metadata": {"task_id": "example-task"},
        "responses_create_params": {"input": [{"role": "user", "content": "test"}]},
        "response": {
            "output": [],
            "id": "response",
            "created_at": 0,
            "model": "test",
            "object": "response",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        },
    }


# --- pytest-summary regex parsing -------------------------------------------------------------


@pytest.mark.parametrize(
    ("summary", "expected"),
    [
        ("12 passed, 3 failed in 4.21s", (12, 3, 0)),
        ("0 passed, 1 error in 0.02s", (0, 0, 1)),
        ("5 passed in 1.10s", (5, 0, 0)),
        ("no summary line at all", (0, 0, 0)),
        ("=== 2 failed, 1 passed, 1 error in 0.5s ===", (1, 2, 1)),
    ],
)
def test_parse_pytest_summary(summary: str, expected: tuple[int, int, int]) -> None:
    assert _parse_pytest_summary(summary) == expected


def test_parse_pytest_summary_takes_last_occurrence() -> None:
    # A command before the final pytest run (e.g. pip install output) might also emit numbers
    # that look like a summary; only the last occurrence should be used.
    combined_output = "1 passed in the install step\n...\n7 passed, 2 failed in 3.00s"
    assert _parse_pytest_summary(combined_output) == (7, 2, 0)


# --- reward formula ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tests_passed", "test_case_count", "expected"),
    [
        (5, 10, 0.5),
        (10, 10, 1.0),
        (15, 10, 1.0),  # clamped to 1.0
        (0, 10, 0.0),
        (5, 0, 0.0),  # divide-by-zero guard
    ],
)
def test_compute_reward(tests_passed: int, test_case_count: int, expected: float) -> None:
    assert _compute_reward(tests_passed, test_case_count) == expected


# --- task resolution -----------------------------------------------------------------------------


def test_conflicting_task_ids_fail() -> None:
    request = _request()
    request["verifier_metadata"] = {"task_id": "different-task"}

    with pytest.raises(ValueError, match="Conflicting"):
        _resolve_task_id(NL2RepoBenchVerifyRequest.model_validate(request))


def test_request_image_matches_pinned_task_image(task_assets: Path) -> None:
    server = NL2RepoBenchResourcesServer(config=_config(task_assets), server_client=MagicMock(spec=ServerClient))

    task = _resolve_task(NL2RepoBenchVerifyRequest.model_validate(_request()), server._task_store)

    assert task_id(task) == "example-task"
    assert task_image(task) == UPSTREAM_IMAGE


def test_request_image_must_match_pinned_task_image(task_assets: Path) -> None:
    server = NL2RepoBenchResourcesServer(config=_config(task_assets), server_client=MagicMock(spec=ServerClient))
    image = "ghcr.io/multimodal-art-projection/nl2repobench/example-task:v2"

    with pytest.raises(ValueError, match="does not match the pinned image"):
        _resolve_task(NL2RepoBenchVerifyRequest.model_validate(_request() | {"image": image}), server._task_store)


# --- sandbox network policy ----------------------------------------------------------------------


def test_model_endpoint_is_the_only_added_egress_target(monkeypatch: MonkeyPatch, task_assets: Path) -> None:
    config = _restricted_config(task_assets)
    config.sandbox_model_server = ModelServerRef(type="responses_api_models", name="policy_model")
    monkeypatch.setattr("resources_servers.nl2repobench.app.get_global_config_dict", _global_config)
    config.sandbox_config = {
        "provider_options": {
            "network_policy": {"defaultAction": "deny", "egress": [{"action": "deny", "target": "example.com"}]}
        }
    }
    server = NL2RepoBenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    assert server._provider_options(phase="agent")["network_policy"] == {
        "defaultAction": "deny",
        "egress": [
            {"action": "deny", "target": "example.com"},
            {"action": "allow", "target": "model.internal"},
        ],
    }


def test_loopback_model_endpoint_is_rejected(monkeypatch: MonkeyPatch, task_assets: Path) -> None:
    config = _restricted_config(task_assets)
    config.sandbox_model_server = ModelServerRef(type="responses_api_models", name="policy_model")
    monkeypatch.setattr(
        "resources_servers.nl2repobench.app.get_global_config_dict", lambda: _global_config("127.0.0.1")
    )
    server = NL2RepoBenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    # A configuration error no retry fixes: a 422 seed refusal, not a 500 that re-dispatches the episode.
    with pytest.raises(ResourcesSeedRejectedError, match="loopback model host"):
        server._provider_options(phase="agent")


@pytest.mark.parametrize(("mode", "enforce"), [("no_internet_preinstalled", False), ("official", True)])
def test_restricted_modes_refuse_a_provider_without_network_policies(
    monkeypatch: MonkeyPatch, task_assets: Path, mode: str, enforce: bool
) -> None:
    config = _config(task_assets)
    config.agent_network_mode = mode
    config.enforce_agent_no_network = enforce
    config.sandbox_model_server = ModelServerRef(type="responses_api_models", name="policy_model")
    monkeypatch.setattr(
        "resources_servers.nl2repobench.app.get_global_config_dict", lambda: _global_config(provider="docker")
    )
    server = NL2RepoBenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    # Docker ignores network_policy, so the agent would get open egress: the seed is refused instead.
    with pytest.raises(ResourcesSeedRejectedError, match="'docker' cannot apply"):
        server._provider_options(phase="agent")
    assert server._provider_options(phase="verifier") == {}


def test_network_policy_is_scoped_to_agent_sandbox(monkeypatch: MonkeyPatch, task_assets: Path) -> None:
    config = _restricted_config(task_assets)
    config.sandbox_model_server = None
    config.sandbox_config = {"provider_options": {"network_policy": {"defaultAction": "allow", "egress": []}}}
    monkeypatch.setattr("resources_servers.nl2repobench.app.get_global_config_dict", _global_config)
    server = NL2RepoBenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    assert server._provider_options(phase="verifier") == {}
    # Deny-all with no model host would fail every model call of the run: refused at seed instead.
    with pytest.raises(ResourcesSeedRejectedError, match="needs sandbox_model_server"):
        server._provider_options(phase="agent")


def test_offline_modes_replace_an_inherited_allow_policy(monkeypatch: MonkeyPatch, task_assets: Path) -> None:
    monkeypatch.setattr("resources_servers.nl2repobench.app.get_global_config_dict", _global_config)
    inherited = {
        "defaultAction": "allow",
        "egress": [
            {"action": "allow", "target": "pypi.org"},
            {"action": "deny", "target": "example.com"},
        ],
    }
    for mode, enforce in (("no_internet_preinstalled", False), ("official", True)):
        config = _config(task_assets)
        config.agent_network_mode = mode
        config.enforce_agent_no_network = enforce
        config.sandbox_model_server = ModelServerRef(type="responses_api_models", name="policy_model")
        config.sandbox_config = {"provider_options": {"network_policy": inherited}}
        server = NL2RepoBenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        # deny-all plus the model host; the unrelated allow rule and the allow-all default are gone
        assert server._provider_options(phase="agent")["network_policy"] == {
            "defaultAction": "deny",
            "egress": [
                {"action": "deny", "target": "example.com"},
                {"action": "allow", "target": "model.internal"},
            ],
        }
        assert server._provider_options(phase="verifier") == {}


def test_official_mode_without_enforcement_keeps_the_inherited_policy_absent(task_assets: Path) -> None:
    config = _config(task_assets)
    config.enforce_agent_no_network = False
    config.sandbox_config = {"provider_options": {"network_policy": {"defaultAction": "allow", "egress": []}}}
    server = NL2RepoBenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    # the open-egress official mode drops the policy altogether, as before
    assert "network_policy" not in server._provider_options(phase="agent")


# --- seed_session / verify flow ------------------------------------------------------------------


async def test_seed_session_creates_agent_sandbox(task_assets: Path, monkeypatch: MonkeyPatch) -> None:
    server = NL2RepoBenchResourcesServer(config=_config(task_assets), server_client=MagicMock(spec=ServerClient))
    sandbox = AsyncMock()
    sandbox.serialize.return_value = {"sandbox_id": "agent-sandbox", "workdir": "/workspace"}
    sandbox.exec.return_value = MagicMock(return_code=0, stdout="", stderr="")
    monkeypatch.setattr(app_module, "get_global_config_dict", _global_config)
    monkeypatch.setattr(app_module, "resolve_provider_config", lambda *_: {"opensandbox": {}})
    monkeypatch.setattr(app_module, "resolve_provider_metadata", lambda *_: {})
    monkeypatch.setattr(app_module, "AsyncSandbox", MagicMock(return_value=sandbox))
    request = MagicMock()
    request.session = {SESSION_ID_KEY: "test-session"}

    response = await server.seed_session(request, NL2RepoBenchSeedSessionRequest.model_validate(_request()))

    assert response.sandbox_handle == "agent-sandbox"
    assert response.sandbox_descriptor == {"sandbox_id": "agent-sandbox", "workdir": "/workspace"}
    assert "test-session" in server._agent_sessions
    spec = sandbox.start.await_args.args[0]
    # Default agent_network_mode is "official", which boots the agent from the generic,
    # task-agnostic sandbox image (not the task's own pinned grading image) - see
    # _AGENT_SANDBOX_IMAGE in app.py.
    assert spec.image == app_module._AGENT_SANDBOX_IMAGE
    assert spec.workdir == "/workspace"


async def test_a_typed_seed_lends_the_agent_sandbox_to_the_session_harness(
    task_assets: Path, monkeypatch: MonkeyPatch
) -> None:
    """An Environment Server episode gets the agent sandbox as sandbox_access, with a descriptor that cannot stop it;
    verify collects and stops it, and a seed retried after that is refused."""
    server = NL2RepoBenchResourcesServer(config=_config(task_assets), server_client=MagicMock(spec=ServerClient))
    sandbox = AsyncMock()
    sandbox.serialize.side_effect = lambda scope=None: {"sandbox_id": "agent-sandbox", "scope": scope or "full"}
    sandbox.exec.return_value = MagicMock(return_code=0, stdout="", stderr="")
    monkeypatch.setattr(app_module, "get_global_config_dict", _global_config)
    monkeypatch.setattr(app_module, "resolve_provider_config", lambda *_: {"opensandbox": {}})
    monkeypatch.setattr(app_module, "resolve_provider_metadata", lambda *_: {})
    monkeypatch.setattr(app_module, "AsyncSandbox", MagicMock(return_value=sandbox))
    monkeypatch.setattr(server, "_collect_workspace", AsyncMock(return_value=(None, "nothing to collect")))
    request = MagicMock()
    request.session = {}
    task_data = {key: value for key, value in _request().items() if key != "response"}
    seed = ResourcesSeedSessionRequest(
        resources_session_id="episode-session",
        episode_id=EpisodeId(rollout_id="0-0"),
        task_id=TaskId(taskset="nl2repobench:benchmark", task_id="example-task"),
        task_data=task_data,
    )

    response = await server.seed_session(request, seed)

    access = response.sandbox_access
    assert access is not None and access.workdir == "/workspace"
    assert access.connection.provider_config_ref == "test"
    assert access.connection.descriptor == {"sandbox_id": "agent-sandbox", "scope": "operate"}
    assert request.session[SESSION_ID_KEY] == "episode-session"
    # A retried seed of the same episode lends the same live sandbox again.
    assert (await server.seed_session(request, seed)).sandbox_access == access
    sandbox.start.assert_awaited_once()

    await server.verify(request, NL2RepoBenchVerifyRequest.model_validate(_request()))
    sandbox.stop.assert_awaited_once()
    with pytest.raises(ResourcesSessionConflictError):
        await server.seed_session(request, seed)


async def test_verify_without_seed_returns_incomplete_result(task_assets: Path) -> None:
    server = NL2RepoBenchResourcesServer(config=_config(task_assets), server_client=MagicMock(spec=ServerClient))
    request = MagicMock()
    request.session = {SESSION_ID_KEY: "missing-session"}

    response = await server.verify(request, NL2RepoBenchVerifyRequest.model_validate(_request()))

    assert response.reward == 0.0
    assert response.evaluation_completed is False
    assert "no agent session found" in (response.verifier_error or "")
    # a lost session says nothing about the agent: masked, not a scored 0
    assert response.mask_sample is True
    assert response.failure_kind == "session_lost"


async def test_verify_collects_workspace_and_runs_tests_in_fresh_sandbox(
    task_assets: Path, monkeypatch: MonkeyPatch
) -> None:
    server = NL2RepoBenchResourcesServer(config=_config(task_assets), server_client=MagicMock(spec=ServerClient))
    agent_sandbox = AsyncMock()
    agent_sandbox.serialize.return_value = {"sandbox_id": "agent-sandbox"}
    verifier_sandbox = AsyncMock()
    monkeypatch.setattr(server, "_create_sandbox", AsyncMock(side_effect=[agent_sandbox, verifier_sandbox]))
    collect_workspace = AsyncMock(return_value=(b"tar-bytes", None))
    monkeypatch.setattr(server, "_collect_workspace", collect_workspace)
    monkeypatch.setattr(
        server,
        "_run_verifier",
        AsyncMock(
            return_value=VerifierResult(
                evaluation_completed=True,
                reward=1.0,
                tests_passed=5,
                tests_failed=0,
                tests_error=0,
                test_case_count=5,
                success_rate=1.0,
                test_output="5 passed in 1.0s",
            )
        ),
    )
    request = MagicMock()
    request.session = {SESSION_ID_KEY: "test-session"}

    await server.seed_session(request, NL2RepoBenchSeedSessionRequest.model_validate(_request()))
    response = await server.verify(request, NL2RepoBenchVerifyRequest.model_validate(_request()))

    assert response.reward == 1.0
    assert response.evaluation_completed is True
    assert response.tests_passed == 5
    assert response.test_case_count == 5
    assert response.workspace_sha256 is not None
    assert response.workspace_bytes == len(b"tar-bytes")
    agent_sandbox.stop.assert_awaited_once()
    verifier_sandbox.stop.assert_awaited_once()
    assert server._agent_sessions == {}


async def test_verify_workspace_collection_failure_short_circuits(task_assets: Path, monkeypatch: MonkeyPatch) -> None:
    server = NL2RepoBenchResourcesServer(config=_config(task_assets), server_client=MagicMock(spec=ServerClient))
    agent_sandbox = AsyncMock()
    agent_sandbox.serialize.return_value = {"sandbox_id": "agent-sandbox"}
    create_sandbox = AsyncMock(return_value=agent_sandbox)
    monkeypatch.setattr(server, "_create_sandbox", create_sandbox)
    monkeypatch.setattr(server, "_collect_workspace", AsyncMock(return_value=(None, "tar command failed")))
    request = MagicMock()
    request.session = {SESSION_ID_KEY: "test-session"}

    await server.seed_session(request, NL2RepoBenchSeedSessionRequest.model_validate(_request()))
    response = await server.verify(request, NL2RepoBenchVerifyRequest.model_validate(_request()))

    assert response.reward == 0.0
    assert response.evaluation_completed is False
    assert response.verifier_error == "tar command failed"
    assert create_sandbox.await_count == 1
    agent_sandbox.stop.assert_awaited_once()


def test_collect_command_excludes_precede_positional_args(tmp_path) -> None:
    # GNU tar requires --exclude flags to precede positional args (-C DIR .);
    # once an --exclude follows them it's rejected (exit code 2) or silently
    # ignored, depending on tar version. Caught via a real (non-mocked) tar
    # invocation during a live smoke test — mocked sandbox tests can't catch
    # this since they never actually shell out to tar.
    command = app_module._COLLECT_COMMAND
    exclude_idx = command.index("--exclude")
    positional_idx = command.index("-C /workspace")
    assert exclude_idx < positional_idx

    result = subprocess.run(
        f"mkdir -p {tmp_path}/workspace/.git {tmp_path}/workspace/src && echo x > {tmp_path}/workspace/src/a.py "
        f"&& echo y > {tmp_path}/workspace/.git/HEAD && "
        + command.replace("/tmp/workspace.tar.gz", f"{tmp_path}/workspace.tar.gz").replace(
            "-C /workspace", f"-C {tmp_path}/workspace"
        ),
        shell=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    listing = subprocess.run(
        f"tar tzf {tmp_path}/workspace.tar.gz",
        shell=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "src/a.py" in listing
    assert ".git" not in listing


# --- grading_protocol="upstream" -----------------------------------------------------------------


def _resolved_task(task_assets: Path):
    server = NL2RepoBenchResourcesServer(config=_config(task_assets), server_client=MagicMock(spec=ServerClient))
    return _resolve_task(NL2RepoBenchVerifyRequest.model_validate(_request()), server._task_store)


def _exec_result(return_code: int, stdout: str = "") -> MagicMock:
    return MagicMock(return_code=return_code, stdout=stdout, stderr="", error_type=None)


def test_parse_pytest_upstream_sums_every_line_of_every_pytest_command() -> None:
    outputs = ["===== 3 passed, 1 failed in 0.4s =====", "2 passed\nnoise\n1 error in 0.1s"]
    assert app_module._parse_pytest_upstream(outputs) == (5, 1, 1)
    assert app_module._parse_pytest_upstream([]) == (0, 0, 0)


def test_upstream_restore_command_deletes_agent_packaging_and_tests_but_keeps_image_files(tmp_path: Path) -> None:
    # The grading image's /workspace already holds the real packaging and held-out tests; the agent's copies of those
    # must be dropped so the image's versions are what pip and pytest see, while the agent's code is laid on top.
    image_ws = tmp_path / "workspace"
    (image_ws / "tests").mkdir(parents=True)
    (image_ws / "pyproject.toml").write_text("real", encoding="utf-8")
    (image_ws / "tests" / "test_real.py").write_text("real test", encoding="utf-8")
    agent = tmp_path / "agent"
    (agent / "pkg").mkdir(parents=True)
    (agent / "tests").mkdir()
    (agent / "pyproject.toml").write_text("broken", encoding="utf-8")
    (agent / "pkg" / "setup.cfg").write_text("nested packaging file", encoding="utf-8")
    (agent / "pkg" / "__init__.py").write_text("code", encoding="utf-8")
    (agent / "tests" / "test_agent.py").write_text("agent test", encoding="utf-8")
    tar_path = tmp_path / "workspace.tar.gz"
    subprocess.run(["tar", "czf", str(tar_path), "-C", str(agent), "."], check=True)

    command = app_module._upstream_restore_command(
        ["tests"], tar_path=str(tar_path), workspace=str(image_ws), stage=str(tmp_path / "stage")
    )
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert (image_ws / "pyproject.toml").read_text(encoding="utf-8") == "real"
    assert (image_ws / "tests" / "test_real.py").exists()
    assert not (image_ws / "tests" / "test_agent.py").exists()
    assert (image_ws / "pkg" / "__init__.py").read_text(encoding="utf-8") == "code"
    assert not (image_ws / "pkg" / "setup.cfg").exists()


async def test_upstream_protocol_runs_every_command_and_scores_pytest(task_assets: Path) -> None:
    config = _config(task_assets).model_copy(update={"grading_protocol": "upstream"})
    server = NL2RepoBenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    sandbox = AsyncMock()
    sandbox.exec.side_effect = [
        _exec_result(0),  # mkdir -p /workspace
        _exec_result(0),  # restore
        _exec_result(1, "ERROR: install failed"),  # pip install -e .
        _exec_result(1, "===== 3 passed, 2 failed in 0.5s ====="),  # pytest
    ]

    result = await server._run_verifier(sandbox, _resolved_task(task_assets), b"tar-bytes")

    commands = [call.args[0] for call in sandbox.exec.await_args_list]
    assert "find " in commands[1] and "-name pyproject.toml" in commands[1]
    assert all(
        command.startswith("bash -o pipefail -c ") and "export PYTHONPATH=/workspace" in command
        for command in commands[2:]
    )
    assert result.verifier_error is None
    assert (result.tests_passed, result.tests_failed) == (3, 2)
    assert result.reward == 3 / 5
    assert result.test_output.startswith("[grading_protocol: upstream]")


def test_parse_pytest_outer_counts_only_each_commands_final_summary() -> None:
    nested = (
        "collected 184 items\n"
        "E   and: '========== 10 passed, 1 warning in 0.06s =========='\n"
        "---- Captured stdout call ----\n"
        "========================= 1 failed, 1 passed in 0.09s ==========================\n"
        "============ 35 failed, 149 passed, 27 skipped in 126.57s (0:02:06) ============\n"
    )
    coloured = "\x1b[31m===== \x1b[31m\x1b[1m243 failed\x1b[0m, \x1b[32m408 passed\x1b[0m\x1b[31m in 4.28s\x1b[0m\x1b[31m =====\n"
    quiet = "73 failed, 61 passed, 16 warnings, 4 errors in 1.80s\n"
    frozen_clock = "============== 41 failed, 92 passed, 6 skipped in -467150442.48s ===============\n"
    assert app_module._parse_pytest_outer([nested]) == (149, 35, 0)
    assert app_module._parse_pytest_outer([coloured]) == (408, 243, 0)
    assert app_module._parse_pytest_outer([quiet]) == (61, 73, 4)
    assert app_module._parse_pytest_outer([frozen_clock]) == (92, 41, 0)
    assert app_module._parse_pytest_outer([nested, quiet]) == (210, 108, 4)
    assert app_module._parse_pytest_outer(["no summary at all"]) == (0, 0, 0)
    # upstream's parser adds the nested summaries too
    assert app_module._parse_pytest_upstream([nested])[0] == 160


async def test_upstream_protocol_clean_reward_ignores_nested_summaries(task_assets: Path) -> None:
    config = _config(task_assets).model_copy(update={"grading_protocol": "upstream"})
    server = NL2RepoBenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    sandbox = AsyncMock()
    sandbox.exec.side_effect = [
        _exec_result(0),
        _exec_result(0),
        _exec_result(0),  # pip install -e .
        # a test that runs pytest itself prints "10 passed" before the outer summary; 7 tests collected, 5 stored
        _exec_result(1, "E   10 passed in 0.1s\n10 passed in 0.1s\n===== 3 passed, 4 failed in 0.5s ====="),
    ]

    result = await server._run_verifier(sandbox, _resolved_task(task_assets), b"tar-bytes")

    assert result.reward == 1.0  # upstream: (10 + 3) / 5, capped
    assert result.clean_reward == 3 / 7  # outer summary over max(stored 5, collected 7)


def test_clean_reward_excludes_vendored_targets_and_falls_back_on_failure() -> None:
    graded = app_module.VerifierResult(evaluation_completed=True, reward=1.0, clean_reward=0.8)
    failed = app_module.VerifierResult(evaluation_completed=False, reward=0.0)
    exclude = app_module.NL2RepoBenchResourcesServerConfig.model_fields["clean_exclude_tasks"].default_factory()
    assert set(exclude) == {"six", "tenacity", "typing_extensions", "more-Itertools"}
    assert app_module._clean_reward_for("tenacity", graded, exclude) is None
    assert app_module._clean_reward_for("aiofiles", graded, exclude) == 0.8
    assert app_module._clean_reward_for("aiofiles", failed, exclude) == 0.0


async def test_default_protocol_still_zeroes_on_install_failure(task_assets: Path) -> None:
    server = NL2RepoBenchResourcesServer(config=_config(task_assets), server_client=MagicMock(spec=ServerClient))
    sandbox = AsyncMock()
    sandbox.exec.side_effect = [_exec_result(0), _exec_result(0), _exec_result(1, "ERROR: install failed")]

    result = await server._run_verifier(sandbox, _resolved_task(task_assets), b"tar-bytes")

    assert sandbox.exec.await_args_list[1].args[0] == "tar xzf /tmp/workspace.tar.gz -C /workspace"
    assert result.verifier_error == "install command failed: pip install -e ."
    assert result.tests_passed == 0
    assert result.reward == 0.0


async def test_trace_only_saves_the_workspace_and_grades_nothing(
    task_assets: Path, tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    config = _config(task_assets).model_copy(update={"trace_only": True, "save_workspace_dir": tmp_path / "ws"})
    server = NL2RepoBenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    agent_sandbox = AsyncMock()
    agent_sandbox.serialize.return_value = {"sandbox_id": "agent-sandbox"}
    create = AsyncMock(side_effect=[agent_sandbox])  # a verifier sandbox would be a second call
    monkeypatch.setattr(server, "_create_sandbox", create)
    monkeypatch.setattr(server, "_collect_workspace", AsyncMock(return_value=(b"tar-bytes", None)))
    run_verifier = AsyncMock()
    monkeypatch.setattr(server, "_run_verifier", run_verifier)
    request = MagicMock()
    request.session = {SESSION_ID_KEY: "s1"}

    await server.seed_session(request, NL2RepoBenchSeedSessionRequest.model_validate(_request()))
    response = await server.verify(request, NL2RepoBenchVerifyRequest.model_validate(_request()))

    assert response.reward == 0.0 and response.evaluation_completed is False and response.verifier_error is None
    assert response.model_dump()["graded"] is False
    assert create.await_count == 1 and run_verifier.await_count == 0
    assert (tmp_path / "ws" / "example-task" / "s1.tar.gz").read_bytes() == b"tar-bytes"


async def test_task_workdir_is_where_the_agent_works(task_assets: Path, monkeypatch: MonkeyPatch) -> None:
    (task_assets / "example-task" / "workdir.txt").write_text("/workspace/example\n", encoding="utf-8")
    server = NL2RepoBenchResourcesServer(config=_config(task_assets), server_client=MagicMock(spec=ServerClient))
    sandbox = AsyncMock()
    sandbox.serialize.return_value = {"sandbox_id": "agent-sandbox"}
    sandbox.exec.return_value = MagicMock(return_code=0, stdout="", stderr="")
    monkeypatch.setattr(app_module, "get_global_config_dict", _global_config)
    monkeypatch.setattr(app_module, "resolve_provider_config", lambda *_: {"opensandbox": {}})
    monkeypatch.setattr(app_module, "resolve_provider_metadata", lambda *_: {})
    monkeypatch.setattr(app_module, "AsyncSandbox", MagicMock(return_value=sandbox))
    request = MagicMock()
    request.session = {SESSION_ID_KEY: "s1"}

    response = await server.seed_session(request, NL2RepoBenchSeedSessionRequest.model_validate(_request()))

    assert response.workdir == "/workspace/example"
    commands = [c.kwargs.get("command") or c.args[0] for c in sandbox.exec.await_args_list]
    assert "mkdir -p /workspace/example" in commands
    assert sandbox.upload.await_args.args[1] == "/workspace/example/start.md"
    assert server._agent_sessions["s1"].workdir == "/workspace/example"
    assert "-C /workspace/example ." in app_module._collect_command("/workspace/example")
    # The tar itself is unchanged for /workspace; the command wraps it with the archive checks and its size.
    assert app_module._COLLECT_COMMAND in app_module._collect_command("/workspace")


async def test_sandbox_metadata_from_config_reaches_the_spec(task_assets: Path, monkeypatch: MonkeyPatch) -> None:
    """nemo.nvidia.com/oom-kill: process (configs/nl2repobench.yaml) must reach OpenSandbox at creation."""
    config = _config(task_assets).model_copy(
        update={"sandbox_config": {"metadata": {"nemo.nvidia.com/oom-kill": "process"}}}
    )
    server = NL2RepoBenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    sandbox = AsyncMock()
    sandbox.serialize.return_value = {"sandbox_id": "agent-sandbox"}
    sandbox.exec.return_value = MagicMock(return_code=0, stdout="", stderr="")
    monkeypatch.setattr(app_module, "get_global_config_dict", _global_config)
    monkeypatch.setattr(app_module, "resolve_provider_config", lambda *_: {"opensandbox": {}})
    monkeypatch.setattr(app_module, "resolve_provider_metadata", lambda *_: {})
    monkeypatch.setattr(app_module, "AsyncSandbox", MagicMock(return_value=sandbox))
    request = MagicMock()
    request.session = {SESSION_ID_KEY: "s1"}
    await server.seed_session(request, NL2RepoBenchSeedSessionRequest.model_validate(_request()))
    assert sandbox.start.await_args.args[0].metadata["nemo.nvidia.com/oom-kill"] == "process"


async def test_verify_image_registry_template_picks_the_grading_image(
    task_assets: Path, tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """verify_image_registry_template (map:<json>) grades in the per-task image it names; the agent image is unchanged."""
    verify_map = tmp_path / "verify_images.json"
    verify_map.write_text('{"example-task": "registry.example/twins/example-task@sha256:abc"}')
    config = _config(task_assets).model_copy(
        update={
            "agent_network_mode": "no_internet_preinstalled",
            "verify_image_registry_template": f"map:{verify_map}",
        }
    )
    server = NL2RepoBenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    sandbox = AsyncMock()
    sandbox.exec.return_value = MagicMock(return_code=0, stdout="", stderr="")
    created = MagicMock(return_value=sandbox)
    monkeypatch.setattr(app_module, "get_global_config_dict", _global_config)
    monkeypatch.setattr(app_module, "resolve_provider_config", lambda *_: {"opensandbox": {}})
    monkeypatch.setattr(app_module, "resolve_provider_metadata", lambda *_: {})
    monkeypatch.setattr(app_module, "AsyncSandbox", created)
    task = _resolve_task(NL2RepoBenchVerifyRequest.model_validate(_request()), server._task_store)
    await server._create_sandbox(task, phase="verify")
    await server._create_sandbox(task, phase="agent")
    specs = [call.args[0] for call in sandbox.start.await_args_list]
    assert specs[0].image == "registry.example/twins/example-task@sha256:abc"
    assert specs[1].image == UPSTREAM_IMAGE


def test_bounded_output_keeps_the_tail_and_the_exit_code() -> None:
    command = (
        'i=0; while [ $i -lt 2000 ]; do echo "progress line $i"; i=$((i+1)); done; '
        'echo "oops" >&2; echo "== 4 passed in 1s =="; exit 3'
    )
    result = subprocess.run(["bash", "-c", _bounded_output(command, max_bytes=200)], capture_output=True, text=True)

    assert result.returncode == 3  # the test run's own exit code survives the pipe
    assert len(result.stdout.encode()) <= 200
    assert result.stdout.rstrip().endswith("== 4 passed in 1s ==")  # the summary line is last, so it is kept
    assert "progress line 0\n" not in result.stdout  # the head was dropped
    assert result.stderr == ""  # stderr was merged into the captured stream


def test_bounded_output_runs_multi_statement_commands_with_a_trailing_comment() -> None:
    result = subprocess.run(
        ["bash", "-c", _bounded_output("export A=1; echo $A  # done")], capture_output=True, text=True
    )

    assert result.returncode == 0 and result.stdout == "1\n"


# --- inconclusive verification is masked, not scored ---------------------------------------------


async def _seeded_server(task_assets: Path, monkeypatch: MonkeyPatch, **config_updates: Any) -> tuple:
    config = _config(task_assets).model_copy(update={"verifier_sandbox_retry_delay_s": 0.0, **config_updates})
    server = NL2RepoBenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    agent_sandbox = AsyncMock()
    agent_sandbox.serialize.return_value = {"sandbox_id": "agent-sandbox"}
    create_sandbox = AsyncMock(side_effect=[agent_sandbox])
    monkeypatch.setattr(server, "_create_sandbox", create_sandbox)
    monkeypatch.setattr(server, "_collect_workspace", AsyncMock(return_value=(b"tar-bytes", None)))
    request = MagicMock()
    request.session = {SESSION_ID_KEY: "test-session"}
    await server.seed_session(request, NL2RepoBenchSeedSessionRequest.model_validate(_request()))
    return server, request, create_sandbox


async def test_verifier_sandbox_start_is_retried_then_succeeds(task_assets: Path, monkeypatch: MonkeyPatch) -> None:
    server, request, create_sandbox = await _seeded_server(task_assets, monkeypatch)
    verifier_sandbox = AsyncMock()
    create_sandbox.side_effect = [RuntimeError("provider busy"), RuntimeError("provider busy"), verifier_sandbox]
    monkeypatch.setattr(
        server,
        "_run_verifier",
        AsyncMock(return_value=VerifierResult(evaluation_completed=True, reward=1.0, test_case_count=5)),
    )

    response = await server.verify(request, NL2RepoBenchVerifyRequest.model_validate(_request()))

    assert response.reward == 1.0 and not getattr(response, "mask_sample", False)
    assert create_sandbox.await_count == 1 + 3  # the agent sandbox, then three verifier starts
    verifier_sandbox.stop.assert_awaited_once()


async def test_verifier_sandbox_that_never_starts_masks_the_row(task_assets: Path, monkeypatch: MonkeyPatch) -> None:
    server, request, create_sandbox = await _seeded_server(task_assets, monkeypatch)
    create_sandbox.side_effect = RuntimeError("provider busy")

    response = await server.verify(request, NL2RepoBenchVerifyRequest.model_validate(_request()))

    assert response.reward == 0.0
    assert response.mask_sample is True
    assert response.failure_kind == "verifier_error"
    assert "provider busy" in response.failure_reason
    assert create_sandbox.await_count == 1 + (1 + server.config.inconclusive_verification_retries)


async def test_provider_error_while_running_tests_masks_the_row(task_assets: Path, monkeypatch: MonkeyPatch) -> None:
    server, request, create_sandbox = await _seeded_server(task_assets, monkeypatch)
    sandboxes = []
    for _ in range((1 + server.config.inconclusive_verification_retries)):
        sandbox = AsyncMock()
        sandbox.exec.return_value = MagicMock(return_code=125, stdout="", stderr="", error_type="sandbox")
        sandboxes.append(sandbox)
    create_sandbox.side_effect = sandboxes

    response = await server.verify(request, NL2RepoBenchVerifyRequest.model_validate(_request()))

    assert response.mask_sample is True and response.reward == 0.0
    assert "sandbox" in response.failure_reason
    assert "3 attempt" in response.failure_reason
    for sandbox in sandboxes:  # every attempt ran on its own sandbox, and each was stopped
        sandbox.stop.assert_awaited_once()


async def test_provider_error_is_retried_on_a_fresh_sandbox_and_then_scored(
    task_assets: Path, monkeypatch: MonkeyPatch
) -> None:
    server, request, create_sandbox = await _seeded_server(task_assets, monkeypatch)
    broken, healthy = AsyncMock(), AsyncMock()
    create_sandbox.side_effect = [broken, healthy]
    run_verifier = AsyncMock(
        side_effect=[
            RuntimeError("the sandbox provider reported 'sandbox'"),
            VerifierResult(evaluation_completed=True, reward=0.5, test_case_count=4, tests_passed=2),
        ]
    )
    monkeypatch.setattr(server, "_run_verifier", run_verifier)

    response = await server.verify(request, NL2RepoBenchVerifyRequest.model_validate(_request()))

    assert response.reward == 0.5 and not getattr(response, "mask_sample", False)
    assert run_verifier.await_args_list[0].args[0] is broken and run_verifier.await_args_list[1].args[0] is healthy
    broken.stop.assert_awaited_once()
    healthy.stop.assert_awaited_once()


async def test_ordinary_failing_tests_are_not_retried_or_masked(task_assets: Path, monkeypatch: MonkeyPatch) -> None:
    server, request, create_sandbox = await _seeded_server(task_assets, monkeypatch)
    verifier_sandbox = AsyncMock()
    create_sandbox.side_effect = [verifier_sandbox]
    monkeypatch.setattr(
        server,
        "_run_verifier",
        AsyncMock(
            return_value=VerifierResult(
                evaluation_completed=True, reward=0.0, test_case_count=4, verifier_error="install command failed: pip"
            )
        ),
    )

    response = await server.verify(request, NL2RepoBenchVerifyRequest.model_validate(_request()))

    assert response.reward == 0.0 and response.evaluation_completed is True
    assert not getattr(response, "mask_sample", False)
    assert create_sandbox.await_count == 2  # the agent sandbox and a single verifier sandbox


async def test_a_test_command_timeout_is_still_a_scored_result(task_assets: Path) -> None:
    server = NL2RepoBenchResourcesServer(config=_config(task_assets), server_client=MagicMock(spec=ServerClient))
    task = server._task_store.get("example-task")
    calls = []

    async def exec_(command: str, **kwargs: Any) -> MagicMock:
        calls.append(command)
        if len(calls) <= 2:  # mkdir and the workspace restore
            return _exec_result(0)
        return MagicMock(return_code=124, stdout="", stderr="", error_type="timeout")  # the agent's code hung

    sandbox = AsyncMock()
    sandbox.exec.side_effect = exec_

    result = await server._run_verifier(sandbox, task, b"tar")

    assert result.evaluation_completed is True  # scored, not masked: a timeout is a test outcome


# --- sandbox lifecycle ---------------------------------------------------------------------------


async def test_failed_agent_init_stops_the_sandbox(task_assets: Path, monkeypatch: MonkeyPatch) -> None:
    server = NL2RepoBenchResourcesServer(config=_config(task_assets), server_client=MagicMock(spec=ServerClient))
    sandbox = AsyncMock()
    sandbox.exec.return_value = _exec_result(1)  # mkdir -p /workspace fails after the sandbox started
    monkeypatch.setattr(app_module, "get_global_config_dict", _global_config)
    monkeypatch.setattr(app_module, "resolve_provider_config", lambda *_: {"opensandbox": {}})
    monkeypatch.setattr(app_module, "resolve_provider_metadata", lambda *_: {})
    monkeypatch.setattr(app_module, "AsyncSandbox", MagicMock(return_value=sandbox))
    request = MagicMock()
    request.session = {SESSION_ID_KEY: "test-session"}

    with pytest.raises(RuntimeError, match="Failed to create"):
        await server.seed_session(request, NL2RepoBenchSeedSessionRequest.model_validate(_request()))

    sandbox.stop.assert_awaited()
    assert server._agent_sessions == {}


async def test_release_session_state_stops_the_live_sandbox(task_assets: Path, monkeypatch: MonkeyPatch) -> None:
    server, request, _ = await _seeded_server(task_assets, monkeypatch)
    agent_sandbox = server._agent_sessions["test-session"].sandbox
    assert server.has_session_state("test-session")

    await server.release_session_state("test-session")

    agent_sandbox.stop.assert_awaited_once()
    assert not server.has_session_state("test-session")
    await server.release_session_state("test-session")  # idempotent
    await server.release_session_state("never-seeded")


async def test_release_session_state_keeps_the_session_when_stop_fails(
    task_assets: Path, monkeypatch: MonkeyPatch
) -> None:
    server, request, _ = await _seeded_server(task_assets, monkeypatch)
    agent_sandbox = server._agent_sessions["test-session"].sandbox
    agent_sandbox.stop.side_effect = [RuntimeError("provider down"), None]

    with pytest.raises(RuntimeError, match="provider down"):
        await server.release_session_state("test-session")
    assert server.has_session_state("test-session")  # kept for the retry

    await server.release_session_state("test-session")
    assert not server.has_session_state("test-session")


async def test_shutdown_stops_every_live_agent_sandbox(task_assets: Path, monkeypatch: MonkeyPatch) -> None:
    server, request, _ = await _seeded_server(task_assets, monkeypatch)
    first = server._agent_sessions["test-session"]
    second_sandbox = AsyncMock()
    server._agent_sessions["other-session"] = app_module.AgentSandboxSession(
        task_id="t",
        image="i",
        sandbox=second_sandbox,
        sandbox_handle="h",
        sandbox_descriptor={},
    )
    second_sandbox.stop.side_effect = RuntimeError("provider down")  # one failing stop must not strand the others

    await server.shutdown()

    first.sandbox.stop.assert_awaited_once()
    second_sandbox.stop.assert_awaited_once()
    assert server._agent_sessions == {}


def test_the_app_stops_sandboxes_when_it_shuts_down(task_assets: Path, monkeypatch: MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    server = NL2RepoBenchResourcesServer(config=_config(task_assets), server_client=MagicMock(spec=ServerClient))
    sandbox = AsyncMock()
    server._agent_sessions["s"] = app_module.AgentSandboxSession(
        task_id="t", image="i", sandbox=sandbox, sandbox_handle="h", sandbox_descriptor={}
    )

    with TestClient(server.setup_webserver()):
        sandbox.stop.assert_not_awaited()
    sandbox.stop.assert_awaited_once()


@pytest.mark.parametrize("failing_step", ["mkdir", "upload"])
async def test_agent_init_failure_or_cancellation_stops_the_sandbox(
    task_assets: Path, monkeypatch: MonkeyPatch, failing_step: str
) -> None:
    for error in (RuntimeError("boom"), asyncio.CancelledError()):
        server = NL2RepoBenchResourcesServer(config=_config(task_assets), server_client=MagicMock(spec=ServerClient))
        sandbox = AsyncMock()
        sandbox.exec.return_value = _exec_result(0)
        if failing_step == "mkdir":
            sandbox.exec.side_effect = error
        else:
            sandbox.upload.side_effect = error
        monkeypatch.setattr(app_module, "get_global_config_dict", _global_config)
        monkeypatch.setattr(app_module, "resolve_provider_config", lambda *_: {"opensandbox": {}})
        monkeypatch.setattr(app_module, "resolve_provider_metadata", lambda *_: {})
        monkeypatch.setattr(app_module, "AsyncSandbox", MagicMock(return_value=sandbox))
        request = MagicMock()
        request.session = {SESSION_ID_KEY: "test-session"}

        with pytest.raises(type(error)):
            await server.seed_session(request, NL2RepoBenchSeedSessionRequest.model_validate(_request()))

        sandbox.stop.assert_awaited()  # stopped before the error left _create_sandbox, so nothing is leaked
        assert server._agent_sessions == {}


def test_close_session_over_http_stops_the_sandbox_seeded_over_http(
    task_assets: Path, monkeypatch: MonkeyPatch
) -> None:
    """The routes an Environment Server uses: /seed_session answers with the handle, /close_session releases it."""
    from fastapi.testclient import TestClient

    from nemo_gym.core import EpisodeId

    server = NL2RepoBenchResourcesServer(config=_config(task_assets), server_client=MagicMock(spec=ServerClient))
    sandbox = AsyncMock()
    sandbox.serialize.return_value = {"sandbox_id": "agent-sandbox"}
    monkeypatch.setattr(server, "_create_sandbox", AsyncMock(return_value=sandbox))

    with TestClient(server.setup_webserver()) as client:
        seeded = client.post("/seed_session", json=_request())
        assert seeded.status_code == 200, seeded.text
        assert seeded.json()["sandbox_handle"] == "agent-sandbox"
        assert seeded.json()["workdir"] == "/workspace"
        [session_id] = server._agent_sessions
        sandbox.stop.assert_not_awaited()

        close = {"resources_session_id": session_id, "episode_id": EpisodeId(rollout_id="r").model_dump(mode="json")}
        assert client.post("/close_session", json=close).status_code == 200
        sandbox.stop.assert_awaited_once()
        assert server._agent_sessions == {}
        assert client.post("/close_session", json=close).status_code == 200  # a repeated close is harmless
        sandbox.stop.assert_awaited_once()


# --- server-side failures are masked; agent-side ones stay scored ---------------------------------


def _grading_sandbox() -> AsyncMock:
    """A verifier sandbox whose every command succeeds and prints a passing pytest summary."""
    sandbox = AsyncMock()
    sandbox.exec.return_value = _exec_result(0, "===== 5 passed in 0.1s =====")
    return sandbox


def _setup_failure_sandbox(kind: str) -> AsyncMock:
    sandbox = AsyncMock()
    if kind == "mkdir_fails":
        sandbox.exec.return_value = MagicMock(
            return_code=1, stdout="", stderr="mkdir: read-only file system", error_type=None
        )
    else:
        sandbox.exec.return_value = MagicMock(return_code=124, stdout="", stderr="", error_type="timeout")
    return sandbox


@pytest.mark.parametrize("kind", ["mkdir_fails", "mkdir_times_out"])
async def test_verifier_setup_failure_on_every_fresh_sandbox_masks_the_row(
    task_assets: Path, monkeypatch: MonkeyPatch, kind: str
) -> None:
    server, request, create_sandbox = await _seeded_server(task_assets, monkeypatch)
    attempts = 1 + server.config.inconclusive_verification_retries
    sandboxes = [_setup_failure_sandbox(kind) for _ in range(attempts)]
    create_sandbox.side_effect = sandboxes

    response = await server.verify(request, NL2RepoBenchVerifyRequest.model_validate(_request()))

    assert response.mask_sample is True and response.reward == 0.0
    assert response.failure_kind == "verifier_error"
    assert f"{attempts} attempt" in response.failure_reason
    assert create_sandbox.await_count == 1 + attempts  # the agent sandbox, then one fresh verifier per attempt
    for sandbox in sandboxes:
        sandbox.stop.assert_awaited_once()


async def test_a_later_healthy_verifier_grades_normally_after_a_mkdir_failure(
    task_assets: Path, monkeypatch: MonkeyPatch
) -> None:
    server, request, create_sandbox = await _seeded_server(task_assets, monkeypatch)
    create_sandbox.side_effect = [_setup_failure_sandbox("mkdir_fails"), _grading_sandbox()]

    response = await server.verify(request, NL2RepoBenchVerifyRequest.model_validate(_request()))

    assert not getattr(response, "mask_sample", False)
    assert response.evaluation_completed is True and response.tests_passed == 5 and response.reward == 1.0


async def test_a_test_command_timeout_is_scored_not_masked(task_assets: Path, monkeypatch: MonkeyPatch) -> None:
    server, request, create_sandbox = await _seeded_server(task_assets, monkeypatch)
    calls = []

    async def exec_(command: str, **kwargs: Any) -> MagicMock:
        calls.append(command)
        if len(calls) <= 2:  # mkdir and the workspace restore
            return _exec_result(0)
        return MagicMock(return_code=124, stdout="", stderr="", error_type="timeout")

    verifier = AsyncMock()
    verifier.exec.side_effect = exec_
    create_sandbox.side_effect = [verifier]

    response = await server.verify(request, NL2RepoBenchVerifyRequest.model_validate(_request()))

    assert response.evaluation_completed is True and not getattr(response, "mask_sample", False)
    assert create_sandbox.await_count == 2  # no retry


def _collected(size: int) -> MagicMock:
    """The collect command's result when the archive in the sandbox has `size` bytes."""
    return MagicMock(return_code=0, stdout=f"{app_module._COLLECT_SIZE_MARKER}{size}\n", stderr="", error_type=None)


def _use_real_collection(server: NL2RepoBenchResourcesServer, monkeypatch: MonkeyPatch) -> None:
    async def collect(sandbox: Any, workdir: str = "/workspace") -> tuple:
        return await NL2RepoBenchResourcesServer._collect_workspace(server, sandbox, workdir)

    monkeypatch.setattr(server, "_collect_workspace", collect)


async def test_a_typed_provider_failure_during_collection_is_masked_and_no_verifier_runs(
    task_assets: Path, monkeypatch: MonkeyPatch
) -> None:
    server, request, create_sandbox = await _seeded_server(task_assets, monkeypatch)
    agent_sandbox = server._agent_sessions["test-session"].sandbox
    event = NgFailureEvent(boundary="sandbox", failure_kind=PROVIDER_UNAVAILABLE, failure_reason="provider down")
    monkeypatch.setattr(server, "_collect_workspace", AsyncMock(side_effect=event))

    response = await server.verify(request, NL2RepoBenchVerifyRequest.model_validate(_request()))

    assert response.mask_sample is True and response.reward == 0.0
    assert response.failure_kind == PROVIDER_UNAVAILABLE
    assert "could not collect" in response.failure_reason
    assert create_sandbox.await_count == 1  # only the agent sandbox: no verifier ran
    agent_sandbox.stop.assert_awaited_once()


async def test_a_collection_exec_the_provider_could_not_run_is_masked(
    task_assets: Path, monkeypatch: MonkeyPatch
) -> None:
    server, request, create_sandbox = await _seeded_server(task_assets, monkeypatch)
    agent_sandbox = server._agent_sessions["test-session"].sandbox
    agent_sandbox.exec.return_value = MagicMock(return_code=125, stdout="", stderr="", error_type="sandbox")
    _use_real_collection(server, monkeypatch)

    response = await server.verify(request, NL2RepoBenchVerifyRequest.model_validate(_request()))

    assert response.mask_sample is True and response.failure_kind == "verifier_error"
    assert create_sandbox.await_count == 1


async def test_an_ordinary_tar_failure_during_collection_is_still_scored(
    task_assets: Path, monkeypatch: MonkeyPatch
) -> None:
    server, request, create_sandbox = await _seeded_server(task_assets, monkeypatch)
    agent_sandbox = server._agent_sessions["test-session"].sandbox
    agent_sandbox.exec.return_value = MagicMock(return_code=2, stdout="", stderr="tar: Cannot stat", error_type=None)
    _use_real_collection(server, monkeypatch)

    response = await server.verify(request, NL2RepoBenchVerifyRequest.model_validate(_request()))

    assert response.reward == 0.0 and not getattr(response, "mask_sample", False)
    assert "exited with code 2" in response.verifier_error
    assert create_sandbox.await_count == 1


async def test_an_over_limit_archive_is_scored_and_never_downloaded(
    task_assets: Path, monkeypatch: MonkeyPatch
) -> None:
    server, request, create_sandbox = await _seeded_server(task_assets, monkeypatch, workspace_tar_max_bytes=3)
    agent_sandbox = server._agent_sessions["test-session"].sandbox
    agent_sandbox.exec.return_value = _collected(10)
    _use_real_collection(server, monkeypatch)

    response = await server.verify(request, NL2RepoBenchVerifyRequest.model_validate(_request()))

    assert response.reward == 0.0 and not getattr(response, "mask_sample", False)
    assert "exceeds workspace_tar_max_bytes" in response.verifier_error
    # The size is read in the sandbox, so an oversized archive never reaches this server's disk or memory.
    agent_sandbox.download.assert_not_awaited()


@pytest.mark.parametrize(
    ("result", "reason"),
    [
        (MagicMock(return_code=124, stdout="", stderr="", error_type="timeout"), "reported 'timeout'"),
        (MagicMock(return_code=0, stdout="", stderr="", error_type=None), "did not report the archive size"),
    ],
)
async def test_a_collection_that_times_out_or_reports_no_size_is_masked(
    task_assets: Path, monkeypatch: MonkeyPatch, result: MagicMock, reason: str
) -> None:
    server, request, create_sandbox = await _seeded_server(task_assets, monkeypatch)
    agent_sandbox = server._agent_sessions["test-session"].sandbox
    agent_sandbox.exec.return_value = result
    _use_real_collection(server, monkeypatch)

    response = await server.verify(request, NL2RepoBenchVerifyRequest.model_validate(_request()))

    # This server's own tar step failed, not the agent's work: masked, and no verifier runs.
    assert response.mask_sample is True and response.failure_kind == "verifier_error"
    assert reason in response.failure_reason
    assert create_sandbox.await_count == 1


async def test_a_download_shorter_than_the_archive_is_masked(task_assets: Path, monkeypatch: MonkeyPatch) -> None:
    server, request, create_sandbox = await _seeded_server(task_assets, monkeypatch)
    agent_sandbox = server._agent_sessions["test-session"].sandbox
    agent_sandbox.exec.return_value = _collected(10)

    async def download(remote: str, local: Path) -> None:
        Path(local).write_bytes(b"x" * 4)

    agent_sandbox.download.side_effect = download
    _use_real_collection(server, monkeypatch)

    response = await server.verify(request, NL2RepoBenchVerifyRequest.model_validate(_request()))

    assert response.mask_sample is True and "returned 4 bytes" in response.failure_reason
    assert create_sandbox.await_count == 1


def test_the_collect_command_refuses_a_planted_archive_and_reports_the_size(tmp_path: Path) -> None:
    """The real command under bash: the archive is rebuilt, a symlink in its place is removed first, and the last line
    carries the size the server reads."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "pkg.py").write_text("x = 1\n")
    archive = tmp_path / "workspace.tar.gz"
    archive.symlink_to("/dev/zero")
    command = app_module._collect_command(str(workspace)).replace("/tmp/workspace.tar.gz", str(archive))

    result = subprocess.run(["bash", "-c", command], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert not archive.is_symlink()
    assert app_module._collected_size(result.stdout) == archive.stat().st_size > 0


# --- trace-only rows are masked ------------------------------------------------------------------


async def test_trace_only_rows_are_masked_so_they_only_count_toward_coverage(
    task_assets: Path, monkeypatch: MonkeyPatch
) -> None:
    server, request, create_sandbox = await _seeded_server(task_assets, monkeypatch, trace_only=True)

    response = await server.verify(request, NL2RepoBenchVerifyRequest.model_validate(_request()))

    assert response.mask_sample is True and response.failure_kind == "judge_skipped"
    assert response.reward == 0.0 and response.graded is False and response.evaluation_completed is False
    assert create_sandbox.await_count == 1  # nothing was graded


async def test_graded_rows_are_not_masked_when_trace_only_is_off(task_assets: Path, monkeypatch: MonkeyPatch) -> None:
    server, request, create_sandbox = await _seeded_server(task_assets, monkeypatch)
    create_sandbox.side_effect = [_grading_sandbox()]

    response = await server.verify(request, NL2RepoBenchVerifyRequest.model_validate(_request()))

    assert response.graded is True and response.reward == 1.0 and not getattr(response, "mask_sample", False)


# --- a failed stop stays reachable --------------------------------------------------------------


async def test_a_failed_stop_after_verify_keeps_the_session_for_a_later_close(
    task_assets: Path, monkeypatch: MonkeyPatch
) -> None:
    server, request, create_sandbox = await _seeded_server(task_assets, monkeypatch)
    agent_sandbox = server._agent_sessions["test-session"].sandbox
    agent_sandbox.stop.side_effect = [RuntimeError("provider down"), None]
    create_sandbox.side_effect = [_grading_sandbox()]

    response = await server.verify(request, NL2RepoBenchVerifyRequest.model_validate(_request()))

    assert response.reward == 1.0  # the grade is not lost to a failed teardown
    assert server.has_session_state("test-session")  # kept: the handle is still reachable
    close = ResourcesCloseSessionRequest(resources_session_id="test-session", episode_id=EpisodeId(rollout_id="r"))
    await server.close_resources_session(close)
    assert agent_sandbox.stop.await_count == 2
    assert not server.has_session_state("test-session")


async def test_a_failed_stop_after_verify_is_retried_at_shutdown(task_assets: Path, monkeypatch: MonkeyPatch) -> None:
    server, request, create_sandbox = await _seeded_server(task_assets, monkeypatch)
    agent_sandbox = server._agent_sessions["test-session"].sandbox
    agent_sandbox.stop.side_effect = [RuntimeError("provider down"), None]
    create_sandbox.side_effect = [_grading_sandbox()]
    await server.verify(request, NL2RepoBenchVerifyRequest.model_validate(_request()))
    assert server.has_session_state("test-session")

    await server.shutdown()

    assert agent_sandbox.stop.await_count == 2
    assert server._agent_sessions == {}


async def test_a_close_cannot_interrupt_the_workspace_snapshot(task_assets: Path, monkeypatch: MonkeyPatch) -> None:
    server, request, create_sandbox = await _seeded_server(task_assets, monkeypatch)
    agent_sandbox = server._agent_sessions["test-session"].sandbox
    create_sandbox.side_effect = [_grading_sandbox()]
    collecting, finish = asyncio.Event(), asyncio.Event()
    events: list[str] = []

    async def collect(sandbox: Any, workdir: str = "/workspace") -> tuple:
        collecting.set()
        await finish.wait()
        events.append("collected")
        return b"tar-bytes", None

    async def stop() -> None:
        events.append("stopped")

    monkeypatch.setattr(server, "_collect_workspace", collect)
    agent_sandbox.stop.side_effect = stop

    verifying = asyncio.create_task(server.verify(request, NL2RepoBenchVerifyRequest.model_validate(_request())))
    await collecting.wait()
    close = ResourcesCloseSessionRequest(resources_session_id="test-session", episode_id=EpisodeId(rollout_id="r"))
    closing = asyncio.create_task(server.close_resources_session(close))
    await asyncio.sleep(0.05)
    assert not closing.done() and events == []  # the close waits: the sandbox is untouched mid-snapshot
    finish.set()
    await asyncio.gather(verifying, closing)

    assert events == ["collected", "stopped"]  # collected first, then released exactly once


# --- upstream restore: agent symlinks cannot replace the image's packaging --------------------------


def test_upstream_restore_keeps_the_images_packaging_when_the_agent_symlinks_over_it(tmp_path: Path) -> None:
    workspace, agent = tmp_path / "workspace", tmp_path / "agent"
    workspace.mkdir()
    agent.mkdir()
    (workspace / "setup.py").write_text("image setup.py\n")
    (workspace / "pyproject.toml").write_text("image pyproject.toml\n")
    (workspace / "requirements.txt").write_text("image requirements\n")
    (agent / "setup.py").write_text("agent setup.py\n")  # a regular packaging file: deleted, as upstream does
    (agent / "evil.toml").write_text("agent pyproject\n")
    (agent / "pyproject.toml").symlink_to("evil.toml")  # a packaging-named symlink must not replace the image's
    (agent / "requirements.txt").symlink_to("/etc/hostname")  # nor one that points outside
    (agent / "pkg").mkdir()
    (agent / "pkg" / "setup.cfg").symlink_to("../evil.toml")  # nor a nested one
    (agent / "data.txt").write_text("data\n")
    (agent / "data_link").symlink_to("data.txt")  # a non-packaging symlink is kept
    tar_path = tmp_path / "workspace.tar.gz"
    subprocess.run(["tar", "czf", str(tar_path), "-C", str(agent), "."], check=True)

    command = app_module._upstream_restore_command(
        [], tar_path=str(tar_path), workspace=str(workspace), stage=str(tmp_path / "stage")
    )
    subprocess.run(["bash", "-c", command], check=True, capture_output=True)

    for name, text in (("setup.py", "image setup.py"), ("pyproject.toml", "image pyproject.toml")):
        assert not (workspace / name).is_symlink() and (workspace / name).read_text().strip() == text
    assert not (workspace / "requirements.txt").is_symlink()
    assert (workspace / "requirements.txt").read_text().strip() == "image requirements"
    assert not (workspace / "pkg" / "setup.cfg").exists() and not (workspace / "pkg" / "setup.cfg").is_symlink()
    assert (workspace / "data_link").is_symlink() and (workspace / "data.txt").read_text() == "data\n"


# --- the seeded spec ---------------------------------------------------------------------------------


async def _seed_and_capture_uploads(task_assets: Path, monkeypatch: MonkeyPatch, mode: str, workdir: str) -> dict:
    (task_assets / "example-task" / "workdir.txt").write_text(workdir, encoding="utf-8")
    config = _config(task_assets).model_copy(update={"agent_network_mode": mode})
    server = NL2RepoBenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    uploads: dict[str, tuple[str, int]] = {}

    async def upload(local: Path, remote: str) -> None:
        uploads[remote] = (Path(local).read_text(), stat.S_IMODE(Path(local).stat().st_mode))

    sandbox = AsyncMock()
    sandbox.exec.return_value = _exec_result(0)
    sandbox.serialize.return_value = {"sandbox_id": "agent-sandbox"}
    sandbox.upload.side_effect = upload
    monkeypatch.setattr(app_module, "get_global_config_dict", _global_config)
    monkeypatch.setattr(app_module, "resolve_provider_config", lambda *_: {"opensandbox": {}})
    monkeypatch.setattr(app_module, "resolve_provider_metadata", lambda *_: {})
    monkeypatch.setattr(app_module, "AsyncSandbox", MagicMock(return_value=sandbox))
    request = MagicMock()
    request.session = {SESSION_ID_KEY: "test-session"}
    await server.seed_session(request, NL2RepoBenchSeedSessionRequest.model_validate(_request()))
    return uploads


@pytest.mark.parametrize("workdir", ["/workspace", "/workspace/repo/nested"])
async def test_the_seeded_spec_is_world_readable_and_exact_when_the_network_is_open(
    task_assets: Path, monkeypatch: MonkeyPatch, workdir: str
) -> None:
    spec = (task_assets / "example-task" / "start.md").read_text()

    uploads = await _seed_and_capture_uploads(task_assets, monkeypatch, "official", workdir)

    text, mode = uploads[f"{workdir}/start.md"]
    assert text == spec  # exactly the upstream spec, nothing added
    assert mode & 0o044 == 0o044  # group and others can read it: a non-root agent can too
    assert mode & 0o022 == 0  # and nobody else can write it


async def test_only_the_no_internet_mode_tells_the_agent_there_is_no_internet(
    task_assets: Path, monkeypatch: MonkeyPatch
) -> None:
    spec = (task_assets / "example-task" / "start.md").read_text()
    for open_mode in ("official", "block_target", "block_target_v2"):
        uploads = await _seed_and_capture_uploads(task_assets, monkeypatch, open_mode, "/workspace")
        assert "no internet" not in uploads["/workspace/start.md"][0].lower()

    uploads = await _seed_and_capture_uploads(task_assets, monkeypatch, "no_internet_preinstalled", "/workspace")

    text, mode = uploads["/workspace/start.md"]
    assert text.startswith(spec.rstrip()) and text.rstrip().endswith(app_module.OFFLINE_SPEC_NOTICE)
    assert mode & 0o044 == 0o044


# --- review: the session is read inside the lock; Docker outages; failed stops stay owned ------------------------


async def test_a_verify_that_arrives_while_a_close_is_stopping_the_sandbox_is_masked_as_lost(
    task_assets: Path, monkeypatch: MonkeyPatch
) -> None:
    server, request, create_sandbox = await _seeded_server(task_assets, monkeypatch)
    agent_sandbox = server._agent_sessions["test-session"].sandbox
    stopping, finish_stop = asyncio.Event(), asyncio.Event()
    collected: list[Any] = []

    async def stop() -> None:
        stopping.set()
        await finish_stop.wait()  # the close yields here while it holds the session lock

    async def collect(sandbox: Any, workdir: str = "/workspace") -> tuple:
        collected.append(sandbox)
        return b"tar-bytes", None

    agent_sandbox.stop.side_effect = stop
    monkeypatch.setattr(server, "_collect_workspace", collect)
    close = ResourcesCloseSessionRequest(resources_session_id="test-session", episode_id=EpisodeId(rollout_id="r"))
    closing = asyncio.create_task(server.close_resources_session(close))
    await stopping.wait()

    verifying = asyncio.create_task(server.verify(request, NL2RepoBenchVerifyRequest.model_validate(_request())))
    await asyncio.sleep(0.05)
    assert not verifying.done()  # verify waits for the lock; it has not looked at the table yet
    finish_stop.set()
    await closing
    response = await verifying

    assert response.mask_sample is True and response.failure_kind == "session_lost"
    assert response.reward == 0.0
    assert collected == []  # no collection through the stopped handle
    assert create_sandbox.await_count == 1  # the agent sandbox only: no verifier was allocated


def _docker_download_sandbox(monkeypatch: MonkeyPatch, tmp_path: Path, stderr: str) -> AsyncMock:
    """An agent sandbox whose tar command succeeds and whose download goes through the real Docker provider's
    download_file, with `docker cp` failing with `stderr`."""
    from nemo_gym.sandbox.providers.docker import provider as docker_provider

    monkeypatch.setattr(docker_provider, "_require_docker", lambda: "/usr/bin/docker")
    provider = docker_provider.DockerProvider(exec={"exec_shell": "sh"})

    async def run(argv: list[str], *, timeout_s: float | None, stdin: bytes | None = None) -> tuple[int, str, str]:
        return 1, "", stderr

    monkeypatch.setattr(provider, "_run", run)
    handle = SandboxHandle(
        sandbox_id="nemo-gym-x",
        provider_name="docker",
        raw=docker_provider._DockerContainer(name="nemo-gym-x", image="img", shell="sh", env={}, published_ports=()),
    )
    sandbox = AsyncMock()
    sandbox.serialize.return_value = {"sandbox_id": "nemo-gym-x"}
    sandbox.exec.return_value = _collected(10)  # the tar command succeeded and the archive is in the sandbox

    async def download(remote: str, local: Path) -> None:
        await provider.download_file(handle, remote, Path(local))

    sandbox.download.side_effect = download
    return sandbox


async def test_a_docker_daemon_outage_during_the_download_masks_the_row_and_allocates_no_verifier(
    task_assets: Path, tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    server, request, create_sandbox = await _seeded_server(task_assets, monkeypatch)
    agent_sandbox = _docker_download_sandbox(
        monkeypatch, tmp_path, "Cannot connect to the Docker daemon at unix:///var/run/docker.sock."
    )
    server._agent_sessions["test-session"].sandbox = agent_sandbox
    # _seeded_server stubs _collect_workspace; these tests exercise the real one.
    monkeypatch.setattr(server, "_collect_workspace", type(server)._collect_workspace.__get__(server))
    verify_task_requests = create_sandbox.await_count

    response = await server.verify(request, NL2RepoBenchVerifyRequest.model_validate(_request()))

    assert response.mask_sample is True and response.reward == 0.0
    assert response.failure_kind == PROVIDER_UNAVAILABLE
    assert create_sandbox.await_count == verify_task_requests  # no verifier sandbox allocated


async def test_a_download_that_fails_after_the_archive_was_confirmed_is_masked(
    task_assets: Path, tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    server, request, create_sandbox = await _seeded_server(task_assets, monkeypatch)
    agent_sandbox = _docker_download_sandbox(
        monkeypatch,
        tmp_path,
        "Error response from daemon: Could not find the file /tmp/workspace.tar.gz in container nemo-gym-x",
    )
    server._agent_sessions["test-session"].sandbox = agent_sandbox
    # _seeded_server stubs _collect_workspace; these tests exercise the real one.
    monkeypatch.setattr(server, "_collect_workspace", type(server)._collect_workspace.__get__(server))

    response = await server.verify(request, NL2RepoBenchVerifyRequest.model_validate(_request()))

    # The collect command already confirmed a regular archive in the sandbox (a missing one fails there and is scored),
    # so a transfer that then fails is this server's: masked, never a scored 0.
    assert response.mask_sample is True and response.failure_kind == "verifier_error"
    assert "docker cp download" in (response.verifier_error or "")
    assert create_sandbox.await_count == 1


@pytest.mark.parametrize("failing", ["upload", "serialize"])
async def test_a_setup_failure_whose_stop_also_fails_is_stopped_at_shutdown(
    task_assets: Path, monkeypatch: MonkeyPatch, failing: str
) -> None:
    server = NL2RepoBenchResourcesServer(config=_config(task_assets), server_client=MagicMock(spec=ServerClient))
    sandbox = AsyncMock()
    sandbox.exec.return_value = _exec_result(0)
    setattr(getattr(sandbox, failing), "side_effect", RuntimeError(f"{failing} failed"))
    sandbox.stop.side_effect = [RuntimeError("provider down"), None]  # the first stop fails
    monkeypatch.setattr(app_module, "get_global_config_dict", _global_config)
    monkeypatch.setattr(app_module, "resolve_provider_config", lambda *_: {"opensandbox": {}})
    monkeypatch.setattr(app_module, "resolve_provider_metadata", lambda *_: {})
    monkeypatch.setattr(app_module, "AsyncSandbox", MagicMock(return_value=sandbox))
    request = MagicMock()
    request.session = {SESSION_ID_KEY: "test-session"}

    with pytest.raises(RuntimeError, match=f"{failing} failed"):
        await server.seed_session(request, NL2RepoBenchSeedSessionRequest.model_validate(_request()))

    assert server._agent_sessions == {}  # no session was published...
    assert sandbox.stop.await_count == 1  # ...and the one stop failed, so nothing has stopped it yet
    await server.shutdown()  # but the server still owns it
    assert sandbox.stop.await_count == 2
    assert server._unstopped_sandboxes == {}


async def test_a_verifier_whose_first_stop_fails_keeps_its_score_and_is_stopped_at_shutdown(
    task_assets: Path, monkeypatch: MonkeyPatch
) -> None:
    server, request, create_sandbox = await _seeded_server(task_assets, monkeypatch)
    grading = _grading_sandbox()
    grading.stop.side_effect = [RuntimeError("provider down"), None]
    create_sandbox.side_effect = [grading]

    response = await server.verify(request, NL2RepoBenchVerifyRequest.model_validate(_request()))

    assert response.reward == 1.0 and not getattr(response, "mask_sample", False)  # the score survives
    assert grading.stop.await_count == 1
    await server.shutdown()
    assert grading.stop.await_count == 2  # the failed stop was retried
    assert server._unstopped_sandboxes == {}
