# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline tests for episode-code substitution, execution, and schemas."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_gym.server_utils import ServerClient
from resources_servers.cuagym import app as cuagym_app
from resources_servers.cuagym import browser_worker as cuagym_browser_worker
from resources_servers.cuagym.episode_code import (
    SID_PLACEHOLDER,
    app_port,
    app_url,
    parse_reward,
    placeholder_values,
    run_code,
    substitute,
)
from resources_servers.cuagym.hub_apps import APP_DIRS, PLACEHOLDER_MAP
from resources_servers.cuagym.schemas import (
    CUAGYM_EVAL_STUB,
    CuaGymResourcesServerConfig,
    task_info_from_row,
)
from resources_servers.webarena.schemas import WebArenaTaskRow


HUB = "http://198.51.100.7"


def test_app_ports_are_contiguous_and_complete() -> None:
    assert len(APP_DIRS) == 98
    assert APP_DIRS == sorted(APP_DIRS)
    assert app_port(APP_DIRS[0], 8000) == 8000
    assert app_port(APP_DIRS[-1], 8000) == 8097


def test_placeholders_resolve_to_deployed_apps() -> None:
    values = placeholder_values(HUB, 8000)
    assert values, "placeholder map must not be empty"
    for placeholder, value in values.items():
        spec = PLACEHOLDER_MAP[placeholder]
        port = 8000 + APP_DIRS.index(spec["app_dir"])
        if spec["kind"] == "host":
            assert value == f"198.51.100.7:{port}"
        else:
            assert value == f"{HUB}:{port}"


def test_unknown_app_dir_raises() -> None:
    with pytest.raises(ValueError):
        app_port("definitely_not_a_mock", 8000)


def test_substitute_fills_sid_and_urls() -> None:
    code = f"sid = \"{SID_PLACEHOLDER}\"\nBASE_URL = '__CUA_GYM_SLACK_URL__'\nprint(BASE_URL, sid)\n"
    out = substitute(code, sid="sid-123", hub_base_url=HUB, base_port=8000)
    slack_port = 8000 + APP_DIRS.index("slack_mock")
    assert 'sid = "sid-123"' in out
    assert f"{HUB}:{slack_port}" in out
    assert "__CUA_GYM_" not in out
    assert SID_PLACEHOLDER not in out


def test_substitute_rejects_unresolved_placeholders() -> None:
    code = f'sid = "{SID_PLACEHOLDER}"\nURL = "__CUA_GYM_NOT_A_REAL_APP_URL__"\n'
    with pytest.raises(ValueError, match="unresolved"):
        substitute(code, sid="sid-123", hub_base_url=HUB, base_port=8000)


def test_run_code_executes_source_from_stdin(tmp_path: Path) -> None:
    result = run_code("import sys\nprint('REWARD: 0.5')\n", timeout_seconds=30)
    assert result.returncode == 0
    assert parse_reward(result.stdout) == 0.5
    # No files are involved in sid handoff or execution.
    assert list(tmp_path.iterdir()) == []


def test_run_code_does_not_leak_server_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEMO_GYM_CONFIG_DICT", '{"policy_api_key": "super-secret"}')
    monkeypatch.setenv("SOME_CLUSTER_TOKEN", "also-secret")
    probe = "import os\nprint('LEAK' if 'NEMO_GYM_CONFIG_DICT' in os.environ or 'SOME_CLUSTER_TOKEN' in os.environ else 'CLEAN')\n"
    result = run_code(probe, timeout_seconds=30)
    assert "CLEAN" in result.stdout, result.stdout


def test_run_code_runs_in_scratch_directory() -> None:
    probe = "import os, pathlib\nopen('side_effect.txt', 'w').write('x')\nprint(os.getcwd())\n"
    result = run_code(probe, timeout_seconds=30)
    assert result.returncode == 0
    cwd = result.stdout.strip().splitlines()[-1]
    # The scratch dir is removed with the run, so relative writes cannot
    # accumulate in the server's working tree.
    assert not Path(cwd).exists()


def test_run_code_passes_extra_env() -> None:
    probe = "import os\nprint(os.environ.get('CUA_GYM_AGENT_ANSWER', ''))\n"
    result = run_code(probe, timeout_seconds=30, extra_env={"CUA_GYM_AGENT_ANSWER": "hello"})
    assert "hello" in result.stdout


def test_best_effort_reset_hub_state(monkeypatch: pytest.MonkeyPatch) -> None:
    response = MagicMock()
    post = MagicMock(return_value=response)
    monkeypatch.setattr(cuagym_browser_worker.requests, "post", post)

    cuagym_browser_worker._best_effort_reset_hub_state(HUB, "sid-123")

    post.assert_called_once_with(
        f"{HUB}/post",
        params={"sid": "sid-123"},
        json={"action": "reset"},
        timeout=cuagym_browser_worker._HUB_STATE_RESET_TIMEOUT_SECONDS,
    )
    response.raise_for_status.assert_called_once_with()


def test_best_effort_reset_hub_state_ignores_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post = MagicMock(side_effect=cuagym_browser_worker.requests.Timeout)
    monkeypatch.setattr(cuagym_browser_worker.requests, "post", post)

    cuagym_browser_worker._best_effort_reset_hub_state(HUB, "sid-123")


@pytest.mark.parametrize("status", ["success", "failure"])
def test_terminate_properly_reward_adds_bonus(status: str) -> None:
    assert (
        cuagym_browser_worker._apply_terminate_properly_reward(
            0.0,
            status=status,
            terminate_properly_reward=0.1,
        )
        == 0.1
    )
    assert (
        cuagym_browser_worker._apply_terminate_properly_reward(
            0.7,
            status=status,
            terminate_properly_reward=0.1,
        )
        == pytest.approx(0.8)
    )


def test_terminate_properly_reward_requires_valid_terminate() -> None:
    assert (
        cuagym_browser_worker._apply_terminate_properly_reward(
            0.7,
            status=None,
            terminate_properly_reward=0.1,
        )
        == 0.0
    )


@pytest.mark.parametrize(
    ("stdout", "expected_reward"),
    [
        ("REWARD: 0.0\n", 0.1),
        ("REWARD: 0.7\n", 0.8),
        ("reward unavailable\n", 0.1),
    ],
)
def test_evaluate_adds_terminate_reward_bonus(
    monkeypatch: pytest.MonkeyPatch,
    stdout: str,
    expected_reward: float,
) -> None:
    worker_class = cuagym_browser_worker.CuaGymBrowserWorker.__ray_metadata__.modified_class
    worker = object.__new__(worker_class)
    worker.config = MagicMock(
        hub_base_url=HUB,
        hub_base_port=8000,
        reward_timeout_seconds=120,
        terminate_properly_reward=0.1,
    )
    worker.reward_code = 'print("REWARD")'
    worker.sid = "sid-123"
    monkeypatch.setattr(
        cuagym_browser_worker,
        "run_code",
        MagicMock(
            return_value=MagicMock(
                returncode=0,
                stdout=stdout,
                stderr="",
            )
        ),
    )

    result = worker.evaluate(status="success", answer=None)

    assert result["reward"] == pytest.approx(expected_reward)


def test_evaluate_without_valid_terminate_skips_reward_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_class = cuagym_browser_worker.CuaGymBrowserWorker.__ray_metadata__.modified_class
    worker = object.__new__(worker_class)
    worker.config = MagicMock(
        hub_base_url=HUB,
        hub_base_port=8000,
        reward_timeout_seconds=120,
        terminate_properly_reward=0.1,
    )
    worker.reward_code = 'print("REWARD: 0.7")'
    worker.sid = "sid-123"
    run_code_mock = MagicMock()
    monkeypatch.setattr(cuagym_browser_worker, "run_code", run_code_mock)

    result = worker.evaluate(status=None, answer=None)

    assert result["reward"] == 0.0
    assert "reward evaluation skipped" in result["evaluator_details"]["message"]
    run_code_mock.assert_not_called()


def test_evaluate_applies_terminate_reward_after_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_class = cuagym_browser_worker.CuaGymBrowserWorker.__ray_metadata__.modified_class
    worker = object.__new__(worker_class)
    worker.config = MagicMock(
        hub_base_url=HUB,
        hub_base_port=8000,
        reward_timeout_seconds=120,
        terminate_properly_reward=0.1,
    )
    worker.reward_code = 'print("REWARD")'
    worker.sid = "sid-123"
    monkeypatch.setattr(
        cuagym_browser_worker,
        "run_code",
        MagicMock(
            side_effect=subprocess.TimeoutExpired(
                cmd="reward",
                timeout=120,
            )
        ),
    )

    result = worker.evaluate(status="success", answer=None)

    assert result["reward"] == 0.1
    assert "timed out" in result["evaluator_details"]["message"]
    assert "terminate reward bonus applied" in result["evaluator_details"]["message"]


def test_generation_node_ids_resolve_from_ray_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "NEMO_RL_GENERATION_NODE_IPS",
        json.dumps(["10.0.0.2", "10.0.0.1", "10.0.0.2"]),
    )
    monkeypatch.setattr(
        cuagym_app.ray,
        "nodes",
        lambda: [
            {"Alive": True, "NodeManagerAddress": "10.0.0.1", "NodeID": "node-1"},
            {"Alive": True, "NodeManagerAddress": "10.0.0.2", "NodeID": "node-2"},
            {"Alive": False, "NodeManagerAddress": "10.0.0.3", "NodeID": "node-3"},
        ],
    )

    assert cuagym_app._generation_node_ids() == ("node-2", "node-1")


def test_generation_node_ids_reject_missing_ray_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEMO_RL_GENERATION_NODE_IPS", json.dumps(["10.0.0.9"]))
    monkeypatch.setattr(cuagym_app.ray, "nodes", lambda: [])

    with pytest.raises(RuntimeError, match="not alive"):
        cuagym_app._generation_node_ids()


def test_model_post_init_preserves_parent_lifecycle_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cuagym_app, "endpoint_overrides", MagicMock(return_value={}))
    pool = MagicMock()
    monkeypatch.setattr(cuagym_app, "BrowserActorPool", MagicMock(return_value=pool))
    config = CuaGymResourcesServerConfig(
        name="cuagym",
        host="127.0.0.1",
        port=12345,
        entrypoint="app.py",
        num_workers=1,
        initial_browser_workers=0,
        max_browser_workers=1,
    )

    server = cuagym_app.CuaGymResourcesServer(
        config=config,
        server_client=MagicMock(spec=ServerClient),
    )

    assert server._pool is pool
    assert server._deduplicator is not None
    assert server._terminal_operations is not None


def test_spread_gen_places_browser_actors_round_robin(monkeypatch: pytest.MonkeyPatch) -> None:
    node_1 = "1" * 56
    node_2 = "2" * 56
    monkeypatch.setenv("NEMO_RL_GENERATION_NODE_IPS", json.dumps(["10.0.0.1", "10.0.0.2"]))
    monkeypatch.setattr(
        cuagym_app.ray,
        "nodes",
        lambda: [
            {"Alive": True, "NodeManagerAddress": "10.0.0.1", "NodeID": node_1},
            {"Alive": True, "NodeManagerAddress": "10.0.0.2", "NodeID": node_2},
        ],
    )
    monkeypatch.setattr(cuagym_app, "BrowserActorPool", MagicMock())
    config = CuaGymResourcesServerConfig.model_validate(
        {
            "name": "cuagym",
            "host": "127.0.0.1",
            "port": 12345,
            "entrypoint": "app.py",
            "num_workers": 1,
            "node_placement_strategy": "spread_gen",
        }
    )
    server = cuagym_app.CuaGymResourcesServer.model_construct(config=config)
    actor_builder = MagicMock()
    actor_builder.remote.side_effect = ["worker-1", "worker-2", "worker-3"]
    options = MagicMock(return_value=actor_builder)
    monkeypatch.setattr(cuagym_app.CuaGymBrowserWorker, "options", options)

    assert [server._create_worker() for _ in range(3)] == ["worker-1", "worker-2", "worker-3"]
    strategies = [call.kwargs["scheduling_strategy"] for call in options.call_args_list]
    assert [strategy.node_id for strategy in strategies] == [node_1, node_2, node_1]
    assert all(not strategy.soft for strategy in strategies)


def test_timeout_config_must_fit_inside_browser_budgets() -> None:
    base = {
        "name": "cuagym",
        "host": "127.0.0.1",
        "port": 12345,
        "entrypoint": "app.py",
        "num_workers": 1,
        "hub_base_url": HUB,
        "browser_operation_timeout_seconds": 180.0,
        "browser_evaluation_timeout_seconds": 300.0,
    }
    ok = CuaGymResourcesServerConfig.model_validate(
        {**base, "setup_timeout_seconds": 90, "reward_timeout_seconds": 120}
    )
    assert ok.hub_base_url == HUB
    assert ok.node_placement_strategy == "spread_all"
    assert (
        CuaGymResourcesServerConfig.model_validate(
            {**base, "node_placement_strategy": "spread_gen"}
        ).node_placement_strategy
        == "spread_gen"
    )
    with pytest.raises(ValueError, match="node_placement_strategy"):
        CuaGymResourcesServerConfig.model_validate({**base, "node_placement_strategy": "invalid"})
    with pytest.raises(ValueError, match="setup_timeout_seconds"):
        CuaGymResourcesServerConfig.model_validate({**base, "setup_timeout_seconds": 180})
    with pytest.raises(ValueError, match="reward_timeout_seconds"):
        CuaGymResourcesServerConfig.model_validate(
            {**base, "setup_timeout_seconds": 90, "reward_timeout_seconds": 300}
        )
    # Defaults alone must already satisfy the inherited browser budgets.
    defaults = CuaGymResourcesServerConfig.model_validate(
        {k: v for k, v in base.items() if not k.startswith("browser_")}
    )
    assert defaults.setup_timeout_seconds < defaults.browser_operation_timeout_seconds
    assert defaults.reward_timeout_seconds < defaults.browser_evaluation_timeout_seconds
    assert defaults.terminate_properly_reward == 0.0
    with pytest.raises(ValueError, match="terminate_properly_reward"):
        CuaGymResourcesServerConfig.model_validate({**base, "terminate_properly_reward": -0.1})
    with pytest.raises(ValueError, match="terminate_properly_reward"):
        CuaGymResourcesServerConfig.model_validate({**base, "terminate_properly_reward": 1.1})


def test_run_code_reports_failures() -> None:
    result = run_code("raise RuntimeError('boom')\n", timeout_seconds=30)
    assert result.returncode != 0
    assert "boom" in result.stderr


def test_parse_reward() -> None:
    assert parse_reward("blah\nREWARD: 0.85\n") == 0.85
    assert parse_reward("REWARD: 0.2\nlater\nREWARD: 1.0") == 1.0
    assert parse_reward("REWARD: 7") == 1.0  # clamped
    assert parse_reward("REWARD: -1") == 0.0  # clamped
    assert parse_reward("no reward here") is None
    assert parse_reward("") is None


def test_task_row_round_trip() -> None:
    payload = {
        "task_id": "bundle-0001",
        "dataset": "cuagym",
        "dataset_version": "v1",
        "sites": [APP_DIRS[0]],
        "start_urls": [],
        "intent": "do the thing",
        "eval": dict(CUAGYM_EVAL_STUB),
        "cuagym": {
            "bundle_id": "bundle-0001",
            "app_dir": APP_DIRS[0],
            "initial_setup": f'sid = "{SID_PLACEHOLDER}"\n',
            "eval_reward_code": f'sid = "{SID_PLACEHOLDER}"\nprint("REWARD: 0.0")\n',
        },
    }
    row = WebArenaTaskRow.model_validate(payload)
    info = task_info_from_row(row)
    assert info.bundle_id == "bundle-0001"
    assert info.app_dir == APP_DIRS[0]
    assert SID_PLACEHOLDER in info.eval_reward_code
    assert app_url(info.app_dir, HUB, 8000) == f"{HUB}:8000"


def test_example_rows_have_no_gui_launch_or_sid_files() -> None:
    """Guards the two conversion blockers: truncated setups and mangled sid reads.

    A setup whose GUI-launch block was stripped by deleting to end-of-file loses
    its trailing state POST, and a reward whose sid read was over-eagerly
    replaced loses its scoring logic. Both show up here as missing hub calls.
    """
    data = Path(__file__).resolve().parents[1] / "data/example.jsonl"
    rows = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    for raw in rows:
        info = task_info_from_row(WebArenaTaskRow.model_validate(raw["task_payload"]))
        setup, reward = info.initial_setup or "", info.eval_reward_code
        assert "launch_gui" not in setup and "google-chrome" not in setup
        assert "/tmp/task_web" not in setup + reward
        # Setup must still inject state, and reward must still read it back.
        assert "/post?sid=" in setup, f"{info.bundle_id}: setup lost its state injection"
        assert "?sid=" in reward, f"{info.bundle_id}: reward lost its state query"
        assert "REWARD:" in reward


def test_example_rows_are_self_contained() -> None:
    data = Path(__file__).resolve().parents[1] / "data/example.jsonl"
    rows = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    assert len(rows) == 5
    for raw in rows:
        row = WebArenaTaskRow.model_validate(raw["task_payload"])
        info = task_info_from_row(row)
        assert info.app_dir in APP_DIRS
        for code in (info.initial_setup, info.eval_reward_code):
            assert code, "episode code must be inlined in the row"
            assert SID_PLACEHOLDER in code
            assert "/tmp/task_web" not in code
            assert "launch_gui" not in code and "google-chrome" not in code
            compile(code, "<row-code>", "exec")
        assert "REWARD:" in info.eval_reward_code
