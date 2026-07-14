# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_SCRIPT = REPO_ROOT / "responses_api_agents/osworld_agent/scripts/run_multienv_osworld_agent.sh"
OMNI_RUN_SCRIPT = REPO_ROOT / "responses_api_agents/osworld_agent/scripts/run_omni_mini_vllm.sh"
PREFLIGHT_SCRIPT = REPO_ROOT / "responses_api_agents/osworld_agent/scripts/preflight_osworld_run.py"
HOST_CHECK_SCRIPT = REPO_ROOT / "responses_api_agents/osworld_agent/scripts/check_host_prerequisites.sh"
VM_PREPARE_SCRIPT = REPO_ROOT / "responses_api_agents/osworld_agent/scripts/prepare_osworld_vm.sh"


def _read_run_env(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


@pytest.mark.parametrize(
    ("requested", "expect_shortened"),
    [
        ("/tmp/ray-ok", False),
        (
            "/tmp/this-is-an-intentionally-very-long-ray-temporary-directory-"
            "that-would-overflow-the-linux-af-unix-socket-limit",
            True,
        ),
    ],
)
def test_multienv_ray_tmpdir_respects_unix_socket_limit(
    tmp_path: Path,
    requested: str,
    expect_shortened: bool,
) -> None:
    run_dir = tmp_path / "run"
    server_venv_root = tmp_path / "server-venvs"
    uv_bin = tmp_path / "bin" / "uv"
    uv_bin.parent.mkdir()
    uv_bin.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    uv_bin.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "RUN_TAG": f"ray-path-test-{expect_shortened}",
            "RUN_DIR": str(run_dir),
            "RAY_TMPDIR": requested,
            "SERVER_VENV_ROOT": str(server_venv_root),
            "UV_BIN": str(uv_bin),
        }
    )

    completed = subprocess.run(
        ["bash", str(RUN_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    run_env = _read_run_env(run_dir / "run.env")

    assert run_env["RAY_TMPDIR_REQUESTED"] == requested
    assert run_env["SERVER_VENV_ROOT"] == str(server_venv_root)
    assert run_env["UV_BIN"] == str(uv_bin)
    assert f"++uv_venv_dir={server_venv_root}" in completed.stdout
    assert f"uv:          {uv_bin}" in completed.stdout
    if expect_shortened:
        assert run_env["RAY_TMPDIR"].startswith("/tmp/ngray-")
        assert "RAY_TMPDIR is too long" in completed.stderr
    else:
        assert run_env["RAY_TMPDIR"] == requested
        assert completed.stderr == ""

    socket_probe = (
        f"{run_env['RAY_TMPDIR'].rstrip('/')}/ray/session_2099-12-31_23-59-59_999999_999999/sockets/plasma_store"
    )
    assert len(socket_probe) <= 107


def test_omni_configs_declare_adapter_agent_class(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text('{"verifier_metadata":{"task_id":"task-1","osworld_task":{"id":"task-1"}}}\n')
    configs = ",".join(
        [
            "responses_api_agents/osworld_agent/configs/osworld_agent.yaml",
            "responses_api_agents/osworld_agent/configs/osworld_agent_omni_mini.yaml",
        ]
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(PREFLIGHT_SCRIPT),
            "--config-paths",
            configs,
            "--input-jsonl",
            str(input_path),
            "--expected-rows",
            "1",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"preflight": "ok"' in completed.stdout
    assert "responses_api_agents.osworld_agent.adapter_agents.NemotronV3NanoOmniAgent" in completed.stdout


def test_multienv_preflight_does_not_require_prebuilt_agent_runtime(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text('{"verifier_metadata":{"task_id":"task-1","osworld_task":{"id":"task-1"}}}\n')
    run_dir = tmp_path / "run"
    uv_bin = tmp_path / "uv"
    uv_bin.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    uv_bin.chmod(0o755)
    configs = ",".join(
        [
            "responses_api_agents/osworld_agent/configs/osworld_agent.yaml",
            "responses_api_agents/osworld_agent/configs/osworld_agent_omni_mini.yaml",
        ]
    )
    env = os.environ.copy()
    env.update(
        {
            "RUN_DIR": str(run_dir),
            "INPUT_JSONL": str(input_path),
            "EXPECTED_INPUT_ROWS": "1",
            "CONFIG_PATHS": configs,
            "RUNNER_NAME": "nemotron_v3_nano_omni_agent",
            # ng_run owns this path and has not created it yet. The launcher
            # preflight must not require it before ng_run gets that chance.
            "SERVER_VENV_ROOT": str(tmp_path / "missing-server-venvs"),
            "PYTHON_BIN": sys.executable,
            "UV_BIN": str(uv_bin),
            "PREFLIGHT_ONLY": "1",
            "RECORD_VIDEO": "0",
            "TASK_ARTIFACTS": "0",
        }
    )

    completed = subprocess.run(
        ["bash", str(RUN_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    run_env = _read_run_env(run_dir / "run.env")
    assert run_env["SERVER_VENV_ROOT"] == str(tmp_path / "missing-server-venvs")
    assert '"preflight": "ok"' in completed.stdout
    assert "responses_api_agents.osworld_agent.adapter_agents.NemotronV3NanoOmniAgent" in completed.stdout


def test_pointer_launcher_preflight_does_not_import_runtime_package(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text('{"verifier_metadata":{"task_id":"task-1","osworld_task":{"id":"task-1"}}}\n')
    fake_package = tmp_path / "mm_agents"
    fake_package.mkdir()
    (fake_package / "__init__.py").write_text("", encoding="utf-8")
    (fake_package / "pointer.py").write_text(
        'raise AssertionError("launcher preflight must not import the runtime package")\n',
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path)
    configs = ",".join(
        [
            "responses_api_agents/osworld_agent/configs/osworld_agent.yaml",
            "responses_api_agents/osworld_agent/configs/osworld_agent_pointer.yaml",
        ]
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(PREFLIGHT_SCRIPT),
            "--config-paths",
            configs,
            "--input-jsonl",
            str(input_path),
            "--runner-name",
            "pointer_agent",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"preflight": "ok"' in completed.stdout
    assert "mm_agents.pointer.PointerAgent" in completed.stdout


def test_omni_runner_defaults_match_the_three_image_recipe(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "PREFLIGHT": "0",
            "RUN_DIR": str(run_dir),
            "SERVER_VENV_ROOT": str(tmp_path / "server-venvs"),
            # GitHub Actions defines RUNNER_NAME for its own worker. Pin the
            # adapter runner so this test exercises the public script default.
            "RUNNER_NAME": "nemotron_v3_nano_omni_agent",
        }
    )

    subprocess.run(
        ["bash", str(OMNI_RUN_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    run_env = _read_run_env(run_dir / "run.env")
    script = OMNI_RUN_SCRIPT.read_text(encoding="utf-8")
    assert run_env["RUNNER_NAME"] == "nemotron_v3_nano_omni_agent"
    assert run_env["MAX_OUTPUT_TOKENS"] == "4096"
    assert 'OMNI_MINI_PREFLIGHT_IMAGE_COUNT="${OMNI_MINI_PREFLIGHT_IMAGE_COUNT:-3}"' in script
    assert '--image-count "${OMNI_MINI_PREFLIGHT_IMAGE_COUNT}"' in script


@pytest.mark.parametrize("script", [HOST_CHECK_SCRIPT, VM_PREPARE_SCRIPT])
def test_public_host_setup_scripts_are_syntax_valid_and_portable(script: Path) -> None:
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_vm_prepare_script_pins_the_verified_image_identity() -> None:
    text = VM_PREPARE_SCRIPT.read_text(encoding="utf-8")
    assert "6bf667a852b3c307f61d9f09c42559351f45e0607e428b4997becf534cf4d313" in text  # pragma: allowlist secret
    assert "24460197888" in text
    assert "--continue-at -" in text
