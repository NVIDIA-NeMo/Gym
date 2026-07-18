# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_SCRIPT = REPO_ROOT / "benchmarks/osworld/tools/run_multienv_osworld_agent.sh"
OMNI_RUN_SCRIPT = REPO_ROOT / "benchmarks/osworld/tools/run_omni_mini_vllm.sh"
PREFLIGHT_SCRIPT = REPO_ROOT / "benchmarks/osworld/tools/preflight_osworld_run.py"
HOST_CHECK_SCRIPT = REPO_ROOT / "benchmarks/osworld/tools/check_host_prerequisites.sh"
VM_PREPARE_SCRIPT = REPO_ROOT / "benchmarks/osworld/tools/prepare_osworld_vm.sh"


def _read_run_env(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def _fake_uv(tmp_path: Path) -> Path:
    uv_bin = tmp_path / "bin" / "uv"
    uv_bin.parent.mkdir(exist_ok=True)
    uv_bin.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    uv_bin.chmod(0o755)
    return uv_bin


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
    uv_bin = _fake_uv(tmp_path)
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


def test_multienv_dry_run_writes_complete_terminal_lifecycle(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "RUN_DIR": str(run_dir),
            "RUN_ATTEMPT_ID": "fresh-test",
            "RUNNER_NAME": "gym_pyautogui",
            "RECORD_VIDEO": "0",
            "TASK_ARTIFACTS": "0",
            "OSWORLD_ENABLE_PROXY": "0",
            "PROXY_CONFIG_FILE": "",
            "UV_BIN": str(_fake_uv(tmp_path)),
        }
    )

    subprocess.run(
        ["bash", str(RUN_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert (run_dir / "started_at.txt").read_text().strip()
    assert (run_dir / "finished_at.txt").read_text().strip()
    assert (run_dir / "launcher.pid").read_text().strip().isdigit()
    assert (run_dir / "exit_code.txt").read_text() == "0\n"
    resolved = (run_dir / "resolved-command.log").read_text(encoding="utf-8")
    assert "--- ng_run command ---" in resolved
    assert "--- ng_collect_rollouts command ---" in resolved
    run_env = _read_run_env(run_dir / "run.env")
    assert run_env["RUN_ATTEMPT_ID"] == "fresh-test"
    assert run_env["RUN_LIFECYCLE_DIR"] == str(run_dir)
    assert run_env["MATERIALIZED_INPUT_JSONL"] == str(run_dir / "rollouts_materialized_inputs.jsonl")
    assert run_env["OSWORLD_ENABLE_PROXY"] == "0"
    assert run_env["PROXY_CONFIG_CONFIGURED"] == "0"


def test_multienv_proxy_switch_validates_and_records_non_secret_provenance(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    proxy_path = tmp_path / "proxy.json"
    raw = b'[{"host":"proxy.example.com","port":3128}]\n'
    proxy_path.write_bytes(raw)
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "RUN_DIR": str(run_dir),
            "RUNNER_NAME": "gym_pyautogui",
            "RECORD_VIDEO": "0",
            "TASK_ARTIFACTS": "0",
            "OSWORLD_ENABLE_PROXY": "1",
            "PROXY_CONFIG_FILE": str(proxy_path),
            "PYTHON_BIN": sys.executable,
            "UV_BIN": str(_fake_uv(tmp_path)),
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
    assert run_env["OSWORLD_ENABLE_PROXY"] == "1"
    assert run_env["PROXY_CONFIG_FILE"] == str(proxy_path)
    assert run_env["PROXY_CONFIG_CONFIGURED"] == "1"
    assert run_env["PROXY_CONFIG_SHA256"] == hashlib.sha256(raw).hexdigest()
    assert run_env["PROXY_CONFIG_ENTRY_COUNT"] == "1"
    resolved = (run_dir / "resolved-command.log").read_text(encoding="utf-8")
    assert "osworld_agent.enable_proxy=true" in resolved
    assert f"osworld_agent.proxy_config_file={proxy_path}" in resolved
    assert "proxy:      enabled=1 configured=1 entries=1" in completed.stdout


def test_multienv_proxy_enable_requires_a_config_and_writes_terminal_markers(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    env = os.environ.copy()
    env.pop("PROXY_CONFIG_FILE", None)
    env.update(
        {
            "DRY_RUN": "1",
            "RUN_DIR": str(run_dir),
            "OSWORLD_ENABLE_PROXY": "1",
            "UV_BIN": str(_fake_uv(tmp_path)),
        }
    )

    completed = subprocess.run(
        ["bash", str(RUN_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "PROXY_CONFIG_FILE is required" in completed.stderr
    assert (run_dir / "finished_at.txt").is_file()
    assert (run_dir / "exit_code.txt").read_text() == "2\n"


def test_multienv_refuses_to_mutate_an_existing_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    sentinel = run_dir / "owned-by-previous-run.txt"
    sentinel.write_text("unchanged\n", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "RUN_DIR": str(run_dir),
            "RUN_ATTEMPT_ID": "must-not-exist",
            "RUNNER_NAME": "gym_pyautogui",
            "UV_BIN": str(_fake_uv(tmp_path)),
        }
    )

    completed = subprocess.run(
        ["bash", str(RUN_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "immutable run already exists" in completed.stderr
    assert sentinel.read_text(encoding="utf-8") == "unchanged\n"
    assert sorted(path.name for path in run_dir.iterdir()) == [sentinel.name]


def test_multienv_preflight_failure_still_writes_terminal_markers(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    failing_python = tmp_path / "preflight-failure"
    failing_python.write_text("#!/usr/bin/env bash\nexit 7\n", encoding="utf-8")
    failing_python.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "RUN_DIR": str(run_dir),
            "RUN_ATTEMPT_ID": "failed-preflight",
            "RUNNER_NAME": "prompt_agent",
            "PYTHON_BIN": str(failing_python),
            "PREFLIGHT_ONLY": "1",
            "RECORD_VIDEO": "0",
            "TASK_ARTIFACTS": "0",
            "UV_BIN": str(_fake_uv(tmp_path)),
        }
    )

    completed = subprocess.run(
        ["bash", str(RUN_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 7
    assert (run_dir / "started_at.txt").is_file()
    assert (run_dir / "finished_at.txt").is_file()
    assert (run_dir / "exit_code.txt").read_text() == "7\n"
    assert (run_dir / "resolved-command.log").is_file()


def test_multienv_resume_uses_a_new_attempt_directory_and_requires_prior_failure(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "rollouts.jsonl").write_text("", encoding="utf-8")
    (run_dir / "rollouts_materialized_inputs.jsonl").write_text("", encoding="utf-8")
    (run_dir / "started_at.txt").write_text("2026-07-13T00:00:00Z\n", encoding="utf-8")
    (run_dir / "finished_at.txt").write_text("2026-07-13T01:00:00Z\n", encoding="utf-8")
    (run_dir / "exit_code.txt").write_text("1\n", encoding="utf-8")
    original_terminal_files = {
        name: (run_dir / name).read_bytes() for name in ("started_at.txt", "finished_at.txt", "exit_code.txt")
    }
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "RESUME_FROM_CACHE": "1",
            "RUN_DIR": str(run_dir),
            "RUN_ATTEMPT_ID": "resume-1",
            "RUNNER_NAME": "gym_pyautogui",
            "RECORD_VIDEO": "0",
            "TASK_ARTIFACTS": "0",
            "UV_BIN": str(_fake_uv(tmp_path)),
        }
    )

    subprocess.run(
        ["bash", str(RUN_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    attempt_dir = run_dir / "resume-attempts" / "resume-1"
    assert (attempt_dir / "exit_code.txt").read_text() == "0\n"
    assert _read_run_env(attempt_dir / "run.env")["RUN_LIFECYCLE_DIR"] == str(attempt_dir)
    assert "+resume_from_cache=true" in (attempt_dir / "resolved-command.log").read_text()
    assert {name: (run_dir / name).read_bytes() for name in original_terminal_files} == original_terminal_files

    env["RUN_ATTEMPT_ID"] = "resume-2"
    completed = subprocess.run(
        ["bash", str(RUN_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "refusing to resume a successfully completed run" in completed.stderr
    assert not (run_dir / "resume-attempts" / "resume-2").exists()


def test_omni_configs_declare_adapter_agent_class(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text('{"verifier_metadata":{"task_id":"task-1","osworld_task":{"id":"task-1"}}}\n')
    configs = ",".join(
        [
            "responses_api_agents/osworld_agent/configs/osworld_agent.yaml",
            "benchmarks/osworld/configs/osworld_agent_omni_mini.yaml",
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
            "benchmarks/osworld/configs/osworld_agent_omni_mini.yaml",
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
            "benchmarks/osworld/configs/osworld_agent_pointer.yaml",
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
