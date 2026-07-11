import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_SCRIPT = REPO_ROOT / "responses_api_agents/osworld_agent/scripts/run_multienv_osworld_agent.sh"
PREFLIGHT_SCRIPT = REPO_ROOT / "responses_api_agents/osworld_agent/scripts/preflight_osworld_run.py"


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
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "RUN_TAG": f"ray-path-test-{expect_shortened}",
            "RUN_DIR": str(run_dir),
            "RAY_TMPDIR": requested,
            "SERVER_VENV_ROOT": str(server_venv_root),
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
    assert f"++uv_venv_dir={server_venv_root}" in completed.stdout
    if expect_shortened:
        assert run_env["RAY_TMPDIR"].startswith("/tmp/ngray-")
        assert "RAY_TMPDIR is too long" in completed.stderr
    else:
        assert run_env["RAY_TMPDIR"] == requested
        assert completed.stderr == ""

    socket_probe = (
        f"{run_env['RAY_TMPDIR'].rstrip('/')}/ray/"
        "session_2099-12-31_23-59-59_999999_999999/sockets/plasma_store"
    )
    assert len(socket_probe) <= 107


def test_omni_configs_reference_importable_adapter_agents(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        '{"verifier_metadata":{"task_id":"task-1","osworld_task":{"id":"task-1"}}}\n'
    )
    configs = ",".join(
        [
            "responses_api_agents/osworld_agent/configs/osworld_agent.yaml",
            "responses_api_agents/osworld_agent/configs/osworld_agent_omni_mini.yaml",
        ]
    )

    completed = subprocess.run(
        [
            str(REPO_ROOT / ".venv/bin/python"),
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
    assert (
        "responses_api_agents.osworld_agent.adapter_agents.NemotronV3NanoOmniAgent"
        in completed.stdout
    )
