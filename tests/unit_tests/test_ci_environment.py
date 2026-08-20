# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
FULL_TEST_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "full-test-suite.yml"
GITLAB_PIPELINE = REPO_ROOT / ".gitlab-ci.yml"
IS_RETRYABLE_FULL_SUITE_FAILURE = REPO_ROOT / "scripts" / "ci" / "is_retryable_full_suite_failure.sh"
RECLAIM_RUNNER_DISK = REPO_ROOT / "scripts" / "ci" / "reclaim_runner_disk.sh"
RETRY_FULL_TEST_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "retry-full-test-suite.yml"
SANITIZER = REPO_ROOT / "scripts" / "ci" / "sanitize_env.sh"
SERVER_TESTS = REPO_ROOT / "scripts" / "ci" / "server_tests.sh"
SETUP_DEV = REPO_ROOT / "scripts" / "ci" / "setup_dev.sh"
UNIT_TEST_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "unit-tests.yml"

BEHAVIOR_CHANGING_ENV = {
    "GYM_CI_DEV_VENV_DIR": "/tmp/injected-driver-venv",
    "SKIP": "ruff",
    "NEMO_GYM_EXTRA_ROOTS": "/tmp/external-gym",
    "NEMO_GYM_CONFIG_DICT": '{"search_dir": "/tmp/external-gym"}',
    "NEMO_GYM_ALLOW_PRERELEASE": "true",
    "PYTHONPATH": "/tmp/python",
    "PYTHONSAFEPATH": "1",
    "PYTEST_ADDOPTS": "-m injected-selection",
}


def _environment_after_sanitizing(stage: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(BEHAVIOR_CHANGING_ENV)
    env["GYM_CI_PRESERVED_SENTINEL"] = "preserved"
    command = f'source "{SANITIZER}"; gym_ci_sanitize_environment "{stage}"; env -0'
    result = subprocess.run(
        ["bash", "-c", command],
        check=True,
        capture_output=True,
        env=env,
    )
    return {
        key.decode(): value.decode()
        for item in result.stdout.split(b"\0")
        if item
        for key, value in [item.split(b"=", 1)]
    }


@pytest.mark.parametrize(
    ("stage", "removed"),
    [
        ("lint", {"SKIP"}),
        (
            "core",
            {"GYM_CI_DEV_VENV_DIR", "NEMO_GYM_EXTRA_ROOTS", "NEMO_GYM_CONFIG_DICT", "PYTHONPATH"},
        ),
        (
            "server",
            {
                "GYM_CI_DEV_VENV_DIR",
                "NEMO_GYM_EXTRA_ROOTS",
                "NEMO_GYM_CONFIG_DICT",
                "NEMO_GYM_ALLOW_PRERELEASE",
                "PYTHONPATH",
                "PYTHONSAFEPATH",
                "PYTEST_ADDOPTS",
            },
        ),
    ],
)
def test_ci_stage_removes_only_its_behavior_changing_environment(stage: str, removed: set[str]) -> None:
    sanitized = _environment_after_sanitizing(stage)

    assert removed.isdisjoint(sanitized)
    assert sanitized["GYM_CI_PRESERVED_SENTINEL"] == "preserved"
    for name, value in BEHAVIOR_CHANGING_ENV.items():
        if name not in removed:
            assert sanitized[name] == value


def test_ci_environment_sanitizer_rejects_unknown_stage() -> None:
    result = subprocess.run(
        ["bash", "-c", f'source "{SANITIZER}"; gym_ci_sanitize_environment unknown'],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "unknown Gym CI stage: unknown" in result.stderr


@pytest.mark.parametrize("venv_dir", ["relative-driver-venv", "/"])
def test_setup_dev_rejects_unsafe_driver_venv(venv_dir: str) -> None:
    env = os.environ.copy()
    env["GYM_CI_DEV_VENV_DIR"] = venv_dir

    result = subprocess.run([str(SETUP_DEV)], capture_output=True, text=True, env=env)

    assert result.returncode == 2
    assert f"GYM_CI_DEV_VENV_DIR must be an absolute non-root path: {venv_dir}" in result.stderr


def test_gitlab_adapter_selects_current_contract_version() -> None:
    expected_version = (REPO_ROOT / "scripts" / "ci" / "contract-version").read_text().strip()

    assert f'GYM_CI_CONTRACT_VERSION: "{expected_version}"' in GITLAB_PIPELINE.read_text()


def test_gitlab_adapter_selects_cpu_short_partition() -> None:
    assert 'GYM_SLURM_PARTITION: "cpu_short"' in GITLAB_PIPELINE.read_text()


@pytest.mark.parametrize(
    "failure",
    [
        "fatal: server certificate verification failed. CAfile: none CRLfile: none",
        "fatal: unable to access repository: Could not resolve host: github.com",
        "error: No space left on device (os error 28)",
        "Insufficient runner disk space: 1024 KiB available; 10485760 KiB required",
        "The hosted runner lost communication with the server",
    ],
)
def test_full_suite_failure_classifier_accepts_infrastructure_failures(tmp_path: Path, failure: str) -> None:
    log_path = tmp_path / "failed.log"
    log_path.write_text(failure)

    result = subprocess.run([str(IS_RETRYABLE_FULL_SUITE_FAILURE), str(log_path)], capture_output=True, text=True)

    assert result.returncode == 0
    assert "recognized runner or transport failure" in result.stdout


@pytest.mark.parametrize(
    "failure",
    [
        "FAILED tests/unit_tests/test_cli.py::test_command - AssertionError",
        "No matching distribution found for deterministic-package==1.2.3",
        "ruff check failed: unused import",
    ],
)
def test_full_suite_failure_classifier_rejects_deterministic_failures(tmp_path: Path, failure: str) -> None:
    log_path = tmp_path / "failed.log"
    log_path.write_text(failure)

    result = subprocess.run([str(IS_RETRYABLE_FULL_SUITE_FAILURE), str(log_path)], capture_output=True, text=True)

    assert result.returncode == 1
    assert "do not contain a recognized" in result.stdout


def test_full_suite_retry_is_bounded_and_reruns_only_failed_jobs() -> None:
    workflow = RETRY_FULL_TEST_WORKFLOW.read_text()

    assert "github.event.workflow_run.run_attempt == 1" in workflow
    assert "head_repository.full_name == github.repository" in workflow
    assert "head_branch == github.event.repository.default_branch" in workflow
    assert "is_retryable_full_suite_failure.sh" in workflow
    assert "rerun-failed-jobs" in workflow
    assert 'select(.conclusion == "failure" and .name != "Server suite")' in workflow
    assert "At least one failed job was deterministic" in workflow
    assert "actions: write" in workflow


def test_github_full_test_jobs_reclaim_disk_before_dependency_restore() -> None:
    for workflow_path, expected_jobs in [(FULL_TEST_WORKFLOW, 2), (UNIT_TEST_WORKFLOW, 2)]:
        workflow = workflow_path.read_text()
        assert workflow.count("run: ./scripts/ci/reclaim_runner_disk.sh") == expected_jobs
        assert workflow.count("Reclaim runner disk") == expected_jobs
        sections = workflow.split("- name: Reclaim runner disk")
        for section in sections[1:]:
            assert section.index("reclaim_runner_disk.sh") < section.index("Cache uv dependencies")


def test_runner_disk_reclamation_fails_fast_when_space_is_still_low(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(bin_dir / "sudo", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(
        bin_dir / "df",
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == "-Pk" ]]; then\n'
        "  printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\n/dev/root 1000 901 99 91%% /\\n'\n"
        "else\n"
        "  printf 'Filesystem Size Used Avail Use%% Mounted on\\n/dev/root 1G 901M 99M 91%% /\\n'\n"
        "fi\n",
    )
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "RUNNER_ENVIRONMENT": "github-hosted",
            "GITHUB_WORKSPACE": str(tmp_path),
            "GYM_CI_MIN_FREE_DISK_KB": "100",
        }
    )

    result = subprocess.run([str(RECLAIM_RUNNER_DISK)], capture_output=True, text=True, env=env)

    assert result.returncode == 1
    assert "Insufficient runner disk space: 99 KiB available; 100 KiB required" in result.stderr


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents)
    path.chmod(0o755)


def test_server_tests_propagates_absolute_cache_and_venv_roots(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    ci_dir = repo_root / "scripts" / "ci"
    shutil.copytree(REPO_ROOT / "scripts" / "ci", ci_dir)
    shutil.copy2(REPO_ROOT / ".python-version", repo_root / ".python-version")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture_path = tmp_path / "ng-test-all.args"

    _write_executable(
        bin_dir / "curl",
        """#!/usr/bin/env bash
cat <<'INSTALL'
set -eu
mkdir -p "${UV_UNMANAGED_INSTALL}"
cat > "${UV_UNMANAGED_INSTALL}/uv" <<'UV'
#!/usr/bin/env bash
set -eu
case "${1:-}" in
    --version) printf '%s\\n' 'uv 0.11.29' ;;
    cache) printf '%s\\n' "${UV_CACHE_DIR:-${HOME}/.cache/uv}" ;;
    venv)
        venv_dir="${@: -1}"
        mkdir -p "${venv_dir}/bin"
        : > "${venv_dir}/bin/activate"
        : > "${venv_dir}/bin/python"
        chmod +x "${venv_dir}/bin/python"
        ;;
    sync) ;;
    *) printf 'unexpected fake uv command: %s\\n' "$*" >&2; exit 2 ;;
esac
UV
chmod +x "${UV_UNMANAGED_INSTALL}/uv"
INSTALL
""",
    )
    _write_executable(
        bin_dir / "ng_test_all",
        """#!/usr/bin/env bash
set -eu
printf 'UV_CACHE_DIR=%s\\n' "${UV_CACHE_DIR}" > "${GYM_CI_CAPTURE}"
printf 'UV_LINK_MODE=%s\\n' "${UV_LINK_MODE:-}" >> "${GYM_CI_CAPTURE}"
printf 'GYM_CI_DEV_VENV_DIR=%s\\n' "${GYM_CI_DEV_VENV_DIR:-}" >> "${GYM_CI_CAPTURE}"
printf 'ARG=%s\\n' "$@" >> "${GYM_CI_CAPTURE}"
""",
    )

    node_local_root = tmp_path / "node-local"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "HOME": str(tmp_path / "home"),
            "UV_CACHE_DIR": "relative-cache",
            "GYM_CI_UV_VENV_DIR": str(node_local_root),
            "GYM_CI_CAPTURE": str(capture_path),
        }
    )
    subprocess.run([str(ci_dir / "server_tests.sh"), "2", "8"], check=True, env=env)

    captured = capture_path.read_text().splitlines()
    assert f"UV_CACHE_DIR={repo_root}/relative-cache" in captured
    assert "UV_LINK_MODE=copy" in captured
    assert f"GYM_CI_DEV_VENV_DIR={node_local_root}/.driver-venv" in captured
    assert "ARG=+uv_cache_dir=" + str(repo_root / "relative-cache") in captured
    assert "ARG=+uv_venv_dir=" + str(node_local_root) in captured
    assert "ARG=+shard_index=2" in captured
    assert "ARG=+num_shards=8" in captured
    assert not (node_local_root / ".driver-venv").exists()


@pytest.mark.parametrize("venv_root", ["relative-venvs", "/"])
def test_server_tests_rejects_unsafe_venv_root(venv_root: str) -> None:
    env = os.environ.copy()
    env["GYM_CI_UV_VENV_DIR"] = venv_root

    result = subprocess.run([str(SERVER_TESTS), "0", "8"], capture_output=True, text=True, env=env)

    assert result.returncode == 2
    assert f"GYM_CI_UV_VENV_DIR must be an absolute non-root path: {venv_root}" in result.stderr
