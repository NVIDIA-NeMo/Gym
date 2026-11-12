# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import full_ci
import pytest
from full_ci_history import ci_contract, legacy_command


LEGACY = """name: Unit tests
env:
  NUM_SHARDS: "8"
jobs:
  test:
    runs-on: nemo-ci-aws-cpu-4-16
    timeout-minutes: 45
    steps:
      - name: Core + sandbox unit tests (coverage gate)
        run: |
          ./scripts/ci/core_unit_tests.sh
          source ./scripts/ci/setup_dev.sh
          uv sync --extra dev --extra ray --extra sandbox --extra openshell --extra mlflow --extra wandb
          source .venv/bin/activate
          pytest --cov=. --cov-append --cov-report=term-missing \\
            --cov-fail-under="$(python3 scripts/ci/cov_fail_under.py)" --durations=10 -m sandbox
  session-routing-e2e:
    runs-on: ubuntu-latest
    timeout-minutes: 20
  server-suite:
    runs-on: nemo-ci-aws-cpu-4-16
    timeout-minutes: 45
    strategy:
      matrix:
        shard: [0, 1, 2, 3, 4, 5, 6, 7]
"""


def source(tmp_path, text=LEGACY):
    workflow = tmp_path / ".github/workflows"
    workflow.mkdir(parents=True)
    (workflow / "unit-tests.yml").write_text(text)
    (workflow / "code-linting.yml").write_text("jobs:\n  lint-check:\n    runs-on: ubuntu-latest\n")
    setup = tmp_path / "scripts/ci/setup_dev.sh"
    setup.parent.mkdir(parents=True)
    setup.write_text('setup_uv_install_url="https://astral.sh/uv/0.11.29/install.sh"\n')
    return tmp_path


def test_legacy_scope_retains_single_core_job_and_own_gate(tmp_path):
    root = source(tmp_path)
    contract = ci_contract(root)
    assert contract["layout"] == "legacy-combined"
    assert contract["core_shards"] == 1
    assert contract["server_shards"] == 8
    assert contract["coverage"] is None
    assert contract["uv_version"] == "0.11.29"
    assert contract["jobs"]["servers"]["timeout"] == 45
    assert "sandbox" not in contract["jobs"]
    command = legacy_command(root)
    assert command.startswith("./scripts/ci/core_unit_tests.sh\n")
    assert "--cov-append" in command
    assert "cov_fail_under.py" in command


def test_legacy_instrumentation_keeps_real_command_and_separate_junit(tmp_path, monkeypatch):
    root = source(tmp_path)
    monkeypatch.chdir(root)
    command = full_ci.command_for("legacy-core", shard=0, shards=1, servers=[], output=Path("results/full-ci"))
    assert command.endswith(legacy_command(root))
    assert "COVERAGE_FILE=results/full-ci/.coverage.legacy-core" in command
    assert "PYTEST_ADDOPTS=--junitxml=results/full-ci/junit/sandbox.xml" in command
    assert "GYM_CI_JUNIT_DIR=results/full-ci/junit" in command


def test_ambiguous_or_unknown_historical_layout_fails_closed(tmp_path):
    root = source(tmp_path, LEGACY + "  core-coverage:\n    runs-on: ubuntu-latest\n")
    with pytest.raises(ValueError, match="Ambiguous"):
        ci_contract(root)
    (root / ".github/workflows/unit-tests.yml").write_text(
        LEGACY.replace("shard: [0, 1, 2, 3, 4, 5, 6, 7]", "shard: [0, 2]")
    )
    with pytest.raises(ValueError, match="zero-based"):
        ci_contract(root)
    (root / ".github/workflows/unit-tests.yml").write_text(LEGACY.replace("nemo-ci-aws-cpu-4-16", "unapproved-runner"))
    with pytest.raises(ValueError, match="runner"):
        ci_contract(root)


def test_mismatching_server_total_fails_before_dispatch(tmp_path):
    root = source(tmp_path, LEGACY.replace('NUM_SHARDS: "8"', 'NUM_SHARDS: "4"'))
    with pytest.raises(ValueError, match="shard count disagree"):
        ci_contract(root)


def test_legacy_native_stops_after_failed_core_before_sandbox_or_gate(tmp_path, monkeypatch):
    calls = []

    def native(**kwargs):
        calls.append(kwargs)
        return 7

    monkeypatch.setattr(full_ci, "run_native", native)
    monkeypatch.setattr(full_ci, "run_logged", lambda *a, **kw: pytest.fail("gate ran after failed core"))
    assert full_ci.run_legacy_native(output=tmp_path, env={}, execution_mode="incremental") == 7
    assert [c["job"] for c in calls] == ["core"]
    assert calls[0]["shards"] == 1


def test_legacy_native_runs_both_groups_then_preserves_gate_failure(tmp_path, monkeypatch):
    calls = []

    def native(**kwargs):
        calls.append(kwargs)
        return 0

    def gate(command, **kwargs):
        assert [c["job"] for c in calls] == ["core", "sandbox"]
        assert "coverage combine --keep" in command[2]
        assert ".coverage.core-0" in command[2] and ".coverage.sandbox" in command[2]
        assert "cov_fail_under.py" in command[2]
        return 2

    monkeypatch.setattr(full_ci, "run_native", native)
    monkeypatch.setattr(full_ci, "run_logged", gate)
    assert full_ci.run_legacy_native(output=tmp_path, env={}, execution_mode="forced") == 2
    assert calls[0]["native_output"] == tmp_path / "native/core"
    assert calls[1]["native_output"] == tmp_path / "native/sandbox"
    assert all(c["execution_mode"] == "forced" for c in calls)


def test_old_global_config_location_uses_sampled_constants(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = Path("nemo_gym")
    root.mkdir()
    (root / "global_config.py").write_text('A = "historical_servers"\nSERVER_TYPE_KEYS: tuple[str, ...] = (A,)\n')
    assert full_ci.server_types() == ("historical_servers",)
