# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
VM_PREPARE_SCRIPT = REPO_ROOT / "benchmarks/osworld/tools/prepare_osworld_vm.sh"
CHECK_ENVIRONMENT_SCRIPT = REPO_ROOT / "benchmarks/osworld/tools/check_environment.sh"
MODEL_PROBE_SCRIPT = REPO_ROOT / "benchmarks/osworld/tools/probe_model_endpoint.py"
START_CONTROL_SCRIPT = REPO_ROOT / "benchmarks/osworld/tools/start_control.sh"
RUN_EVAL_SCRIPT = REPO_ROOT / "benchmarks/osworld/tools/run_eval.sh"
CLEANUP_RUN_SCRIPT = REPO_ROOT / "benchmarks/osworld/tools/cleanup_run.sh"
SANDBOX_CONFIG = REPO_ROOT / "benchmarks/osworld/configs/osworld_sandbox.yaml"


@pytest.mark.parametrize(
    "script",
    [VM_PREPARE_SCRIPT, CHECK_ENVIRONMENT_SCRIPT, START_CONTROL_SCRIPT, RUN_EVAL_SCRIPT, CLEANUP_RUN_SCRIPT],
)
def test_public_host_setup_scripts_are_syntax_valid_and_portable(script: Path) -> None:
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_vm_prepare_script_pins_the_verified_image_identity() -> None:
    text = VM_PREPARE_SCRIPT.read_text(encoding="utf-8")
    assert "6bf667a852b3c307f61d9f09c42559351f45e0607e428b4997becf534cf4d313" in text  # pragma: allowlist secret
    assert "24460197888" in text
    assert "--continue-at -" in text


def test_runtime_wrappers_delegate_to_current_gym_commands() -> None:
    start_control = START_CONTROL_SCRIPT.read_text(encoding="utf-8")
    assert "env start \\" in start_control
    assert "model-io.jsonl" not in start_control
    assert "eval run --no-serve \\" in RUN_EVAL_SCRIPT.read_text(encoding="utf-8")


def test_remote_docker_requires_a_reachable_publish_host() -> None:
    start_text = START_CONTROL_SCRIPT.read_text(encoding="utf-8")
    sandbox_text = SANDBOX_CONFIG.read_text(encoding="utf-8")

    assert "DOCKER_HOST" in start_text
    assert "OSWORLD_SANDBOX_PUBLISH_HOST" in start_text
    assert "docker info" in start_text
    assert "${oc.env:OSWORLD_SANDBOX_PUBLISH_HOST,127.0.0.1}" in sandbox_text


def test_role_checks_cover_environment_and_model_contracts() -> None:
    environment_text = CHECK_ENVIRONMENT_SCRIPT.read_text(encoding="utf-8")
    model_text = MODEL_PROBE_SCRIPT.read_text(encoding="utf-8")

    assert "--ssh" in environment_text
    assert "/dev/kvm" in environment_text
    assert "EXPECTED_VM_SHA256" in environment_text
    assert "/models" in model_text
    assert "/chat/completions" in model_text
    compile(model_text, str(MODEL_PROBE_SCRIPT), "exec")


def test_cleanup_is_scoped_to_the_run_id() -> None:
    text = CLEANUP_RUN_SCRIPT.read_text(encoding="utf-8")
    assert "process_belongs_to_run" in text
    assert "Ignoring stale" in text
    assert 'rm -f "${pid_file}"' in text
    assert "label=nemo-gym.run-id=${RUN_ID}" in text
    assert "nemo-gym.workload=osworld" in text
    assert "logs and results were preserved" in text
