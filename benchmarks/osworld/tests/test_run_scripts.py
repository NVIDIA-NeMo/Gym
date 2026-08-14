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
OPENSANDBOX_CLEANUP_SCRIPT = REPO_ROOT / "benchmarks/osworld/tools/cleanup_opensandbox_run.py"
OSWORLD_AGENT_CONFIG = REPO_ROOT / "responses_api_agents/osworld_agent/configs/osworld_agent.yaml"
OSWORLD_AGENT_APP = REPO_ROOT / "responses_api_agents/osworld_agent/app.py"
OSWORLD_AGENT_REQUIREMENTS = REPO_ROOT / "responses_api_agents/osworld_agent/requirements.txt"
OSWORLD_AGENT_OVERRIDES = REPO_ROOT / "responses_api_agents/osworld_agent/overrides.txt"
OSWORLD_RUNTIME_DEPS_SCRIPT = REPO_ROOT / "responses_api_agents/osworld_agent/install_optional_runtime_deps.sh"
OSWORLD_RUNTIME_DEPS_CHECKER = REPO_ROOT / "responses_api_agents/osworld_agent/runtime_dependencies.py"


@pytest.mark.parametrize(
    "script",
    [
        VM_PREPARE_SCRIPT,
        CHECK_ENVIRONMENT_SCRIPT,
        START_CONTROL_SCRIPT,
        RUN_EVAL_SCRIPT,
        CLEANUP_RUN_SCRIPT,
        OSWORLD_RUNTIME_DEPS_SCRIPT,
    ],
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
    assert "NEMO_GYM_RUN_ID=${NEMO_GYM_RUN_ID:-${RUN_ID}}" in start_control
    assert "eval run --no-serve \\" in RUN_EVAL_SCRIPT.read_text(encoding="utf-8")


def test_start_control_preflights_native_build_toolchain() -> None:
    text = START_CONTROL_SCRIPT.read_text(encoding="utf-8")

    assert "command -v cc" in text
    assert "Python.h" in text
    assert "python3-dev" in text


def test_start_control_requires_explicit_osworld_runtime_setup() -> None:
    text = START_CONTROL_SCRIPT.read_text(encoding="utf-8")

    assert "runtime_dependencies.py" in text
    assert "OSWORLD_AGENT_VENV" in text
    assert "gym env prefetch" in text
    assert "install_optional_runtime_deps.sh" in text
    assert "uv pip install" not in text
    assert "require_optional_runtime_dependencies()" in OSWORLD_AGENT_APP.read_text(encoding="utf-8")


def test_managed_osworld_agent_installs_opensandbox_sdk() -> None:
    requirements = OSWORLD_AGENT_REQUIREMENTS.read_text(encoding="utf-8").splitlines()
    overrides = OSWORLD_AGENT_OVERRIDES.read_text(encoding="utf-8").splitlines()
    runtime_script = OSWORLD_RUNTIME_DEPS_SCRIPT.read_text(encoding="utf-8")

    assert "-e nemo-gym[dev] @ ../../" in requirements
    assert "opensandbox>=0.1.15" in requirements
    assert "tenacity>=9.1.4" in requirements
    assert not any(line.startswith("cryptography") for line in requirements)
    assert not any(line.startswith("flask") for line in requirements)
    assert not any(line.startswith("opencv-") for line in requirements)
    assert "torch==2.11.0" in overrides
    assert "matplotlib==3.10.6" in overrides
    assert "agp-client; sys_platform == 'never'" in overrides
    assert "--no-config" in runtime_script
    assert '"numpy<2"' in runtime_script
    assert "cryptography~=46.0" in runtime_script
    assert "opencv-python-headless~=4.8.1.78" in runtime_script
    assert "torchvision==0.26.0" in runtime_script


def test_remote_docker_requires_a_reachable_publish_host() -> None:
    start_text = START_CONTROL_SCRIPT.read_text(encoding="utf-8")
    sandbox_text = OSWORLD_AGENT_CONFIG.read_text(encoding="utf-8")

    assert "DOCKER_HOST" in start_text
    assert "OSWORLD_SANDBOX_PUBLISH_HOST" in start_text
    assert "OSWORLD_SANDBOX_PUBLISH_HOST:-127.0.0.1" in start_text
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
    compile(
        OSWORLD_RUNTIME_DEPS_CHECKER.read_text(encoding="utf-8"),
        str(OSWORLD_RUNTIME_DEPS_CHECKER),
        "exec",
    )


def test_cleanup_is_scoped_to_the_run_id() -> None:
    text = CLEANUP_RUN_SCRIPT.read_text(encoding="utf-8")
    assert "process_belongs_to_run" in text
    assert "Ignoring stale" in text
    assert 'rm -f "${pid_file}"' in text
    assert "label=nemo-gym.run-id=${RUN_ID}" in text
    assert "nemo-gym.workload=osworld" in text
    assert '"${OPENSANDBOX_CLEANUP}" --run-id "${RUN_ID}" --reap' in text
    assert "logs and results were preserved" in text

    opensandbox_text = OPENSANDBOX_CLEANUP_SCRIPT.read_text(encoding="utf-8")
    compile(opensandbox_text, str(OPENSANDBOX_CLEANUP_SCRIPT), "exec")
    assert 'RUN_METADATA_KEYS = ("run-id", "nemo-gym.nvidia.com/run")' in opensandbox_text
    assert "SandboxManagerSync" in opensandbox_text
