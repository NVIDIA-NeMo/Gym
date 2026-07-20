# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
VM_PREPARE_SCRIPT = REPO_ROOT / "benchmarks/osworld/tools/prepare_osworld_vm.sh"
START_CONTROL_SCRIPT = REPO_ROOT / "benchmarks/osworld/tools/start_control.sh"
RUN_EVAL_SCRIPT = REPO_ROOT / "benchmarks/osworld/tools/run_eval.sh"


@pytest.mark.parametrize(
    "script",
    [VM_PREPARE_SCRIPT, START_CONTROL_SCRIPT, RUN_EVAL_SCRIPT],
)
def test_public_host_setup_scripts_are_syntax_valid_and_portable(script: Path) -> None:
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_vm_prepare_script_pins_the_verified_image_identity() -> None:
    text = VM_PREPARE_SCRIPT.read_text(encoding="utf-8")
    assert "6bf667a852b3c307f61d9f09c42559351f45e0607e428b4997becf534cf4d313" in text  # pragma: allowlist secret
    assert "24460197888" in text
    assert "--continue-at -" in text


def test_runtime_wrappers_delegate_to_current_gym_commands() -> None:
    assert 'env start \\' in START_CONTROL_SCRIPT.read_text(encoding="utf-8")
    assert 'eval run --no-serve \\' in RUN_EVAL_SCRIPT.read_text(encoding="utf-8")
