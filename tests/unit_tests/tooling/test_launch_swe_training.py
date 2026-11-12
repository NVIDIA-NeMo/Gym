# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise submission arguments without contacting Slurm."""

import os
import subprocess

import pytest
from scripts import prepare_swe_single_agent_turn_recipe as recipe


@pytest.mark.parametrize("gpus", [4, 8])
@pytest.mark.parametrize("dry_run", [True, False])
@pytest.mark.parametrize("config_readable", [True, False])
def test_launcher_reads_allocation_and_stops_when_config_cannot_be_loaded(tmp_path, gpus, dry_run, config_readable):
    # Simulate reading the configured GPU count; input validation is not a launch gate.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "uv").write_text(f"#!/bin/sh\nprintf '{gpus}\\n'\n" if config_readable else "#!/bin/sh\nexit 72\n")
    (bin_dir / "uv").chmod(0o755)
    receipt = tmp_path / "submission-receipt"
    (bin_dir / "sbatch").write_text('#!/bin/sh\nprintf "%s\\n" "$GPUS_PER_NODE" "$@" > "$SUBMISSION_RECEIPT"\n')
    (bin_dir / "sbatch").chmod(0o755)
    env = os.environ | {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "NEMO_RL_ROOT": str(tmp_path / "rl"),
        "CONFIG_PATH": str(tmp_path / "config.yaml"),
        "CONTAINER": str(tmp_path / "image.sqsh"),
        "CONTAINER_PROVENANCE": str(tmp_path / "image.json"),
        "SHARED_ROOT": "/",
        "SLURM_ACCOUNT": "test-account",
        "SLURM_PARTITION": "test-partition",
        "OPENSANDBOX_DOMAIN": "https://example.invalid",
        "OPENSANDBOX_API_KEY": "not-a-real-key",
        "SUBMISSION_RECEIPT": str(receipt),
    }
    result = subprocess.run(
        [
            "bash",
            str(recipe.ROOT / "scripts/launch_swe_single_agent_turn_training.sh"),
            *(["--dry-run"] if dry_run else []),
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if not config_readable:
        assert result.returncode == 72 and not receipt.exists()
        assert "Submit:" not in result.stdout
        return
    assert result.returncode == 0, result.stderr
    assert "--nodes=1" in result.stdout and "--exclusive" in result.stdout
    assert f"--gres=gpu:{gpus}" in result.stdout
    assert "run_grpo_single_controller.py" in result.stdout
    assert "check_swe_training_recipe.py" not in result.stdout
    assert "not-a-real-key" not in result.stdout
    if dry_run:
        assert not receipt.exists()
    else:
        submitted = receipt.read_text().splitlines()
        assert submitted[0] == str(gpus) and f"--gres=gpu:{gpus}" in submitted
