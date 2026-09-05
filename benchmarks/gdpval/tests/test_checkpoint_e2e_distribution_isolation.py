# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from pathlib import Path


PACKAGE = Path(__file__).parents[1] / "hsg" / "checkpoint_e2e"


def test_fingerprint_probe_requires_and_receipts_judge_owned_distribution() -> None:
    probe = (PACKAGE / "fingerprint_probe.py").read_text()
    assert 'parser.add_argument("--distribution-path", type=Path, required=True)' in probe
    assert '"distribution_path": str(args.distribution_path)' in probe
    assert '"distribution_sha256": distribution_sha256' in probe
    assert 'expected_distribution = args.preprocessed_input.parent.parent / "occupation_distribution.json"' in probe
    assert 'distribution_path = _regular(distribution_path, "occupation distribution")' in probe


def test_prepare_and_live_judge_share_frozen_distribution_contract() -> None:
    for name in ("prepare_existing_fingerprint.sh", "prepare_rejudge_fingerprint.sh"):
        script = (PACKAGE / name).read_text()
        assert 'DISTRIBUTION_PATH="$JUDGE_DIR/occupation_distribution.json"' in script or (
            "DISTRIBUTION_PATH=$JUDGE_DIR/occupation_distribution.json" in script
        )
        assert '--distribution-path "$DISTRIBUTION_PATH"' in script
        assert 'chmod 0400 "$DISTRIBUTION_PATH"' in script
        assert 'd["distribution_path"]' in script
        assert 'd["distribution_sha256"]' in script

    judge = (PACKAGE / "judge.sbatch").read_text()
    assert 'DISTRIBUTION_PATH="$JUDGE_DIR/occupation_distribution.json"' in judge
    assert '++multistage.distribution_path="$DISTRIBUTION_PATH"' in judge
    assert "$(stat -c '%a' \"$DISTRIBUTION_PATH\") == 400" in judge
    assert 'd["distribution_path"]' in judge
    assert 'd["distribution_sha256"]' in judge
    assert judge.index("provider-free fingerprint input/distribution path/hash drift") < judge.index(
        ': "${JUDGE_BASE_URL:?JUDGE_BASE_URL is required}"'
    )
    assert judge.index("provider-free fingerprint input/distribution path/hash drift") < judge.index(
        "models_json=$(curl"
    )


def test_rejudge_controller_revalidates_distribution_receipt() -> None:
    script = (PACKAGE / "rejudge_controller.sbatch").read_text()
    assert "DISTRIBUTION_PATH=$JUDGE_DIR/occupation_distribution.json" in script
    assert '$(file_mode "$DISTRIBUTION_PATH") == 400' in script
    assert 'd["distribution_path"]' in script
    assert 'd["distribution_sha256"]' in script


def test_distribution_isolation_shell_entrypoints_parse() -> None:
    scripts = [
        PACKAGE / "prepare_existing_fingerprint.sh",
        PACKAGE / "prepare_rejudge_fingerprint.sh",
        PACKAGE / "judge.sbatch",
        PACKAGE / "rejudge_controller.sbatch",
    ]
    subprocess.run(["bash", "-n", *map(str, scripts)], check=True)
