# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from nemo_gym.environment_manifest import EnvironmentManifest


def complete_scaffold_fixture(path: Path, manifest: EnvironmentManifest) -> None:
    """Replace a generated fixture's TODOs with the scaffold scorer's expected results."""

    minimum, maximum = manifest.reward.range
    full_reward, zero_reward = (maximum, minimum) if manifest.reward.higher_is_better else (minimum, maximum)
    cases = [json.loads(line) for line in path.read_text().splitlines()]
    for case in cases:
        case["expected_status"] = 422 if case["case"] == "malformed" else 200
        if case["case"] in {"full_reward", "determinism_reseed"}:
            case["expected_reward"] = full_reward
        elif case["case"] == "zero_reward":
            case["expected_reward"] = zero_reward
        else:
            case.pop("expected_reward", None)
    path.write_text("".join(json.dumps(case) + "\n" for case in cases))
