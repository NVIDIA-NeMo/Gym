# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenEvolve evaluator entrypoint for NeMo Gym OpenHands candidate files."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from openevolve_gym.openhands_evaluator import (
    DEFAULT_NEMO_GYM_REPO,
    OpenHandsEvaluationConfig,
    evaluate_candidate,
)


def evaluate(program_path: str) -> dict:
    """Evaluate one candidate instruction file through NeMo Gym."""
    candidate_path = Path(program_path).expanduser().resolve()
    output_dir = Path(
        os.environ.get(
            "OPENEVOLVE_OPENHANDS_OUTPUT_DIR",
            f"results/openevolve_openhands/{candidate_path.stem}",
        )
    ).expanduser()
    config = OpenHandsEvaluationConfig(
        nemo_gym_repo=Path(os.environ.get("NEMO_GYM_REPO", str(DEFAULT_NEMO_GYM_REPO))),
        policy_base_url=os.environ.get("POLICY_BASE_URL", "http://localhost:8001/v1"),
        policy_api_key=os.environ.get("POLICY_API_KEY", "dummy"),
        policy_model_name=os.environ.get("POLICY_MODEL_NAME", "dummy-model"),
    )
    return evaluate_candidate(
        candidate_path,
        output_dir,
        config=config,
        execute=_env_flag("OPENEVOLVE_OPENHANDS_EXECUTE"),
    )


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python contrib/open_evolve/evaluator.py <candidate.md>")
    print(json.dumps(evaluate(sys.argv[1]), indent=2, sort_keys=True))
