# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-role runtime assertions split from OSWorld's dependency-policy tests."""

import tomllib
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.version import Version

from nemo_gym.orchestration.server_venv import setup_env_command


ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "responses_api_models/vllm_model"


def test_vllm_model_venv_keeps_the_parent_python(tmp_path: Path) -> None:
    # The oldest parent Gym supports must satisfy the model's .python-version, so its venv is built with the
    # parent's exact interpreter rather than whichever patch release uv finds first.
    with (ROOT / "pyproject.toml").open("rb") as source:
        python_floor = tomllib.load(source)["project"]["requires-python"].removeprefix(">=")
    config = {
        "head_server_deps": [],
        "head_server_constraints": [],
        "python_version": python_floor,
        "skip_venv_if_present": False,
        "uv_venv_dir": str(tmp_path),
    }

    assert f"--python {python_floor} " in setup_env_command(MODEL_DIR, config, "vllm_model")
    assert Version("3.14") not in SpecifierSet((MODEL_DIR / ".python-version").read_text(encoding="utf-8").strip())
