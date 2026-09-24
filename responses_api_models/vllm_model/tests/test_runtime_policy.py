# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-role runtime assertions split from OSWorld's dependency-policy tests."""

import tomllib
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.version import Version


ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "responses_api_models/vllm_model"


def test_vllm_model_uses_a_compatible_python_range() -> None:
    with (ROOT / "pyproject.toml").open("rb") as source:
        python_floor = tomllib.load(source)["project"]["requires-python"].removeprefix(">=")
    supported = SpecifierSet((MODEL_DIR / ".python-version").read_text(encoding="utf-8").strip())
    assert Version(python_floor) in supported
    assert Version("3.14") not in supported


def test_vllm_model_uses_standard_uv_runtime_files() -> None:
    legacy_markers = {
        "uv-managed-python.txt",
        "uv-overrides.txt",
        "uv-python-version.txt",
        "uv-torch-backend.txt",
    }
    assert not [path for path in MODEL_DIR.iterdir() if path.name in legacy_markers]
