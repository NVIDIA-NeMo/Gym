# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

from nemo_gym._config_aliases import legacy_config_path_alias


def test_legacy_import_namespace_resolves_canonical_backend() -> None:
    canonical = importlib.import_module("model_backends.openai_model")
    legacy = importlib.import_module("responses_api_models.openai_model")

    assert Path(canonical.__file__).resolve() == Path(legacy.__file__).resolve()


def test_legacy_model_config_path_maps_to_canonical_location() -> None:
    assert legacy_config_path_alias("responses_api_models/openai_model/configs/openai_model.yaml") == (
        "model_backends/openai_model/configs/openai_model.yaml"
    )


def test_absolute_legacy_path_is_not_rewritten() -> None:
    assert legacy_config_path_alias("/tmp/responses_api_models/custom/configs/custom.yaml") is None


def _write_backend(root: Path, layout: str, name: str, value: str) -> None:
    backend = root / layout / name
    backend.mkdir(parents=True)
    (backend / "__init__.py").write_text(f"VALUE = {value!r}\n")


def test_third_party_imports_match_discovery_precedence(tmp_path: Path) -> None:
    high_priority = tmp_path / "high"
    low_priority = tmp_path / "low"

    _write_backend(high_priority, "model_backends", "canonical_only", "canonical")
    _write_backend(high_priority, "responses_api_models", "legacy_only", "legacy")
    _write_backend(high_priority, "model_backends", "same_root", "canonical")
    _write_backend(high_priority, "responses_api_models", "same_root", "legacy")
    _write_backend(high_priority, "responses_api_models", "cross_root", "high-legacy")
    _write_backend(low_priority, "model_backends", "cross_root", "low-canonical")

    script = """
import importlib
import json
import nemo_gym

names = (
    "model_backends.canonical_only",
    "responses_api_models.legacy_only",
    "responses_api_models.same_root",
    "responses_api_models.cross_root",
)
print(json.dumps({name: importlib.import_module(name).VALUE for name in names}, sort_keys=True))
"""
    env = os.environ.copy()
    env["NEMO_GYM_EXTRA_ROOTS"] = os.pathsep.join((str(high_priority), str(low_priority)))
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[2],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {
        "model_backends.canonical_only": "canonical",
        "responses_api_models.cross_root": "high-legacy",
        "responses_api_models.legacy_only": "legacy",
        "responses_api_models.same_root": "canonical",
    }
