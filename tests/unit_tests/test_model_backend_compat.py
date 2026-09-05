# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
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
