# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependencies the vLLM server venv must provide."""

import importlib.metadata
import sys

import pytest
from packaging.requirements import Requirement


pytestmark = pytest.mark.skipif(
    sys.platform == "darwin", reason="Darwin pins vllm 0.11 and keeps the codec packages excluded"
)


def test_vllm_runtime_requirements_installed() -> None:
    # Gym's shared policy never-installs codec and copyleft packages; vLLM requires several of them to serve.
    missing = []
    for raw in importlib.metadata.requires("vllm") or []:
        requirement = Requirement(raw)
        if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
            continue
        try:
            importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(str(requirement))
    assert missing == []


def test_vllm_serve_import_chain() -> None:
    # vllm serve imports both at startup for every model, text-only ones included.
    from torchvision.transforms import InterpolationMode
    from transformers import MistralCommonBackend

    assert InterpolationMode.BILINEAR.value == "bilinear"
    assert MistralCommonBackend.__name__ == "MistralCommonBackend"
