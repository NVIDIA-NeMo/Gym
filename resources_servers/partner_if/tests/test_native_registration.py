# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapters that own sandboxes install the sandbox SDK in their own environments."""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement


def test_the_sandbox_owning_server_installs_the_sdk_in_its_own_environment():
    project = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    requirements = map(Requirement, project["project"]["dependencies"])
    extras = {extra for requirement in requirements if requirement.name == "nemo-gym" for extra in requirement.extras}
    assert "sandbox" in extras
