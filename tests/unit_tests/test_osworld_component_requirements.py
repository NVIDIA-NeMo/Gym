# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
OSWORLD_REVISION = "5d8c93592be588ab8e9909bed82f3b480fd4f533"
COMPONENT_REQUIREMENTS = (
    "resources_servers/osworld/requirements.txt",
    "responses_api_agents/nemotron_osworld/requirements.txt",
)


@pytest.mark.parametrize("relative_path", COMPONENT_REQUIREMENTS)
def test_osworld_component_runtime_is_reproducibly_pinned(relative_path: str) -> None:
    requirements = (REPO_ROOT / relative_path).read_text(encoding="utf-8")

    assert f"OSWorld/archive/{OSWORLD_REVISION}.tar.gz" in requirements
    assert "ray[default]==2.56.1" in requirements
    assert "opensandbox==0.1.15" in requirements
