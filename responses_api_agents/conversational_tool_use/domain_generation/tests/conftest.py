# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from responses_api_agents.conversational_tool_use.domain_generation import assets


@pytest.fixture(autouse=True)
def prepared_domain_prompts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prompts_dir = tmp_path / "domain_prompts"
    prompts_dir.mkdir()
    (prompts_dir / "domain_generation.txt").write_text("Generate domains.", encoding="utf-8")
    monkeypatch.setattr(assets, "PROMPTS_DIR", prompts_dir)
