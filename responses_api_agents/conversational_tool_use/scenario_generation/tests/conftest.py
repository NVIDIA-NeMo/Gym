# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from responses_api_agents.conversational_tool_use.scenario_generation import assets


@pytest.fixture(autouse=True)
def prepared_scenario_prompts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prompts_dir = tmp_path / "scenario_prompts"
    prompts_dir.mkdir()
    (prompts_dir / "scenario_system.txt").write_text(
        "Policy: {domain_policy}\nScope: {policy_scope_instruction}",
        encoding="utf-8",
    )
    (prompts_dir / "scenario_user.txt").write_text(
        "Please create {scenario_count} different customer scenarios using {scenarios_schema}",
        encoding="utf-8",
    )
    monkeypatch.setattr(assets, "PROMPTS_DIR", prompts_dir)
