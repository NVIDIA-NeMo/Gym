# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from responses_api_agents.conversational_tool_use.simulation import prompt


@pytest.fixture(autouse=True)
def prepared_simulation_prompts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prompts_dir = tmp_path / "simulation_prompts"
    prompts_dir.mkdir()
    (prompts_dir / "agent_system.txt").write_text(
        "Agent policy: {domain_policy}",
        encoding="utf-8",
    )
    (prompts_dir / "agent_parallel_system.txt").write_text(
        "Parallel agent policy: {domain_policy}",
        encoding="utf-8",
    )
    monkeypatch.setattr(prompt, "PROMPTS_DIR", prompts_dir)
