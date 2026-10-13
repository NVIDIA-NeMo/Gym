# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import pytest

from responses_api_agents.conversational_tool_use.policy_tool_generation import assets


PROMPTS = {
    "cohesion_judge.txt": "Judge cohesion for {domain}: {policy}\n{tools}",
    "general_policy.txt": "Generate a general policy for {domain} at {timestamp}.",
    "general_policy_refine.txt": "Refine {domain}: {policy}\n{reference_policies}",
    "general_tools.txt": "Generate tools for {domain}: {policy}",
    "golden_judge.txt": "Choose the better policy and tool set.",
    "proactive_policy.txt": "Generate a proactive policy for {domain} at {timestamp}.",
    "proactive_policy_refine.txt": "Refine proactive policy for {domain}: {policy}",
    "proactive_tools.txt": "Generate proactive tools for {domain}: {policy}",
    "tools_refine.txt": "Refine tools for {domain}: {policy}\n{tools}",
}


@pytest.fixture(autouse=True)
def prepared_policy_tool_assets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep unit tests offline after runtime assets move to Hugging Face."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    for filename, content in PROMPTS.items():
        (prompts_dir / filename).write_text(content, encoding="utf-8")

    golden_dir = tmp_path / "golden_policies"
    golden_dir.mkdir()
    for index in range(1, 9):
        (golden_dir / f"policy-{index}.md").write_text(f"reference policy {index}\n", encoding="utf-8")
        (golden_dir / f"tools_{index}.jsonl").write_text(
            f'{{"name":"reference_tool_{index}","doc":"Reference tool {index}"}}\n',
            encoding="utf-8",
        )
    monkeypatch.setattr(assets, "PROMPTS_DIR", prompts_dir)
    monkeypatch.setattr(assets, "GOLDENS_DIR", golden_dir)
    return golden_dir
