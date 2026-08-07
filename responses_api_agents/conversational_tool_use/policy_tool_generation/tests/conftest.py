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


@pytest.fixture(autouse=True)
def prepared_policy_tool_references(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep unit tests offline after runtime references move to Hugging Face."""
    golden_dir = tmp_path / "golden_policies"
    golden_dir.mkdir()
    for index in range(1, 9):
        (golden_dir / f"policy-{index}.md").write_text(f"reference policy {index}\n", encoding="utf-8")
        (golden_dir / f"tools_{index}.jsonl").write_text(
            f'{{"name":"reference_tool_{index}","doc":"Reference tool {index}"}}\n',
            encoding="utf-8",
        )
    monkeypatch.setattr(assets, "GOLDENS_DIR", golden_dir)
    return golden_dir
