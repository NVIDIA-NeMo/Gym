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

from __future__ import annotations

from responses_api_agents.conversational_tool_use.policy_tool_generation import assets
from responses_api_agents.conversational_tool_use.policy_tool_generation.compat import (
    format_domain_name,
    parse_judgment,
    parse_policy,
    parse_tools,
    validate_tools,
)


def test_prepared_prompt_and_reference_filenames() -> None:
    active_paths = sorted(assets.PROMPTS_DIR.glob("*.txt"))
    assert {path.name for path in active_paths} == set(assets.PROMPT_FILENAMES)

    assert assets.GOLDEN_FILENAMES == tuple(
        filename for index in range(1, 9) for filename in (f"policy-{index}.md", f"tools_{index}.jsonl")
    )
    all_paths = [*active_paths, *assets.GOLDENS_DIR.iterdir()]
    assert all(not any(character.isspace() for character in path.name) for path in all_paths)
    markdown_names = sorted(path.name for path in assets.GOLDENS_DIR.glob("*.md"))
    assert markdown_names == [f"policy-{index}.md" for index in range(1, 9)]
    assert all("_" not in name for name in markdown_names)
    general_assets = assets.load_assets("general")
    proactive_assets = assets.load_assets("proactive")
    assert len(general_assets.golden_pairs) == 8
    assert len(proactive_assets.golden_pairs) == 8
    assert general_assets.policy_prompt == "Generate a general policy for {domain} at {timestamp}."
    assert proactive_assets.policy_prompt == "Generate a proactive policy for {domain} at {timestamp}."


def test_case_sensitive_last_tag_and_json_repair_parsing() -> None:
    assert parse_policy("<policy>first</policy><policy> final </policy>") == "final"
    assert parse_policy("<POLICY>wrong case</POLICY>") is None
    assert parse_tools("<tools>{name: lookup, doc: test}\n</tools>") == [{"name": "lookup", "doc": "test"}]
    assert parse_tools("<tools></tools>") == []
    assert parse_tools("<TOOLS>{}</TOOLS>") is None
    assert parse_judgment("missing") is False


def test_permissive_tool_validation_and_domain_rendering() -> None:
    tool = {
        "name": "lookup",
        "doc": "Look up an item.",
        "params": None,
        "returns": None,
        "ignored_extra": {"retained_in_artifact": True},
    }
    assert validate_tools([])
    assert validate_tools([tool, tool])
    assert not validate_tools([{"name": "missing doc"}])
    assert not validate_tools(
        [
            {
                "name": "bad_schema",
                "doc": "Bad schema.",
                "params": {"type": "not-a-json-schema-type"},
                "returns": None,
            }
        ]
    )
    assert format_domain_name("(Home & Office)/Help Desk") == "Home__Office-Help_Desk"
