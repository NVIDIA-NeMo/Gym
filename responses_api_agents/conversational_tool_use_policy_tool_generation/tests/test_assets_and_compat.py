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

import hashlib
from pathlib import Path

from responses_api_agents.conversational_tool_use_policy_tool_generation.assets import (
    GOLDENS_DIR,
    PROMPTS_DIR,
    load_assets,
)
from responses_api_agents.conversational_tool_use_policy_tool_generation.compat import (
    format_domain_name,
    parse_judgment,
    parse_policy,
    parse_tools,
    validate_tools,
)


ACTIVE_PROMPT_HASHES = {
    "cohesion_judge.txt": "a0070c4d9688df277c5f65e2b3a22112a1dea2b4f02c173ff4758529f4d912d9",  # pragma: allowlist secret
    "general_policy.txt": "4da0ed416c152dffd975b46e480c3e83843b6bfe037c2222dc70cb5e471bad88",  # pragma: allowlist secret
    "general_policy_refine.txt": "cbe16bf60332a12a03083758fe31dafd5cb779abebf29214ef1315de526b8c1b",  # pragma: allowlist secret
    "general_tools.txt": "5b8af59584760ce76523286ef17865f5b195385afeb3494fcd867cd41d90b17a",  # pragma: allowlist secret
    "golden_judge.txt": "ca5ce1481ff65e3f913dce4f225598f50180241edac0f6317dfaa6a896441977",  # pragma: allowlist secret
    "proactive_policy.txt": "00c93189b9411fabde7439a5fdf36d14412e473e6b8851824043a7e4c74109be",  # pragma: allowlist secret
    "proactive_policy_refine.txt": "544fce4d3fc534634f04db0e2e01d7a5691612d01b6f3abd9bc27d039e88d9d5",  # pragma: allowlist secret
    "proactive_tools.txt": "2fd77edd5d5fdc99beceb58aede864b2a41ef3fecef102eb5b0a3c7bc1a7895d",  # pragma: allowlist secret
    "tools_refine.txt": "38965e0f6909b3799863153aab6980ef694da5128145a5aa4ff88bbbe2782738",  # pragma: allowlist secret
}


def _tree_hash(directory: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    paths = sorted(path for path in directory.iterdir() if path.is_file())
    for path in paths:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return len(paths), digest.hexdigest()


def test_prompt_and_reference_bytes_and_filenames() -> None:
    active_paths = sorted(PROMPTS_DIR.glob("*.txt"))
    assert {path.name for path in active_paths} == set(ACTIVE_PROMPT_HASHES)
    for path in active_paths:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == ACTIVE_PROMPT_HASHES[path.name]

    assert _tree_hash(GOLDENS_DIR) == (
        16,
        "c1c621e88f763dab8fa23e6721180376d65b1386b99e662d32c652dcf28e1cd6",  # pragma: allowlist secret
    )
    assert _tree_hash(PROMPTS_DIR / "archive") == (
        42,
        "fd4d674dea96fee4d258daef1defa55b4aa42dfe6bc7720ac2b0e44aa41c5d90",  # pragma: allowlist secret
    )
    all_paths = [*active_paths, *GOLDENS_DIR.iterdir(), *(PROMPTS_DIR / "archive").iterdir()]
    assert all(not any(character.isspace() for character in path.name) for path in all_paths)
    markdown_names = sorted(path.name for path in GOLDENS_DIR.glob("*.md"))
    assert markdown_names == [f"policy-{index}.md" for index in range(1, 9)]
    assert all("_" not in name for name in markdown_names)
    assert len(load_assets("general").golden_pairs) == 8
    assert len(load_assets("proactive").golden_pairs) == 8


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
