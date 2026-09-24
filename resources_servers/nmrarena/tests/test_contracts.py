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

"""Contracts between files that nothing imports across at runtime.

The vendored prompt must stay upstream's, the shipped YAML must agree with the class
defaults, and the committed example rows must be what the preparer emits.
"""

import hashlib
import json
import sys
from pathlib import Path

import yaml
from app import NMRArenaResourcesServerConfig
from prompting import MAX_OUTPUT_TOKENS, NUM_CANDIDATES, PROMPTS_DIR, TEMPERATURE, build_messages, system_prompt


SERVER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER_DIR / "scripts"))
import make_example_data  # noqa: E402


# SHA-256 of ``_SYSTEM_TEMPLATE`` (``dataset/llm_track.ipynb``, cell 9) and ``LICENSE`` of
# odanchem/NMRArena at commit 8b4ca8a8953185c00f0c4d7fa3c16c23aa616326.
UPSTREAM_SHA256 = {
    "system_prompt.txt": "9cedddc7fdd9269f40146d3b718e2d49c9b9b7b1b91f52824842883e7deb8cff",  # pragma: allowlist secret
    "LICENSE": "7546512862305e3dce266818bfaff4a4ecc4ca18469208b9c3c3ae9d4a4c779b",  # pragma: allowlist secret
}


class TestVendoredPrompt:
    def test_files_are_byte_identical_to_upstream(self) -> None:
        for name, expected in UPSTREAM_SHA256.items():
            assert hashlib.sha256((PROMPTS_DIR / name).read_bytes()).hexdigest() == expected, name

    def test_template_formats_without_leftover_placeholders(self) -> None:
        text = system_prompt(10)
        assert "{n}" not in text and "{min_slots}" not in text
        assert '{"candidates": [{"rank": 1, "smiles": "..."}, {"rank": 2, "smiles": "..."}]}' in text
        assert text.startswith(
            "You are a senior organic chemist with decades of hands-on experience in NMR structure elucidation. \n"
        )

    def test_upstream_constants(self) -> None:
        assert (NUM_CANDIDATES, TEMPERATURE, MAX_OUTPUT_TOKENS) == (10, 1.0, 24576)


class TestShippedConfigAgreesWithClassDefaults:
    def test_execution_knobs(self) -> None:
        shipped = yaml.safe_load((SERVER_DIR / "configs" / "nmrarena.yaml").read_text())["nmrarena"][
            "resources_servers"
        ]["nmrarena"]
        defaults = NMRArenaResourcesServerConfig.model_fields
        for knob in ("num_candidates", "max_smiles_chars", "strict_candidates", "salvage_truncated_json"):
            assert shipped[knob] == defaults[knob].default, knob
        assert shipped["num_candidates"] == NUM_CANDIDATES

    def test_agent_datasets_declare_a_license(self) -> None:
        agent = yaml.safe_load((SERVER_DIR / "configs" / "nmrarena.yaml").read_text())["nmrarena_simple_agent"]
        for ds in agent["responses_api_agents"]["simple_agent"]["datasets"]:
            assert ds["license"] == "MIT"


class TestExampleData:
    def test_committed_example_matches_the_generator(self) -> None:
        rows = [
            json.loads(line)
            for line in (SERVER_DIR / "data" / "example.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert len(rows) == 5
        for row, (cls, rec) in zip(rows, make_example_data.EXAMPLES):
            assert row["responses_create_params"]["input"] == build_messages(rec["h_nmr"], rec["c_nmr"])
            assert row["verifier_metadata"]["smiles"] == rec["smiles"]
            assert row["verifier_metadata"]["primary_class"] == cls
            assert row["verifier_metadata"]["compound_id"].startswith("synthetic-")
