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

The shipped YAML overrides the class defaults; the agent's repair bound must equal the
server's; the example rows must be what the preparer emits; ``task_data.py`` must name
every field the server requires.
"""

import json
import sys
from pathlib import Path

import yaml
from app import REQUIRED_TASK_FIELDS, VeriSoftBenchResourcesServerConfig
from task_data import TaskData


SERVER_DIR = Path(__file__).resolve().parents[1]
GYM_ROOT = SERVER_DIR.parents[1]
sys.path.insert(0, str(SERVER_DIR / "scripts"))
import prepare_verisoftbench as prep  # noqa: E402


def _config():
    return yaml.safe_load((SERVER_DIR / "configs" / "verisoftbench.yaml").read_text())


class TestShippedConfig:
    def test_yaml_agrees_with_class_defaults_for_every_execution_knob(self) -> None:
        server = _config()["verisoftbench"]["resources_servers"]["verisoftbench"]
        defaults = VeriSoftBenchResourcesServerConfig.model_fields
        for knob in (
            "max_repair_rounds",
            "max_concurrent_compilations",
            "conflict_attempts_initial",
            "conflict_attempts_repair",
            "require_container_at_startup",
        ):
            assert server[knob] == defaults[knob].default, knob

    def test_agent_repair_bound_matches_server(self) -> None:
        cfg = _config()
        server = cfg["verisoftbench"]["resources_servers"]["verisoftbench"]
        agent = cfg["verisoftbench_agent"]["responses_api_agents"]["verisoftbench_agent"]
        assert agent["max_repair_rounds"] == server["max_repair_rounds"] == 3
        standalone = yaml.safe_load(
            (GYM_ROOT / "responses_api_agents/verisoftbench_agent/configs/verisoftbench_agent.yaml").read_text()
        )
        assert (
            standalone["verisoftbench_agent"]["responses_api_agents"]["verisoftbench_agent"]["max_repair_rounds"] == 3
        )

    def test_resources_server_block_comes_first(self) -> None:
        # The README-table hook reads the first top-level key.
        assert next(iter(_config())) == "verisoftbench"


class TestExampleData:
    def test_example_rows_are_preparer_shaped_and_synthetic(self) -> None:
        rows = [
            json.loads(line)
            for line in (SERVER_DIR / "data" / "example.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert len(rows) == 5
        for row in rows:
            # `gym dataset collate` rewrites the preparer's agent_ref into task_source; the tracked file carries that schema.
            assert row["task_source"] == prep.AGENT_REF["name"]
            meta = row["verifier_metadata"]
            for field in REQUIRED_TASK_FIELDS:
                assert field in meta, field
            TaskData.model_validate(meta)
            assert meta["thm_name"].startswith("GymExample."), "example theorems are authored here, not upstream's"
            assert meta["id"] >= 100_000, "example ids must not collide with upstream's 1..500"
            assert [m["role"] for m in row["responses_create_params"]["input"]] == ["system", "user"]


class TestTaskData:
    def test_every_field_the_server_requires_is_documented(self) -> None:
        for field in REQUIRED_TASK_FIELDS:
            assert field in TaskData.model_fields
