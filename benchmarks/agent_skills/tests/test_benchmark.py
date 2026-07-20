# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import yaml

from benchmarks.agent_skills.prepare import DATASET_PATH, prepare
from nemo_gym.skills import parse_skill_md


REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = REPO_ROOT / "benchmarks/agent_skills"


def test_prepare_validates_checked_in_dataset() -> None:
    assert prepare() == DATASET_PATH
    rows = [json.loads(line) for line in DATASET_PATH.read_text().splitlines() if line.strip()]
    assert rows
    assert {row["verifier_metadata"]["check_suite_id"] for row in rows} == {"create-environment-sql-v1"}


def test_create_environment_skill_metadata() -> None:
    metadata = parse_skill_md(REPO_ROOT / ".agents/skills/nemo-gym-create-environment/SKILL.md")
    assert metadata.name == "nemo-gym-create-environment"
    assert "create" in metadata.description.lower()
    assert "resources server" in metadata.description.lower()


def test_benchmark_config_wires_hidden_suite_and_agent() -> None:
    resources_config = yaml.safe_load(
        (REPO_ROOT / "resources_servers/agent_skills/configs/agent_skills.yaml").read_text()
    )
    suite = resources_config["agent_skills"]["resources_servers"]["agent_skills"]["check_suites"][
        "create-environment-sql-v1"
    ]
    assert suite["hidden_file_paths"]["run_checks.py"].endswith("create_environment_sql.py")
    assert suite["check_user"] == "nobody"

    benchmark_config = yaml.safe_load((BENCHMARK_DIR / "config.yaml").read_text())
    agent = benchmark_config["agent_skills_create_environment_claude_code_agent"]["responses_api_agents"][
        "claude_code_agent"
    ]
    assert agent["resources_server"]["name"] == "agent_skills_create_environment_resources_server"
    assert agent["datasets"][0]["prepare_script"] == "benchmarks/agent_skills/prepare.py"


def test_fixture_image_removes_project_skill_directories() -> None:
    dockerfile = (BENCHMARK_DIR / "fixtures/create-environment-sql/Dockerfile").read_text()
    assert "rm -rf .agents/skills .claude/skills .codex/skills" in dockerfile
    assert "git commit --allow-empty" in dockerfile
