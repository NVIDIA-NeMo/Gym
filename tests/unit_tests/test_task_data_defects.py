# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Defect 1 and Defect 2 in dataset collate schema validation.

Defect 1: when an agent references a resources server but the RS is not in the
server_instance_configs list (as happens after load_and_validate_server_instance_configs
returns only agents), _owning_resources_server_impl returns None and validation is
silently skipped.

Defect 2: when running from a cwd outside the tree, mkdir creates a data-only
`resources_servers/<name>/` directory under cwd. `find_server_dir` with `Path.is_dir`
picks up that shadow dir (which has no `task_data.py`), causing validation to be
silently skipped.
"""

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_gym.config_types import (
    ResourcesServerInstanceConfig,
    ResponsesAPIAgentServerInstanceConfig,
)
from nemo_gym.task_data import find_server_dir
from nemo_gym.train_data_utils import TrainDataProcessor


class TestDefect1AgentDeclaredSchemaValidation:
    """Defect 1: agent references RS, but RS not in config list → validation skipped.

    Production: `load_and_validate_server_instance_configs` returns only agents.
    The referenced resources server is absent from that list.
    The validator should still find and use the RS's schema.
    """

    @staticmethod
    def _agent(impl, rs_ref):
        inner = {"entrypoint": "app.py"} | ({"resources_server": rs_ref} if rs_ref else {})
        return ResponsesAPIAgentServerInstanceConfig(
            name=f"{impl}_instance",
            server_type_config_dict=OmegaConf.create({}),
            responses_api_agents={impl: inner},
        )

    @staticmethod
    def _dataset():
        from types import SimpleNamespace

        return SimpleNamespace(jsonl_fpath="data/example.jsonl")

    def test_find_server_dir_returns_none_for_shadow_dir(self, tmp_path, monkeypatch):
        """Defect 2: bare directory without markers should NOT be returned."""
        shadow_dir = tmp_path / "resources_servers" / "only_shadow"
        shadow_dir.mkdir(parents=True)
        # No task_data.py, no requirements.txt — just a bare directory

        monkeypatch.chdir(tmp_path)
        result = find_server_dir("only_shadow", base_folder="resources_servers")
        assert result is None, (
            "Defect 2: find_server_dir should NOT return a shadow directory "
            "that lacks marker files (requirements.txt/pyproject.toml/task_data.py)"
        )

    def test_find_server_dir_ignores_shadow_but_finds_real(self, tmp_path, monkeypatch):
        """A shadow dir should not interfere with finding a real server."""
        shadow_dir = tmp_path / "resources_servers" / "shadow_rs"
        shadow_dir.mkdir(parents=True)

        real_dir = tmp_path / "resources_servers" / "real_rs"
        real_dir.mkdir(parents=True)
        (real_dir / "task_data.py").write_text(
            "from pydantic import BaseModel\nclass TaskData(BaseModel):\n    pass\n"
        )
        (real_dir / "requirements.txt").write_text("")

        monkeypatch.chdir(tmp_path)
        result = find_server_dir("real_rs", base_folder="resources_servers")
        assert result is not None
        assert result.name == "real_rs"

    def test_agent_with_rs_ref_but_rs_absent_from_config_still_validates(self, tmp_path, monkeypatch):
        """The core Defect 1 test: agent references RS, but RS is NOT in the config list."""
        # Set up the resources server's schema
        rs_dir = tmp_path / "resources_servers" / "test_rs"
        rs_dir.mkdir(parents=True)
        (rs_dir / "task_data.py").write_text(
            "from pydantic import BaseModel, ConfigDict\n"
            "class TaskData(BaseModel):\n"
            "    model_config = ConfigDict(extra='forbid')\n"
            "    grading_mode: str\n"
        )
        (rs_dir / "requirements.txt").write_text("")

        # Set up the agent that references the RS
        agent_dir = tmp_path / "responses_api_agents" / "test_agent"
        agent_dir.mkdir(parents=True)
        (agent_dir / "task_data.py").write_text(
            "from pydantic import BaseModel\nclass TaskData(BaseModel):\n    pass\n"
        )

        monkeypatch.chdir(tmp_path)
        agent = self._agent("test_agent", {"type": "resources_servers", "name": "test_rs"})
        rs = ResourcesServerInstanceConfig(
            name="test_rs",
            server_type_config_dict=OmegaConf.create({}),
            resources_servers={"test_rs": {"entrypoint": "app.py", "domain": "other"}},
        )
        # Simulate the production state: full_server_instance_configs contains both
        # agents AND resources servers, but the caller only passes the filtered agent list.
        processor = TrainDataProcessor()
        processor.full_server_instance_configs = [agent, rs]
        validator = processor._task_data_validator_for(agent, self._dataset(), [agent])

        # Before the fix, validator is None (validation silently skipped).
        # After the fix, validator is not None and should catch the violation.
        assert validator is not None, (
            "Defect 1: validator should NOT be None when agent references an RS "
            "even if the RS is not in the config list"
        )
        assert validator.report.server_name == "test_rs"

    def test_agent_with_rs_ref_but_rs_absent_catches_schema_violation(self, tmp_path, monkeypatch):
        """Defect 1 + validation: a row violating the RS schema should raise ValueError."""
        # Set up the resources server's schema with a strict field
        rs_dir = tmp_path / "resources_servers" / "strict_rs"
        rs_dir.mkdir(parents=True)
        (rs_dir / "task_data.py").write_text(
            "from pydantic import BaseModel, ConfigDict, Field\n"
            "class TaskData(BaseModel):\n"
            "    model_config = ConfigDict(extra='forbid')\n"
            "    answer: str = Field(..., min_length=1)\n"
        )
        (rs_dir / "requirements.txt").write_text("")

        # Set up the agent that references the RS
        agent_dir = tmp_path / "responses_api_agents" / "strict_agent"
        agent_dir.mkdir(parents=True)

        # Create the dataset file with a schema-violating row
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True)
        (data_dir / "example.jsonl").write_text(
            json.dumps({
                "responses_create_params": {"input": []},
                "answer": "",  # Empty string violates min_length=1
            })
        )

        monkeypatch.chdir(tmp_path)
        agent = self._agent("strict_agent", {"type": "resources_servers", "name": "strict_rs"})
        rs = ResourcesServerInstanceConfig(
            name="strict_rs",
            server_type_config_dict=OmegaConf.create({}),
            resources_servers={"strict_rs": {"entrypoint": "app.py", "domain": "other"}},
        )
        processor = TrainDataProcessor()
        processor.full_server_instance_configs = [agent, rs]
        validator = processor._task_data_validator_for(agent, self._dataset(), [agent])

        assert validator is not None, (
            "Defect 1: validator should NOT be None when agent references an RS "
            "even if the RS is not in the config list"
        )
        assert validator.report.server_name == "strict_rs"

        # Validate the row — should fail
        validator.validate_row(0, {"responses_create_params": {"input": []}, "answer": ""})
        assert not validator.report.clean, "Defect 1: schema violation should be detected"
        assert validator.report.error_rows > 0, "Defect 1: error rows should be > 0"


class TestDefect2ShadowDirectory:
    """Defect 2: mkdir creates shadow dirs that fool find_server_dir."""

    def test_find_server_dir_returns_none_for_empty_shadow(self, tmp_path, monkeypatch):
        """Empty shadow directory should not be returned."""
        shadow_dir = tmp_path / "resources_servers" / "empty_shadow"
        shadow_dir.mkdir(parents=True)

        monkeypatch.chdir(tmp_path)
        result = find_server_dir("empty_shadow", base_folder="resources_servers")
        assert result is None, (
            "Defect 2: empty shadow directory should not be returned"
        )

    def test_find_server_dir_returns_none_for_task_data_only_no_marker(self, tmp_path, monkeypatch):
        """A directory with only task_data.py but no marker should still work
        (task_data.py is itself a valid marker)."""
        rs_dir = tmp_path / "resources_servers" / "task_data_only"
        rs_dir.mkdir(parents=True)
        (rs_dir / "task_data.py").write_text(
            "from pydantic import BaseModel\nclass TaskData(BaseModel):\n    pass\n"
        )

        monkeypatch.chdir(tmp_path)
        result = find_server_dir("task_data_only", base_folder="resources_servers")
        assert result is not None, "task_data.py alone should be recognized as a valid server"

    def test_find_server_dir_returns_none_for_pure_mkdir_shadow(self, tmp_path, monkeypatch):
        """A directory created by mkdir with no files at all should not be returned."""
        shadow_dir = tmp_path / "resources_servers" / "pure_shadow"
        shadow_dir.mkdir(parents=True)
        # Create a subdirectory (as mkdir -p would) but no files
        (shadow_dir / "subdir").mkdir()

        monkeypatch.chdir(tmp_path)
        result = find_server_dir("pure_shadow", base_folder="resources_servers")
        assert result is None, (
            "Defect 2: directory with only subdirectories (no marker files) should not be returned"
        )
