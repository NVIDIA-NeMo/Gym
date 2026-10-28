# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from nemo_gym.server_utils import ServerClient
from responses_api_agents.opencode_multimodal_agent.combined import (
    OpenCodeCombinedPromptAgent,
    OpenCodeCombinedPromptConfig,
)
from responses_api_agents.opencode_multimodal_agent.tests.test_app import agent


def combined_config(monkeypatch, tmp_path, **updates):
    native = agent(monkeypatch)
    return OpenCodeCombinedPromptConfig.model_validate(
        native.config.model_dump()
        | {
            "system_prompt": None,
            "entrypoint": "combined.py",
            "sandbox": {
                "bwrap_command": ["bwrap"],
                "python_env": tmp_path,
                "plugin_dependencies": tmp_path,
            },
        }
        | updates
    )


def test_combined_changes_only_prompt_policy(monkeypatch, tmp_path):
    cfg = combined_config(monkeypatch, tmp_path)
    obj = OpenCodeCombinedPromptAgent(config=cfg, server_client=MagicMock(spec=ServerClient))
    native = agent(monkeypatch)._build_opencode_config("r")
    combined = obj._build_opencode_config("r")
    expected = (Path(__file__).parents[1] / "prompts/combined.txt").read_text()
    assert combined["agent"]["build"].pop("prompt") == expected
    assert combined.pop("plugin") == ["file:///testbed/.combined-system-prompt.mjs"]
    assert combined == native
    assert cfg.system_prompt is None
    assert cfg.max_turns == 8 and cfg.command_exec_timeout == 300
    assert not cfg.preserve_image_format
    obj._write_opencode_config(tmp_path)
    assert (
        "output.system.splice(0, output.system.length, prompt)"
        in (tmp_path / ".combined-system-prompt.mjs").read_text()
    )


def test_combined_rejects_user_prefix(monkeypatch, tmp_path):
    with pytest.raises(ValidationError):
        combined_config(monkeypatch, tmp_path, system_prompt="duplicate prefix")
