# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the copyable Finance Agent v2 setup offline, without importing servers."""

import re
from pathlib import Path

import pytest
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_ROOT = REPO_ROOT / "resources_servers/finance_agent_v2"
TOOL_KEYS = {
    "sec_api_key": "SEC_API_KEY",
    "tavily_api_key": "TAVILY_API_KEY",
    "pricing_data_api_key": "TIINGO_API_KEY",
}


@pytest.mark.parametrize(
    "enabled_variables",
    [(), ("SEC_API_KEY",), ("TAVILY_API_KEY",), ("TIINGO_API_KEY",), tuple(TOOL_KEYS.values())],
    ids=["none", "sec-only", "tavily-only", "tiingo-only", "all"],
)
def test_readme_optional_tool_credentials_resolve_like_shipped_config(monkeypatch, enabled_variables):
    # Never read the developer's credentials or env.yaml; every relevant value is
    # either removed or replaced with an inert test string, and no server starts.
    for variable in TOOL_KEYS.values():
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.delenv("FINANCE_AGENT_V2_CACHE_DIR", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-model-key")
    monkeypatch.setenv("POLICY_DROP_REASONING_ITEMS", "false")
    for variable in enabled_variables:
        monkeypatch.setenv(variable, f"offline-{variable.lower()}")

    readme = (SERVER_ROOT / "README.md").read_text()
    snippets = re.findall(r"```yaml\n(# env.yaml\n.*?)\n```", readme, flags=re.DOTALL)
    assert len(snippets) == 1, "Expected one copyable env.yaml example"
    documented_env = OmegaConf.create(snippets[0])
    shipped = OmegaConf.load(SERVER_ROOT / "configs/finance_agent_v2.yaml")
    model = OmegaConf.load(REPO_ROOT / "responses_api_models/openai_model/configs/openai_model.yaml")
    resolved = OmegaConf.to_container(OmegaConf.merge(shipped, model, documented_env), resolve=True)
    resources = resolved["finance_agent_v2_resources_server"]["resources_servers"]["finance_agent_v2"]
    shipped_resources = OmegaConf.to_container(shipped, resolve=True)["finance_agent_v2_resources_server"][
        "resources_servers"
    ]["finance_agent_v2"]

    for key, variable in TOOL_KEYS.items():
        expected = f"offline-{variable.lower()}" if variable in enabled_variables else None
        assert resources[key] == expected
        assert resources[key] == shipped_resources[key]
    assert resolved["policy_model"]["responses_api_models"]["openai_model"]["openai_api_key"] == "offline-model-key"
    assert (
        resolved["search_judge_model"]["responses_api_models"]["openai_model"]["openai_api_key"] == "offline-model-key"
    )
