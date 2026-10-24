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
"""Ensure the guidance interventions do not silently replace native profiles."""

import copy
import json
import tomllib

import pytest

from environments.skill_trigger.launch import authorize_prompt, codex_profile, hermes_profile


def test_codex_guidance_uses_catalog_without_inventing_prompt_text(tmp_path):
    base = {"harness": "codex", "model": "test-model", "base_url": "http://gateway/v1"}
    _, home, baseline = codex_profile(base, root=tmp_path)
    descriptor = {
        "slug": "test-model",
        "include_skills_usage_instructions": True,
        "model_messages": {"instructions_template": "captured effective baseline"},
    }
    enabled = base | {"harness": "codex_guidance", "model_catalog": {"models": [descriptor]}}
    before = copy.deepcopy(enabled)
    _, _, guidance = codex_profile(enabled, root=tmp_path)
    parsed = tomllib.loads(guidance)
    assert parsed.pop("model_catalog_json") == str(home / "models.json")
    assert parsed == tomllib.loads(baseline)
    assert enabled == before
    assert "captured effective baseline" not in guidance
    assert "developer_instructions" not in parsed
    assert parsed["model_providers"]["benchmark"]["request_max_retries"] == 0
    assert parsed["skills"]["bundled"]["enabled"] is False


@pytest.mark.parametrize(
    "catalog",
    [
        None,
        {"models": []},
        {"models": [{"slug": "test-model", "include_skills_usage_instructions": False}]},
        {
            "models": [
                {"slug": "test-model", "include_skills_usage_instructions": True},
                {"slug": "test-model", "include_skills_usage_instructions": True},
            ]
        },
    ],
)
def test_codex_guidance_requires_one_explicit_enabled_model_descriptor(tmp_path, catalog):
    config = {
        "harness": "codex_guidance",
        "model": "test-model",
        "base_url": "http://gateway/v1",
        "model_catalog": catalog,
    }
    with pytest.raises(ValueError):
        codex_profile(config, root=tmp_path)


def test_hermes_override_changes_only_tool_use_enforcement():
    native = hermes_profile(guidance=False)
    guided = hermes_profile(guidance=True)
    assert guided["agent"].pop("tool_use_enforcement") is True
    assert guided == native
    assert "execution_guidance" not in native["agent"]
    assert "tool_use_enforcement" not in native["agent"]
    # JSON serialization is accepted as YAML by the original Hermes loader.
    assert json.loads(json.dumps(native)) == native


def test_prompt_release_requires_explicit_host_signal(monkeypatch):
    from io import StringIO

    monkeypatch.setattr("sys.stdin", StringIO(""))
    with pytest.raises(RuntimeError, match="observer"):
        authorize_prompt()
    monkeypatch.setattr("sys.stdin", StringIO("run\n"))
    authorize_prompt()
