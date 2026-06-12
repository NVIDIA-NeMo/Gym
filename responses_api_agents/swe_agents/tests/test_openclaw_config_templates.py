# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json

import pytest

from responses_api_agents.swe_agents.openclaw.config_templates import build_openclaw_json


def _kwargs(**overrides):
    base = dict(
        workspace_path="/testbed",
        model_name="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        upstream_base_url="http://127.0.0.1:0/v1",
        upstream_api_key="dummy",
        gateway_auth_token="00000000-0000-0000-0000-000000000000",
    )
    base.update(overrides)
    return base


def test_static_config_structure():
    """Structural invariants of the rendered config that don't depend on optional inputs:
    gateway wiring, models-block provider wiring, the tools allow-list, and skills/plugins."""
    cfg = build_openclaw_json(**_kwargs())

    # Gateway: token auth + locked-down local mode.
    assert cfg["gateway"]["auth"] == {"mode": "token", "token": "00000000-0000-0000-0000-000000000000"}
    assert cfg["gateway"]["mode"] == "local"
    assert cfg["gateway"]["bind"] == "loopback"
    assert cfg["gateway"]["controlUi"] == {"enabled": False}

    # discovery/update are fully off so the agent never phones home mid-run.
    assert cfg["discovery"] == {"mdns": {"mode": "off"}, "wideArea": {"enabled": False}}
    assert cfg["update"] == {"auto": {"enabled": False}, "checkOnStart": False}

    # Models block: the vllm provider points at the upstream and exposes the policy model.
    assert cfg["models"]["mode"] == "replace"
    vllm = cfg["models"]["providers"]["vllm"]
    assert vllm["baseUrl"] == "http://127.0.0.1:0/v1"
    assert vllm["apiKey"] == "dummy"
    assert vllm["api"] == "openai-responses"
    # reasoning is HARDCODED False on purpose — flipping it on for vLLM sends the system
    # prompt as role:"developer" (HTTP 400) and injects untranslatable reasoning-effort params.
    # Thinking still round-trips on-policy with reasoning:false (wet-test verified).
    assert vllm["models"] == [
        {
            "id": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
            "name": "policy",
            "api": "openai-responses",
            "input": ["text"],
            "reasoning": False,
        }
    ]

    # Agents defaults: workspace + policy-model wiring; bootstrap/heartbeat/startup off.
    d = cfg["agents"]["defaults"]
    assert d["workspace"] == "/testbed"
    assert d["models"] == {"vllm/Qwen/Qwen3-Coder-30B-A3B-Instruct": {"alias": "policy"}}
    assert d["model"] == {"primary": "vllm/Qwen/Qwen3-Coder-30B-A3B-Instruct"}
    assert d["skipBootstrap"] is True
    assert d["heartbeat"] == {"every": "0m"}
    assert d["startupContext"] == {"enabled": False}

    # Tools: apply_patch is OpenAI/Codex-only in openclaw; it is gated off for the
    # vllm/Qwen provider, so we don't advertise it (avoids a misleading warning).
    t = cfg["tools"]
    assert t["allow"] == ["read", "write", "edit", "exec", "process"]
    assert "apply_patch" not in t["allow"]
    assert t["fs"] == {"workspaceOnly": True}
    assert t["exec"]["ask"] == "off"
    assert t["exec"]["security"] == "full"
    assert t["loopDetection"] == {"enabled": False}

    # Skills triple-defense: zero skills in prompt, only a nonexistent bundled skill allowed,
    # and every known bundled skill explicitly disabled.
    s = cfg["skills"]
    assert s["limits"] == {"maxSkillsInPrompt": 0, "maxSkillsPromptChars": 0}
    assert s["allowBundled"] == ["__nonexistent_skill__"]
    for key in ("healthcheck", "node-connect", "skill-creator", "taskflow", "taskflow-inbox-triage", "weather"):
        assert s["entries"][key] == {"enabled": False}

    # Plugins: allowlist gates everything except the vllm plugin; memory slot disabled.
    p = cfg["plugins"]
    assert p["allow"] == ["vllm"]
    assert p["bundledDiscovery"] == "allowlist"
    assert p["slots"] == {"memory": "none"}


def test_build_openclaw_json_idle_timeout_omitted_by_default():
    """No timeoutSeconds emitted unless requested (preserve OpenClaw's provider default)."""
    cfg = build_openclaw_json(**_kwargs())
    assert "timeoutSeconds" not in cfg["models"]["providers"]["vllm"]


def test_build_openclaw_json_idle_timeout_emitted_when_set():
    """model_idle_timeout_seconds raises the llm-idle-timeout watchdog so slow (cold-start)
    turns are not preempted into a same-turn retry (the Qwen3.5-9B duplicate-turn root cause)."""
    cfg = build_openclaw_json(**_kwargs(model_idle_timeout_seconds=1200))
    assert cfg["models"]["providers"]["vllm"]["timeoutSeconds"] == 1200
    # The model block is otherwise unchanged (reasoning stays hardcoded False).
    assert cfg["models"]["providers"]["vllm"]["models"][0]["reasoning"] is False


def test_pathprepend_includes_agent_env_bin_after_wrapper():
    """The dataset's runtime-env bin is prepended so the agent's `python` is the repo
    interpreter — openclaw's exec rebuilds PATH from a sanitized base, and pathPrepend is
    the only lever that survives into the agent's commands. The security-wrapper dir MUST
    stay FIRST so it still intercepts denylisted commands (git/curl/...)."""
    cfg = build_openclaw_json(**_kwargs(agent_env_bin="/opt/miniconda3/envs/testbed/bin"))
    assert cfg["tools"]["exec"]["pathPrepend"] == ["/openclaw_setup/bin", "/opt/miniconda3/envs/testbed/bin"]


def test_pathprepend_without_agent_env_bin_is_wrapper_only():
    cfg = build_openclaw_json(**_kwargs(agent_env_bin=None))
    assert cfg["tools"]["exec"]["pathPrepend"] == ["/openclaw_setup/bin"]


def test_sampling_params_emitted_with_provider_specific_keys():
    # OpenClaw's openai-responses transport forwards only temperature (verbatim) and maxTokens
    # (-> max_output_tokens) from agents.defaults.params; verified via wet smoke. top_p is NOT
    # config-settable (the transport never wires it) — it's injected by the stream shim instead.
    cfg = build_openclaw_json(**_kwargs(temperature=0.0, max_output_tokens=131072))
    assert cfg["agents"]["defaults"]["params"] == {
        "temperature": 0.0,
        "maxTokens": 131072,
    }


def test_no_sampling_params_means_no_params_key():
    cfg = build_openclaw_json(**_kwargs())
    assert "params" not in cfg["agents"]["defaults"]


def test_partial_sampling_params_only_set_provided():
    cfg = build_openclaw_json(**_kwargs(max_output_tokens=4096))
    assert cfg["agents"]["defaults"]["params"] == {"maxTokens": 4096}


def test_build_openclaw_json_is_json_serialisable():
    cfg = build_openclaw_json(**_kwargs())
    json.dumps(cfg)  # must not raise


def test_build_openclaw_json_rejects_unknown_kwarg():
    with pytest.raises(TypeError):
        build_openclaw_json(**_kwargs(), unknown_field=42)
