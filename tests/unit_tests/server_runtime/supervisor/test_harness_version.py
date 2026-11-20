# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from nemo_gym.config.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.server_runtime.supervisor import harness_version as versions


@pytest.mark.parametrize(
    ("agent", "field", "key", "release"),
    [
        ("opencode_agent", "opencode_version", "tag_name", "v2.3.2"),
        ("openclaw_agent", "openclaw_version", "version", "2026.11.5"),
        ("pi_agent", "pi_version", "version", "1.5.2"),
    ],
)
def test_latest_is_pinned_before_worker_configs_are_serialized(monkeypatch, agent, field, key, release):
    lookups = []

    def lookup(url, lookup_key):
        lookups.append((url, lookup_key))
        return release

    monkeypatch.setattr(versions, "_lookup", lookup)
    config = OmegaConf.create({field: "old", "harness_version": "latest"})
    assert versions.resolve_harness_version(agent, config) is None
    assert OmegaConf.create(OmegaConf.to_yaml(config))[field] == release.removeprefix("v")
    versions.resolve_harness_version(agent, config)
    assert len(lookups) == 1
    assert lookups[0][1] == key


def test_hermes_fork_pin_selects_worker_source_without_a_registry_lookup(monkeypatch):
    monkeypatch.setattr(versions, "_lookup", lambda *_: pytest.fail("exact pins do not need a release lookup"))
    sha = "a" * 40
    config = OmegaConf.create({"harness_version": sha, "hermes_repo": "https://github.com/example/hermes"})
    assert (
        versions.resolve_harness_version("hermes_agent", config) == f"git+https://github.com/example/hermes.git@{sha}"
    )


def test_hermes_latest_follows_the_selected_repository(monkeypatch):
    sha = "b" * 40
    urls = []

    def lookup(url, key):
        urls.append(url)
        return "v0.23.3" if key == "tag_name" else sha

    monkeypatch.setattr(versions, "_lookup", lookup)
    config = OmegaConf.create({"hermes_version": "latest", "hermes_repo": "https://github.com/example/hermes.git"})
    assert versions.resolve_harness_version("hermes_agent", config).endswith("@" + sha)
    assert urls == [
        "https://api.github.com/repos/example/hermes/releases/latest",
        "https://api.github.com/repos/example/hermes/commits/v0.23.3",
    ]
    assert config.harness_version == config.hermes_version == sha


def test_no_override_keeps_existing_defaults(monkeypatch):
    monkeypatch.setattr(versions, "_lookup", lambda *_: pytest.fail("unexpected network lookup"))
    config = OmegaConf.create({"pi_version": "0.80.2"})
    assert versions.resolve_harness_version("pi_agent", config) is None
    assert dict(config) == {"pi_version": "0.80.2"}
    assert versions.resolve_harness_version("hermes_agent", OmegaConf.create({})) is None


def test_unsupported_selector_fails_before_launch():
    with pytest.raises(ValueError, match="does not support"):
        versions.resolve_harness_version("simple_agent", OmegaConf.create({"harness_version": "latest"}))


@pytest.mark.parametrize(
    ("preset", "pin"),
    [
        ("benchmarks/deepsearchqa/configs/deepsearchqa_openclaw.yaml", "2026.11.5"),
        ("benchmarks/deepsearchqa/configs/deepsearchqa_pi.yaml", "0.80.2"),
        ("benchmarks/gdpval/openclaw.yaml", "2026.11.5"),
        ("benchmarks/gdpval/pi.yaml", "1.5.2"),
        ("benchmarks/job_bench/pi.yaml", "1.5.2"),
        ("benchmarks/swe_atlas_qa/openclaw.yaml", "2026.11.5"),
        ("benchmarks/widesearch/configs/widesearch_openclaw.yaml", "2026.11.5"),
        ("benchmarks/widesearch/configs/widesearch_pi.yaml", "0.80.2"),
        ("benchmarks/workspace_bench/pi.yaml", "1.5.2"),
        ("environments/openclaw_math/config.yaml", "2026.11.5"),
        ("environments/openclaw_reasoning_gym/config.yaml", "2026.11.5"),
        ("environments/pi_math/config.yaml", "1.5.2"),
        ("environments/pi_reasoning_gym/config.yaml", "1.5.2"),
        ("responses_api_agents/anyswe_agent/configs/anyswe_openclaw.yaml", "2026.11.5"),
        ("responses_api_agents/anyswe_agent/configs/anyswe_pi.yaml", "1.5.2"),
        ("responses_api_agents/anyterminal_agent/configs/anyterminal_openclaw.yaml", "2026.11.5"),
        ("responses_api_agents/openclaw_agent/configs/openclaw_agent.yaml", "2026.11.5"),
        ("responses_api_agents/pi_agent/configs/pi_agent.yaml", "1.5.2"),
        ("responses_api_agents/pi_agent/configs/profiles/offline_science.yaml", "1.5.2"),
        ("responses_api_agents/pi_sandboxed_agent/configs/benchmark.yaml", "1.5.2"),
    ],
)
def test_preset_pin_reaches_the_installer(preset, pin, monkeypatch):
    root = Path(__file__).resolve().parents[4]
    config = OmegaConf.load(root / preset)
    (instance,) = GlobalConfigDictParser()._agent_instances(config)
    server = instance.server_config
    pins = server.get("agent_kwargs", server)
    key = "openclaw_version" if "openclaw" in preset else "pi_version"
    package = "openclaw" if key == "openclaw_version" else "@earendil-works%2fpi-coding-agent"
    assert pins.harness_version == pin
    assert key not in pins

    lookup = Mock(return_value="2026.12.1")
    monkeypatch.setattr(versions, "_lookup", lookup)
    versions.resolve_harness_version(instance.agent_type, server)
    assert pins[key] == pin
    lookup.assert_not_called()
    pins.harness_version = "latest"
    versions.resolve_harness_version(instance.agent_type, server)
    versions.resolve_harness_version(instance.agent_type, server)
    assert pins[key] == "2026.12.1"
    lookup.assert_called_once_with(f"https://registry.npmjs.org/{package}/latest", "version")
    if pins is not server:
        assert "harness_version" not in pins  # kwargs sent to older in-sandbox Gym releases


@pytest.mark.parametrize(
    ("preset", "pin_path", "legacy_key", "old_pin"),
    [
        (
            "responses_api_agents/openclaw_agent/configs/openclaw_agent.yaml",
            "openclaw_agent.responses_api_agents.openclaw_agent",
            "openclaw_version",
            "2026.6.11",
        ),
        (
            "responses_api_agents/anyterminal_agent/configs/anyterminal_openclaw.yaml",
            "anyterminal_openclaw.responses_api_agents.anyterminal_agent.agent_kwargs",
            "openclaw_version",
            "2026.6.11",
        ),
        (
            "responses_api_agents/pi_agent/configs/pi_agent.yaml",
            "pi_agent.responses_api_agents.pi_agent",
            "pi_version",
            "0.80.2",
        ),
        (
            "benchmarks/widesearch/configs/widesearch_pi.yaml",
            "widesearch_pi_benchmark.responses_api_agents.harness_exa_search.agent_kwargs",
            "pi_version",
            "0.85.1",
        ),
    ],
)
def test_legacy_override_on_a_migrated_preset(preset, pin_path, legacy_key, old_pin):
    overrides = OmegaConf.create({"config_paths": [preset]})
    OmegaConf.update(overrides, f"{pin_path}.{legacy_key}", old_pin)
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.merge(
                GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT, overrides
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    pins = OmegaConf.select(config, pin_path)
    assert pins[legacy_key] == old_pin
    assert "harness_version" not in pins


@pytest.mark.parametrize(
    ("base_key", "override_key"), [("pi_version", "harness_version"), ("harness_version", "pi_version")]
)
def test_pin_override_on_a_composed_instance(base_key, override_key):
    base = OmegaConf.create({base_key: "1.5.2"})
    overrides = OmegaConf.create({"task_pi": {"responses_api_agents": {"pi_agent": {override_key: "0.80.2"}}}})
    GlobalConfigDictParser._apply_held_agent_override(overrides, "task_pi", "pi_agent", base)
    versions.resolve_harness_version("pi_agent", base)
    assert base.pi_version == "0.80.2"
