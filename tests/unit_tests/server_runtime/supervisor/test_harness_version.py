# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from omegaconf import OmegaConf

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
    ("agent", "version", "expected"),
    [
        ("pool_agent", "v1.0.16", "v1.0.16"),
        ("copilot_agent", "1.2.3", "1.2.3"),
        ("goose_agent", "v1.54.0", "1.54.0"),
    ],
)
def test_published_versions_reach_the_native_installer(agent, version, expected, monkeypatch):
    lookups = []

    def lookup(url, key):
        lookups.append(url)
        return version

    monkeypatch.setattr(versions, "_lookup", lookup)
    config = OmegaConf.create({"harness_version": "latest", "npm_registry": "https://mirror.example"})
    versions.resolve_harness_version(agent, config)
    field = {"pool_agent": "pool_version", "copilot_agent": "copilot_version", "goose_agent": "goose_version"}[agent]
    assert config[field] == expected
    if agent == "copilot_agent":
        assert lookups == ["https://mirror.example/@github%2fcopilot/latest"]


def test_python_release_overrides_both_openhands_packages_in_a_private_environment(tmp_path):
    config = OmegaConf.create({"harness_version": "1.59.1"})
    versions.resolve_harness_version("openhands_agent", config)
    base = OmegaConf.create({"uv_venv_dir": str(tmp_path), "head_server_deps": ["parent==1"]})
    setup = versions.python_harness_setup("openhands_agent", config, base)
    assert setup.uv_venv_dir != base.uv_venv_dir
    assert setup.head_server_deps == ["parent==1", "openhands-sdk==1.59.1", "openhands-tools==1.59.1"]
    assert setup.head_server_constraints == ["openhands-sdk==1.59.1", "openhands-tools==1.59.1"]
    assert base.head_server_deps == ["parent==1"]


def test_nooa_fork_and_sandbox_archive_use_the_same_commit(tmp_path):
    sha = "a" * 40
    repo = "https://github.com/example/labs-OO-Agents"
    cfg = OmegaConf.create({"harness_version": sha, "harness_repo": repo})
    versions.resolve_harness_version("nooa_agent", cfg)
    base = OmegaConf.create({"uv_venv_dir": str(tmp_path)})
    setup = versions.python_harness_setup("nooa_agent", cfg, base)
    assert list(setup.head_server_deps) == [
        f"nooa @ git+{repo}.git@{sha}",
        f"nooa-cli @ git+{repo}.git@{sha}#subdirectory=packages/nooa-cli",
        f"nooa-bench @ git+{repo}.git@{sha}#subdirectory=packages/nooa-bench",
    ]
    assert versions.harness_environment("nooa_agent", cfg, hermes_source=None) == {
        "NEMO_GYM_NOOA_SOURCE": f"{repo}/archive/{sha}.tar.gz"
    }
