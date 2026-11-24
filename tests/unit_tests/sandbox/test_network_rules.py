# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The typed network rule of a sandbox block, and how each provider declares and applies the rules it supports."""

from typing import Literal

import pytest
from pydantic import ValidationError

from nemo_gym.sandbox.config import SandboxConfig
from nemo_gym.sandbox.providers import registry
from nemo_gym.sandbox.providers.base import SandboxCapabilities
from nemo_gym.sandbox.providers.docker.provider import DockerCreateConfig, DockerProvider
from nemo_gym.sandbox.providers.opensandbox.provider import OpenSandboxProvider
from nemo_gym.sandbox.providers.registry import get_provider_class


class TestSandboxConfigNetwork:
    @pytest.mark.parametrize(
        "block",
        [
            {},
            {"network_access": "inherit"},
            {"network_access": "no_network"},
            {"network_access": "model_only", "model_server": "policy_model"},
            {"network_access": "model_and_tools", "model_server": "policy_model", "tool_servers": ["search"]},
            {"network_access": "hosts_only", "allowed_hosts": ["proxy.internal"]},
            {"network_access": "model_only", "model_server": "policy_model", "allowed_hosts": ["proxy.internal"]},
            # Without a typed rule, a provider's own network option keeps working for the deprecation release.
            {"provider_options": {"network_policy": {"defaultAction": "deny"}}},
        ],
    )
    def test_a_consistent_block_loads(self, block: dict) -> None:
        assert SandboxConfig.model_validate(block).network_access == block.get("network_access")

    @pytest.mark.parametrize(
        ("block", "message"),
        [
            ({"network_access": "model_only"}, "model_server"),
            ({"network_access": "no_network", "model_server": "policy_model"}, "model_server"),
            ({"network_access": "model_only", "model_server": "m", "tool_servers": ["t"]}, "tool_servers"),
            ({"network_access": "model_and_tools", "model_server": "m"}, "tool_servers"),
            ({"network_access": "hosts_only"}, "allowed_hosts"),
            ({"network_access": "inherit", "allowed_hosts": ["h"]}, "allowed_hosts"),
            ({"network_access": "no_network", "allowed_hosts": ["h"]}, "hosts_only"),
            ({"allowed_hosts": ["h"]}, "allowed_hosts"),
            ({"network_access": "no_network", "provider_options": {"network_policy": {}}}, "set only network"),
            ({"network_access": "inherit", "provider_options": {"network_access": "none"}}, "set only network"),
            ({"network_access": "open"}, "network"),
        ],
    )
    def test_an_inconsistent_block_is_refused(self, block: dict, message: str) -> None:
        with pytest.raises(ValidationError, match=message):
            SandboxConfig.model_validate(block)


class TestProviderCapabilities:
    def test_the_defaults_are_what_every_provider_supports(self) -> None:
        assert SandboxCapabilities() == SandboxCapabilities(
            snapshot_keeps=frozenset(), network_rules=frozenset({"inherit"}), reconnect="always"
        )

    def test_opensandbox_snapshots_keep_files_and_docker_takes_none(self) -> None:
        assert OpenSandboxProvider.capabilities.snapshot_keeps == {"files"}
        assert DockerProvider.capabilities.snapshot_keeps == frozenset()


class TestOpenSandboxNetworkOptions:
    def _options(self, rule: str, hosts: list[str], options: dict | None = None) -> dict:
        return object.__new__(OpenSandboxProvider).network_options(rule, hosts, options or {})

    def test_a_restricted_rule_denies_by_default_and_allows_only_its_hosts(self) -> None:
        options = self._options("model_only", ["model.internal.", "model.internal", "proxy"], {"volumes": ["v"]})

        assert options == {
            "volumes": ["v"],
            "network_policy": {
                "defaultAction": "deny",
                "egress": [
                    {"action": "allow", "target": "model.internal"},
                    {"action": "allow", "target": "proxy"},
                ],
            },
        }

    def test_no_network_denies_all_egress(self) -> None:
        assert self._options("no_network", [])["network_policy"] == {"defaultAction": "deny", "egress": []}

    def test_inherit_changes_nothing(self) -> None:
        assert self._options("inherit", [], {"volumes": ["v"]}) == {"volumes": ["v"]}


class TestDockerNetworkOptions:
    def _provider(self, network: str | None = None) -> DockerProvider:
        provider = object.__new__(DockerProvider)
        provider._create_config = DockerCreateConfig(network=network)
        return provider

    def test_no_network_runs_the_container_without_a_network(self) -> None:
        options = self._provider().network_options("no_network", [], {"run_args": ["--cap-drop", "ALL"]})

        assert options == {"run_args": ["--cap-drop", "ALL", "--network", "none"]}

    def test_a_rule_that_allows_hosts_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cannot restrict egress to a list of hosts"):
            self._provider().network_options("model_only", ["model.internal"], {})

    def test_no_network_is_refused_when_every_container_joins_another_network(self) -> None:
        with pytest.raises(ValueError, match="cannot also start one with no network"):
            self._provider(network="gym").network_options("no_network", [], {})


class TestARuleNeverReachesASpecWithoutItsPolicy:
    @pytest.mark.parametrize(
        "block", [{"network_access": "no_network"}, {"network_access": "hosts_only", "allowed_hosts": ["h"]}]
    )
    def test_spec_alone_refuses_a_restricted_rule(self, block: dict) -> None:
        with pytest.raises(ValueError, match="OwnedSandboxes.spec"):
            SandboxConfig.model_validate(block).spec({"docker": {}}, image="image", workdir=None)

    @pytest.mark.parametrize("block", [{}, {"network_access": "inherit"}])
    def test_spec_alone_builds_an_open_or_unset_rule(self, block: dict) -> None:
        spec = SandboxConfig.model_validate({**block, "provider_options": {"volumes": ["v"]}}).spec(
            {"docker": {}}, image="image", workdir=None
        )

        assert spec.provider_options == {"volumes": ["v"]}

    def test_the_owners_options_replace_the_provider_options(self) -> None:
        spec = SandboxConfig(network_access="no_network", provider_options={"volumes": ["v"]}).spec(
            {"docker": {}},
            image="image",
            workdir=None,
            _network_options={"run_args": ["--network", "none"]},
        )

        assert spec.provider_options == {"run_args": ["--network", "none"]}


class _OwnNetwork(SandboxConfig):
    network: Literal["package_index"] = "package_index"  # type: ignore[assignment]


def test_a_subclass_with_its_own_network_setting_keeps_its_own_checks() -> None:
    assert _OwnNetwork().network == "package_index"
    with pytest.raises(ValidationError, match="allowed_hosts"):
        _OwnNetwork(allowed_hosts=["pypi.org"])
    with pytest.raises(ValidationError, match="cannot both be set"):
        _OwnNetwork(network_access="no_network")


TABLE = [
    ("opensandbox", {"inherit", "no_network", "model_only", "model_and_tools", "hosts_only"}, "always"),
    ("docker", {"inherit", "no_network"}, "same_host"),
    ("e2b", {"inherit", "no_network", "model_only", "model_and_tools", "hosts_only"}, "always"),
    ("openshell", {"inherit"}, "always"),
    ("daytona", {"inherit"}, "never"),
    ("apptainer", {"inherit"}, "same_host"),
    ("bwrap", {"inherit"}, "same_host"),
    ("enroot", {"inherit"}, "same_host"),
    ("local", {"inherit"}, "never"),
    ("ecs_fargate", {"inherit"}, "same_host"),
]
BUILT_IN_PROVIDERS = sorted(registry._BUILTIN_PROVIDER_LOADERS)


def test_the_design_table_lists_every_built_in_provider() -> None:
    # A new built-in provider has to declare what it can do, and be added to the table.
    assert {name for name, _, _ in TABLE} == set(BUILT_IN_PROVIDERS)


@pytest.mark.parametrize(("name", "rules", "reconnect"), TABLE)
def test_every_built_in_provider_declares_the_design_tables_values(name: str, rules: set, reconnect: str) -> None:
    capabilities = get_provider_class(name).capabilities

    assert capabilities.network_rules == rules
    assert capabilities.reconnect == reconnect


@pytest.mark.parametrize("name", BUILT_IN_PROVIDERS)
def test_a_provider_declares_snapshots_exactly_when_it_takes_them(name: str) -> None:
    provider_class = get_provider_class(name)

    takes_snapshots = all(hasattr(provider_class, method) for method in ("snapshot", "wait_snapshot_ready"))
    assert bool(provider_class.capabilities.snapshot_keeps) == takes_snapshots
    assert bool(provider_class.capabilities.network_rules - {"inherit"}) == hasattr(provider_class, "network_options")


@pytest.mark.parametrize("name", BUILT_IN_PROVIDERS)
def test_a_provider_declares_reconnect_exactly_when_another_process_can_connect(name: str) -> None:
    provider_class = get_provider_class(name)

    connects = hasattr(provider_class, "serialize_handle") and hasattr(provider_class, "connect")
    assert (provider_class.capabilities.reconnect != "never") == connects


class TestDockerAppliesNoNetworkOnce:
    def _provider(self, network: str | None = None) -> DockerProvider:
        provider = object.__new__(DockerProvider)
        provider._create_config = DockerCreateConfig(network=network)
        return provider

    def test_a_provider_that_already_runs_without_a_network_adds_nothing(self) -> None:
        assert self._provider(network="none").network_options("no_network", [], {"volumes": ["v"]}) == {
            "volumes": ["v"]
        }

    @pytest.mark.parametrize("run_args", [["--network", "host"], ["--net=bridge"], "--network=gym"])
    def test_run_args_that_pick_a_network_are_refused(self, run_args) -> None:
        with pytest.raises(ValueError, match="pick a network themselves"):
            self._provider().network_options("no_network", [], {"run_args": run_args})


@pytest.mark.parametrize("host", ["https://pypi.org/simple", "host:443", "*", "host/path", "", "bad host"])
def test_allowed_hosts_refuses_targets_that_cannot_match(host: str) -> None:
    with pytest.raises(ValidationError, match="bare host"):
        SandboxConfig(network_access="hosts_only", allowed_hosts=[host])


@pytest.mark.parametrize("host", ["pypi.org", "*.example.org", "192.0.2.1", "192.0.2.0/24", "2001:db8::/32"])
def test_allowed_hosts_accepts_hosts_wildcards_and_ip_networks(host: str) -> None:
    assert SandboxConfig(network_access="hosts_only", allowed_hosts=[host]).allowed_hosts == [host]


def test_e2b_applies_restricted_rules_through_its_existing_adapter() -> None:
    from nemo_gym.sandbox.providers.e2b.provider import E2BProvider, _network_options

    provider = object.__new__(E2BProvider)
    options = provider.network_options("hosts_only", ["pypi.org"], {"template": "harness"})
    assert options["template"] == "harness"
    assert _network_options(options["network_policy"]) == {"deny_out": ["0.0.0.0/0"], "allow_out": ["pypi.org"]}


def test_docker_refuses_a_network_hidden_in_provider_run_args() -> None:
    provider = object.__new__(DockerProvider)
    provider._create_config = DockerCreateConfig(extra_run_args=["--network=host"])
    with pytest.raises(ValueError, match="extra_run_args pick a network"):
        provider.network_options("no_network", [], {})


def test_public_spec_cannot_bypass_a_restricted_rule_with_empty_options() -> None:
    config = SandboxConfig(network_access="hosts_only", allowed_hosts=["pypi.org"])
    with pytest.raises(TypeError, match="network_options"):
        config.spec({"opensandbox": {}}, image=None, workdir=None, network_options={})
