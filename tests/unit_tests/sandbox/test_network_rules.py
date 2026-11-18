# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The typed network rule of a sandbox block, and how each provider declares and applies the rules it supports."""

from typing import Literal

import pytest
from pydantic import ValidationError

from nemo_gym.sandbox import SandboxCapabilities, SandboxConfig, get_provider_class
from nemo_gym.sandbox.providers.docker.provider import DockerCreateConfig, DockerProvider
from nemo_gym.sandbox.providers.opensandbox.provider import OpenSandboxProvider


class TestSandboxConfigNetwork:
    @pytest.mark.parametrize(
        "block",
        [
            {},
            {"network": "inherit"},
            {"network": "no_network"},
            {"network": "model_only", "model_server": "policy_model"},
            {"network": "model_and_tools", "model_server": "policy_model", "tool_servers": ["search"]},
            {"network": "hosts_only", "allowed_hosts": ["proxy.internal"]},
            {"network": "model_only", "model_server": "policy_model", "allowed_hosts": ["proxy.internal"]},
            # Without a typed rule, a provider's own network option keeps working for the deprecation release.
            {"provider_options": {"network_policy": {"defaultAction": "deny"}}},
        ],
    )
    def test_a_consistent_block_loads(self, block: dict) -> None:
        assert SandboxConfig.model_validate(block).network == block.get("network")

    @pytest.mark.parametrize(
        ("block", "message"),
        [
            ({"network": "model_only"}, "model_server"),
            ({"network": "no_network", "model_server": "policy_model"}, "model_server"),
            ({"network": "model_only", "model_server": "m", "tool_servers": ["t"]}, "tool_servers"),
            ({"network": "model_and_tools", "model_server": "m"}, "tool_servers"),
            ({"network": "hosts_only"}, "allowed_hosts"),
            ({"network": "inherit", "allowed_hosts": ["h"]}, "allowed_hosts"),
            ({"network": "no_network", "allowed_hosts": ["h"]}, "hosts_only"),
            ({"allowed_hosts": ["h"]}, "allowed_hosts"),
            ({"network": "no_network", "provider_options": {"network_policy": {}}}, "set only network"),
            ({"network": "inherit", "provider_options": {"network": "none"}}, "set only network"),
            ({"network": "open"}, "network"),
        ],
    )
    def test_an_inconsistent_block_is_refused(self, block: dict, message: str) -> None:
        with pytest.raises(ValidationError, match=message):
            SandboxConfig.model_validate(block)


class TestProviderCapabilities:
    def test_the_defaults_are_what_every_provider_supports(self) -> None:
        assert SandboxCapabilities() == SandboxCapabilities(
            snapshot_keeps=frozenset(), network_rules=frozenset({"inherit"}), reconnect="always", close_by_label=False
        )

    @pytest.mark.parametrize(
        ("name", "rules", "reconnect"),
        [
            ("opensandbox", {"inherit", "no_network", "model_only", "model_and_tools", "hosts_only"}, "always"),
            ("docker", {"inherit", "no_network"}, "same_host"),
            ("e2b", {"inherit"}, "always"),
            ("local", {"inherit"}, "never"),
        ],
    )
    def test_the_providers_declare_what_the_design_lists(self, name: str, rules: set, reconnect: str) -> None:
        capabilities = get_provider_class(name).capabilities

        assert capabilities.network_rules == rules
        assert capabilities.reconnect == reconnect

    def test_only_opensandbox_snapshots_keep_files(self) -> None:
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
    @pytest.mark.parametrize("block", [{"network": "no_network"}, {"network": "hosts_only", "allowed_hosts": ["h"]}])
    def test_spec_alone_refuses_a_restricted_rule(self, block: dict) -> None:
        with pytest.raises(ValueError, match="OwnedSandboxes.spec"):
            SandboxConfig.model_validate(block).spec({"docker": {}}, image="image", workdir=None)

    @pytest.mark.parametrize("block", [{}, {"network": "inherit"}])
    def test_spec_alone_builds_an_open_or_unset_rule(self, block: dict) -> None:
        spec = SandboxConfig.model_validate({**block, "provider_options": {"volumes": ["v"]}}).spec(
            {"docker": {}}, image="image", workdir=None
        )

        assert spec.provider_options == {"volumes": ["v"]}

    def test_the_owners_options_replace_the_provider_options(self) -> None:
        spec = SandboxConfig(network="no_network", provider_options={"volumes": ["v"]}).spec(
            {"docker": {}},
            image="image",
            workdir=None,
            network_options={"run_args": ["--network", "none"]},
        )

        assert spec.provider_options == {"run_args": ["--network", "none"]}


class _OwnNetwork(SandboxConfig):
    network: Literal["package_index"] = "package_index"  # type: ignore[assignment]


def test_a_subclass_with_its_own_network_setting_keeps_its_own_checks() -> None:
    # The typed rule's checks would refuse allowed_hosts without a restricted rule; this subclass owns its network.
    assert _OwnNetwork(allowed_hosts=["pypi.org"]).allowed_hosts == ["pypi.org"]


@pytest.mark.parametrize(
    ("name", "rules", "reconnect"),
    [
        ("opensandbox", {"inherit", "no_network", "model_only", "model_and_tools", "hosts_only"}, "always"),
        ("docker", {"inherit", "no_network"}, "same_host"),
        ("e2b", {"inherit"}, "always"),
        ("openshell", {"inherit"}, "always"),
        ("daytona", {"inherit"}, "always"),
        ("apptainer", {"inherit"}, "same_host"),
        ("enroot", {"inherit"}, "same_host"),
        ("local", {"inherit"}, "never"),
        ("ecs_fargate", {"inherit"}, "never"),
    ],
)
def test_every_built_in_provider_declares_the_design_tables_values(name: str, rules: set, reconnect: str) -> None:
    capabilities = get_provider_class(name).capabilities

    assert capabilities.network_rules == rules
    assert capabilities.reconnect == reconnect


@pytest.mark.parametrize(
    "name", ["opensandbox", "docker", "e2b", "openshell", "daytona", "apptainer", "enroot", "local", "ecs_fargate"]
)
def test_a_provider_declares_snapshots_exactly_when_it_takes_them(name: str) -> None:
    provider_class = get_provider_class(name)

    takes_snapshots = all(hasattr(provider_class, method) for method in ("snapshot", "wait_snapshot_ready"))
    assert bool(provider_class.capabilities.snapshot_keeps) == takes_snapshots
    assert bool(provider_class.capabilities.network_rules - {"inherit"}) == hasattr(provider_class, "network_options")


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
