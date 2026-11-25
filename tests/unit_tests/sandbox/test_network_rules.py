# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The typed network rule of a sandbox block, and how each provider declares and applies the rules it supports."""

from typing import Any, Literal

import pytest
from pydantic import ValidationError

from nemo_gym.sandbox.config import NetworkRuleOptions, SandboxConfig
from nemo_gym.sandbox.providers import registry
from nemo_gym.sandbox.providers.base import NETWORK_RULES, NetworkRule, SandboxCapabilities
from nemo_gym.sandbox.providers.docker.provider import DockerCreateConfig, DockerProvider
from nemo_gym.sandbox.providers.e2b.provider import E2BProvider
from nemo_gym.sandbox.providers.e2b.provider import _network_options as e2b_network_options
from nemo_gym.sandbox.providers.opensandbox.provider import OpenSandboxProvider
from nemo_gym.sandbox.providers.registry import get_provider_class


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
            ({"network": "model_only", "model_server": ""}, "model_server"),
            ({"network": "model_only", "model_server": "  "}, "model_server"),
            ({"network": "model_and_tools", "model_server": "", "tool_servers": ["t"]}, "model_server"),
            ({"network": "model_and_tools", "model_server": "  ", "tool_servers": ["t"]}, "model_server"),
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
            snapshot_keeps=frozenset(), network_rules=frozenset({"inherit"}), reconnect="always"
        )

    def test_opensandbox_snapshots_keep_files_and_docker_takes_none(self) -> None:
        assert OpenSandboxProvider.capabilities.snapshot_keeps == {"files"}
        assert DockerProvider.capabilities.snapshot_keeps == frozenset()


class TestOpenSandboxNetworkOptions:
    def _options(self, rule: NetworkRule, hosts: list[str], options: dict | None = None) -> dict:
        provider = OpenSandboxProvider(networking={"enforces_network_policy": True})
        assert provider.capabilities.network_rules == NETWORK_RULES
        return provider.network_options(rule, hosts, options or {})

    @pytest.mark.parametrize(
        "networking", [{}, {"enforces_network_policy": False}, {"enforces_network_policy": "true"}]
    )
    @pytest.mark.parametrize("rule", sorted(NETWORK_RULES - {"inherit"}))
    def test_restricted_rules_require_an_operator_declaration(self, networking: dict, rule: NetworkRule) -> None:
        provider = OpenSandboxProvider(networking=networking)
        assert provider.capabilities.network_rules == {"inherit"}
        with pytest.raises(ValueError, match="networking.enforces_network_policy"):
            provider.network_options(rule, [], {})

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
        assert OpenSandboxProvider().network_options("inherit", [], {"volumes": ["v"]}) == {"volumes": ["v"]}


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
            network_options=NetworkRuleOptions("no_network", {"run_args": ["--network", "none"]}),
        )

        assert spec.provider_options == {"run_args": ["--network", "none"]}

    @pytest.mark.parametrize("options", [{}, NetworkRuleOptions("model_only", {})])
    def test_options_not_built_for_the_blocks_rule_are_refused(self, options: Any) -> None:
        # An empty mapping would start the sandbox with the provider's default egress.
        with pytest.raises(ValueError, match="takes the NetworkRuleOptions the owner built for it"):
            SandboxConfig(network="no_network").spec(
                {"docker": {}}, image="image", workdir=None, network_options=options
            )


class _OwnNetwork(SandboxConfig):
    network: Literal["package_index"] = "package_index"  # type: ignore[assignment]


def test_a_subclass_with_its_own_network_setting_keeps_its_own_meaning() -> None:
    assert _OwnNetwork().network == "package_index"


@pytest.mark.parametrize(
    "fields", [{"allowed_hosts": ["pypi.org"]}, {"model_server": "policy"}, {"tool_servers": ["search"]}]
)
def test_the_typed_rules_fields_are_refused_beside_a_subclasss_own_network_setting(fields: dict) -> None:
    # They would load and do nothing.
    with pytest.raises(ValidationError, match="go with a network rule"):
        _OwnNetwork(**fields)


@pytest.mark.parametrize(
    "host", ["pypi.org", "*.pypi.org", "files.pythonhosted.org.", "10.0.0.5", "10.0.0.0/8", "2001:db8::1"]
)
def test_allowed_hosts_take_what_an_egress_rule_matches(host: str) -> None:
    assert SandboxConfig(network="hosts_only", allowed_hosts=[host]).allowed_hosts == [host]


@pytest.mark.parametrize(
    "host",
    [
        "https://pypi.org/simple",
        "pypi.org:443",
        "pypi.org/simple",
        "*",
        "localhost",
        "127.0.0.1",
        "::1",
        "0.0.0.0",
        "0.0.0.0/0",
        "-bad-.org",
        "",
    ],
)
def test_allowed_hosts_refuse_what_no_egress_rule_would_match(host: str) -> None:
    with pytest.raises(ValidationError, match="allowed_hosts entry"):
        SandboxConfig(network="hosts_only", allowed_hosts=[host])


TABLE = [
    ("opensandbox", {"inherit"}, "always"),
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
    if provider_class.capabilities.network_rules - {"inherit"}:
        assert hasattr(provider_class, "network_options")


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

    @pytest.mark.parametrize("extra_run_args", [["--network", "host"], ["--net=bridge"]])
    def test_a_provider_whose_extra_run_args_pick_a_network_refuses_no_network(self, extra_run_args) -> None:
        provider = object.__new__(DockerProvider)
        provider._create_config = DockerCreateConfig(extra_run_args=extra_run_args)

        with pytest.raises(ValueError, match="create.extra_run_args pick a network"):
            provider.network_options("no_network", [], {})


class TestE2BAppliesEveryRuleAsANetworkPolicy:
    def test_a_restricted_rule_becomes_a_deny_by_default_policy_e2b_maps(self) -> None:
        options = object.__new__(E2BProvider).network_options("model_only", ["10.0.0.5"], {"template": "t"})

        assert options == {
            "template": "t",
            "network_policy": {"defaultAction": "deny", "egress": [{"action": "allow", "target": "10.0.0.5"}]},
        }
        assert e2b_network_options(options["network_policy"]) == {"deny_out": ["0.0.0.0/0"], "allow_out": ["10.0.0.5"]}

    def test_no_network_denies_all_egress(self) -> None:
        options = object.__new__(E2BProvider).network_options("no_network", [], {})

        assert e2b_network_options(options["network_policy"]) == {"deny_out": ["0.0.0.0/0"], "allow_out": []}

    def test_inherit_leaves_the_options(self) -> None:
        assert object.__new__(E2BProvider).network_options("inherit", [], {"template": "t"}) == {"template": "t"}
