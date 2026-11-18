# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``phase_network_policy``: a Harbor phase's ``network_mode`` as the OpenSandbox egress policy, failing closed;
``with_harness_hosts``: the agent's policy opened to the hosts its harness needs."""

import pytest

from environment_servers.harbor.network_policy import phase_network_policy, with_harness_hosts


class TestPhaseNetworkPolicy:
    """`phase_network_policy`: a Harbor phase's `network_mode` as the OpenSandbox egress policy, failing closed."""

    def test_public_phases_have_no_policy(self):
        assert phase_network_policy("opensandbox", None) is None
        assert phase_network_policy("docker", "public") is None
        assert phase_network_policy(None, "public", []) is None
        with pytest.raises(ValueError, match="only valid when network_mode='allowlist'"):
            phase_network_policy("opensandbox", "public", ["example.org"])

    def test_no_network_denies_everything(self):
        assert phase_network_policy("opensandbox", "no-network") == {"defaultAction": "deny", "egress": []}
        with pytest.raises(ValueError, match="only valid when network_mode='allowlist'"):
            phase_network_policy("opensandbox", "no-network", ["example.org"])

    def test_allowlist_hosts_become_ordered_allow_rules(self):
        policy = phase_network_policy(
            "opensandbox",
            "allowlist",
            ["PyPI.org", "*.nvidia.com", "pypi.org.", "10.0.0.7", "10.0.0.7.", "2001:DB8:0::1"],
        )
        assert policy == {
            "defaultAction": "deny",
            "egress": [
                {"action": "allow", "target": "pypi.org"},
                {"action": "allow", "target": "*.nvidia.com"},
                {"action": "allow", "target": "10.0.0.7"},
                {"action": "allow", "target": "2001:db8::1"},
            ],
        }
        # An empty allowlist is a deny-all policy, as Harbor resolves it.
        assert phase_network_policy("opensandbox", "allowlist", None) == {
            "defaultAction": "deny",
            "egress": [],
        }

    @pytest.mark.parametrize(
        ("host", "reason"),
        [
            ("10.0.0.0/8", "CIDR ranges are not supported"),
            ("", "must not be empty"),
            ("http://pypi.org", "not a host name"),
            ("pypi org", "not a host name"),
            ("*pypi.org", "not a host name"),
            # The spellings Harbor's `normalize_allowed_hosts` refuses at task parse are refused here too, rather
            # than forwarded verbatim as rules the sidecar never matches.
            ("*.10.0.0.1", "wildcards must target host names, not IP addresses"),
            ("fe80::1%eth0", "zone ids are not supported"),
            ("[2001:db8::1]", "written without brackets"),
        ],
    )
    def test_entries_the_policy_cannot_express_are_refused(self, host, reason):
        with pytest.raises(ValueError, match=reason):
            phase_network_policy("opensandbox", "allowlist", ["ok.example", host])

    @pytest.mark.parametrize("provider", ["docker", "daytona", None])
    def test_other_providers_fail_closed(self, provider):
        with pytest.raises(ValueError, match="requires the OpenSandbox network-policy provider"):
            phase_network_policy(provider, "no-network")
        with pytest.raises(ValueError, match="requires the OpenSandbox"):
            phase_network_policy(provider, "allowlist", ["example.org"])

    def test_unknown_mode_is_refused(self):
        with pytest.raises(ValueError, match="Unknown network_mode"):
            phase_network_policy("opensandbox", "firewalled")


class TestWithHarnessHosts:
    """`with_harness_hosts`: the hosts a harness reaches from the agent's sandbox, after the task's own rules."""

    def test_harness_hosts_follow_the_tasks_rules_without_repeats(self):
        policy = phase_network_policy("opensandbox", "allowlist", ["example.com"])
        opened = with_harness_hosts(policy, ["10.1.2.3", "Registry.npmjs.org.", "example.com", "10.1.2.3"])
        assert opened == {
            "defaultAction": "deny",
            "egress": [
                {"action": "allow", "target": "example.com"},
                {"action": "allow", "target": "10.1.2.3"},
                {"action": "allow", "target": "registry.npmjs.org"},
            ],
        }
        # The task's policy is not changed in place.
        assert policy["egress"] == [{"action": "allow", "target": "example.com"}]

    def test_no_hosts_leave_the_tasks_policy(self):
        policy = phase_network_policy("opensandbox", "no-network")
        assert with_harness_hosts(policy, []) == {"defaultAction": "deny", "egress": []}

    @pytest.mark.parametrize(
        ("host", "reason"),
        [
            ("127.0.0.1", "loopback or unspecified address"),
            ("::1", "loopback or unspecified address"),
            ("0.0.0.0", "loopback or unspecified address"),
            ("LocalHost.", "loopback or unspecified address"),
            ("", "must not be empty"),
            ("[::1]", "written without brackets"),
            ("http://model:8000", "is not a host name"),
        ],
    )
    def test_hosts_the_sandbox_cannot_reach_are_refused(self, host, reason):
        with pytest.raises(ValueError, match=f"host the harness needs.*{reason}|{reason}"):
            with_harness_hosts(phase_network_policy("opensandbox", "no-network"), ["10.1.2.3", host])
