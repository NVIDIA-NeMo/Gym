# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``phase_network_policy``: a Harbor phase's ``network_mode`` as the OpenSandbox egress policy, failing closed."""

import pytest

from environment_servers.harbor.network_policy import phase_network_policy


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
            "opensandbox", "allowlist", ["PyPI.org", "*.nvidia.com", "pypi.org.", "10.0.0.7", "2001:DB8:0::1"]
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
