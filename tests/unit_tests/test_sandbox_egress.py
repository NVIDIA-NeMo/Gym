# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.providers.base import SandboxEgressRules, SandboxHandle, SandboxSpec


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mode": "deny"}, "mode must be"),
        ({"mode": "allow_all", "allowed_domains": ("a",)}, "allow_all cannot"),
        ({"mode": "allow", "blocked_domains": ("a",)}, "require mode=block"),
        ({"mode": "block", "allowed_cidrs": ("10.0.0.0/8",)}, "cannot be used with mode=block"),
        ({"mode": "block"}, "at least one blocked rule"),
    ],
)
def test_invalid_egress_rules_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SandboxEgressRules(**kwargs)


def test_egress_rules_reject_a_bare_string():
    with pytest.raises(TypeError, match="not a string"):
        SandboxEgressRules(mode="allow", allowed_domains="pypi.org")


async def test_sandbox_update_egress_rules_delegates_or_reports_missing_capability():
    class _Plain:
        name = "plain"

        async def create(self, spec):
            return SandboxHandle("box-1", self.name, None)

        async def close(self, handle):
            return None

        async def aclose(self):
            return None

    class _WithEgress(_Plain):
        name = "egress"

        def __init__(self) -> None:
            self.updates: list[tuple[str, SandboxEgressRules, Any]] = []

        async def update_egress_rules(self, handle, rules, *, timeout_s=None):
            self.updates.append((handle.sandbox_id, rules, timeout_s))

    rules = SandboxEgressRules(mode="allow")
    with_egress = _WithEgress()
    sandbox = AsyncSandbox(with_egress)
    with pytest.raises(RuntimeError, match="not been started"):
        await sandbox.update_egress_rules(rules)
    await sandbox.start(SandboxSpec(image="img"))
    await sandbox.update_egress_rules(rules, timeout_s=5)
    assert with_egress.updates == [("box-1", rules, 5)]
    await sandbox.stop()

    plain = AsyncSandbox(_Plain())
    await plain.start(SandboxSpec(image="img"))
    with pytest.raises(NotImplementedError, match="'plain' does not support runtime egress rule updates"):
        await plain.update_egress_rules(rules)
    await plain.stop()

    unsupported = AsyncSandbox(_Plain())
    with pytest.raises(NotImplementedError, match="'plain' does not support egress rules"):
        await unsupported.start(SandboxSpec(image="img", egress_rules=rules))


def test_sandbox_spec_rejects_invalid_egress_rules():
    with pytest.raises(TypeError, match="egress_rules must be"):
        SandboxSpec(image="img", egress_rules="block")
    with pytest.raises(ValueError, match="mode must be"):
        SandboxSpec(image="img", egress_rules={"mode": "deny"})


def test_sandbox_spec_coerces_egress_mapping():
    spec = SandboxSpec(image="example:latest", egress_rules={"mode": "allow", "allowed_domains": ["example.com"]})
    assert spec.egress_rules == SandboxEgressRules(mode="allow", allowed_domains=("example.com",))
