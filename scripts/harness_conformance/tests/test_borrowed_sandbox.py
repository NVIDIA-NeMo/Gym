# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Borrowers execute in the fixture's workspace and release it without destroying it."""

import json
import os

import pytest
from scripts.harness_conformance.local_sandbox import ProbeLocalProvider, borrowable_local_provider

from nemo_gym.sandbox import AsyncSandbox, SandboxSpec
from nemo_gym.sandbox.providers.registry import get_provider_class


async def test_borrowed_commands_share_workspace_and_owner_controls_cleanup(tmp_path):
    owner = AsyncSandbox(ProbeLocalProvider(workspace_root=str(tmp_path)))
    await owner.start(SandboxSpec())
    descriptor = json.loads(json.dumps(await owner.serialize()))
    try:
        borrower = await AsyncSandbox.connect(descriptor, provider=ProbeLocalProvider())
        try:
            result = await borrower.exec("printf conformance > result; cat result")
            assert result.error_type is None and result.return_code == 0
            assert result.stdout == "conformance"
        finally:
            await borrower.disconnect()
        result = await owner.exec("cat result")
        assert result.error_type is None and result.return_code == 0
        assert result.stdout == "conformance"
    finally:
        await owner.stop()
    with pytest.raises(ValueError, match="must already exist"):
        await AsyncSandbox.connect(descriptor, provider=ProbeLocalProvider())


async def test_explicit_user_must_match_actual_execution_identity(tmp_path):
    sandbox = AsyncSandbox(ProbeLocalProvider())
    await sandbox.start(SandboxSpec(workdir=str(tmp_path)))
    try:
        result = await sandbox.exec("id -u", user=os.geteuid())
        assert result.return_code == 0 and int(result.stdout) == os.geteuid()
        with pytest.raises(ValueError, match="current user"):
            await sandbox.exec("touch should-not-exist", user=os.geteuid() + 1)
        assert not (tmp_path / "should-not-exist").exists()
    finally:
        await sandbox.stop()


def test_provider_override_is_restored_on_failure():
    previous = get_provider_class("local")
    with pytest.raises(RuntimeError, match="setup failed"):
        with borrowable_local_provider():
            assert get_provider_class("local") is ProbeLocalProvider
            raise RuntimeError("setup failed")
    assert get_provider_class("local") is previous
