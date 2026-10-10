# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""anyswe starts Apptainer sandboxes with ``--no-home`` so the task image keeps its own ``/root``."""

import asyncio
import copy
import shutil

import pytest

from nemo_gym.sandbox.providers import SandboxSpec
from nemo_gym.sandbox.providers.apptainer import provider as apptainer_provider
from responses_api_agents.anyswe_agent.app import _resolve_sandbox_provider


APPTAINER = {
    "apptainer": {
        "create": {"mount_point": "/sandbox", "extra_start_args": ["--userns", "--writable-tmpfs"]},
        "exec": {"fakeroot_for_root": True},
    }
}


class TestResolveSandboxProvider:
    def test_inline_apptainer_config_gets_no_home(self) -> None:
        original = copy.deepcopy(APPTAINER)
        resolved = _resolve_sandbox_provider(APPTAINER, {})
        create = resolved["apptainer"]["create"]
        assert create["extra_start_args"] == ["--userns", "--writable-tmpfs", "--no-home"]
        assert create["mount_point"] == "/sandbox"
        assert resolved["apptainer"]["exec"] == {"fakeroot_for_root": True}
        assert APPTAINER == original  # the agent config is not mutated

    def test_named_config_and_empty_create(self) -> None:
        named = {"sandbox": {"apptainer": {}, "default_metadata": {"team": "x"}}}
        resolved = _resolve_sandbox_provider("sandbox", named)
        assert resolved == {"apptainer": {"create": {"extra_start_args": ["--no-home"]}}}

    def test_no_home_is_not_added_twice(self) -> None:
        once = _resolve_sandbox_provider(APPTAINER, {})
        assert _resolve_sandbox_provider(once, {})["apptainer"]["create"]["extra_start_args"].count("--no-home") == 1

    def test_dataclass_create_config(self) -> None:
        create = apptainer_provider.ApptainerCreateConfig(extra_start_args=["--writable-tmpfs"])
        resolved = _resolve_sandbox_provider({"apptainer": {"create": create}}, {})
        assert resolved["apptainer"]["create"]["extra_start_args"] == ["--writable-tmpfs", "--no-home"]

    @pytest.mark.parametrize("provider", [{"opensandbox": {"api_url": "x"}}, {"docker": {}}])
    def test_other_providers_unchanged(self, provider: dict) -> None:
        assert _resolve_sandbox_provider(provider, {}) == provider

    def test_apptainer_instance_start_carries_no_home(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(apptainer_provider, "_require_apptainer", lambda _bin_path=None: "/usr/bin/apptainer")
        resolved = _resolve_sandbox_provider(APPTAINER, {})
        provider = apptainer_provider.ApptainerProvider(**resolved["apptainer"], probe={"command": None})
        calls: list[list[str]] = []

        async def fake_run(argv, **kwargs):
            calls.append(list(argv))
            return 0, "", ""

        monkeypatch.setattr(provider, "_run", fake_run)
        handle = asyncio.run(provider.create(SandboxSpec(image="docker://r2e/task:latest")))
        shutil.rmtree(handle.raw.staging_dir, ignore_errors=True)

        start = calls[0]
        assert start[:3] == ["/usr/bin/apptainer", "instance", "start"]
        assert "--no-home" in start
        assert start.index("--no-home") < start.index("docker://r2e/task:latest")
