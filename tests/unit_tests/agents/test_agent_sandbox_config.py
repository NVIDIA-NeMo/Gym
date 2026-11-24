# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The typed sandbox config of an agent's own sandbox: the keys agent recipes set today, and the spec it builds."""

import pytest

from nemo_gym.agents.utils.agent_sandbox_config import AgentSandboxConfig


class TestAgentSandboxConfig:
    def test_the_keys_agent_recipes_set_today_load(self) -> None:
        config = AgentSandboxConfig.model_validate(
            {
                "image": "harness:1",
                "workdir": "/app",
                "files": {"/app/AGENTS.md": "be brief"},
                "entrypoint": ["sleep", "infinity"],
                "derive_cpu_env": False,
                "ttl_s": 3600,
                "ready_timeout_s": 300,
                "resources": {"cpu": 2},
                "env": {"A": "1"},
                "metadata": {"team": "gym"},
                "provider_options": {"volumes": []},
                "network": "inherit",
            }
        )

        assert (config.image, config.workdir, config.entrypoint) == ("harness:1", "/app", ["sleep", "infinity"])

    def test_an_unknown_key_is_refused(self) -> None:
        with pytest.raises(ValueError, match="extra"):
            AgentSandboxConfig.model_validate({"imagee": "harness:1"})

    def test_the_spec_carries_the_agents_own_image_files_and_entrypoint(self) -> None:
        config = AgentSandboxConfig(
            image="harness:1",
            workdir="/app",
            files={"/app/a": "a"},
            entrypoint=["sleep", "infinity"],
            resources={"cpu": 2},
            env={"OMP_NUM_THREADS": "8"},
            derive_cpu_env=True,
        )

        spec = config.spec({"fake": {}}, files={"/app/b": "b"})

        assert (spec.image, spec.workdir, spec.entrypoint) == ("harness:1", "/app", ["sleep", "infinity"])
        assert spec.files == {"/app/a": "a", "/app/b": "b"}
        assert spec.env["OMP_NUM_THREADS"] == "8", "the recipe's own value wins over the CPU caps"
        assert spec.env["GOMAXPROCS"] == "2"

    def test_the_cpu_caps_are_off_unless_asked_for(self) -> None:
        # Most agents set no CPU caps today, so moving onto this config must not add them.
        spec = AgentSandboxConfig(resources={"cpu": 2}).spec({"fake": {}})

        assert spec.env == {}

    def test_the_user_and_ports_reach_the_spec(self) -> None:
        spec = AgentSandboxConfig(user="agent", ports=[8080]).spec({"fake": {}})

        assert (spec.user, list(spec.ports)) == ("agent", [8080])
