# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sandboxed harness configs, composed as `gym env start` composes them with a sandbox provider and a model."""

import inspect
import logging

import pytest
from omegaconf import OmegaConf
from opensandbox import Sandbox


# OfficeEnv installs apart from the server's locked dependencies until it is published: without it these tests skip.
pytest.importorskip("officeenv", reason="OfficeEnv is not installed (README.md, 'The OfficeEnv version')")

from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServerConfig
from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.sandbox import create_provider, resolve_provider_config
from nemo_gym.sandbox.providers.opensandbox import provider as opensandbox_provider
from resources_servers.officeenv.app import OfficeEnvSandboxConfig


CONFIGS = "resources_servers/officeenv/configs"
DOCKER = "nemo_gym/sandbox/providers/docker/configs/docker.yaml"
# The configs of the harnesses that run at most `concurrency` episodes at once, with the provider each runs on
# (officeenv_opensandbox.yaml is Pi's on OpenSandbox). mini-SWE and Terminus 2 run every episode they are given.
CAPPED = {
    "pi": ([f"{CONFIGS}/officeenv_sandbox.yaml", DOCKER], "pi_agent"),
    "pi-opensandbox": ([f"{CONFIGS}/officeenv_opensandbox.yaml"], "pi_agent"),
    "codex": ([f"{CONFIGS}/officeenv_sandbox_codex.yaml", DOCKER], "codex_agent"),
    "opencode": ([f"{CONFIGS}/officeenv_sandbox_opencode.yaml", DOCKER], "opencode_agent"),
    "openclaw": ([f"{CONFIGS}/officeenv_sandbox_openclaw.yaml", DOCKER], "openclaw_agent"),
}
# Every sandboxed harness config, by the harness it runs
SANDBOXED = {name: paths for name, (paths, _) in CAPPED.items()} | {
    "miniswe": [f"{CONFIGS}/officeenv_sandbox_miniswe.yaml", DOCKER],
    "terminus-2": [f"{CONFIGS}/officeenv_sandbox_terminus_2.yaml", DOCKER],
}


def _compose(monkeypatch, config_paths: list[str], overrides: dict | None = None) -> dict:
    """The configs as `gym env start` composes them with a model server; ``overrides`` win over the config files, as
    the command line's do."""
    monkeypatch.setenv("OPENSANDBOX_DOMAIN", "sandbox.invalid")
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "unused")  # pragma: allowlist secret
    initial = OmegaConf.create(
        {
            "config_paths": ["responses_api_models/vllm_model/configs/vllm_model_for_training.yaml", *config_paths],
            "policy_base_url": "http://127.0.0.1:1/v1",
            "policy_api_key": "unused",  # pragma: allowlist secret
            "policy_model_name": "policy",
        }
        | (overrides or {})
    )
    parsed = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=initial, skip_load_from_cli=True, skip_load_from_dotenv=True, offline=True
        )
    )
    return OmegaConf.to_container(parsed, resolve=True)


def _environment(config: dict) -> dict:
    return config["single_agent_turn_legacy"]["environment_servers"]["single_agent_turn_legacy"]


@pytest.mark.parametrize("config_paths,agent", CAPPED.values(), ids=CAPPED)
def test_the_server_admits_no_more_episodes_than_the_harness_runs(monkeypatch, config_paths: list, agent: str) -> None:
    # The seed lends an episode its sandbox, whose TTL runs from then, and the harness runs at most `concurrency`
    # episodes at once: an episode admitted beyond them waits for the harness while its sandbox's TTL runs. On an
    # OpenSandbox cell, 128 episodes seeded at once under Pi's 8, with a 1 h TTL, lost 30 sandboxes that way
    # ("Sandbox not found").
    config = _compose(monkeypatch, config_paths)
    environment = _environment(config)
    assert environment["agent_server"]["name"] == agent
    admitted = environment["max_concurrent_episodes"]
    assert admitted is not None  # unset, every episode a collection sends is seeded at once
    assert admitted == config[agent]["responses_api_agents"][agent]["concurrency"]
    # the rest wait before their seed, holding no sandbox, as long as a large collection keeps the harness busy
    assert environment["queue_timeout_seconds"] >= 86400
    SingleAgentTurnLegacyEnvironmentServerConfig.model_validate(environment | {"name": "single_agent_turn_legacy"})


@pytest.mark.parametrize("config_paths", SANDBOXED.values(), ids=SANDBOXED)
def test_a_runs_end_leaves_the_server_time_to_stop_its_sandboxes(monkeypatch, config_paths: list) -> None:
    # Gym's run command kills the servers still running this long after signalling them to stop (1 s by default):
    # the OfficeEnv server stops the sandboxes it lent within it, and a sandbox left running stays on the cell until
    # its TTL. A smoke's job end that allowed 1 s left all 64 of its sandboxes.
    assert _compose(monkeypatch, config_paths)["server_shutdown_timeout_seconds"] == 30
    overridden = _compose(monkeypatch, config_paths, {"server_shutdown_timeout_seconds": 70})
    assert overridden["server_shutdown_timeout_seconds"] == 70  # the command line still decides


@pytest.mark.parametrize("concurrency", [None, 64])
def test_the_admitted_episodes_follow_pis_concurrency(monkeypatch, concurrency: int | None) -> None:
    pi = {"pi_agent": {"responses_api_agents": {"pi_agent": {"concurrency": concurrency}}}}
    config = _compose(monkeypatch, CAPPED["pi-opensandbox"][0], pi if concurrency else None)
    assert _environment(config)["max_concurrent_episodes"] == (concurrency or 8)  # 8: Pi's default


def test_a_sandbox_create_outlasts_its_request_and_readiness_wait(monkeypatch, caplog) -> None:
    # One create attempt on OpenSandbox holds the request, which blocks while the cell places the sandbox, and the
    # readiness wait that starts once the request returns: a budget below both gives up on a sandbox the cell is still
    # starting within its bounds, deletes it and starts again, and the provider warns.
    config = _compose(monkeypatch, CAPPED["pi-opensandbox"][0])
    monkeypatch.setattr(opensandbox_provider, "_BUDGET_WARNINGS", set())
    server = config["oe"]["resources_servers"]["officeenv"]
    sandbox = OfficeEnvSandboxConfig(**server["sandbox_config"])
    spec = sandbox.spec(server["sandbox_provider"], image=sandbox.image, workdir=sandbox.workdir, named_configs=config)
    provider = create_provider(resolve_provider_config(server["sandbox_provider"], config))
    with caplog.at_level(logging.WARNING):
        budget, request = provider._create_budget(spec)
    # the spec's readiness wait, or the SDK's own when the config sets none
    ready = spec.ready_timeout_s or inspect.signature(Sandbox.create).parameters["ready_timeout"].default.seconds
    assert budget >= request + ready
    assert "create.timeout_s" not in caplog.text
