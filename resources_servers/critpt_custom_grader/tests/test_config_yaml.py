# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The shipped YAML must parse offline with strict environment-server routing and load the server model.

The server model refuses, at startup, a sandbox workdir the execution policy disagrees with, a create budget that
cannot hold the readiness wait plus the two setup transfers, a sandbox ttl a job can outlive and any key that is
neither its own nor one Gym reads from a resources-server block.

No grading image is shipped: ``sandbox_spec.image`` resolves to CRITPT_CUSTOM_GRADER_IMAGE when that is set and to
None otherwise, which the server model accepts.
"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from resources_servers.critpt_custom_grader import app as adapter
from resources_servers.critpt_custom_grader.execution import ExecutionPolicy


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "critpt_custom_grader.yaml"
# A value that must never appear in a validation error: the input could hold an env-resolved credential. It is short
# and shallow on purpose, since pydantic would elide a long or deeply nested input on its own.
ECHO_MARKER = "echo-zq7"


def test_shipped_config_parses_offline_with_strict_routing():
    initial = OmegaConf.merge(
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        {"config_paths": [str(CONFIG)], "error_on_agent_without_environment_server": True},
    )
    resolved = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=initial, skip_load_from_cli=True, skip_load_from_dotenv=True, offline=True
        )
    )
    assert "critpt_custom_grader_environment_server" in resolved
    relay = resolved["critpt_custom_grader_environment_server"]["environment_servers"]["legacy_agent"]
    assert relay["agent_server"]["name"] == "critpt_custom_grader_simple_agent"
    block = resolved["critpt_custom_grader"]["resources_servers"]["critpt_custom_grader"]
    assert block["sandbox_provider"] == "sandbox" and block["execution"]["exec_user"] == "grader"
    assert "???" not in OmegaConf.to_yaml(resolved["critpt_custom_grader"])


def test_shipped_config_loads_the_server_config_model():
    raw = OmegaConf.to_container(OmegaConf.load(CONFIG), resolve=True)["critpt_custom_grader"]["resources_servers"][
        "critpt_custom_grader"
    ]
    config = adapter.CritPtCustomGraderConfig(host="127.0.0.1", port=8000, name="critpt_custom_grader", **raw)
    assert config.comparison.default_rtol == "5e-12" and config.comparison.default_atol == "0"
    assert config.sandbox_spec.workdir == "/work" and config.sandbox_spec.resources == {"cpu": 1, "memory_mib": 4096}
    assert config.execution.max_concurrent_jobs == 4 and config.startup_probe is True
    assert config.REVERIFY_MODE.value == "stateless"


def server_config(sandbox_spec=None, **execution):
    return adapter.CritPtCustomGraderConfig(
        host="127.0.0.1",
        port=8000,
        name="critpt_custom_grader",
        entrypoint="app.py",
        sandbox_spec={"image": "example/grader:test", **(sandbox_spec or {})},
        execution=execution,
    )


def test_startup_config_requires_one_workdir_and_create_headroom():
    with pytest.raises(ValueError, match=r"execution\.workdir.*sandbox_spec\.workdir"):
        server_config({"workdir": "/grader"})
    with pytest.raises(ValueError, match=r"execution\.create_timeout_s.*sandbox_spec\.ready_timeout_s"):
        server_config({"ready_timeout_s": 600}, create_timeout_s=719.0, transfer_timeout_s=60.0)
    # Both bounds are inclusive; a provider-default readiness wait (null) leaves only the two transfers to cover.
    config = server_config(
        {"workdir": "/grader", "ready_timeout_s": 600},
        workdir="/grader",
        create_timeout_s=720.0,
        transfer_timeout_s=60.0,
    )
    assert config.sandbox_spec.workdir == config.execution.workdir == "/grader"
    config = server_config({"ready_timeout_s": None}, create_timeout_s=120.0, transfer_timeout_s=60.0)
    assert config.sandbox_spec.ready_timeout_s is None
    with pytest.raises(ValueError, match=r"execution\.create_timeout_s"):
        server_config({"ready_timeout_s": None}, create_timeout_s=119.0, transfer_timeout_s=60.0)


def test_startup_config_requires_the_ttl_to_outlive_a_job():
    # Default policy: the job deadline is its derived minimum, and one stop may follow it.
    policy = ExecutionPolicy()
    bound = policy.minimum_job_timeout_s() + policy.cleanup_timeout_s
    assert server_config({"ttl_s": bound}).sandbox_spec.ttl_s == bound
    with pytest.raises(ValueError, match=r"sandbox_spec\.ttl_s.*execution\.job_timeout_s"):
        server_config({"ttl_s": bound - 1})
    # An explicit job_timeout_s moves the bound; a null ttl leaves the lifetime to the provider's default.
    job_timeout_s = policy.minimum_job_timeout_s() + 105.0
    bound = job_timeout_s + policy.cleanup_timeout_s
    assert server_config({"ttl_s": bound}, job_timeout_s=job_timeout_s).sandbox_spec.ttl_s == bound
    with pytest.raises(ValueError, match=r"sandbox_spec\.ttl_s.*execution\.cleanup_timeout_s"):
        server_config({"ttl_s": bound - 1}, job_timeout_s=job_timeout_s)
    assert server_config({"ttl_s": None}).sandbox_spec.ttl_s is None


def shipped_block():
    return OmegaConf.to_container(OmegaConf.load(CONFIG), resolve=True)["critpt_custom_grader"]["resources_servers"][
        "critpt_custom_grader"
    ]


def loaded(block):
    """What server_utils.load_config_from_global_config validates: the populated YAML block plus the instance name."""
    return adapter.CritPtCustomGraderConfig.model_validate(
        {**block, "host": "127.0.0.1", "port": 8000, "name": "critpt_custom_grader"}
    )


def test_startup_config_forbids_a_misspelt_key_instead_of_using_the_default_provider():
    block = shipped_block()
    del block["sandbox_provider"]
    # A misspelt inline mapping: its values (here a marker, in practice perhaps a credential) are never echoed.
    block["sandbox_providr"] = {"opensandbox": ECHO_MARKER}
    with pytest.raises(ValueError, match="sandbox_providr") as caught:
        loaded(block)
    assert ECHO_MARKER not in str(caught.value)


def test_startup_config_accepts_the_framework_pairing_keys():
    block = {**shipped_block(), "allowed_agents": ["simple_agent"], "allowed_model_types": ["openai_model"]}
    config = loaded(block)
    assert config.allowed_agents == ["simple_agent"] and config.allowed_model_types == ["openai_model"]
    # A bare-string ++ override is legal: Gym reads it as a one-entry list.
    assert loaded({**block, "allowed_agents": "simple_agent"}).allowed_agents == "simple_agent"
    # The metadata and manifest labels Gym reads from the block are accepted too; the grader ignores all four.
    labelled = loaded({**block, "verified_url": "https://example.invalid/report", "grading_mode": "binary"})
    assert labelled.verified_url == "https://example.invalid/report" and labelled.grading_mode == "binary"
    block["sandbox_providr"] = block.pop("sandbox_provider")
    with pytest.raises(ValueError, match="sandbox_providr"):
        loaded(block)


def test_shipped_config_ships_no_image(monkeypatch):
    monkeypatch.delenv("CRITPT_CUSTOM_GRADER_IMAGE", raising=False)
    block = shipped_block()
    assert block["sandbox_spec"]["image"] is None
    assert loaded(block).sandbox_spec.image is None


def test_shipped_config_takes_the_image_from_the_environment(monkeypatch):
    monkeypatch.setenv("CRITPT_CUSTOM_GRADER_IMAGE", "registry.example.invalid/grader-rt:test")
    block = shipped_block()
    assert block["sandbox_spec"]["image"] == "registry.example.invalid/grader-rt:test"
    assert loaded(block).sandbox_spec.image == "registry.example.invalid/grader-rt:test"
