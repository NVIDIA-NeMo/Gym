# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig


CONFIG_DIR = Path(__file__).parents[1] / "configs"


@pytest.mark.parametrize("filename", ["enterpriseops_gym.yaml", "enterpriseops_gym_turnlog.yaml"])
def test_composed_configs_keep_each_agent_explicitly_fronted(filename: str) -> None:
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.merge(
                GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                {
                    "config_paths": [str(CONFIG_DIR / filename)],
                    "error_on_agent_without_environment_server": True,
                },
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    bindings = {
        name: instance.environment_servers.legacy_agent.agent_server.name
        for name, instance in config.items()
        if isinstance(instance, DictConfig) and "environment_servers" in instance
    }
    expected = {"enterpriseops_gym_environment_server": "enterpriseops_gym_simple_agent"}
    if filename == "enterpriseops_gym_turnlog.yaml":
        expected["enterpriseops_gym_turn_logging_environment_server"] = "enterpriseops_gym_turn_logging_agent"
    assert bindings == expected


def test_base_config_composes_enterpriseops_apptainer_profile() -> None:
    base_config = (CONFIG_DIR / "enterpriseops_gym.yaml").read_text()

    assert "resources_servers/enterpriseops_gym/configs/enterpriseops_gym_apptainer.yaml" in base_config


def test_base_config_keeps_sandbox_spec_provider_neutral() -> None:
    base_config = (CONFIG_DIR / "enterpriseops_gym.yaml").read_text()

    assert "provider_options:" not in base_config
    assert "/etc/hosts" not in base_config


def test_base_config_uses_optional_native_sif_directory_override() -> None:
    base_config = (CONFIG_DIR / "enterpriseops_gym.yaml").read_text()

    assert (
        "native_sif_dir: ${oc.env:ENTERPRISEOPS_NATIVE_SIF_DIR,'~/.cache/nemo_gym/enterpriseops_gym/images'}"
    ) in base_config
    assert (CONFIG_DIR / "enterpriseops_gym_apptainer.yaml").is_file()


def test_base_config_resolves_default_native_sif_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ENTERPRISEOPS_NATIVE_SIF_DIR", raising=False)

    config = OmegaConf.load(CONFIG_DIR / "enterpriseops_gym.yaml")

    assert config.enterpriseops_gym.resources_servers.enterpriseops_gym.native_sif_dir == (
        "~/.cache/nemo_gym/enterpriseops_gym/images"
    )


def test_shared_sandbox_spec_is_provider_neutral() -> None:
    base_config = (CONFIG_DIR / "enterpriseops_gym.yaml").read_text()
    sandbox_spec = base_config.split("      sandbox_spec:\n", 1)[1].split("      service_start_timeout_seconds", 1)[0]

    assert "provider_options" not in sandbox_spec
    assert "/etc/hosts" not in sandbox_spec
    assert "binds:" not in sandbox_spec
