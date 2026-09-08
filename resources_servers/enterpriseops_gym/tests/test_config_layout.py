# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


CONFIG_DIR = Path(__file__).parents[1] / "configs"


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
        "native_sif_dir: ${oc.env:ENTERPRISEOPS_NATIVE_SIF_DIR,~/.cache/nemo_gym/enterpriseops_gym/images}"
    ) in base_config
    assert (CONFIG_DIR / "enterpriseops_gym_apptainer.yaml").is_file()


def test_shared_sandbox_spec_is_provider_neutral() -> None:
    base_config = (CONFIG_DIR / "enterpriseops_gym.yaml").read_text()
    sandbox_spec = base_config.split("      sandbox_spec:\n", 1)[1].split("      service_start_timeout_seconds", 1)[0]

    assert "provider_options" not in sandbox_spec
    assert "/etc/hosts" not in sandbox_spec
    assert "binds:" not in sandbox_spec
