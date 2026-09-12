# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_gym.sandbox.utils import CPU_CAP_ENV_VARS, cpu_cap_env, rewrite_image


def test_cpu_cap_env_empty_when_unset() -> None:
    assert cpu_cap_env(None) == {}


def test_cpu_cap_env_floors_to_at_least_one_core() -> None:
    assert cpu_cap_env(0.5) == {name: "1" for name in CPU_CAP_ENV_VARS}
    assert cpu_cap_env(2.7) == {name: "2" for name in CPU_CAP_ENV_VARS}


def test_rewrite_image_applies_first_matching_prefix() -> None:
    rewrites = [{"from": "nvcr.io/nemo", "to": "mirror.example/nemo"}]
    assert rewrite_image("nvcr.io/nemo/gym:latest", rewrites) == "mirror.example/nemo/gym:latest"


def test_rewrite_image_passthrough_and_none() -> None:
    assert rewrite_image("ghcr.io/other/img:1", [{"from": "nvcr.io", "to": "m"}]) == "ghcr.io/other/img:1"
    assert rewrite_image(None, [{"from": "nvcr.io", "to": "m"}]) is None
