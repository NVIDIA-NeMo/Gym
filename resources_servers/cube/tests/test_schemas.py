# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cube resources server config schema."""

from nemo_gym.config_types import Domain
from resources_servers.cube.schemas import CubeResourcesServerConfig


def test_env_domain_default_with_gym_domain():
    c = CubeResourcesServerConfig.model_validate(
        {
            "name": "s",
            "host": "0.0.0.0",
            "port": 1,
            "entrypoint": "app.py",
            "domain": "other",
        }
    )
    assert c.domain == Domain.OTHER
    assert c.env_domain == "osworld"


def test_env_domain_explicit():
    c = CubeResourcesServerConfig.model_validate(
        {
            "name": "s",
            "host": "0.0.0.0",
            "port": 1,
            "entrypoint": "app.py",
            "domain": "other",
            "env_domain": "osworld",
        }
    )
    assert c.env_domain == "osworld"
    assert c.domain == Domain.OTHER
