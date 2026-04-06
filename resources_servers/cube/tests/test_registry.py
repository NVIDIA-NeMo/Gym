# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from resources_servers.cube.domains.base import CubeEnvironmentBase
from resources_servers.cube.registry import instantiate_domain, register_domain


def test_instantiate_osworld_domain():
    p = instantiate_domain("osworld")
    assert isinstance(p, CubeEnvironmentBase)


def test_unknown_domain_raises():
    with pytest.raises(ValueError, match="Unknown cube.resources_servers domain"):
        instantiate_domain("nonexistent_env_xyz")


def test_register_custom_domain():
    class DummyEnv(CubeEnvironmentBase):
        def ensure_loaded(self, server):
            pass

        def empty_reset_obs_detail(self) -> str:
            return "dummy"

    register_domain("dummy_test_env", DummyEnv)
    try:
        p = instantiate_domain("dummy_test_env")
        assert isinstance(p, DummyEnv)
    finally:
        # avoid polluting global registry for other tests
        from resources_servers.cube import registry as reg

        reg._REGISTERED_DOMAIN_CLASSES.pop("dummy_test_env", None)
