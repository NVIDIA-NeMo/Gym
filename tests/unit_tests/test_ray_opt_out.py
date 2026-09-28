# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_gym.server_utils import _WARNED_IMPLICIT_RAY_SERVERS, _server_uses_ray


def test_omitted_ray_flag_preserves_compatibility(caplog) -> None:
    class LegacyServer:
        ray_enabled = None

    _WARNED_IMPLICIT_RAY_SERVERS.clear()
    assert _server_uses_ray(LegacyServer) is True
    assert "Ray remains enabled for backward compatibility" in caplog.text
    assert "future release will default it to false" in caplog.text


def test_explicit_ray_declarations_do_not_warn(caplog) -> None:
    class RayServer:
        ray_enabled = True

    class NonRayServer:
        ray_enabled = False

    assert _server_uses_ray(RayServer) is True
    assert _server_uses_ray(NonRayServer) is False
    assert caplog.text == ""
