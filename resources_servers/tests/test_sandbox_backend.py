# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for skills_sidecar vs gym_sandbox dual-path helpers."""

import pytest

from resources_servers.sandbox_backend import (
    SANDBOX_BACKEND_GYM,
    SANDBOX_BACKEND_SKILLS,
    build_sandbox_spec_from_mapping,
    normalize_sandbox_backend,
    resolve_gym_provider_config,
)


def test_normalize_sandbox_backend_default() -> None:
    assert normalize_sandbox_backend(None) == SANDBOX_BACKEND_SKILLS
    assert normalize_sandbox_backend("") == SANDBOX_BACKEND_SKILLS
    assert normalize_sandbox_backend("skills_sidecar") == SANDBOX_BACKEND_SKILLS
    assert normalize_sandbox_backend("gym_sandbox") == SANDBOX_BACKEND_GYM


def test_normalize_sandbox_backend_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown sandbox_backend"):
        normalize_sandbox_backend("docker")


def test_resolve_gym_provider_config_requires_provider() -> None:
    with pytest.raises(ValueError, match="requires sandbox_provider"):
        resolve_gym_provider_config(None, {})


def test_resolve_gym_provider_config_inline() -> None:
    cfg = resolve_gym_provider_config({"opensandbox": {"connection": {"domain": "x"}}}, None)
    assert cfg == {"opensandbox": {"connection": {"domain": "x"}}}


def test_resolve_gym_provider_config_named() -> None:
    named = {"sandbox": {"opensandbox": {"connection": {"domain": "x"}}}}
    cfg = resolve_gym_provider_config("sandbox", named)
    assert "opensandbox" in cfg


def test_build_sandbox_spec_unknown_key() -> None:
    with pytest.raises(ValueError, match="Unknown sandbox_spec keys"):
        build_sandbox_spec_from_mapping({"not_a_real_field": 1})
