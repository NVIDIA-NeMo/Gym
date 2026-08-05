# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json

import pytest

from responses_api_agents.osworld_agent.proxy import (
    inspect_proxy_config_file,
    parse_env_bool,
    task_requires_proxy,
)


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_parse_env_bool_accepts_explicit_true_values(value: str) -> None:
    assert parse_env_bool("OSWORLD_ENABLE_PROXY", False, environ={"OSWORLD_ENABLE_PROXY": value}) is True


@pytest.mark.parametrize("value", ["0", "false", "NO", "off"])
def test_parse_env_bool_accepts_explicit_false_values(value: str) -> None:
    assert parse_env_bool("OSWORLD_ENABLE_PROXY", True, environ={"OSWORLD_ENABLE_PROXY": value}) is False


def test_parse_env_bool_uses_default_and_rejects_ambiguous_values() -> None:
    assert parse_env_bool("OSWORLD_ENABLE_PROXY", True, environ={}) is True
    with pytest.raises(ValueError, match="must be one of"):
        parse_env_bool("OSWORLD_ENABLE_PROXY", False, environ={"OSWORLD_ENABLE_PROXY": "enabled"})


def test_task_proxy_requirement_is_strict_boolean() -> None:
    assert task_requires_proxy({"proxy": True}) is True
    assert task_requires_proxy({}) is False
    with pytest.raises(ValueError, match="must be boolean"):
        task_requires_proxy({"proxy": "false"})


def test_proxy_config_inspection_returns_only_non_secret_provenance(tmp_path) -> None:
    config_path = tmp_path / "proxy.json"
    payload = [
        {"host": "proxy.example.com", "port": 3128, "protocol": "http"},
        {
            "host": "authenticated.example.com",
            "port": 8080,
            "protocol": "http",
            "username": "proxy-user",
            "password": "proxy-password",  # pragma: allowlist secret
        },
    ]
    raw = (json.dumps(payload) + "\n").encode()
    config_path.write_bytes(raw)

    info = inspect_proxy_config_file(str(config_path))

    assert info.path == str(config_path)
    assert info.sha256 == hashlib.sha256(raw).hexdigest()
    assert info.entry_count == 2
    assert "proxy-user" not in repr(info)
    assert "proxy-password" not in repr(info)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        [{"host": "proxy.example.com", "port": 3128, "username": "user"}],
        [{"host": "proxy;hostname", "port": 3128}],
        [{"host": "proxy.example.com", "port": 70000}],
        [{"host": "proxy.example.com", "port": 3128, "protocol": "socks5"}],
    ],
)
def test_proxy_config_rejects_incomplete_or_unsafe_entries(tmp_path, payload) -> None:
    config_path = tmp_path / "proxy.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        inspect_proxy_config_file(str(config_path))
