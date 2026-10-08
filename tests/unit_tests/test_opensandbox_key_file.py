# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_gym.sandbox.providers.opensandbox import provider as opensandbox_provider
from nemo_gym.sandbox.providers.opensandbox.provider import OpenSandboxConnectionConfig, OpenSandboxProvider


def make_provider(path: Path) -> OpenSandboxProvider:
    return OpenSandboxProvider(connection={"domain": "sandbox.example.test", "api_key_file": str(path)})


def test_file_backed_key_is_not_serialized(tmp_path: Path) -> None:
    path = tmp_path / "key"
    path.write_text("fake-test-credential\n")
    path.chmod(0o600)
    provider = make_provider(path)
    assert provider._resolve_api_key() == "fake-test-credential"
    assert provider._connection.api_key is None
    assert "fake-test-credential" not in str(asdict(provider._connection))


@pytest.mark.parametrize("proxy", [True, False])
def test_file_key_reaches_sdk_without_exposing_it_to_direct_endpoints(
    tmp_path: Path, proxy: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Unit tests must not require the optional runtime SDK installed by the
    # sandbox extra. Live runs separately validate the real SDK boundary.
    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (None, lambda **kwargs: SimpleNamespace(**{"headers": {}, **kwargs}), None, None, None),
    )
    path = tmp_path / "key"
    path.write_text("fake-test-credential")
    path.chmod(0o600)
    provider = OpenSandboxProvider(
        connection={
            "domain": "sandbox.example.test",
            "api_key_file": str(path),
            "use_server_proxy": proxy,
            "tls_verify": True,
        }
    )
    config = provider._connection_config()
    assert config.api_key == "fake-test-credential"
    if proxy:
        assert config.headers["OPEN-SANDBOX-API-KEY"] == "fake-test-credential"
    else:
        assert "OPEN-SANDBOX-API-KEY" not in (config.headers or {})
    assert provider._connection.api_key is None


def test_rejects_conflicting_credential_sources(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="either api_key or api_key_file"):
        OpenSandboxConnectionConfig(api_key="fake-test-credential", api_key_file=str(tmp_path / "key"))


@pytest.mark.parametrize("mode", [0o640, 0o644])
def test_rejects_readable_key(tmp_path: Path, mode: int) -> None:
    path = tmp_path / "key"
    path.write_text("fake-test-credential")
    path.chmod(mode)
    with pytest.raises(PermissionError, match="owner-only"):
        make_provider(path)._resolve_api_key()


def test_missing_key_has_safe_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="API key file is missing"):
        make_provider(tmp_path / "missing")._resolve_api_key()


def test_empty_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "key"
    path.write_text("\n")
    path.chmod(0o600)
    with pytest.raises(ValueError, match="API key file is empty"):
        make_provider(path)._resolve_api_key()
