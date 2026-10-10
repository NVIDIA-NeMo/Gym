# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from nemo_gym.sandbox.providers.opensandbox.provider import OpenSandboxProvider, _to_image_spec


IMAGE = "registry.example.test/team/task@sha256:fixture"
CREDENTIALS = {"registry": "registry.example.test", "username": "fixture-user", "password": "fixture-password"}


def provider_with_file(tmp_path: Path, contents: object = CREDENTIALS, mode: int = 0o600) -> OpenSandboxProvider:
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(contents))
    path.chmod(mode)
    return OpenSandboxProvider(create={"image_auth_file": str(path)})


def test_registry_file_reaches_sdk_without_entering_stored_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Exercise conversion without adding the optional runtime SDK to core CI.
    models = ModuleType("opensandbox.models.sandboxes")
    models.SandboxImageAuth = SimpleNamespace
    models.SandboxImageSpec = lambda image, **kwargs: SimpleNamespace(image=image, **kwargs)
    monkeypatch.setitem(sys.modules, "opensandbox.models.sandboxes", models)
    provider = provider_with_file(tmp_path)
    image = _to_image_spec(IMAGE, provider._resolve_image_auth(IMAGE, None))
    assert image.auth.username == "fixture-user"
    assert image.auth.password == "fixture-password"
    assert "fixture-password" not in json.dumps(asdict(provider._create))


def test_inline_registry_auth_remains_supported() -> None:
    inline = {"username": "fixture-user", "password": "fixture-password"}
    assert OpenSandboxProvider()._resolve_image_auth(IMAGE, inline) is inline


def test_registry_file_rejects_conflicting_auth(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="either image_auth or create.image_auth_file"):
        provider_with_file(tmp_path)._resolve_image_auth(IMAGE, {})


@pytest.mark.parametrize(
    "image", ["ubuntu:24.04", "another.example.test/team/task:tag", "registry.example.test.attacker/task"]
)
def test_registry_file_cannot_leak_credentials_to_another_registry(tmp_path: Path, image: str) -> None:
    with pytest.raises(ValueError, match="do not match"):
        provider_with_file(tmp_path)._resolve_image_auth(image, None)


@pytest.mark.parametrize("mode", [0o640, 0o644])
def test_registry_file_requires_private_permissions(tmp_path: Path, mode: int) -> None:
    with pytest.raises(PermissionError, match="owner-only"):
        provider_with_file(tmp_path, mode=mode)._resolve_image_auth(IMAGE, None)


@pytest.mark.parametrize("contents", [None, [], {}, {**CREDENTIALS, "password": ""}, {**CREDENTIALS, "username": 7}])
def test_registry_file_rejects_invalid_structure_without_echoing_credentials(tmp_path: Path, contents: object) -> None:
    with pytest.raises(ValueError, match="requires registry, username, and password") as error:
        provider_with_file(tmp_path, contents)._resolve_image_auth(IMAGE, None)
    assert "fixture-password" not in str(error.value)


def test_missing_registry_file_has_safe_error(tmp_path: Path) -> None:
    provider = OpenSandboxProvider(create={"image_auth_file": str(tmp_path / "missing")})
    with pytest.raises(FileNotFoundError, match="credential file is missing"):
        provider._resolve_image_auth(IMAGE, None)


def test_malformed_registry_file_has_safe_error(tmp_path: Path) -> None:
    provider = provider_with_file(tmp_path)
    (tmp_path / "registry.json").write_text("invalid fixture-password JSON")
    with pytest.raises(ValueError, match="valid JSON") as error:
        provider._resolve_image_auth(IMAGE, None)
    assert "fixture-password" not in str(error.value)
