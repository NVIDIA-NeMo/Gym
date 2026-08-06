# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared credential-key detection and safe configuration serialization."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


CREDENTIAL_KEY_NAMES = frozenset(
    {
        "access_key_id",
        "access_token",
        "api_key",
        "apikey",
        "api_token",
        "auth_header",
        "auth_token",
        "authorization",
        "authorization_header",
        "bearer_token",
        "client_secret",
        "credential",
        "credentials",
        "csrf_token",
        "gateway_token",
        "github_token",
        "gitlab_token",
        "hf_token",
        "id_token",
        "jwt",
        "jwt_secret",
        "jwt_token",
        "oauth_token",
        "password",
        "passwd",
        "private_key",
        "refresh_token",
        "secret",
        "secret_access_key",
        "secret_key",
        "session_token",
    }
)
CREDENTIAL_KEY_SUFFIXES = tuple(f"_{name}" for name in sorted(CREDENTIAL_KEY_NAMES))


def normalize_credential_key(key: object) -> str:
    return str(key).strip().casefold().replace("-", "_").replace(".", "_").replace(" ", "_")


def is_credential_key(key: object) -> bool:
    """Recognize credentials while retaining score settings such as ``max_output_tokens``."""

    normalized = normalize_credential_key(key)
    return normalized in CREDENTIAL_KEY_NAMES or normalized.endswith(CREDENTIAL_KEY_SUFFIXES)


def is_literal_credential_value(value: Any) -> bool:
    """Return whether a credential value would persist secret material in an artifact."""

    if value in (None, "", [], {}):
        return False
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.casefold() in {"???", "empty", "<api_key>", "<token>"}:
            return False
        return not (stripped.startswith("${") and stripped.endswith("}"))
    return True


def literal_credential_paths(value: Any, *, prefix: str = "") -> tuple[str, ...]:
    """Find credential-bearing fields whose values are not deferred interpolations."""

    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            location = f"{prefix}.{key}" if prefix else str(key)
            if is_credential_key(key) and is_literal_credential_value(item):
                found.append(location)
            else:
                found.extend(literal_credential_paths(item, prefix=location))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            found.extend(literal_credential_paths(item, prefix=f"{prefix}[{index}]"))
    return tuple(found)


def redact_credentials(value: Any, unresolved: Any = None) -> Any:
    """Recursively remove literal credentials while retaining deferred interpolation expressions."""

    if isinstance(value, Mapping):
        unresolved_mapping = unresolved if isinstance(unresolved, Mapping) else {}
        result: dict[str, Any] = {}
        for key, item in value.items():
            unresolved_item = unresolved_mapping.get(key)
            if is_credential_key(key):
                result[str(key)] = (
                    unresolved_item
                    if isinstance(unresolved_item, str)
                    and unresolved_item.strip().startswith("${")
                    and unresolved_item.strip().endswith("}")
                    else None
                )
            else:
                result[str(key)] = redact_credentials(item, unresolved_item)
        return result
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        unresolved_items = unresolved if isinstance(unresolved, Sequence) else ()
        return [
            redact_credentials(item, unresolved_items[index] if index < len(unresolved_items) else None)
            for index, item in enumerate(value)
        ]
    return value


def canonical_redacted_config_json(value: Any) -> str:
    """Serialize resolved configuration deterministically without literal credentials."""

    from omegaconf import OmegaConf

    if OmegaConf.is_config(value):
        unresolved = OmegaConf.to_container(value, resolve=False)
        resolved = OmegaConf.to_container(value, resolve=True)
    else:
        unresolved = value
        resolved = value
    return json.dumps(
        redact_credentials(resolved, unresolved),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


__all__ = [
    "CREDENTIAL_KEY_NAMES",
    "CREDENTIAL_KEY_SUFFIXES",
    "canonical_redacted_config_json",
    "is_credential_key",
    "is_literal_credential_value",
    "literal_credential_paths",
    "normalize_credential_key",
    "redact_credentials",
]
