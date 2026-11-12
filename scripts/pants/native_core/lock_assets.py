# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select immutable, validated dependency assets for an exact sampled source."""

import hashlib
import json
import re
from pathlib import Path


INPUTS = ("uv.lock", "pyproject.toml", ".python-version")


def uv_version_from_text(text: str) -> str:
    versions = set(re.findall(r"https://astral\.sh/uv/([0-9]+\.[0-9]+\.[0-9]+)/install\.sh", text))
    if len(versions) != 1:
        raise ValueError("Sampled setup_dev.sh must pin exactly one uv installer version")
    return versions.pop()


def source_uv_version(source: Path) -> str:
    return uv_version_from_text((source / "scripts/ci/setup_dev.sh").read_text())


def fingerprint(source: Path) -> dict[str, str]:
    return {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in INPUTS}


def bundle_key(inputs: dict[str, str], uv_version: str) -> str:
    payload = json.dumps({"inputs": inputs, "uv_version": uv_version}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def lock_assets(source: Path, controller: Path | None = None) -> Path:
    """Fail before setup unless both resolves match this exact source and toolchain."""
    from prepare import EXTRAS, validate_lock

    controller = controller or Path(__file__).resolve().parent
    inputs = fingerprint(source)
    version = source_uv_version(source)
    key = bundle_key(inputs, version)
    for assets in (controller, controller / "lock-bundles" / key):
        manifest_path = assets / "lock-inputs.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("inputs") != inputs or manifest.get("uv_version") != version:
            continue
        if manifest.get("pants_version") != "2.30.2" or manifest.get("pex_version") != "2.101.6":
            raise ValueError("Dependency bundle uses a different Pants/PEX toolchain")
        for resolve in EXTRAS:
            actual = validate_lock(assets / f"{resolve}.requirements.txt", assets / f"{resolve}.lock")
            if actual != manifest["resolves"][resolve]:
                raise ValueError(f"Dependency bundle validation differs for {resolve}")
        return assets
    raise ValueError(f"No validated dependency bundle for sampled inputs: {key} (uv {version})")
