# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate and materialize data-only replay assets from the private registry."""

import base64
import binascii
import hashlib
import json
from pathlib import Path

from responses_api_agents.ngc_recorded_context.runtime_contract import RuntimeContract


REPLAY_FILES = frozenset(
    {
        "case.json",
        "bundle.json",
        "coverage-v2.json.gz",
        "source-inventory.json",
        "source-extra-v4.json.gz",
        "temporal-world.json",
        "actor-params.json",
    }
)


def read_bundle(path: Path) -> dict[str, bytes]:
    """Reject unknown members, invalid base64 and bytes that differ from their digest."""
    bundle = json.loads(path.read_text())
    if set(bundle) != {"schema_version", "files"} or bundle["schema_version"] not in {
        "partner-if-replay-assets-v1",
        "partner-if-replay-assets-v2",
    }:
        raise ValueError("Unexpected replay asset schema")
    members = REPLAY_FILES | ({"runtime.json"} if bundle["schema_version"] == "partner-if-replay-assets-v2" else set())
    if set(bundle["files"]) != members:
        raise ValueError("Replay asset members differ from the declared data files")
    result = {}
    for name, record in bundle["files"].items():
        if set(record) != {"sha256", "data"}:
            raise ValueError("Unexpected replay asset fields")
        try:
            data = base64.b64decode(record["data"], validate=True)
        except (binascii.Error, TypeError) as error:
            raise ValueError("Invalid replay asset encoding") from error
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError("Replay asset checksum mismatch")
        result[name] = data
    if "runtime.json" in result:
        runtime = RuntimeContract.model_validate_json(result["runtime.json"])
        if set(runtime.input_hashes) != REPLAY_FILES or any(
            hashlib.sha256(result[name]).hexdigest() != digest for name, digest in runtime.input_hashes.items()
        ):
            raise ValueError("Runtime contract differs from bundled replay assets")
    return result


def materialize_bundle(source: Path, target: Path) -> None:
    """Write a verified bundle to a newly staged directory; never execute its data."""
    files = read_bundle(source)
    directory = target / "ngc-recorded-context"
    directory.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        (directory / name).write_bytes(data)


def prepared_bundle_matches(source: Path, target: Path) -> bool:
    """Verify actual cached bytes rather than trusting a completion marker."""
    try:
        files = read_bundle(source)
        return all((target / "ngc-recorded-context" / name).read_bytes() == data for name, data in files.items())
    except (OSError, ValueError, TypeError, KeyError):
        return False


def prepare_candidate(source: Path, target: Path, *, expected_sha256: str) -> Path:
    """Materialize a pinned candidate separately from the published release cache."""
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("Candidate replay asset digest differs from its declared pin")
    if "runtime.json" not in read_bundle(source):
        raise ValueError("A standalone candidate requires a version-2 runtime contract")
    if target.exists():
        raise FileExistsError(target)
    materialize_bundle(source, target)
    return target


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare a pinned local replay candidate without registry writes")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    print(prepare_candidate(arguments.input, arguments.output_dir, expected_sha256=arguments.sha256))
