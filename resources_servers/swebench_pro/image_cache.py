# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare SIFs from pinned registry images and verify their local provenance."""

import argparse
import fcntl
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path


def digest_hex(digest: str) -> str:
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ValueError("Local Pro images require a pinned sha256 registry digest")
    return digest.removeprefix("sha256:")


def sif_checksum(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def verify_local_image(path: Path, source_uri: str) -> dict:
    """The manifest records the trusted pull; the SIF hash detects later stale/replaced files.

    An OCI digest and a SIF checksum hash different formats. They cannot be
    compared directly, so provenance is recorded at conversion time.
    """
    manifest_path = path.with_suffix(path.suffix + ".json")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"Missing or invalid image provenance: {manifest_path}; prepare this cache again") from exc
    if manifest.get("source_uri") != source_uri:
        raise ValueError(f"Cached image source does not match the dataset digest: {path}")
    actual = sif_checksum(path)
    if manifest.get("sif_sha256") != actual:
        raise ValueError(f"Cached SIF checksum mismatch: {path}")
    return {"image": str(path.resolve()), "source_uri": source_uri, "sif_sha256": actual}


def prepare_image(image_dir: Path, repository: str, digest: str) -> dict:
    path = image_dir / f"{digest_hex(digest)}.sif"
    source_uri = f"docker://{repository}@{digest}"
    image_dir.mkdir(parents=True, exist_ok=True)
    # Concurrent pipeline preparations share this cache. Publish the SIF and
    # manifest under one lock so another preparation cannot observe a partial pair.
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _prepare_image(path, source_uri)


def _prepare_image(path: Path, source_uri: str) -> dict:
    if path.exists():
        return verify_local_image(path, source_uri)
    # Publish only a complete conversion. A failed pull never becomes a reusable cache entry.
    with tempfile.TemporaryDirectory(prefix=".pro-pull-", dir=path.parent) as temporary:
        candidate = Path(temporary) / path.name
        subprocess.run(["apptainer", "pull", "--disable-cache", str(candidate), source_uri], check=True)
        manifest = {"source_uri": source_uri, "sif_sha256": sif_checksum(candidate)}
        candidate_manifest = candidate.with_suffix(".sif.json")
        candidate_manifest.write_text(json.dumps(manifest, indent=2))
        candidate.replace(path)
        candidate_manifest.replace(path.with_suffix(".sif.json"))
    return verify_local_image(path, source_uri)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--instance-id", action="append", required=True)
    parser.add_argument("--repository", default="docker.io/jefzda/sweap-images")
    args = parser.parse_args()
    rows = {row["instance_id"]: row for row in map(json.loads, args.dataset.read_text().splitlines())}
    for instance_id in args.instance_id:
        info = prepare_image(args.image_dir, args.repository, rows[instance_id]["image_digest"])
        print(json.dumps({"instance_id": instance_id, **info}), flush=True)


if __name__ == "__main__":
    main()
