# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage trusted TB2 test bundles only after the separate verifier is created."""

import asyncio
import os
import tarfile
import tempfile
from pathlib import Path
from uuid import uuid4


REMOTE_TESTS_BUNDLE = "/tmp/nemo-gym-verifier-tests.tgz"


def pack_tests(source: Path, destination: Path, max_bytes: int) -> None:
    source = source.resolve()
    if not (source / "test.sh").is_file():
        raise ValueError(f"Verifier test bundle is missing test.sh: {source}")
    total = count = 0
    with tarfile.open(destination, "w:gz") as archive:
        for directory, dirs, names in os.walk(source, followlinks=False):
            for name in sorted(dirs + names):
                path = Path(directory) / name
                count += 1
                if count > 10000:
                    raise ValueError("Verifier test bundle exceeds 10,000 entries")
                if path.is_symlink():
                    raise ValueError(f"Verifier test bundle contains a symlink: {path}")
                if path.is_file():
                    total += path.stat().st_size
                    if total > max_bytes:
                        raise ValueError("Verifier test bundle exceeds verifier_tests_max_bytes")
                    archive.add(path, arcname="tests/" + path.relative_to(source).as_posix(), recursive=False)


async def stage_verifier_tests(sandbox, tests_dir: Path, max_bytes: int) -> None:
    with tempfile.TemporaryDirectory(prefix="nemo-gym-verifier-tests-") as temporary:
        bundle = Path(temporary) / "tests.tgz"
        await asyncio.to_thread(pack_tests, tests_dir, bundle, max_bytes)
        await sandbox.upload(bundle, REMOTE_TESTS_BUNDLE)
    backup = "/tmp/nemo-gym-image-tests-" + uuid4().hex
    # Preserve any image-baked tests; only the explicitly supplied bundle is run.
    result = await sandbox.exec(
        f": ng-tb4-install-tests; if [ -e /tests ] || [ -L /tests ]; then mv /tests {backup}; fi"
        f" && tar -xzf {REMOTE_TESTS_BUNDLE} -C / && chmod +x /tests/test.sh",
        timeout_s=300,
    )
    if result.error_type or result.return_code != 0:
        raise RuntimeError("Installing tests into the separate verifier failed")
