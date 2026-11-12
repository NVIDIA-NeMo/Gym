# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free downloads shared by the P0 reporter and read-only docs job."""

import subprocess
from pathlib import Path


def download_artifacts(harnesses: list[str], *, repo: str, run_id: str, attempt: str, output: Path) -> None:
    """Download one attempt only; failed/partial downloads invalidate that harness's result."""
    for harness in harnesses:
        try:
            subprocess.run(
                [
                    "gh",
                    "run",
                    "download",
                    run_id,
                    "--repo",
                    repo,
                    "--name",
                    f"harness-conformance-{harness}-{run_id}-{attempt}",
                    "--dir",
                    str(output / harness),
                ],
                check=True,
            )
        except subprocess.CalledProcessError:
            # Do not let partially extracted or older files qualify this attempt.
            (output / harness / "probes" / "ci-result.json").unlink(missing_ok=True)
            print(f"::warning::Could not download {harness} evidence; its measurement is unavailable.")
