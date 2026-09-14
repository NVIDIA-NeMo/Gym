# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministically fetch and verify the pinned FACTS Benchmark Suite technical report (see paper/PAPER.md)."""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path


ARXIV_ID = "2512.10791"
VERSION = "v1"
URL = f"https://arxiv.org/pdf/{ARXIV_ID}{VERSION}"
SHA256 = "db046e76cc1877880843d0e7fd4898422f1064d47f8990b04f3c230229ede6be"  # pragma: allowlist secret
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "paper" / f"{ARXIV_ID}{VERSION}.pdf"


def fetch(output: Path = DEFAULT_OUTPUT) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and hashlib.sha256(output.read_bytes()).hexdigest() == SHA256:
        return output
    request = urllib.request.Request(URL, headers={"User-Agent": "nemo-gym-benchmark-paper-fetch/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        content = response.read()
    digest = hashlib.sha256(content).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"SHA-256 mismatch for {URL}: expected {SHA256}, got {digest}; refusing to write")
    output.write_bytes(content)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(fetch(args.output))


if __name__ == "__main__":
    sys.exit(main())
