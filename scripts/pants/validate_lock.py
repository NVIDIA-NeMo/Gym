# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reject stale or divergent dependency inputs before publishing timing ratios."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


def requirement_lines(text: str) -> list[str]:
    return sorted(line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#"))


def validate(root: Path, *, uv: str = "uv") -> dict[str, object]:
    requirements_path = root / "scripts/pants/pilot.requirements.txt"
    lock_path = root / "scripts/pants/pilot.lock"
    exported = subprocess.run(
        [
            uv,
            "export",
            "--locked",
            "--extra",
            "dev",
            "--no-emit-project",
            "--no-hashes",
            "--no-annotate",
            "--prune",
            "mypy",
            "--prune",
            "pre-commit",
            "--prune",
            "ruff",
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    requirements = requirement_lines(requirements_path.read_text())
    if requirements != requirement_lines(exported):
        raise ValueError("Pilot requirements are stale; run scripts/pants/refresh-locks.sh")
    pins: set[tuple[str, str]] = set()
    unconditional: set[tuple[str, str]] = set()
    for line in requirements:
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)(?:\[[^]]+\])?==([^;\s]+)(?:\s*;.*)?", line)
        if match is None:
            raise ValueError(f"The demo requires exact registry pins, got: {line}")
        pin = (re.sub(r"[-_.]+", "-", match[1]).lower(), match[2])
        pins.add(pin)
        if ";" not in line:
            unconditional.add(pin)
    text = lock_path.read_text()
    lock = json.loads("\n".join(line for line in text.splitlines() if not line.startswith(("#", "//"))))
    resolved = {
        (item["project_name"], item["version"])
        for resolve in lock["locked_resolves"]
        for item in resolve["locked_requirements"]
    }
    if resolved - pins or unconditional - resolved:
        raise ValueError(
            f"Pants/uv pins differ: only Pants={sorted(resolved - pins)}, missing unconditional uv={sorted(unconditional - resolved)}"
        )
    return {
        "valid": True,
        "pinned_distributions": len(pins),
        "requirements_sha256": hashlib.sha256(requirements_path.read_bytes()).hexdigest(),
        "pants_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "uv_lock_sha256": hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uv", default="uv")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(validate(args.root, uv=args.uv), indent=2))


if __name__ == "__main__":
    main()
