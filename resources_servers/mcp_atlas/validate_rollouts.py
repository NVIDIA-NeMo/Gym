# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check actual rollout JSONL for configured service-credential values before publishing it."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


if __package__:
    from .credentials import SERVICE_ENV_NAMES
else:
    from credentials import SERVICE_ENV_NAMES


@dataclass(frozen=True)
class CredentialLeak:
    """A location and credential names; matched values and surrounding text are never retained."""

    path: Path
    line: int
    credential_names: tuple[str, ...]


def _strings(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def scan_rollout_credentials(paths: Sequence[Path], credentials: Mapping[str, str]) -> list[CredentialLeak]:
    """Inspect raw and decoded JSON strings for exact configured values, without exposing those values.

    Only nonblank values are checked. This is an exact-value publication check, not proof against arbitrary
    encoded exfiltration or a replacement for scoped credentials and enforced sandbox isolation.
    """
    values = {name: value for name, value in credentials.items() if value.strip()}
    leaks: list[CredentialLeak] = []
    for path in paths:
        if path.suffix != ".jsonl":
            raise ValueError(f"expected a rollout JSONL file: {path}")
        with path.open() as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError as exc:
                    raise ValueError(f"invalid rollout JSON at {path}:{line_number}") from exc
                texts = [line, *_strings(row)]
                names = tuple(sorted(name for name, value in values.items() if any(value in text for text in texts)))
                if names:
                    leaks.append(CredentialLeak(path, line_number, names))
    return leaks


def main(argv: Sequence[str] | None = None) -> int:
    """Scan supplied rollout files using configured service values; print counts or safe rejection locations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollouts", type=Path, nargs="+", help="Actual rollout JSONL files to inspect")
    args = parser.parse_args(argv)
    credentials = {name: os.environ[name] for name in SERVICE_ENV_NAMES if os.environ.get(name, "").strip()}
    try:
        leaks = scan_rollout_credentials(args.rollouts, credentials)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")
    if leaks:
        for leak in leaks:
            print(f"credential value detected at {leak.path}:{leak.line} ({', '.join(leak.credential_names)})")
        return 1
    print(json.dumps({"checked_files": len(args.rollouts), "checked_credential_values": len(credentials), "leaks": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
