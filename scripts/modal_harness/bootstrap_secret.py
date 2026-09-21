# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create the campaign secret in FDR from a local .env, without printing any value.

`modal secret create NAME KEY=VALUE` puts the value in argv, where it lands in shell
history and in any transcript of the session that ran it. The CLI's `--from-json` reads
from a file instead, so nothing sensitive appears on a command line.

The intermediate file is written to a private temp path, chmod 600, and unlinked in a
`finally` -- it must not outlive the call even if the command fails. `--from-dotenv`
against the real `.env` would work too, but it would upload every key in that file;
this forwards an allowlist so an unrelated credential sitting in the same file never
leaves the machine.

    python scripts/modal_harness/bootstrap_secret.py --env-file /path/to/.env
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


SECRET_NAME = "nemo-gym-campaign-tokens"
ENVIRONMENT = "FDR"

#: Only these are forwarded. An allowlist rather than "everything in .env", so an unrelated
#: credential sitting in the same file never leaves the machine.
WANTED = (
    "MODAL_PROXY_TOKEN",
    "SUPER_VL_MODAL_TOKEN",
    "OPENROUTER_API_KEY_SNORKEL",
)

#: Forwarded when present, but not required. HF_TOKEN is needed only by benchmarks whose
#: defenses load gated HuggingFace repos; demanding it would block everyone else.
OPTIONAL = ("HF_TOKEN",)


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key in WANTED or key in OPTIONAL:
            values[key] = value.strip().strip('"').strip("'")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--name", default=SECRET_NAME)
    parser.add_argument("--environment", default=ENVIRONMENT)
    args = parser.parse_args(argv)

    values = read_env_file(args.env_file)
    missing = [key for key in WANTED if key not in values or not values[key]]
    if missing:
        print(f"missing from {args.env_file}: {', '.join(missing)}", file=sys.stderr)
        return 1

    handle, temp_path = tempfile.mkstemp(suffix=".json")
    try:
        os.fchmod(handle, 0o600)
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(values, stream)
        completed = subprocess.run(
            [
                "modal",
                "secret",
                "create",
                args.name,
                "--from-json",
                temp_path,
                "--env",
                args.environment,
                "--force",
            ],
            capture_output=True,
            text=True,
        )
    finally:
        # Unlink even on failure: a credentials file left in /tmp is the thing this
        # function exists to avoid.
        try:
            os.unlink(temp_path)
        except OSError:
            pass

    if completed.returncode != 0:
        # stderr can quote the request body, so report the status rather than echo it.
        print(f"modal secret create failed (exit {completed.returncode})", file=sys.stderr)
        print(completed.stderr.splitlines()[-1] if completed.stderr else "", file=sys.stderr)
        return completed.returncode

    # Key names and lengths only -- enough to confirm the right things were sent, and not
    # enough to reconstruct any of them.
    print(f"created secret {args.name!r} in environment {args.environment!r} with keys:")
    for key in sorted(values):
        print(f"  {key} ({len(values[key])} chars)")
    for key in OPTIONAL:
        if key not in values:
            print(f"  {key}: absent from {args.env_file} (skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
