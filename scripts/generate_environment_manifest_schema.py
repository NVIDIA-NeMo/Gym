# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate or check the environment manifest JSON Schema."""

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "schemas" / "environment-manifest.schema.json"
sys.path.insert(0, str(REPO_ROOT))

from nemo_gym.environment_manifest import manifest_json_schema  # noqa: E402


def rendered_schema() -> str:
    return json.dumps(manifest_json_schema(), indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail when the checked-in schema is stale.")
    args = parser.parse_args()
    expected = rendered_schema()

    if args.check:
        actual = SCHEMA_PATH.read_text(encoding="utf-8") if SCHEMA_PATH.exists() else ""
        if actual != expected:
            print(
                f"{SCHEMA_PATH.relative_to(REPO_ROOT)} is stale; run "
                "`python scripts/generate_environment_manifest_schema.py`.",
                file=sys.stderr,
            )
            return 1
        return 0

    SCHEMA_PATH.parent.mkdir(exist_ok=True)
    SCHEMA_PATH.write_text(expected, encoding="utf-8")
    print(SCHEMA_PATH.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
