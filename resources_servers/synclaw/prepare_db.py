# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the BioNeMo-SynClaw database: reaction templates, eMolecules plus common-reagent stock, search index.

Usage:

    python resources_servers/synclaw/prepare_db.py [--release YYYY-MM-DD]

Clones BioNeMo-SynClaw at the revision pinned in pyproject.toml and uses its CLI to download an eMolecules
release, load the reaction templates, the eMolecules building blocks and the common reagents, and
write an indexed database to the resources server's default ``db_path``. Requires ``git`` and
``uv``.
"""

import argparse
import json
import re
import shutil
import subprocess
import tomllib
from pathlib import Path


SERVER_DIR = Path(__file__).resolve().parent
BUILD_DIR = SERVER_DIR / ".synclaw_build"
DEFAULT_OUTPUT = SERVER_DIR / "data" / "synclaw_emolecules.sqlite"
COMMON_REAGENTS = "data/building_blocks/common_reagent_building_blocks.txt"
# BioNeMo-SynClaw names no default release because eMolecules retires old ones; this is the tested one.
DEFAULT_EMOLECULES_RELEASE = "2026-09-01"


def pinned_synclaw() -> tuple[str, str]:
    """Return the (repository URL, revision) of the BioNeMo-SynClaw dependency in pyproject.toml."""
    dependencies = tomllib.loads((SERVER_DIR / "pyproject.toml").read_text())["project"]["dependencies"]
    for dependency in dependencies:
        match = re.fullmatch(r"synclaw @ git\+(\S+?)@(\S+)", dependency)
        if match is not None:
            return match.group(1), match.group(2)
    raise RuntimeError("pyproject.toml has no `synclaw @ git+<url>@<revision>` dependency")


def clone_pinned_synclaw(destination: Path) -> None:
    """Fetch the pinned tag or commit, replacing any existing checkout at ``destination``."""
    url, revision = pinned_synclaw()
    shutil.rmtree(destination, ignore_errors=True)
    print(f"Cloning BioNeMo-SynClaw {revision}", flush=True)
    subprocess.run(["git", "init", "--quiet", str(destination)], check=True)
    subprocess.run(["git", "-C", str(destination), "fetch", "--quiet", "--depth", "1", url, revision], check=True)
    subprocess.run(["git", "-C", str(destination), "checkout", "--quiet", "--detach", "FETCH_HEAD"], check=True)


def uv_run(checkout: Path, *command: str) -> dict:
    """Run ``command`` in the BioNeMo-SynClaw checkout's locked environment and parse its JSON stdout."""
    result = subprocess.run(
        ["uv", "run", "--locked", "--project", str(checkout), *command],
        cwd=checkout,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return json.loads(result.stdout)


def synclaw(checkout: Path, *args: str) -> dict:
    return uv_run(checkout, "synclaw", *args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--release",
        default=DEFAULT_EMOLECULES_RELEASE,
        help="eMolecules release date (YYYY-MM-DD); default: %(default)s",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    checkout = BUILD_DIR / "synclaw"
    work_db = BUILD_DIR / "synclaw.work.sqlite"
    work_db.unlink(missing_ok=True)
    BUILD_DIR.mkdir(exist_ok=True)
    clone_pinned_synclaw(checkout)

    print("Downloading and normalizing eMolecules building blocks", flush=True)
    # The archive and converted file live outside the checkout so rebuilds reuse them.
    emolecules = synclaw(checkout, "prepare-emolecules-bbs", "--release", args.release, "--dest-root", str(BUILD_DIR))

    print("Loading reaction templates", flush=True)
    synclaw(checkout, "db", "init", "--db", str(work_db))
    synclaw(checkout, "ingest-reactions", "--db", str(work_db))

    print(f"Loading eMolecules {emolecules['release']} building blocks (slow)", flush=True)
    catalog = synclaw(checkout, "ingest-bbs", emolecules["output"], "--db", str(work_db))
    print(f"  {catalog['inserted']:,} building blocks", flush=True)

    print("Loading common reagents", flush=True)
    reagents = synclaw(checkout, "ingest-bbs", COMMON_REAGENTS, "--db", str(work_db))
    if reagents["invalid"] or reagents["inserted"] + reagents["duplicates"] != reagents["read"]:
        raise RuntimeError(f"common reagents did not all load: {reagents}")
    print(f"  {reagents['inserted']} added, {reagents['duplicates']} already in eMolecules", flush=True)

    print(f"Building the search index at {args.output}", flush=True)
    synclaw(checkout, "db", "prepare-search", "--db", str(work_db), "--output", str(args.output.resolve()))
    # BioNeMo-SynClaw writes through a private temp file; let other users and jobs read the shared database.
    args.output.chmod(0o644)
    work_db.unlink()
    print(f"Done: {args.output}", flush=True)


if __name__ == "__main__":
    main()
