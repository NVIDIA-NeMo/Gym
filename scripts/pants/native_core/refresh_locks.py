# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regenerate both native CI resolves from clean sampled main, then publish together."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path

from lock_assets import source_uv_version
from prepare import EXTRAS, prepare, require_uv_version, validate_lock


HERE = Path(__file__).resolve().parent
INPUTS = ("uv.lock", "pyproject.toml", ".python-version")


def fingerprint(source: Path) -> dict[str, str]:
    return {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in INPUTS}


def clean_source(source: Path) -> str:
    root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=source, text=True).strip()
    if Path(root).resolve() != source.resolve():
        raise ValueError("--source must be a Git checkout root")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=source,
        text=True,
    )
    if dirty:
        raise ValueError("Sampled source has tracked changes; refusing to regenerate from an ambiguous revision")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()


def interpreter_environment(source: Path, uv: str) -> dict[str, str]:
    version = (source / ".python-version").read_text().strip()
    subprocess.run([uv, "python", "install", version], cwd=source, check=True, timeout=300)
    interpreter = subprocess.check_output([uv, "python", "find", version], cwd=source, text=True).strip()
    env = dict(os.environ)
    env["PATH"] = str(Path(interpreter).parent) + os.pathsep + env.get("PATH", "")
    return env


def publish(files: dict[str, bytes], destination: Path) -> None:
    """Replace validated files, manifest last; roll back a failed publication."""
    destination.mkdir(parents=True, exist_ok=True)
    original = {name: (destination / name).read_bytes() if (destination / name).exists() else None for name in files}
    replaced = []
    with tempfile.TemporaryDirectory(prefix=".lock-publication-", dir=destination) as temporary:
        staging = Path(temporary)
        for name, contents in files.items():
            (staging / name).write_bytes(contents)
        try:
            for name in sorted(files, key=lambda name: (name == "lock-inputs.json", name)):
                os.replace(staging / name, destination / name)
                replaced.append(name)
        except BaseException:
            for name in reversed(replaced):
                if original[name] is None:
                    (destination / name).unlink()
                else:
                    (staging / name).write_bytes(original[name])
                    os.replace(staging / name, destination / name)
            raise


def refresh(*, source: Path, pants: str, uv: str, destination: Path) -> dict[str, object]:
    """Generate and validate both complete resolves before changing any output."""
    source = source.resolve()
    revision = clean_source(source)
    initial = fingerprint(source)
    uv_version = source_uv_version(source)
    require_uv_version(uv, uv_version)
    destination.mkdir(parents=True, exist_ok=True)
    mutex = destination / ".refresh-locks.lock"
    with mutex.open("x"):
        pass
    try:
        with tempfile.TemporaryDirectory(prefix="gym-pants-locks-") as temporary:
            checkout = Path(temporary) / "checkout"
            prepare(source, checkout, uv=uv)
            environment = interpreter_environment(checkout, uv)
            subprocess.run(
                [pants, "--pants-config-files=pants.native.toml", "--no-pantsd", "generate-lockfiles"],
                cwd=checkout,
                env=environment,
                check=True,
                timeout=1800,
            )
            assets = checkout / "scripts/pants/native_core"
            resolves = {
                name: validate_lock(assets / f"{name}.requirements.txt", assets / f"{name}.lock") for name in EXTRAS
            }
            if clean_source(source) != revision or fingerprint(source) != initial:
                raise ValueError("Sampled source changed during lock generation; outputs were not published")
            config = tomllib.loads((checkout / "pants.native.toml").read_text())
            pex_version = config["pex-cli"]["version"].removeprefix("v")
            manifest = {
                "inputs": initial,
                "resolves": resolves,
                "uv_version": uv_version,
                "pants_version": config["GLOBAL"]["pants_version"],
                "pex_version": pex_version,
                "pex_release": f"https://github.com/pex-tool/pex/releases/tag/v{pex_version}",
            }
            files = {
                f"{name}.{suffix}": (assets / f"{name}.{suffix}").read_bytes()
                for name in EXTRAS
                for suffix in ("requirements.txt", "lock")
            }
            files["lock-inputs.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
            publish(files, destination)
            return manifest
    finally:
        mutex.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--pants", required=True)
    parser.add_argument("--uv", required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE, help="Defaults to this controller's native assets")
    args = parser.parse_args()
    pants = shutil.which(args.pants) or str(Path(args.pants).resolve())
    uv = shutil.which(args.uv) or str(Path(args.uv).resolve())
    print(json.dumps(refresh(source=args.source, pants=pants, uv=uv, destination=args.output_dir.resolve()), indent=2))


if __name__ == "__main__":
    main()
