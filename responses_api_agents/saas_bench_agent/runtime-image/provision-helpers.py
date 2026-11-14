# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage SVG's host utilities for a runtime image and record exact binary hashes."""

import argparse
import hashlib
import json
import os
import pwd
import re
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--image-rootfs", type=Path, required=True)
    args = parser.parse_args()
    destination = args.destination.resolve()
    try:
        home = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
    except KeyError:
        home = Path.home().resolve()
    if any(destination == p or destination.is_relative_to(p) for p in (Path("/root"), home)):
        raise ValueError("Provisioning output must be outside forbidden homes")
    files = {}

    def copy(source: Path, relative: Path) -> None:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise FileExistsError(f"Refusing to replace an existing helper: {target}")
        if not target.exists():
            shutil.copy2(source, target)
            target.chmod(0o755 if source.stat().st_mode & 0o111 else 0o644)
        files[str(relative)] = {"source": str(source), "sha256": digest}

    for name in ("ip", "ss", "unsquashfs", "mksquashfs"):
        binary = shutil.which(name)
        if not binary:
            raise FileNotFoundError(f"Required host tool: {name}")
        source = Path(binary).resolve()
        copy(source, Path("usr/bin") / name)
        result = subprocess.run(["ldd", str(source)], check=True, capture_output=True, text=True)
        for library in re.findall(r"(?:=>\s+)?(/[^\s]+)", result.stdout):
            path = Path(library)
            if path.is_file() and not (args.image_rootfs / str(path).lstrip("/")).exists():
                copy(path, Path("usr/lib/x86_64-linux-gnu") / path.name)
    (destination / "saas-eval-tools.json").write_text(
        json.dumps({"platform": "SVG amd64", "files": files}, indent=2) + "\n"
    )
    print(f"Provisioned {len(files)} tools/libraries under {destination}")


if __name__ == "__main__":
    main()
