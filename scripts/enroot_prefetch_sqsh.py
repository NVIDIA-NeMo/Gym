#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pre-fetch enroot sqsh files for a list of container images.

Produces sqsh files in exactly the naming scheme expected by EnrootProvider
(sha256(image)[:16].sqsh), so they are picked up from the cache on the
first sandbox create without re-importing.

Usage:
    # Images on the command line
    python scripts/enroot_prefetch_sqsh.py \\
        --sqsh-dir /scratch/sqsh \\
        nvcr.io/nvidia/nemo:24.12 \\
        ubuntu:22.04

    # Images from a file (one per line, # comments allowed)
    python scripts/enroot_prefetch_sqsh.py \\
        --sqsh-dir /scratch/sqsh \\
        --images-file containers.txt \\
        --jobs 4

    # Let the script auto-detect sqsh-dir from NEMO_GYM_ENROOT_SQSH_CACHE
    NEMO_GYM_ENROOT_SQSH_CACHE=/scratch/sqsh \\
        python scripts/enroot_prefetch_sqsh.py ubuntu:22.04
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Must match DOCKER_HUB_HOSTS in nemo_gym/sandbox/providers/enroot/provider.py
_DOCKER_HUB_HOSTS = frozenset({"docker.io", "index.docker.io", "registry-1.docker.io"})


def _translate_docker_uri(image: str) -> str:
    """Mirror of EnrootProvider._translate_docker_uri."""
    first, sep, rest = image.partition("/")
    if sep and first in _DOCKER_HUB_HOSTS:
        return f"docker://{rest}"
    if sep and ("." in first or ":" in first or first == "localhost"):
        return f"docker://{first}#{rest}"
    return f"docker://{image}"


def _resolve_uri(image: str) -> str | None:
    """Return the enroot import URI for image, or None if image is already a local sqsh."""
    if image.endswith(".sqsh") or ("://" not in image and Path(image).exists()):
        return None
    if "://" in image:
        return image
    return _translate_docker_uri(image)


def _sqsh_key(image: str) -> str:
    return hashlib.sha256(image.encode()).hexdigest()[:16]


def _is_valid_sqsh(path: Path) -> bool:
    if not path.exists():
        return False
    st = path.stat()
    return stat.S_ISREG(st.st_mode) and st.st_uid == os.getuid() and st.st_size > 0


def prefetch_image(image: str, sqsh_dir: Path, extra_import_args: list[str]) -> tuple[str, str]:
    """Import one image. Returns (image, status) where status is 'cached', 'imported', or 'failed: ...'."""
    uri = _resolve_uri(image)
    if uri is None:
        return image, "skipped (local sqsh)"

    key = _sqsh_key(image)
    target = sqsh_dir / f"{key}.sqsh"

    if _is_valid_sqsh(target):
        return image, f"cached ({target})"

    enroot = shutil.which("enroot")
    if enroot is None:
        return image, "failed: enroot not found on PATH"

    tmp = sqsh_dir / f".{key}.{uuid.uuid4().hex}.tmp"
    argv = [enroot, "import", "-o", str(tmp), *extra_import_args, uri]
    try:
        result = subprocess.run(argv, capture_output=True, text=True, errors="replace")
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        return image, f"failed: {exc}"

    if result.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        return image, f"failed (code={result.returncode}): {result.stderr.strip()}"

    os.replace(tmp, target)
    return image, f"imported → {target}"


def _load_images_file(path: str) -> list[str]:
    images = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                images.append(line)
    return images


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pre-fetch enroot sqsh files for a list of container images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "images",
        nargs="*",
        metavar="IMAGE",
        help="Container images to pre-fetch (docker image refs or enroot URIs).",
    )
    parser.add_argument(
        "--images-file",
        metavar="FILE",
        help="File with one image per line (# comments allowed). Combined with positional IMAGE args.",
    )
    parser.add_argument(
        "--sqsh-dir",
        metavar="DIR",
        default=os.environ.get("NEMO_GYM_ENROOT_SQSH_CACHE"),
        help=(
            "Directory to store sqsh files. Defaults to $NEMO_GYM_ENROOT_SQSH_CACHE. "
            "Must match the sqsh_cache_dir configured in the EnrootProvider."
        ),
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel import jobs (default: 1). Each job calls 'enroot import'.",
    )
    parser.add_argument(
        "--extra-import-arg",
        action="append",
        default=[],
        dest="extra_import_args",
        metavar="ARG",
        help="Extra argument passed to 'enroot import' (can be repeated).",
    )
    args = parser.parse_args()

    images: list[str] = list(args.images)
    if args.images_file:
        images.extend(_load_images_file(args.images_file))
    images = list(dict.fromkeys(images))  # deduplicate, preserve order

    if not images:
        parser.error("No images specified. Provide IMAGE arguments or --images-file.")

    if not args.sqsh_dir:
        parser.error(
            "No sqsh directory specified. Use --sqsh-dir or set NEMO_GYM_ENROOT_SQSH_CACHE."
        )

    sqsh_dir = Path(args.sqsh_dir)
    sqsh_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    print(f"sqsh dir : {sqsh_dir}")
    print(f"images   : {len(images)}")
    print(f"jobs     : {args.jobs}")
    print()

    failed = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(prefetch_image, image, sqsh_dir, args.extra_import_args): image
            for image in images
        }
        for future in as_completed(futures):
            image, status = future.result()
            ok = not status.startswith("failed")
            marker = "✓" if ok else "✗"
            print(f"  {marker} {image}")
            print(f"      {status}")
            if not ok:
                failed += 1

    print()
    if failed:
        print(f"{failed}/{len(images)} image(s) failed.")
        return 1
    print(f"All {len(images)} image(s) ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
