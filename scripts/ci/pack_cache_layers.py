# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Split a directory tree into tar member lists that each hold at most a byte budget.

gym:main-offline appends the merged server dependency cache to the gym image. A cluster that
pulls the image through a remote registry refuses any layer above that registry's request
ceiling, so the cache goes into several layers instead of one. The first list holds every
directory, so later layers add files to directories that already carry the runtime owner.
Paths that share an inode stay in one list, so tar keeps them as hard links.
"""

import argparse
import os
import stat
from pathlib import Path


def member_groups(root: Path, top: str) -> tuple[list[str], list[tuple[list[str], int]]]:
    """Collect the directories and the non-directory members below ``root / top``.

    Args:
        root: Directory that tar runs from.
        top: Relative path of the tree to pack.

    Returns:
        The directories, parents first, and the other members as ``(paths, bytes)`` groups in
        path order. Hard-linked paths form one group that counts its bytes once.
    """
    directories = []
    groups: dict[object, tuple[list[str], int]] = {}
    for current, names, files in os.walk(root / top):
        names.sort()
        directories.append(Path(current).relative_to(root).as_posix())
        for name in [*sorted(files), *(name for name in names if (Path(current) / name).is_symlink())]:
            path = Path(current) / name
            info = path.lstat()
            relative = path.relative_to(root).as_posix()
            size = info.st_size if stat.S_ISREG(info.st_mode) else 0
            key = (info.st_dev, info.st_ino) if stat.S_ISREG(info.st_mode) and info.st_nlink > 1 else relative
            paths, _ = groups.setdefault(key, ([], size))
            paths.append(relative)
    return directories, sorted(groups.values(), key=lambda group: group[0][0])


def pack(root: Path, top: str, max_bytes: int) -> list[list[str]]:
    """Fill member lists in path order, starting a new list before one would exceed ``max_bytes``.

    Args:
        root: Directory that tar runs from.
        top: Relative path of the tree to pack.
        max_bytes: Uncompressed file bytes one list may hold. A single larger group gets a list
            of its own.

    Returns:
        Member lists in layer order; the first starts with every directory.
    """
    directories, groups = member_groups(root, top)
    layers = [list(directories)]
    filled = 0
    for paths, size in groups:
        if filled and filled + size > max_bytes:
            layers.append([])
            filled = 0
        layers[-1].extend(paths)
        filled += size
    return layers


def main() -> None:
    """Write one NUL-separated member list per layer as ``layer-NN.list``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("top")
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-bytes", type=int, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    layers = pack(arguments.root, arguments.top, arguments.max_bytes)
    for index, members in enumerate(layers):
        (arguments.output / f"layer-{index:02d}.list").write_text("".join(f"{path}\0" for path in members))
    print(f"Packed {arguments.top} into {len(layers)} layers of at most {arguments.max_bytes} bytes.")


if __name__ == "__main__":
    main()
