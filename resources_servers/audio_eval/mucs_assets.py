# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MUCS 2021 scoring maps, downloaded on first use.

The punctuation and transliteration word maps are part of the official MUCS 2021 blind-test release (OpenSLR SLR104,
CC BY-SA 4.0), so they are fetched from the pinned archive instead of being redistributed in this repository. Each
file is checked against a pinned SHA-256 before use.
"""

from __future__ import annotations

import hashlib
import os
import tarfile
import urllib.request
from functools import cache
from pathlib import Path

from nemo_gym import CACHE_DIR


ARCHIVE_URL = "https://www.openslr.org/resources/104/subtask2_blindtest_wReadme.tar.gz"
# SHA-256 of the full archive (1,008,263,116 bytes). The scorer verifies the three members below instead, because
# streaming stops as soon as they are found.
ARCHIVE_SHA256 = "b134ba007e6a310a2ca2a89619257ea6e7de744e135043c8c517cc05a404889a"
# Members of ``subtask2_blindtest_wReadme/inference/``.
ASSET_SHA256 = {
    "punct.mapping.txt": "43927a9aea99289d552bdc509406680688f67e25d52fb6f9debcd27c1e31b1ef",
    "hin2engwords.txt": "af49f1a95114021e678a0e86c7a9c4e85c8b91072ebd538d76293e73b8b5968e",
    "ben2engwords.txt": "8e53663a8b839e042eb9e430877fc9f783efb056d169de2e4c8d7b1fe28e5b80",
}


def assets_dir() -> Path:
    """Cache directory that holds the verified MUCS 2021 maps."""
    return CACHE_DIR / "audio_eval" / "mucs2021"


def _is_valid(path: Path, name: str) -> bool:
    return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == ASSET_SHA256[name]


def fetch_assets(directory: Path) -> None:
    """Download the missing or corrupted maps from the pinned archive into ``directory``."""
    missing = {name for name in ASSET_SHA256 if not _is_valid(directory / name, name)}
    if not missing:
        return
    directory.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(ARCHIVE_URL, headers={"User-Agent": "nemo-gym-audio-eval"})
    with urllib.request.urlopen(request, timeout=300) as source, tarfile.open(fileobj=source, mode="r|gz") as archive:
        for member in archive:
            path = Path(member.name)
            if not member.isfile() or path.parent.name != "inference" or path.name not in missing:
                continue
            data = archive.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != ASSET_SHA256[path.name]:
                raise ValueError(f"MUCS scoring asset {path.name} does not match its pinned SHA-256")
            partial = directory / f"{path.name}.{os.getpid()}.partial"
            partial.write_bytes(data)
            os.replace(partial, directory / path.name)
            missing.discard(path.name)
            if not missing:
                return
    raise ValueError(f"MUCS scoring assets not found in {ARCHIVE_URL}: {sorted(missing)}")


def read_mapping(path: Path) -> dict[str, str]:
    """Read a released mapping with the official first-key-wins behavior."""
    mapping: dict[str, str] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            fields = raw_line.strip().split()
            if not fields:
                raise ValueError(f"Blank mapping row at {path}:{line_number}")
            mapping.setdefault(fields[0], " ".join(fields[1:]))
    return mapping


@cache
def load_official_mapping(name: str) -> dict[str, str]:
    """One official map, downloaded and verified on first use."""
    if name not in ASSET_SHA256:
        raise ValueError(f"Unknown MUCS scoring asset: {name}")
    directory = assets_dir()
    fetch_assets(directory)
    return read_mapping(directory / name)
