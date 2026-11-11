# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recover only Pong's data asset omitted from the ALE 0.10.2 source archive.

Use the exact official PyPI wheel as a data source, never install its x86 code
on ARM. Both archive and ROM checksums are pinned and checked.
"""

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path


WHEEL_URL = "https://files.pythonhosted.org/packages/1e/74/a989e287b2a3822f636e2bd80beb3605bc026fa50b05b8969d4c4a0f0e7e/ale_py-0.10.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
WHEEL_SHA256 = "cba2a904ab80670b9db8965d95dad3d978b4a133595c04b1be42f86c0371fdde"
ROM_SHA256 = "41623e3c26148f96cbe77d56512845d224a63c8fe5bf90e38527155e39ec96d3"


def install(destination):
    destination = Path(destination)
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != ROM_SHA256:
            raise ValueError(f"Existing Pong ROM has an unexpected hash: {destination}")
        return
    with urllib.request.urlopen(WHEEL_URL, timeout=120) as response:
        archive = response.read()
    if hashlib.sha256(archive).hexdigest() != WHEEL_SHA256:
        raise ValueError("ALE wheel has an unexpected hash")
    with zipfile.ZipFile(io.BytesIO(archive)) as wheel:
        rom = wheel.read("ale_py/roms/pong.bin")
    if len(rom) != 2048 or hashlib.sha256(rom).hexdigest() != ROM_SHA256:
        raise ValueError("Pong ROM has an unexpected size or hash")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(rom)


if __name__ == "__main__":
    import ale_py

    destination = Path(ale_py.__file__).parent / "roms/pong.bin"
    install(destination)
    print("Verified Pong ROM:", destination, ROM_SHA256)
