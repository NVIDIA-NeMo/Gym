# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned host-side bootstrap assets for mini-SWE."""

from pathlib import Path

from nemo_gym.sandbox.bootstrap import cached_download, python_asset


async def bootstrap_assets(arch: str, libc: str) -> tuple[Path, Path]:
    python = await python_asset(arch, libc)
    uv = await cached_download(
        f"https://github.com/astral-sh/uv/releases/download/0.10.12/uv-{arch}-unknown-linux-musl.tar.gz"
    )
    return uv, python
