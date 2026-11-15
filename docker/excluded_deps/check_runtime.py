# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reject an incompatible Torch runtime before syncing the opt-in codec layer."""

import importlib.metadata
import sys
import tomllib
from pathlib import Path


def check_runtime(lock: Path) -> None:
    packages = tomllib.loads(lock.read_text())["package"]
    versions = {package["version"] for package in packages if package["name"] == "torch"}
    if len(versions) != 1:
        raise RuntimeError("The codec lock must select exactly one Torch version.")
    expected = versions.pop()
    try:
        installed = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return
    # A CUDA/local wheel suffix does not change the upstream Torch release.
    if installed.partition("+")[0] != expected:
        raise RuntimeError(
            f"The codec lock requires torch=={expected}, but the active environment has torch=={installed}. "
            "Refusing to replace the model runtime. Use a compatible environment or review the codec pins."
        )


if __name__ == "__main__":
    check_runtime(Path(sys.argv[1]))
