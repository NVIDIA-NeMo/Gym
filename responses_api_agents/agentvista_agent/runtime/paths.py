# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep writable paths in the namespace mounted by the evaluation launcher."""

import os
from pathlib import Path


def output_path(value: str | Path) -> Path:
    """Make a path absolute without resolving a container bind alias."""
    return Path(os.path.abspath(value))
