# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the KIDBench multi-turn track.

Both tracks come from one pinned upstream checkout and one grid expansion, so this delegates
to ``benchmarks.kidbench.prepare`` rather than repeating it.
"""

from pathlib import Path

from benchmarks.kidbench.prepare import prepare as prepare_kidbench


def prepare() -> Path:
    return prepare_kidbench(track="multi_turn")
