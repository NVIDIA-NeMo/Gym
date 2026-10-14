# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from benchmarks.claweval.prepare import prepare_split


def prepare(task_ids: list[str] | None = None) -> Path:
    return prepare_split("multi_turn", task_ids=task_ids)
