# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym prepare wrapper for Tau3 banking_knowledge alltools."""

from pathlib import Path

from benchmarks.tau2.prepare_utils import prepare_banking_knowledge


def prepare() -> Path:
    return prepare_banking_knowledge("alltools")
