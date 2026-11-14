# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve public Gym paths without a private source-repository dependency."""

from pathlib import Path


GYM_ROOT = Path(__file__).resolve().parents[2]
