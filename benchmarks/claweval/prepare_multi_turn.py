# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from benchmarks.claweval.prepare import prepare_split


def prepare():
    return prepare_split("multi_turn")
