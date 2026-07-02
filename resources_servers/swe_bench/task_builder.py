# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backward-compatible re-exports — prefer ``resources_servers.swe_bench.task``."""

from resources_servers.swe_bench.task import (
    SweTask,
)
from resources_servers.swe_bench.task import (
    build_task as build_swetask,
)
from resources_servers.swe_bench.task import (
    harness_family_key as benchmark_key,
)
from resources_servers.swe_bench.task import (
    merge_row_metadata as problem_info_from_row,
)


__all__ = ["SweTask", "benchmark_key", "build_swetask", "problem_info_from_row"]
