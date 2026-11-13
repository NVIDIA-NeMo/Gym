# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned reference WebArena evaluator implementation."""

from resources_servers.webarena_browser.reference_evaluation.classic_evaluation import (
    evaluate_classic_task_sync,
)
from resources_servers.webarena_browser.reference_evaluation.eval_snapshots import (
    build_snapshot_context,
    collect_browser_snapshots_sync,
    collect_snapshots,
    merge_snapshots,
)
