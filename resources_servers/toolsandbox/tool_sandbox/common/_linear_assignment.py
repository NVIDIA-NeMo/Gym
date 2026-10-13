# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Dependency-free linear sum assignment (Hungarian / Jonker-Volgenant).

ToolSandbox milestone scoring solves a minimum-cost assignment on a small cost
matrix. Upstream used ``scipy.optimize.linear_sum_assignment``; scipy is
excluded from NeMo Gym's base install, and installing it into the per-server
venv is blocked by uv's project-level dependency exclusion. Since the only thing
the scorer needs is the optimal assignment's total cost (see
``snapshot_similarity`` in ``evaluation.py``, which returns
``exp(-cost[row_ind, col_ind].mean())``), a vendored exact solver gives
identical results without the dependency.

This is a drop-in replacement for the subset of the scipy API the scorer uses:
``row_ind, col_ind = linear_sum_assignment(cost_matrix)`` minimizing total cost,
raising ``ValueError`` when no finite-cost complete assignment exists (matching
scipy's behaviour on infeasible matrices, which the scorer catches to return 0).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def linear_sum_assignment(
    cost_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the linear sum assignment problem (minimization).

    Args:
        cost_matrix: 2-D array of costs. May contain ``inf`` for forbidden
            assignments. For an ``n x m`` matrix, ``min(n, m)`` pairs are
            returned.

    Returns:
        ``(row_ind, col_ind)`` such that ``cost_matrix[row_ind, col_ind]`` is
        the set of chosen assignments, with ``row_ind`` sorted ascending — the
        same convention as ``scipy.optimize.linear_sum_assignment``.

    Raises:
        ValueError: If the matrix is not 2-D, or no complete assignment with
            finite total cost exists (infeasible).
    """
    cost = np.asarray(cost_matrix, dtype=float)
    if cost.ndim != 2:
        raise ValueError("expected a 2-D cost matrix")

    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    # The Jonker-Volgenant shortest-augmenting-path formulation below requires
    # rows <= cols; transpose if needed and swap back at the end.
    transpose = n_rows > n_cols
    work = cost.T if transpose else cost
    n, m = work.shape  # n <= m

    INF = float("inf")
    # 1-indexed potentials/assignments; index 0 is a sentinel column.
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)  # p[j] = row (1-indexed) currently assigned to column j
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, m + 1):
                if not used[j]:
                    cur = work[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            if delta == INF:
                # No finite-cost augmenting path -> assignment is infeasible.
                raise ValueError("cost matrix is infeasible")
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        # Augment along the found path.
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    rows = []
    cols = []
    for j in range(1, m + 1):
        if p[j] != 0:
            rows.append(p[j] - 1)
            cols.append(j - 1)
    row_ind = np.asarray(rows, dtype=int)
    col_ind = np.asarray(cols, dtype=int)

    if transpose:
        row_ind, col_ind = col_ind, row_ind

    order = np.argsort(row_ind, kind="stable")
    return row_ind[order], col_ind[order]
