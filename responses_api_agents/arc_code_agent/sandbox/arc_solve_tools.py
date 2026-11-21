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
# ruff: noqa
# fmt: off
# Vendored sandbox code: kept as run by the original harness; not linted or reformatted.
"""Solve-side helpers preloaded into the sandbox session.

Loaded by the sandbox server with ``exec(open(path).read())`` once at start-up,
before the per-row header that defines ``examples`` and ``test_inputs``. It first
loads the shared ARC helpers (``print_grid``, ``diff_text``, ``components``,
...) from ``arc_sdg_tools.py`` in the same directory, then adds ``check`` and
``predict`` used by the solving prompt.

Only stdout matters: the model sees whatever these functions print.
"""

import os as _os
import signal as _signal
import traceback as _traceback

import numpy as np

_CANDIDATE_DIRS = [
    _os.environ.get("ARC_TOOLS_DIR"),
    _os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else None,
]
for _dir in _CANDIDATE_DIRS:
    if _dir and _os.path.isfile(_os.path.join(_dir, "arc_sdg_tools.py")):
        with open(_os.path.join(_dir, "arc_sdg_tools.py"), encoding="utf-8") as _handle:
            exec(compile(_handle.read(), "arc_sdg_tools.py", "exec"))
        break
else:
    raise FileNotFoundError("arc_sdg_tools.py not found next to arc_solve_tools.py")

CHECK_TIMEOUT_SECONDS = 2


class _Timeout(Exception):
    pass


def _run_with_timeout(func, arg, seconds):
    def _handler(signum, frame):
        raise _Timeout(f"transform exceeded {seconds}s")

    previous = _signal.signal(_signal.SIGALRM, _handler)
    _signal.alarm(seconds)
    try:
        return func(arg)
    finally:
        _signal.alarm(0)
        _signal.signal(_signal.SIGALRM, previous)


def _to_grid(value):
    """Coerce a transform result to a 2D int array, or raise with a clear message."""
    arr = np.asarray(value)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError(f"result must be a non-empty 2D grid, got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        if np.issubdtype(arr.dtype, np.bool_):
            arr = arr.astype(int)
        elif np.issubdtype(arr.dtype, np.floating) and np.all(np.mod(arr, 1) == 0):
            arr = arr.astype(int)
        else:
            raise ValueError(f"result must contain integers, got dtype {arr.dtype}")
    if arr.min() < 0 or arr.max() > 9:
        raise ValueError(f"result values must be 0-9, got {arr.min()}..{arr.max()}")
    return arr.astype(int)


MAX_DIFF_CELLS = 12


def _diff_summary(pred, expected):
    """Compact mismatch report that fits the sandbox's 1,000-character output cap."""
    wrong = np.argwhere(pred != expected)
    rows = sorted({int(r) for r, _ in wrong})
    cells = [f"({int(r)},{int(c)}):{int(pred[r, c])}>{int(expected[r, c])}" for r, c in wrong[:MAX_DIFF_CELLS]]
    more = f" +{len(wrong) - MAX_DIFF_CELLS} more" if len(wrong) > MAX_DIFF_CELLS else ""
    return f"  wrong rows {rows[:10]}{'...' if len(rows) > 10 else ''}; cells pred>exp: " + " ".join(cells) + more


def _pairs(pairs):
    if pairs is None:
        pairs = globals().get("examples")
    if not pairs:
        raise ValueError("no example pairs: pass pairs=[...] or define `examples`")
    return pairs


def check(transform, pairs=None, timeout=CHECK_TIMEOUT_SECONDS, max_failures=2):
    """Run ``transform`` on every example input and report pass/fail per pair.

    For failures prints the exception, a shape mismatch, or a compact list of
    mismatched cells as ``(row,col):predicted>expected`` (first 12), so the whole
    report stays under the sandbox's 1,000-character output cap.
    Returns the number of passing pairs.
    """
    pairs = _pairs(pairs)
    passed = 0
    shown = 0
    for index, pair in enumerate(pairs, start=1):
        expected = np.asarray(pair["output"], dtype=int)
        try:
            pred = _to_grid(_run_with_timeout(transform, np.asarray(pair["input"], dtype=int), timeout))
        except _Timeout as exc:
            print(f"pair {index}: FAIL timeout ({exc})")
            continue
        except Exception as exc:  # noqa: BLE001 - report every failure to the model
            tb = _traceback.extract_tb(exc.__traceback__)
            where = f" at line {tb[-1].lineno}" if tb else ""
            print(f"pair {index}: FAIL error {type(exc).__name__}: {exc}{where}")
            continue
        if pred.shape != expected.shape:
            print(f"pair {index}: FAIL shape mismatch predicted {pred.shape} expected {expected.shape}")
            continue
        if np.array_equal(pred, expected):
            passed += 1
            print(f"pair {index}: pass")
            continue
        wrong = int((pred != expected).sum())
        print(f"pair {index}: FAIL {wrong}/{expected.size} cells differ")
        if shown < max_failures:
            print(_diff_summary(pred, expected))
            shown += 1
    print(f"check: passed {passed}/{len(pairs)}")
    return passed


def predict(transform, inputs=None, timeout=CHECK_TIMEOUT_SECONDS):
    """Apply ``transform`` to each test input and print the result as digit rows.

    Returns the list of predicted grids (``None`` where the transform failed).
    """
    if inputs is None:
        inputs = globals().get("test_inputs")
    if not inputs:
        raise ValueError("no test inputs: pass inputs=[...] or define `test_inputs`")
    results = []
    for index, grid in enumerate(inputs, start=1):
        try:
            pred = _to_grid(_run_with_timeout(transform, np.asarray(grid, dtype=int), timeout))
        except Exception as exc:  # noqa: BLE001
            print(f"test {index}: FAIL {type(exc).__name__}: {exc}")
            results.append(None)
            continue
        print(f"test {index} ({pred.shape[0]}x{pred.shape[1]}):")
        print(grid_to_text(pred))
        results.append(pred)
    return results


def objects(grid, *, bg=0, connectivity=4, by_color=True, limit=20):
    """Compact object list: id color size bbox(r0,c0,r1,c1). Prints at most `limit` rows."""
    comps = components(np.asarray(grid, dtype=int), bg=bg, connectivity=connectivity, by_color=by_color)
    comps = sorted(comps, key=lambda c: (-c.area, c.bbox))
    print(f"{len(comps)} objects (bg={bg}, conn={connectivity}); id color size bbox")
    for index, comp in enumerate(comps[:limit]):
        print(f"  {index} c{comp.color} n{comp.area} {comp.bbox}")
    if len(comps) > limit:
        print(f"  ... {len(comps) - limit} more")
    return comps


def diff(a, b, *, limit=MAX_DIFF_CELLS):
    """Changed cells between two same-shape grids as (r,c):a>b; prints count and first `limit`."""
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    if a.shape != b.shape:
        print(f"shapes differ {a.shape} vs {b.shape}")
        return None
    wrong = np.argwhere(a != b)
    cells = " ".join(f"({int(r)},{int(c)}):{int(a[r, c])}>{int(b[r, c])}" for r, c in wrong[:limit])
    print(f"{len(wrong)}/{a.size} cells differ: {cells}{' ...' if len(wrong) > limit else ''}")
    return wrong


def crop(grid, r0, c0, r1, c1):
    """Print a small region [r0:r1, c0:c1] with row indices; keep it small."""
    g = np.asarray(grid, dtype=int)[r0:r1, c0:c1]
    for offset, row in enumerate(g):
        print(f"{r0 + offset:>2} " + "".join(str(int(v)) for v in row))
    return g


__all__ = list(globals().get("__all__", [])) + ["check", "predict", "objects", "diff", "crop"]
