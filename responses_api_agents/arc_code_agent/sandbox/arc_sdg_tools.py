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
"""Common helper tools for ARC Python tool-use traces.

This module is exec'd inside the sandbox server's namespace. Keep the API small, explicit, and dependency-light: the helpers
should replace repeated validation/debug scaffolding while leaving each puzzle's
specific logic to the model.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence
import signal

import numpy as np


ARC_COLORS = set(range(10))


@dataclass(frozen=True)
class Component:
    """Connected component summary for a single color."""

    color: int
    cells: tuple[tuple[int, int], ...]
    bbox: tuple[int, int, int, int]  # r0, c0, r1, c1 inclusive
    area: int
    height: int
    width: int
    centroid: tuple[float, float]

    @property
    def rows(self) -> tuple[int, ...]:
        return tuple(r for r, _ in self.cells)

    @property
    def cols(self) -> tuple[int, ...]:
        return tuple(c for _, c in self.cells)


def as_grid(grid: Any, name: str = "grid") -> np.ndarray:
    """Return `grid` as a 2D integer numpy array, or raise AssertionError."""

    arr = np.asarray(grid)
    assert arr.ndim == 2, f"{name} must be 2D, got shape {arr.shape}"
    assert arr.dtype.kind in {"i", "u"}, f"{name} must have integer dtype, got {arr.dtype}"
    return arr


def validate_grid(grid: Any, name: str = "grid", max_size: int = 30) -> np.ndarray:
    """Validate ARC grid shape and color range, returning a numpy array."""

    arr = as_grid(grid, name=name)
    h, w = arr.shape
    assert 1 <= h <= max_size, f"{name} height {h} outside 1..{max_size}"
    assert 1 <= w <= max_size, f"{name} width {w} outside 1..{max_size}"
    assert np.all((arr >= 0) & (arr <= 9)), f"{name} has values outside 0..9"
    return arr


def grid_to_text(grid: Any, sep: str = "") -> str:
    """Render a grid as compact text, one row per line."""

    arr = as_grid(grid)
    return "\n".join(sep.join(str(int(x)) for x in row) for row in arr)


grid_to_str = grid_to_text


def print_grid(grid: Any, label: str | None = None, sep: str = "") -> None:
    """Print a compact text rendering of a grid."""

    if label:
        print(label)
    print(grid_to_text(grid, sep=sep))


print_grid_text = print_grid


def print_pair(input_grid: Any, output_grid: Any, *, input_label: str = "input", output_label: str = "output") -> None:
    """Print an input/output pair in the compact ARC text format."""

    print_grid(input_grid, label=input_label)
    print_grid(output_grid, label=output_label)


def diff_text(a: Any, b: Any, same: str = ".", sep: str = "") -> str:
    """Render changed cells as output digits and unchanged cells as `same`."""

    ga = as_grid(a, name="a")
    gb = as_grid(b, name="b")
    assert ga.shape == gb.shape, f"diff shape mismatch: {ga.shape} vs {gb.shape}"
    rows = []
    for r in range(ga.shape[0]):
        vals = [same if int(ga[r, c]) == int(gb[r, c]) else str(int(gb[r, c])) for c in range(ga.shape[1])]
        rows.append(sep.join(vals))
    return "\n".join(rows)


def grid_hash(grid: Any) -> tuple[tuple[int, ...], ...]:
    """Hashable exact grid representation."""

    arr = as_grid(grid)
    return tuple(tuple(int(x) for x in row) for row in arr)


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    """Return whether row/col is inside a grid shape."""

    h, w = shape
    return 0 <= r < h and 0 <= c < w


def bbox_from_cells(cells: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    """Inclusive bounding box for `(row, col)` cells."""

    cells = list(cells)
    assert cells, "bbox_from_cells requires at least one cell"
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def crop_bbox(grid: Any, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Return the inclusive bounding-box crop."""

    arr = as_grid(grid)
    r0, c0, r1, c1 = bbox
    return arr[r0 : r1 + 1, c0 : c1 + 1]


def mask_from_cells(shape: tuple[int, int], cells: Iterable[tuple[int, int]]) -> np.ndarray:
    """Boolean mask with True at listed cells."""

    mask = np.zeros(shape, dtype=bool)
    for r, c in cells:
        if in_bounds(shape, r, c):
            mask[r, c] = True
    return mask


def neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def neighbors8(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return (
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    )


def components(
    grid: Any,
    *,
    colors: Iterable[int] | None = None,
    bg: int | None = 0,
    connectivity: int = 4,
    by_color: bool = True,
) -> list[Component]:
    """Connected components for ARC grids.

    Args:
        grid: 2D integer grid.
        colors: Optional color subset to include.
        bg: Background color to exclude. Set to None to include all colors.
        connectivity: 4 or 8.
        by_color: If true, cells only connect to same-color cells. If false,
            all included non-bg cells connect regardless of color; the component
            color is set to the unique color if one exists, else -1.
    """

    arr = validate_grid(grid)
    assert connectivity in (4, 8), "connectivity must be 4 or 8"
    include_colors = set(colors) if colors is not None else None
    neigh = neighbors4 if connectivity == 4 else neighbors8
    seen = np.zeros(arr.shape, dtype=bool)
    out: list[Component] = []

    for sr in range(arr.shape[0]):
        for sc in range(arr.shape[1]):
            if seen[sr, sc]:
                continue
            color = int(arr[sr, sc])
            if bg is not None and color == bg:
                seen[sr, sc] = True
                continue
            if include_colors is not None and color not in include_colors:
                seen[sr, sc] = True
                continue

            q = deque([(sr, sc)])
            seen[sr, sc] = True
            cells = []
            comp_colors = set()
            while q:
                r, c = q.popleft()
                cur_color = int(arr[r, c])
                cells.append((r, c))
                comp_colors.add(cur_color)
                for nr, nc in neigh(r, c):
                    if not in_bounds(arr.shape, nr, nc) or seen[nr, nc]:
                        continue
                    next_color = int(arr[nr, nc])
                    if bg is not None and next_color == bg:
                        seen[nr, nc] = True
                        continue
                    if include_colors is not None and next_color not in include_colors:
                        seen[nr, nc] = True
                        continue
                    if by_color and next_color != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            bbox = bbox_from_cells(cells)
            r0, c0, r1, c1 = bbox
            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]
            out.append(
                Component(
                    color=color if len(comp_colors) == 1 else -1,
                    cells=tuple(cells),
                    bbox=bbox,
                    area=len(cells),
                    height=r1 - r0 + 1,
                    width=c1 - c0 + 1,
                    centroid=(float(np.mean(rows)), float(np.mean(cols))),
                )
            )
    return out


connected_components = components
find_objects = components


def rect_cells(r0: int, c0: int, height: int, width: int, *, border_only: bool = False) -> list[tuple[int, int]]:
    """Cells in an axis-aligned rectangle."""

    cells = []
    for r in range(r0, r0 + height):
        for c in range(c0, c0 + width):
            if border_only and r not in (r0, r0 + height - 1) and c not in (c0, c0 + width - 1):
                continue
            cells.append((r, c))
    return cells


def line_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    """Integer Bresenham line cells between two endpoints."""

    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    r, c = r0, c0
    cells = []
    if dc > dr:
        err = dc // 2
        while c != c1:
            cells.append((r, c))
            err -= dr
            if err < 0:
                r += sr
                err += dc
            c += sc
    else:
        err = dr // 2
        while r != r1:
            cells.append((r, c))
            err -= dc
            if err < 0:
                c += sc
                err += dr
            r += sr
    cells.append((r1, c1))
    return cells


def diamond_cells(center_r: int, center_c: int, radius: int, *, border_only: bool = True) -> list[tuple[int, int]]:
    """Cells in a Manhattan diamond."""

    assert radius >= 0, "radius must be nonnegative"
    cells = []
    for dr in range(-radius, radius + 1):
        rem = radius - abs(dr)
        for dc in range(-rem, rem + 1):
            if border_only and abs(dr) + abs(dc) != radius:
                continue
            cells.append((center_r + dr, center_c + dc))
    return cells


def can_place(
    grid: Any,
    cells: Iterable[tuple[int, int]],
    *,
    allowed: Iterable[int] = (0,),
    require_in_bounds: bool = True,
) -> bool:
    """Return whether all cells can be placed on colors in `allowed`."""

    arr = as_grid(grid)
    allowed_set = set(int(x) for x in allowed)
    for r, c in cells:
        if not in_bounds(arr.shape, r, c):
            if require_in_bounds:
                return False
            continue
        if int(arr[r, c]) not in allowed_set:
            return False
    return True


def place_cells(
    grid: Any,
    cells: Iterable[tuple[int, int]],
    color: int,
    *,
    copy: bool = True,
    clip: bool = False,
) -> np.ndarray:
    """Paint cells with `color`, optionally copying and clipping out-of-bounds."""

    arr = as_grid(grid).copy() if copy else as_grid(grid)
    assert color in ARC_COLORS, f"invalid ARC color {color}"
    for r, c in cells:
        if in_bounds(arr.shape, r, c):
            arr[r, c] = color
        elif not clip:
            raise AssertionError(f"cell {(r, c)} out of bounds for shape {arr.shape}")
    return arr


def draw_rect(grid: Any, r0: int, c0: int, height: int, width: int, color: int, *, border_only: bool = False) -> np.ndarray:
    return place_cells(grid, rect_cells(r0, c0, height, width, border_only=border_only), color)


def draw_line(grid: Any, r0: int, c0: int, r1: int, c1: int, color: int) -> np.ndarray:
    return place_cells(grid, line_cells(r0, c0, r1, c1), color)


def draw_diamond(grid: Any, center_r: int, center_c: int, radius: int, color: int, *, border_only: bool = True) -> np.ndarray:
    return place_cells(grid, diamond_cells(center_r, center_c, radius, border_only=border_only), color)


def _call_custom_check(fn: Callable, seed: int, input_grid: np.ndarray, output_grid: np.ndarray | None) -> None:
    """Call user check with common signatures."""

    attempts = []
    if output_grid is not None:
        attempts.extend(
            [
                lambda: fn(input_grid, output_grid, seed=seed),
                lambda: fn(input_grid, output_grid),
            ]
        )
    attempts.extend([lambda: fn(input_grid, seed=seed), lambda: fn(input_grid)])
    last_type_error = None
    for attempt in attempts:
        try:
            attempt()
            return
        except TypeError as exc:
            last_type_error = exc
    raise last_type_error if last_type_error is not None else RuntimeError("custom check failed")


def arc_check(
    generate_puzzle_input: Callable[[int], Any],
    generate_puzzle_output: Callable[[Any], Any] | None = None,
    *,
    seeds: Iterable[int] = range(20),
    custom_checks: Sequence[Callable] = (),
    require_change: bool | None = None,
    min_unique_inputs: int | None = None,
    min_unique_outputs: int | None = None,
    examples: int = 3,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run common ARC SDG validation and print a compact report.

    This replaces the repeated boilerplate seen in SDG traces: size/color checks,
    determinism checks, input/output equality checks, diversity counts, failure
    seed reporting, and example grid printing. For stg3 input-only generation,
    pass only `generate_puzzle_input`. For stg2/stg4, pass both functions.
    """

    seed_list = list(seeds)
    assert seed_list, "arc_check requires at least one seed"
    if require_change is None:
        require_change = generate_puzzle_output is not None

    failures = []
    inputs = []
    outputs = []
    shapes_in = []
    shapes_out = []

    for seed in seed_list:
        try:
            inp = validate_grid(generate_puzzle_input(seed), name=f"input(seed={seed})")
            inp2 = validate_grid(generate_puzzle_input(seed), name=f"input2(seed={seed})")
            assert np.array_equal(inp, inp2), f"generate_puzzle_input is not deterministic for seed {seed}"
            out = None
            if generate_puzzle_output is not None:
                out = validate_grid(generate_puzzle_output(inp.copy()), name=f"output(seed={seed})")
                out2 = validate_grid(generate_puzzle_output(inp.copy()), name=f"output2(seed={seed})")
                assert np.array_equal(out, out2), f"generate_puzzle_output is not deterministic for seed {seed}"
                if require_change and inp.shape == out.shape:
                    assert not np.array_equal(inp, out), f"output equals input for seed {seed}"
            for check in custom_checks:
                _call_custom_check(check, seed, inp, out)
            inputs.append(inp)
            outputs.append(out)
            shapes_in.append(inp.shape)
            if out is not None:
                shapes_out.append(out.shape)
        except Exception as exc:  # noqa: BLE001 - model-facing diagnostic helper
            failures.append((seed, type(exc).__name__, str(exc)))

    unique_inputs = len({grid_hash(g) for g in inputs})
    unique_outputs = len({grid_hash(g) for g in outputs if g is not None})
    if min_unique_inputs is not None and unique_inputs < min_unique_inputs:
        failures.append(("all", "AssertionError", f"unique inputs {unique_inputs} < {min_unique_inputs}"))
    if min_unique_outputs is not None and generate_puzzle_output is not None and unique_outputs < min_unique_outputs:
        failures.append(("all", "AssertionError", f"unique outputs {unique_outputs} < {min_unique_outputs}"))

    summary: dict[str, Any] = {
        "seeds": seed_list,
        "passed": len(seed_list) - len([f for f in failures if f[0] != "all"]),
        "failed": len(failures),
        "failures": failures[:10],
        "unique_inputs": unique_inputs,
        "unique_outputs": unique_outputs if generate_puzzle_output is not None else None,
        "input_shapes": sorted(set(shapes_in)),
        "output_shapes": sorted(set(shapes_out)),
    }

    if verbose:
        print(
            "arc_check:",
            f"passed={summary['passed']}/{len(seed_list)}",
            f"failures={len(failures)}",
            f"unique_inputs={unique_inputs}",
            f"unique_outputs={summary['unique_outputs']}",
        )
        print("input_shapes:", summary["input_shapes"])
        if generate_puzzle_output is not None:
            print("output_shapes:", summary["output_shapes"])
        if failures:
            print("first_failures:", failures[:5])
        for seed, inp, out in list(zip(seed_list, inputs, outputs))[:examples]:
            print_grid(inp, label=f"input seed={seed}")
            if out is not None:
                print_grid(out, label=f"output seed={seed}")
                if inp.shape == out.shape:
                    print("diff")
                    print(diff_text(inp, out))

    return summary


def run_bulk_validation(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Alias for arc_check, matching common model-written validation names."""

    return arc_check(*args, **kwargs)


def bulk_validate(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Alias for arc_check, matching common model-written validation names."""

    return arc_check(*args, **kwargs)


@contextmanager
def _alarm_timeout(seconds: int | float | None):
    """Best-effort timeout for model-written generation/test code."""

    if seconds is None or seconds <= 0:
        yield
        return

    def _handler(signum, frame):  # noqa: ARG001 - signal handler signature
        raise TimeoutError(f"timed out after {seconds}s")

    try:
        old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, float(seconds))
    except Exception:
        old_handler = None
    try:
        yield
    finally:
        try:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
        except Exception:
            pass


def _fresh_helper_namespace() -> dict[str, Any]:
    """Namespace matching the deliberate preloaded ARC helper environment."""

    names = {
        "np": np,
        "numpy": np,
    }
    for name in __all__:
        if name in globals() and name != "input_artifact_check":
            names[name] = globals()[name]
    names["__name__"] = "__main__"
    return names


def input_artifact_check(
    code_string: str,
    *,
    seeds: Iterable[int] = range(20),
    min_unique_inputs: int = 10,
    require_tests: bool = True,
    require_multiple_colors: bool = True,
    timeout_seconds: int | float = 2,
    examples: int = 2,
    verbose: bool = True,
) -> dict[str, Any]:
    """Validate a final INPUT_CODE artifact in a fresh helper-only namespace.

    Use this immediately before finalizing merged stg2+stg3 outputs. It checks
    the exact code string that will appear in the final Python block, not the
    current notebook state. The fresh namespace intentionally contains the
    standard ARC helper API from this module, but no puzzle-specific helpers
    unless they are included in `code_string`.
    """

    assert isinstance(code_string, str) and code_string.strip(), "code_string must be a non-empty string"
    seed_list = list(seeds)
    assert seed_list, "input_artifact_check requires at least one seed"

    ns = _fresh_helper_namespace()
    failures: list[tuple[Any, str, str]] = []
    grids: list[np.ndarray] = []
    shapes: list[tuple[int, int]] = []
    color_sets: list[tuple[int, ...]] = []

    try:
        with _alarm_timeout(timeout_seconds):
            exec(code_string, ns)
    except Exception as exc:  # noqa: BLE001 - model-facing diagnostic helper
        failures.append(("exec", type(exc).__name__, str(exc)))

    gen = ns.get("generate_puzzle_input")
    if not callable(gen):
        failures.append(("exec", "AssertionError", "generate_puzzle_input is missing or not callable"))

    tests = [
        fn
        for name, fn in sorted(ns.items())
        if name.startswith("test_") and callable(fn)
    ]
    if require_tests and not tests:
        failures.append(("exec", "AssertionError", "no callable test_* functions found"))

    if callable(gen):
        for seed in seed_list:
            try:
                with _alarm_timeout(timeout_seconds):
                    grid = validate_grid(gen(seed), name=f"input(seed={seed})")
                    grid2 = validate_grid(gen(seed), name=f"input2(seed={seed})")
                    assert np.array_equal(grid, grid2), f"generate_puzzle_input is not deterministic for seed {seed}"
                    for test_fn in tests:
                        _call_custom_check(test_fn, seed, grid, None)
                grids.append(grid)
                shapes.append(grid.shape)
                color_sets.append(tuple(sorted(int(x) for x in np.unique(grid))))
            except Exception as exc:  # noqa: BLE001 - model-facing diagnostic helper
                failures.append((seed, type(exc).__name__, str(exc)))

    unique_inputs = len({grid_hash(g) for g in grids})
    if unique_inputs < min_unique_inputs:
        failures.append(("all", "AssertionError", f"unique inputs {unique_inputs} < {min_unique_inputs}"))
    if require_multiple_colors and grids:
        mono = [seed for seed, grid in zip(seed_list, grids) if len(np.unique(grid)) <= 1]
        if mono:
            failures.append(("all", "AssertionError", f"monochrome generated grids for seeds {mono[:5]}"))

    summary: dict[str, Any] = {
        "seeds": seed_list,
        "passed": len(seed_list) - len([f for f in failures if f[0] not in ("exec", "all")]),
        "failed": len(failures),
        "failures": failures[:10],
        "unique_inputs": unique_inputs,
        "input_shapes": sorted(set(shapes)),
        "color_sets": sorted(set(color_sets)),
        "num_tests": len(tests),
        "test_names": [fn.__name__ for fn in tests],
    }

    if verbose:
        print(
            "input_artifact_check:",
            f"passed={summary['passed']}/{len(seed_list)}",
            f"failures={len(failures)}",
            f"unique_inputs={unique_inputs}",
            f"tests={len(tests)}",
        )
        print("input_shapes:", summary["input_shapes"])
        print("color_sets:", summary["color_sets"][:10])
        if failures:
            print("first_failures:", failures[:5])
        for seed, grid in list(zip(seed_list, grids))[:examples]:
            print_grid(grid, label=f"artifact input seed={seed}")

    return summary


def output_artifact_check(
    code_string: str,
    *,
    seeds: Iterable[int] = range(20),
    generate_input: Callable[[int], Any] | None = None,
    require_change: bool = False,
    min_unique_outputs: int | None = None,
    timeout_seconds: int | float = 2,
    examples: int = 2,
    verbose: bool = True,
) -> dict[str, Any]:
    """Validate a final OUTPUT_CODE artifact in a fresh helper-only namespace.

    Use this immediately before finalizing stg4 outputs. It checks the exact
    output-code string that will appear in the final Python block, not the
    current notebook state. The fresh namespace contains the standard ARC helper
    API plus the preloaded `generate_puzzle_input` function used by stg4.
    """

    assert isinstance(code_string, str) and code_string.strip(), "code_string must be a non-empty string"
    seed_list = list(seeds)
    assert seed_list, "output_artifact_check requires at least one seed"

    if generate_input is None:
        generate_input = globals().get("generate_puzzle_input")
    assert callable(generate_input), "generate_puzzle_input is missing or not callable"

    ns = _fresh_helper_namespace()
    ns["generate_puzzle_input"] = generate_input
    failures: list[tuple[Any, str, str]] = []
    inputs: list[np.ndarray] = []
    outputs: list[np.ndarray] = []
    shapes_in: list[tuple[int, int]] = []
    shapes_out: list[tuple[int, int]] = []

    try:
        with _alarm_timeout(timeout_seconds):
            exec(code_string, ns)
    except Exception as exc:  # noqa: BLE001 - model-facing diagnostic helper
        failures.append(("exec", type(exc).__name__, str(exc)))

    gen_out = ns.get("generate_puzzle_output")
    if not callable(gen_out):
        failures.append(("exec", "AssertionError", "generate_puzzle_output is missing or not callable"))

    if callable(gen_out):
        for seed in seed_list:
            try:
                with _alarm_timeout(timeout_seconds):
                    inp = validate_grid(generate_input(seed), name=f"input(seed={seed})")
                    out = validate_grid(gen_out(inp.copy()), name=f"output(seed={seed})")
                    out2 = validate_grid(gen_out(inp.copy()), name=f"output2(seed={seed})")
                    assert np.array_equal(out, out2), f"generate_puzzle_output is not deterministic for seed {seed}"
                    if require_change and inp.shape == out.shape:
                        assert not np.array_equal(inp, out), f"output equals input for seed {seed}"
                inputs.append(inp)
                outputs.append(out)
                shapes_in.append(inp.shape)
                shapes_out.append(out.shape)
            except Exception as exc:  # noqa: BLE001 - model-facing diagnostic helper
                failures.append((seed, type(exc).__name__, str(exc)))

    unique_outputs = len({grid_hash(g) for g in outputs})
    if min_unique_outputs is not None and unique_outputs < min_unique_outputs:
        failures.append(("all", "AssertionError", f"unique outputs {unique_outputs} < {min_unique_outputs}"))

    summary: dict[str, Any] = {
        "seeds": seed_list,
        "passed": len(seed_list) - len([f for f in failures if f[0] not in ("exec", "all")]),
        "failed": len(failures),
        "failures": failures[:10],
        "unique_outputs": unique_outputs,
        "input_shapes": sorted(set(shapes_in)),
        "output_shapes": sorted(set(shapes_out)),
    }

    if verbose:
        print(
            "output_artifact_check:",
            f"passed={summary['passed']}/{len(seed_list)}",
            f"failures={len(failures)}",
            f"unique_outputs={unique_outputs}",
        )
        print("input_shapes:", summary["input_shapes"])
        print("output_shapes:", summary["output_shapes"])
        if failures:
            print("first_failures:", failures[:5])
        for seed, inp, out in list(zip(seed_list, inputs, outputs))[:examples]:
            print_grid(inp, label=f"artifact input seed={seed}")
            print_grid(out, label=f"artifact output seed={seed}")

    return summary


__all__ = [
    "ARC_COLORS",
    "Component",
    "arc_check",
    "as_grid",
    "bbox_from_cells",
    "can_place",
    "bulk_validate",
    "components",
    "connected_components",
    "crop_bbox",
    "diamond_cells",
    "diff_text",
    "draw_diamond",
    "draw_line",
    "draw_rect",
    "find_objects",
    "grid_hash",
    "grid_to_text",
    "grid_to_str",
    "in_bounds",
    "input_artifact_check",
    "line_cells",
    "mask_from_cells",
    "output_artifact_check",
    "place_cells",
    "print_grid_text",
    "print_grid",
    "print_pair",
    "rect_cells",
    "run_bulk_validation",
    "validate_grid",
]
