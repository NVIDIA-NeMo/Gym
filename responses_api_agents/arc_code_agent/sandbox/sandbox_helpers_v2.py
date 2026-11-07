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
# Extra sandbox helpers (setting "helpers_v2"), run in the sandbox namespace after the per-row header, so `np`, `components` and
# `crop` from arc_solve_tools / arc_sdg_tools are already defined. They cover what the models rebuilt by hand in most traces
# (from a review of model trajectories): connected components as data, bounding boxes, background and colour
# counts, the 8 rotations/reflections, shape normalisation, subgrid search and enclosed regions. This runs before every code run,
# so it only defines functions (scipy is imported inside `holes`, the one helper that needs it).

_VARIANTS = (("identity", lambda g: g), ("rot90", lambda g: np.rot90(g, 1)), ("rot180", lambda g: np.rot90(g, 2)),
             ("rot270", lambda g: np.rot90(g, 3)), ("flipud", np.flipud), ("fliplr", np.fliplr), ("transpose", lambda g: g.T),
             ("antitranspose", lambda g: np.rot90(g, 2).T))


def bg(g):
    """The most common colour of `g` (ties go to the smaller colour)."""
    values, counts = np.unique(np.asarray(g, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def colors(g):
    """[(colour, count), ...] for every colour in `g`, most common first."""
    values, counts = np.unique(np.asarray(g, dtype=int), return_counts=True)
    order = sorted(zip(counts.tolist(), values.tolist()), key=lambda t: (-t[0], t[1]))
    return [(int(v), int(n)) for n, v in order]


def norm(cells):
    """A set of (row, col) cells moved so its top-left is (0, 0); equal shapes give equal sets. Accepts cells, a bool mask or an object."""
    if hasattr(cells, "cells"):
        cells = cells.cells
    elif isinstance(cells, np.ndarray) and cells.dtype == bool:
        cells = [tuple(map(int, rc)) for rc in np.argwhere(cells)]
    cells = [(int(r), int(c)) for r, c in cells]
    if not cells:
        return frozenset()
    r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


class Obj:
    """One connected component: .color (.colour), .size (.area), .bbox (r0, c0, r1, c1 inclusive), .cells, .mask, .shape, .sub,
    .height, .width, .centroid. Fields can also be read as obj['size']."""

    def __init__(self, comp, grid, background):
        self.color = self.colour = int(comp.color)
        self.cells = [tuple(map(int, rc)) for rc in comp.cells]
        self.size = self.area = int(comp.area)
        self.bbox = tuple(int(v) for v in comp.bbox)
        self.height, self.width = int(comp.height), int(comp.width)
        self.centroid = tuple(float(v) for v in comp.centroid)
        self.mask = np.zeros(grid.shape, dtype=bool)
        for r, c in self.cells:
            self.mask[r, c] = True
        self.shape = norm(self.cells)
        r0, c0, r1, c1 = self.bbox
        self.sub = np.where(self.mask[r0:r1 + 1, c0:c1 + 1], grid[r0:r1 + 1, c0:c1 + 1], background)

    def __getitem__(self, key):
        return getattr(self, key)

    def __repr__(self):
        return f"Obj(color={self.color}, size={self.size}, bbox={self.bbox})"


def objects(g, bg="auto", connectivity=4, by_color=True, limit=20, quiet=False):
    """Connected components of `g` as a list of Obj, largest first, printing a short summary unless quiet=True. bg='auto' uses the
    most common colour as background; bg=None keeps every colour."""
    grid = np.asarray(g, dtype=int)
    background = globals()["bg"](grid) if bg == "auto" else bg
    comps = components(grid, bg=background, connectivity=connectivity, by_color=by_color)
    objs = sorted((Obj(c, grid, 0 if background is None else background) for c in comps), key=lambda o: (-o.size, o.bbox))
    if not quiet:
        print(f"{len(objs)} objects (bg={background}, conn={connectivity}); id color size bbox")
        for index, o in enumerate(objs[:limit]):
            print(f"  {index} c{o.color} n{o.size} {o.bbox}")
        if len(objs) > limit:
            print(f"  ... {len(objs) - limit} more")
    return objs


def bbox(x, color=None):
    """(r0, c0, r1, c1) inclusive of a bool mask, of the cells of one colour, or (no colour) of every non-background cell; None if empty."""
    a = np.asarray(x)
    if a.dtype == bool:
        mask = a
    elif color is not None:
        mask = a == color
    else:
        mask = a != globals()["bg"](a)
    hits = np.argwhere(mask)
    if hits.size == 0:
        return None
    (r0, c0), (r1, c1) = hits.min(0), hits.max(0)
    return int(r0), int(c0), int(r1), int(c1)


def variants(g):
    """The 8 rotations and reflections of `g` as [(name, grid), ...]."""
    grid = np.asarray(g)
    return [(name, f(grid)) for name, f in _VARIANTS]


def find(g, pattern, all_variants=False):
    """Every (row, col) where `pattern` occurs exactly in `g`; with all_variants=True, (row, col, variant name) over the 8 variants."""
    grid = np.asarray(g, dtype=int)
    hits = []
    for name, p in (variants(pattern) if all_variants else [("identity", np.asarray(pattern, dtype=int))]):
        ph, pw = p.shape
        for r in range(grid.shape[0] - ph + 1):
            for c in range(grid.shape[1] - pw + 1):
                if np.array_equal(grid[r:r + ph, c:c + pw], p):
                    hits.append((r, c, name) if all_variants else (r, c))
    return hits


def holes(g, color):
    """Bool mask of the cells enclosed by `color` (not reachable from the border without crossing `color`), excluding `color` itself."""
    import scipy.ndimage as _ndi
    wall = np.asarray(g, dtype=int) == color
    return _ndi.binary_fill_holes(wall) & ~wall
