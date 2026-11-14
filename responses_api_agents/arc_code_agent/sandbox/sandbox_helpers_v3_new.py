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
# New sandbox helpers (setting "helpers_v3_new"), run after sandbox_helpers_v2.py (and sandbox_helpers_v3_fix.py when that is on).
# From a review of model helper traces, the utilities the model still wrote by hand most often: components of a
# mask (multi-colour or 8-connected objects; ndimage.label in about 27-34% of attempts), holes and containment, panels split by
# separator lines, straight runs and rays, a rotation/reflection-invariant shape key, and placement offsets. Only defines functions.
import collections as _collections

_Obj_cls, _components_new, _holes_v2, _variants_v2, _norm_new = Obj, components, holes, variants, norm
_bg_new = bg


def _cells_of(x):
    """(row, col) cells of an Obj, a bool mask or an iterable of cells."""
    if hasattr(x, "cells"):
        return [tuple(c) for c in x.cells]
    a = np.asarray(x) if not isinstance(x, (set, frozenset)) else None
    if a is not None and a.dtype == bool and a.ndim == 2:
        return [tuple(map(int, rc)) for rc in np.argwhere(a)]
    return [(int(r), int(c)) for r, c in x]


def regions(mask, connectivity=4, grid=None):
    """Connected regions of a bool mask (for example `g != bg(g)`, which joins touching cells of different colours) as Obj, largest
    first, without printing. With `grid`, `.color` is the most common colour under each region and `.colors` the set of its colours."""
    m = np.asarray(mask).astype(bool)
    base = m.astype(int)
    g = np.asarray(grid, dtype=int) if grid is not None else base
    out = []
    for comp in _components_new(base, bg=0, connectivity=connectivity, by_color=True):
        o = _Obj_cls(comp, g, 0)
        under = _collections.Counter(g[o.mask].tolist())
        o.colors = set(under)
        o.color = o.colour = int(under.most_common(1)[0][0]) if grid is not None else 1
        out.append(o)
    out.sort(key=lambda o: (-o.size, tuple(o.bbox)))
    for index, o in enumerate(out):
        o.id = index
    return out


def holes(x, color=None):
    """holes(g, color): bool mask of the cells enclosed by `color` (as before). holes(obj_or_mask): the enclosed regions inside it,
    one Obj per hole (largest first)."""
    if color is not None:
        return _holes_v2(x, color)
    import scipy.ndimage as _ndi
    m = x.mask if hasattr(x, "mask") else np.asarray(x).astype(bool)
    inner = _ndi.binary_fill_holes(m) & ~m
    return regions(inner, 4)


def _n_holes(self):
    return len(holes(self))


def _touches_border(self):
    m = self.mask
    return bool(m[0, :].any() or m[-1, :].any() or m[:, 0].any() or m[:, -1].any())


_Obj_cls.n_holes = property(_n_holes)
_Obj_cls.touches_border = property(_touches_border)


def inside(a, b):
    """True when every cell of `a` lies in an enclosed hole of `b` (both Obj or bool masks)."""
    import scipy.ndimage as _ndi
    am = a.mask if hasattr(a, "mask") else np.asarray(a).astype(bool)
    bm = b.mask if hasattr(b, "mask") else np.asarray(b).astype(bool)
    hole = _ndi.binary_fill_holes(bm) & ~bm
    return bool(am.any() and hole[am].all())


def panels(g, sep=None):
    """Split `g` on full rows and columns of one colour (the separator; by default the colour with the most such lines, preferring
    a non-background colour). Returns {'sep', 'rows': [(r0, r1)], 'cols': [(c0, c1)], 'grid': [[subgrid, ...], ...]} with inclusive
    ranges and subgrids in row-major order, or None when there are no separator lines. Prints one summary line."""
    grid = np.asarray(g, dtype=int)
    h, w = grid.shape
    if sep is None:
        counts = {}
        for color in np.unique(grid).tolist():
            n = sum(bool((grid[r] == color).all()) for r in range(h)) + sum(bool((grid[:, c] == color).all()) for c in range(w))
            if n:
                counts[color] = n
        if not counts:
            print("panels: no separator lines")
            return None
        b = _bg_new(grid)
        non_bg = {c: n for c, n in counts.items() if c != b}
        sep = max(non_bg or counts, key=lambda c: (counts[c], -c))
    sep_rows = [r for r in range(h) if (grid[r] == sep).all()]
    sep_cols = [c for c in range(w) if (grid[:, c] == sep).all()]

    def spans(n, cut):
        out, start = [], None
        for i in range(n + 1):
            if i < n and i not in cut:
                start = i if start is None else start
            elif start is not None:
                out.append((start, i - 1))
                start = None
        return out

    rows, cols = spans(h, set(sep_rows)), spans(w, set(sep_cols))
    sub = [[grid[r0:r1 + 1, c0:c1 + 1] for c0, c1 in cols] for r0, r1 in rows]
    print(f"panels: sep {sep}, {len(rows)}x{len(cols)}, rows {rows}, cols {cols}")
    return {"sep": sep, "rows": rows, "cols": cols, "grid": sub}


class _Runs(list):
    def __repr__(self):
        head = ", ".join(f"c{c} {a}->{b} n{n}" for c, a, b, _, n in self[:6])
        return f"<{len(self)} runs: {head}{', ...' if len(self) > 6 else ''}>"


def runs(g, color=None, dirs=8, min_len=2):
    """Maximal straight same-colour segments (horizontal and vertical, plus both diagonals when dirs=8) of `color` or of every
    non-background colour, as (colour, (r0, c0), (r1, c1), (dr, dc), length) with length >= min_len, longest first."""
    grid = np.asarray(g, dtype=int)
    h, w = grid.shape
    b = _bg_new(grid)
    steps = [(0, 1), (1, 0), (1, 1), (1, -1)][: 4 if dirs == 8 else 2]
    out = []
    for dr, dc in steps:
        for r in range(h):
            for c in range(w):
                v = int(grid[r, c])
                if (color is not None and v != color) or (color is None and v == b):
                    continue
                pr, pc = r - dr, c - dc
                if 0 <= pr < h and 0 <= pc < w and grid[pr, pc] == v:
                    continue  # not the start of a run
                n = 1
                while 0 <= r + n * dr < h and 0 <= c + n * dc < w and grid[r + n * dr, c + n * dc] == v:
                    n += 1
                if n >= min_len:
                    out.append((v, (r, c), (r + (n - 1) * dr, c + (n - 1) * dc), (dr, dc), n))
    return _Runs(sorted(out, key=lambda t: -t[4]))


def ray(g, r, c, dr, dc, stop=None):
    """Cells from (r+dr, c+dc) onwards in direction (dr, dc) until the grid edge or a blocking cell: by default any non-background
    cell, or with `stop` any cell whose colour is in `stop` (an int or a collection)."""
    grid = np.asarray(g, dtype=int)
    b = _bg_new(grid)
    block = None if stop is None else ({stop} if isinstance(stop, (int, np.integer)) else set(stop))
    out = []
    r, c = r + dr, c + dc
    while 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
        v = int(grid[r, c])
        if (block is None and v != b) or (block is not None and v in block):
            break
        out.append((r, c))
        r, c = r + dr, c + dc
    return out


_CELL_MAPS = (("identity", lambda r, c: (r, c)), ("rot90", lambda r, c: (-c, r)), ("rot180", lambda r, c: (-r, -c)),
              ("rot270", lambda r, c: (c, -r)), ("flipud", lambda r, c: (-r, c)), ("fliplr", lambda r, c: (r, -c)),
              ("transpose", lambda r, c: (c, r)), ("antitranspose", lambda r, c: (-c, -r)))


def _cell_variants(cells):
    return [(name, _norm_new([f(r, c) for r, c in cells])) for name, f in _CELL_MAPS]


def variants(x):
    """The 8 rotations and reflections: of a grid as [(name, grid), ...], or of a set of cells (or an Obj) as [(name, shape), ...]."""
    if hasattr(x, "cells") or isinstance(x, (set, frozenset)) or (isinstance(x, list) and x and isinstance(x[0], tuple)):
        return _cell_variants(_cells_of(x))
    return _variants_v2(x)


def canon(x):
    """A key for a shape (cells, bool mask or Obj) that is the same under every rotation and reflection: the smallest sorted tuple
    of normalised cells over the 8 variants."""
    return min(tuple(sorted(s)) for _, s in _cell_variants(_cells_of(x)))


def fits(shape, region, used=None, variants=False):
    """Offsets (r, c) at which `shape` (cells, mask or Obj; moved to the origin) lies entirely inside `region` (a bool mask or Obj)
    without touching `used` (a bool mask); with variants=True, (r, c, variant name) over the distinct rotations and reflections."""
    reg = region.mask if hasattr(region, "mask") else np.asarray(region).astype(bool)
    free = reg & ~(np.asarray(used).astype(bool) if used is not None else np.zeros_like(reg))
    base = sorted(_norm_new(_cells_of(shape)))
    options = [("identity", base)]
    if variants:
        seen, options = set(), []
        for name, s in _cell_variants(base):
            key = tuple(sorted(s))
            if key not in seen:
                seen.add(key)
                options.append((name, list(key)))
    h, w = free.shape
    out = []
    for name, cells in options:
        ph, pw = max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1
        for r in range(h - ph + 1):
            for c in range(w - pw + 1):
                if all(free[r + dr, c + dc] for dr, dc in cells):
                    out.append((r, c, name) if variants else (r, c))
    return out
