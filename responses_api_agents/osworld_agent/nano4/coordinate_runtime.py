# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone VM-side coordinate adapter; this file uses only Python's stdlib."""

from __future__ import annotations

import functools
import json
import math
from collections.abc import Sequence
from types import ModuleType


AUDIT_PREFIX = "__N4_COORDINATE_AUDIT_V1__"


def resolve_pair(
    values: Sequence[int | float | None], screen_size: tuple[int, int], *, declared_space: str | None = None
) -> tuple[list[int | None], dict[str, object]]:
    """Prefer unit coordinates; the step1500 profile's unlabelled fallback is grid1000.

    Grid/pixel overlap cannot be inferred from the numbers. The profile makes
    that choice explicit, and records ambiguity rather than claiming certainty.
    An explicit coordinate_space comment overrides that prior for the block.
    """
    present = [value for value in values if value is not None]
    if not present or any(type(value) not in (int, float) or not math.isfinite(value) for value in present):
        raise ValueError("Pointer coordinates must be finite numbers; omit x/y to use the current cursor")
    unit = all(0 <= value <= 1 for value in present)
    grid = all(0 <= value <= 1000 for value in present)
    pixel = all(value is None or 0 <= value <= size for value, size in zip(values, screen_size, strict=True))
    if declared_space:
        space = declared_space
        reason = "explicit_coordinate_space_comment"
    elif unit:
        space = "unit"
        reason = "unit_interval"
    elif grid:
        space = "grid1000"
        reason = "step1500_grid1000_fallback"
    elif pixel:
        space = "pixel"
        reason = "outside_grid_inside_screenshot"
    else:
        raise ValueError(
            f"Coordinates {values!r} exceed unit/grid1000 and screenshot {screen_size!r}; "
            "use [0,1] or declare '# coordinate_space: pixel' with valid screenshot coordinates"
        )
    if not {"unit": unit, "grid1000": grid, "pixel": pixel}.get(space, False):
        raise ValueError(f"Coordinates {values!r} are outside declared {space} bounds for {screen_size!r}")
    projected = []
    for value, size in zip(values, screen_size, strict=True):
        if value is None:
            projected.append(None)
        elif space == "unit":
            projected.append(round(value * (size - 1)))
        elif space == "grid1000":
            projected.append(round(value / 1000 * (size - 1)))
        else:
            # Preserve the reference's exact-width/height endpoint sentinel.
            projected.append(min(size - 1, round(value)))
    return projected, {
        "source_space": space,
        "selection_reason": reason,
        "grid_pixel_ambiguous": not unit and grid and pixel,
    }


class CoordinateAdapter:
    """Project at the real call boundary, so variables and Python expressions work."""

    def __init__(
        self,
        module: ModuleType,
        parameters: dict[str, list[str]],
        screen_size: tuple[int, int],
        declared_space: str | None,
        command_sha256: str,
    ) -> None:
        object.__setattr__(self, "_module", module)
        object.__setattr__(self, "_parameters", parameters)
        object.__setattr__(self, "_screen_size", screen_size)
        object.__setattr__(self, "_declared_space", declared_space)
        object.__setattr__(self, "_command_sha256", command_sha256)

    def __setattr__(self, name: str, value: object) -> None:
        setattr(self._module, name, value)

    def __getattr__(self, name: str) -> object:
        original = getattr(self._module, name)
        parameters = self._parameters.get(name)
        if not parameters:
            return original

        @functools.wraps(original)
        def call(*args: object, **kwargs: object) -> object:
            positional = list(args)
            keywords = dict(kwargs)
            positions = [parameters.index(axis) for axis in ("x", "y")]
            values = []
            for axis, position in zip(("x", "y"), positions, strict=True):
                if len(positional) > position and axis in keywords:
                    raise TypeError(f"{name} received {axis} both positionally and by keyword")
                values.append(positional[position] if len(positional) > position else keywords.get(axis))
            tuple_form = isinstance(values[0], (tuple, list)) and len(values[0]) == 2 and values[1] is None
            if tuple_form:
                values = list(values[0])
            if all(value is None for value in values):
                return original(*args, **kwargs)
            projected, decision = resolve_pair(values, self._screen_size, declared_space=self._declared_space)
            if tuple_form:
                if len(positional) > positions[0]:
                    positional[positions[0]] = tuple(projected)
                else:
                    keywords["x"] = tuple(projected)
            else:
                for axis, position, value in zip(("x", "y"), positions, projected, strict=True):
                    if len(positional) > position:
                        positional[position] = value
                    elif axis in keywords:
                        keywords[axis] = value
            event = {
                "event": "coordinate_projection",
                "model_command_sha256": self._command_sha256,
                "function": name,
                "input_xy": values,
                "projected_xy": projected,
                "screen_size": list(self._screen_size),
                "scope": "Projection before device call; does not prove the action or task succeeded",
                **decision,
            }
            print(AUDIT_PREFIX + json.dumps(event, separators=(",", ":")), flush=True)
            return original(*positional, **keywords)

        return call
