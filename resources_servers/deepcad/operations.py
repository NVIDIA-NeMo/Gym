# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Construction-operation inventory, execution evidence, and bounded reward shaping.

No CadQuery import occurs here. Static candidates are inventory hints, not proof of
execution. Only the sandbox's runtime trace is used to reward model operations.
"""

import ast
import hashlib
import math
import sys
from collections import Counter
from types import FrameType, ModuleType


# Modeling operations, not selectors, imports, exporters, or workplane navigation.
OPERATION_NAMES = frozenset(
    "box sphere cylinder cone torus wedge extrude twistExtrude revolve sweep loft "
    "fillet chamfer shell union cut intersect cutBlind cutThruAll hole cboreHole cskHole "
    "rarray polarArray mirror mirrorX mirrorY offset2D interpPlate "
    "parametricCurve parametricSurface slot2D text "
    "spline splineApprox bezier ellipse ellipseArc threePointArc radiusArc tangentArcPoint "
    "makeHelix makeSpline makeBezier makeTorus makeCone makeSphere makeCylinder "
    "makeLoft makePipeShell makeShell makeSolid makeRuledSurface".split()
)


def extract_operation_candidates(code: str) -> dict[str, int]:
    """Count syntactic call sites; callers must retain the static-evidence label.

    Comments and string literals do not count. Imported/bound method aliases are
    recognized. Dynamic getattr and helper-library internals require runtime tracing.
    """
    tree = ast.parse(code)
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in OPERATION_NAMES:
                    aliases[alias.asname or alias.name] = alias.name
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if isinstance(value, ast.Attribute) and value.attr in OPERATION_NAMES:
                for target in targets:
                    if isinstance(target, ast.Name):
                        aliases[target.id] = value.attr
    counts = Counter()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = (
            node.func.attr
            if isinstance(node.func, ast.Attribute)
            else aliases.get(node.func.id if isinstance(node.func, ast.Name) else "")
        )
        if name in OPERATION_NAMES:
            counts[name] += 1
    return dict(sorted(counts.items()))


class OperationTrace:
    """Record successful CadQuery calls made directly by generated Python code.

    The profile hook identifies actual CadQuery function code objects, preventing
    unrelated methods named 'fillet' from counting. It excludes internal helper
    calls and CAD exports. Repeated calls collapse to one rewarded operation.
    """

    def __init__(self, cadquery: ModuleType, filename: str = "model.py") -> None:
        self.filename = filename
        self.executed: set[str] = set()
        self._functions = {}
        self._previous = None
        for class_name in ("Workplane", "Sketch", "Shape", "Solid", "Face", "Wire", "Edge"):
            owner = getattr(cadquery, class_name, None)
            if owner is None:
                continue
            for name in OPERATION_NAMES:
                method = getattr(owner, name, None)
                function = getattr(method, "__func__", method)
                code = getattr(function, "__code__", None)
                if code is not None:
                    self._functions[code] = name

    def _profile(self, frame: FrameType, event: str, result: object) -> None:
        if event != "return" or result is None:
            return
        name = self._functions.get(frame.f_code)
        caller = frame.f_back
        if name and caller is not None and caller.f_code.co_filename == self.filename:
            self.executed.add(name)

    def __enter__(self) -> "OperationTrace":
        self._previous = sys.getprofile()
        sys.setprofile(self._profile)
        return self

    def __exit__(self, *_args: object) -> None:
        sys.setprofile(self._previous)


def operation_bonus(
    base_reward: float,
    reference: list[str],
    executed: list[str],
    *,
    max_fraction: float = 0.10,
) -> tuple[float, list[str]]:
    """Reward coverage of this reference's distinct supported operations."""
    if not all(math.isfinite(x) for x in (base_reward, max_fraction)):
        raise ValueError("Operation bonus requires finite reward and max_fraction")
    if not 0 <= max_fraction <= 0.2:
        raise ValueError("Require 0 <= max_fraction <= 0.2")
    reference_operations = set(reference) & OPERATION_NAMES
    matched = sorted(reference_operations & set(executed))
    coverage = len(matched) / len(reference_operations) if reference_operations else 0.0
    return max(0.0, base_reward) * max_fraction * coverage, matched


def operation_hint(task_id: str, operations: list[str], *, probability: float, seed: int = 42) -> str:
    """Stable Bernoulli choice per task, independent of process order and GRPO K."""
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Hint probability must be in [0, 1]")
    names = sorted(set(operations) & OPERATION_NAMES)
    value = int.from_bytes(hashlib.sha256(f"operation-hint-v1:{seed}:{task_id}".encode()).digest()[:8], "big")
    if not names or value / 2**64 >= probability:
        return ""
    return "Try to use " + "/".join(names) + " operations."


def add_operation_hint(row: dict, operations: list[str], *, probability: float, seed: int = 42) -> dict:
    """Attach target operations to verifier metadata and optionally hint in system text.

    Called during prompt preparation, before GRPO fan-out. The caller owns row.
    """
    metadata = row["verifier_metadata"]
    names = sorted(set(operations) & OPERATION_NAMES)
    metadata["reference_operations"] = names
    hint = operation_hint(metadata["task_id"], names, probability=probability, seed=seed)
    metadata["operation_hint_applied"] = bool(hint)
    metadata["operation_hint_probability"] = probability
    if not hint:
        return row
    messages = row["responses_create_params"]["input"]
    system = next((message for message in messages if message.get("role") == "system"), None)
    if system is None:
        system = {"role": "system", "content": ""}
        messages.insert(0, system)
    content = system["content"]
    if isinstance(content, str):
        system["content"] = content.rstrip() + "\n\n" + hint
    elif isinstance(content, list):
        content.append({"type": "input_text", "text": hint})
    else:
        raise ValueError("Unsupported system-message content for operation hint")
    return row
