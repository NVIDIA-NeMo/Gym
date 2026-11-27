# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static construction-operation inventory and bounded reward shaping.

No generated code executes here. Candidates describe source syntax, not executed
CadQuery calls or their contribution to the final geometry.
"""

import ast
import hashlib
import math
from collections import Counter


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
    recognized. Dynamic getattr and helper-library internals are not recognized.
    Dead code and unrelated methods with supported names can count.
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


def operation_bonus(
    base_reward: float,
    reference: list[str],
    candidates: list[str],
    *,
    max_fraction: float = 0.10,
) -> tuple[float, list[str]]:
    """Reward coverage of this reference's distinct supported operations."""
    if not all(math.isfinite(x) for x in (base_reward, max_fraction)):
        raise ValueError("Operation bonus requires finite reward and max_fraction")
    if not 0 <= max_fraction <= 0.2:
        raise ValueError("Require 0 <= max_fraction <= 0.2")
    reference_operations = set(reference) & OPERATION_NAMES
    matched = sorted(reference_operations & set(candidates))
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
