# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compile the explicit step1500 runtime policy without a training-converter gate."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import re
import tokenize
from pathlib import Path

from .coordinate_runtime import AUDIT_PREFIX


STRICT_COORDINATES = "reference_unit"
STEP1500_COORDINATES = "step1500_compat_v1"
COORDINATE_POLICIES = (STRICT_COORDINATES, STEP1500_COORDINATES)
COMPAT_PROTOCOL_ID = "nemotron-v4-step1500-coordinate-compat-v1"
_SPACE = re.compile(r"#\s*coordinate_space\s*:\s*(\S+)\s*$", re.IGNORECASE)


class _Imports(ast.NodeTransformer):
    def visit_Import(self, node: ast.Import) -> list[ast.stmt]:
        result = []
        for alias in node.names:
            if alias.name == "pyautogui":
                replacement = ast.Assign(
                    targets=[ast.Name(id=alias.asname or "pyautogui", ctx=ast.Store())],
                    value=ast.Name(id="__n4_coordinate_adapter", ctx=ast.Load()),
                )
                result.append(ast.copy_location(replacement, node))
            else:
                result.append(ast.copy_location(ast.Import(names=[alias]), node))
        return result

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom | list[ast.stmt]:
        if node.module != "pyautogui":
            return node
        if any(alias.name == "*" for alias in node.names):
            raise ValueError("Use import pyautogui or explicit from-imports so coordinate projection remains defined")
        return [
            ast.copy_location(
                ast.Assign(
                    targets=[ast.Name(id=alias.asname or alias.name, ctx=ast.Store())],
                    value=ast.Attribute(
                        value=ast.Name(id="__n4_coordinate_adapter", ctx=ast.Load()), attr=alias.name, ctx=ast.Load()
                    ),
                ),
                node,
            )
            for alias in node.names
        ]


def compile_compatible(code: str, screen_size: tuple[int, int], parameters: dict[str, list[str]]) -> str:
    tree = ast.parse(code, mode="exec")
    if not tree.body:
        raise ValueError("exec_py requires nonempty executable Python source")
    declarations = []
    for token in tokenize.generate_tokens(io.StringIO(code).readline):
        if token.type == tokenize.COMMENT and (match := _SPACE.fullmatch(token.string)):
            declarations.append(match.group(1).lower())
    if len(set(declarations)) > 1 or any(value not in {"unit", "grid1000", "pixel"} for value in declarations):
        raise ValueError("Use one coordinate_space declaration: unit, grid1000, or pixel")
    space = declarations[0] if declarations else None
    source = Path(__file__).with_name("coordinate_runtime.py").read_text()
    setup = (
        "__n4_coordinate_namespace = {}\n"
        f"exec({source!r}, __n4_coordinate_namespace)\n"
        "import pyautogui as __n4_original_pyautogui\n"
        "__n4_coordinate_adapter = __n4_coordinate_namespace['CoordinateAdapter']("
        f"__n4_original_pyautogui, {parameters!r}, {tuple(screen_size)!r}, {space!r}, "
        f"{hashlib.sha256(code.encode()).hexdigest()!r})\n"
        "pyautogui = __n4_coordinate_adapter\n"
    )
    transformed = _Imports().visit(tree)
    future = [node for node in transformed.body if isinstance(node, ast.ImportFrom) and node.module == "__future__"]
    body = [node for node in transformed.body if node not in future]
    combined = ast.Module(body=[*future, *ast.parse(setup).body, *body], type_ignores=[])
    ast.fix_missing_locations(combined)
    projected = ast.unparse(combined)
    compile(projected, "<nemotron-v4-step1500-compatible>", "exec")
    return projected


def extract_projection_audit(stdout: str, command: str) -> tuple[str, list[dict[str, object]]]:
    """Keep raw controller logs intact; separate verified audit rows from feedback."""
    command_hash = hashlib.sha256(command.encode()).hexdigest()
    kept, events = [], []
    for line in stdout.splitlines(keepends=True):
        if line.startswith(AUDIT_PREFIX):
            try:
                value = json.loads(line[len(AUDIT_PREFIX) :])
                if value.get("model_command_sha256") == command_hash and value.get("event") == "coordinate_projection":
                    events.append(value)
                    continue
            except (ValueError, AttributeError):
                pass
        kept.append(line)
    return "".join(kept), events
