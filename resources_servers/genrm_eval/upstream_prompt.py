# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fetch a pinned judging prompt without importing or executing upstream Python."""

import ast
from pathlib import Path
from urllib.request import urlopen


def download_prompt(*, repo: str, revision: str, filename: str, name: str, cache_dir: Path) -> str:
    """Extract a literal string assignment or a function's literal return value."""
    cached = cache_dir / revision / filename
    url = f"https://raw.githubusercontent.com/{repo}/{revision}/{filename}"
    if cached.is_file():
        source = cached.read_text()
    else:
        with urlopen(url, timeout=60) as response:
            source = response.read().decode("utf-8")
    for node in ast.parse(source).body:
        value = None
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            value = node.value
        elif isinstance(node, ast.FunctionDef) and node.name == name:
            value = next((statement.value for statement in node.body if isinstance(statement, ast.Return)), None)
        if value is not None:
            prompt = ast.literal_eval(value)
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Expected a nonempty prompt {name!r} in {url}")
            # Cache only successfully parsed sources; upstream code is never executed.
            if not cached.is_file():
                cached.parent.mkdir(parents=True, exist_ok=True)
                cached.write_text(source)
            return prompt
    raise ValueError(f"Could not find literal prompt {name!r} in {url}")
