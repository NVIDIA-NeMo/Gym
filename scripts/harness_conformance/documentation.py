# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free helpers shared by local table generation and Fern publication."""

import os
import tempfile
from pathlib import Path


PAGE = Path("fern/versions/latest/pages/reference/trajectory-capabilities.mdx")
BEGIN = "{/* BEGIN GENERATED TE TABLE */}"
END = "{/* END GENERATED TE TABLE */}"


def replace_block(original: str, block: str) -> str:
    """Replace exactly one generated block without changing surrounding reference text."""
    for text in (original, block):
        if text.count(BEGIN) != 1 or text.count(END) != 1 or text.index(BEGIN) >= text.index(END):
            raise ValueError("reference page needs exactly one ordered pair of generated-table markers")
    if not block.strip().startswith(BEGIN) or not block.strip().endswith(END):
        raise ValueError("documentation artifact must contain only the generated block")
    return original[: original.index(BEGIN)] + block.strip() + original[original.index(END) + len(END) :]


def atomic_write(path: Path, content: str) -> None:
    """Replace an output only after its full contents are available."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(content)
            handle.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
