# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Patch SGLang 0.5.20's sparse Mamba layer IDs for this TP-only HiCache recipe.

Nemotron interleaves cache-bearing layers with feed-forward-only layers. HiCache
must iterate the layer-ID span, not the number of cache-bearing layers: both the
transfer mappers and Mamba's copy-on-write event waits use the original IDs.
The same span places packed MTP KV layers after all target cache layers.
"""

import importlib.util
from pathlib import Path


_OLD = "len(full_layer_mapping | mamba_layer_mapping)"
_NEW = "(max(full_layer_mapping | mamba_layer_mapping, default=-1) + 1)"


def patch_source(source: str) -> str:
    """Update the builder and its reported span, rejecting unexpected source layouts."""
    old_count, new_count = source.count(_OLD), source.count(_NEW)
    if (old_count, new_count) == (0, 2):
        return source
    if (old_count, new_count) != (2, 0):
        raise RuntimeError(
            "Unsupported SGLang HiCache source: expected two unpatched layer-count expressions "
            f"or two patched expressions; found {old_count} unpatched and {new_count} patched. "
            "Review hybrid_pool_assembler.py against this patch before using this container."
        )
    patched = source.replace(_OLD, _NEW)
    compile(patched, "hybrid_pool_assembler.py", "exec")
    return patched


def patch_file(path: Path) -> None:
    """Patch installed source in the worker container and preserve the original file."""
    source = path.read_text()
    patched = patch_source(source)
    if patched == source:
        print(f"HiCache sparse Mamba layer patch already applied: {path}")
        return
    backup = path.with_suffix(".py.gym-original")
    if not backup.exists():
        backup.write_text(source)
    path.write_text(patched)
    print(f"Applied HiCache sparse Mamba layer patch: {path} (original: {backup})")


def main() -> None:
    """Find the active interpreter's SGLang installation without importing its GPU modules."""
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("SGLang is not installed in the worker's Python environment")
    root = Path(next(iter(spec.submodule_search_locations)))
    patch_file(root / "srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py")


if __name__ == "__main__":
    main()
