# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Expose declared checkout metadata inside each native pytest sandbox."""

import ast
import json
import os
import shutil
from pathlib import Path


def restore_symlinks(root: Path, links: dict[str, str]) -> None:
    """Restore Git's exact link text, including directory aliases and dangling links."""
    for name, target in links.items():
        path = root / name
        if path.is_symlink():
            if os.readlink(path) != target:
                raise RuntimeError(f"Native sandbox has a different symlink: {name}")
            continue
        if path.exists():
            raise RuntimeError(f"Native sandbox symlink path is occupied: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(target)


def expose_editable_to_children(root: Path, distribution: Path, installed: Path) -> None:
    """Relocate the actual editable finder into a sandbox-local child Python site."""
    manifest = json.loads((distribution / "editable.json").read_text())
    original = manifest["source_root"]
    tree = ast.parse((distribution / manifest["finder"]).read_text())

    class Relocate(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and (node.value == original or node.value.startswith(original + "/")):
                node.value = str(root) + node.value[len(original) :]
            return node

    relocated = ast.unparse(Relocate().visit(tree))
    if original != str(root) and original in relocated:
        raise RuntimeError("Editable finder retained a path outside the native sandbox")
    child_site = root / ".native-child-site"
    child_site.mkdir()
    (child_site / manifest["finder"]).write_text(relocated + "\n")
    shutil.copy2(distribution / manifest["hook"], child_site / manifest["hook"])
    shutil.copytree(installed, child_site / installed.name)
    (child_site / "sitecustomize.py").write_text(
        "import site\nfrom pathlib import Path\nsite.addsitedir(str(Path(__file__).parent))\n"
    )
    previous = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = str(child_site) + (os.pathsep + previous if previous else "")


def pytest_load_initial_conftests(early_config, parser, args):
    root = Path(__file__).resolve().parents[3]
    metadata = root / "scripts/pants/native_core/git-state"
    restore_symlinks(root, json.loads((metadata.parent / "symlinks.json").read_text()))
    checkout_git = root / ".git"
    if checkout_git.exists():
        raise RuntimeError("Native test sandbox unexpectedly contains Git metadata")
    # Pants materializes files, so Git's required empty refs directory is recreated.
    (metadata / "refs").mkdir(exist_ok=True)
    checkout_git.symlink_to(metadata, target_is_directory=True)
    distributions = list((metadata.parent / "distribution").glob("*.dist-info"))
    if len(distributions) != 1:
        raise RuntimeError("Native sandbox is missing the sampled editable distribution")
    installed = root / distributions[0].name
    shutil.copytree(distributions[0], installed)
    # Relocate the original editable metadata alongside the relocated source.
    (installed / "direct_url.json").write_text(json.dumps({"dir_info": {"editable": True}, "url": root.as_uri()}))
    expose_editable_to_children(root, metadata.parent / "distribution", installed)
