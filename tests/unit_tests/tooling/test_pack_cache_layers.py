# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest
import yaml
from scripts.ci.pack_cache_layers import pack


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github/workflows/full-test-suite.yml"
TOP = "opt/nemo-gym/cache"


def build_cache(root: Path) -> None:
    cache = root / TOP
    for name, size in (("uv/archive-v0/a/torch.so", 700), ("uv/archive-v0/b/numpy.so", 300), ("xdg/npm/x", 200)):
        (cache / name).parent.mkdir(parents=True, exist_ok=True)
        (cache / name).write_bytes(os.urandom(size))
    (cache / "uv/archive-v0/c").mkdir()
    os.link(cache / "uv/archive-v0/a/torch.so", cache / "uv/archive-v0/c/torch.so")
    (cache / "uv/wheels-v5/pypi/torch").mkdir(parents=True)
    (cache / "uv/wheels-v5/pypi/torch/2.9").symlink_to("../../../archive-v0/a")
    (cache / "uv/empty").mkdir()


def snapshot(root: Path) -> dict[str, object]:
    entries = {}
    for path in sorted((root / TOP).rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            entries[relative] = ("link", os.readlink(path))
        elif path.is_dir():
            entries[relative] = ("dir",)
        else:
            entries[relative] = ("file", path.read_bytes())
    return entries


def test_layers_hold_each_member_once_within_budget_with_directories_first(tmp_path: Path) -> None:
    build_cache(tmp_path)
    layers = pack(tmp_path, TOP, 800)
    tree = [TOP, *(path.relative_to(tmp_path).as_posix() for path in (tmp_path / TOP).rglob("*"))]
    directories = [path for path in tree if (tmp_path / path).is_dir() and not (tmp_path / path).is_symlink()]
    assert sorted(path for layer in layers for path in layer) == sorted(tree)
    assert layers[0][0] == TOP
    assert set(layers[0]) - set(directories) == {f"{TOP}/uv/archive-v0/a/torch.so", f"{TOP}/uv/archive-v0/c/torch.so"}
    assert layers[1:] == [
        [f"{TOP}/uv/archive-v0/b/numpy.so", f"{TOP}/uv/wheels-v5/pypi/torch/2.9", f"{TOP}/xdg/npm/x"]
    ]
    assert pack(tmp_path, TOP, 800) == layers


@pytest.mark.skipif(shutil.which("tar") is None, reason="GNU tar is required")
def test_layers_applied_in_order_reproduce_the_tree_owned_by_the_runtime_user(tmp_path: Path) -> None:
    source, target = tmp_path / "source", tmp_path / "target"
    build_cache(source)
    target.mkdir()
    for index, layer in enumerate(pack(source, TOP, 800)):
        members = tmp_path / f"layer-{index}.list"
        members.write_text("".join(f"{path}\0" for path in layer))
        archive = tmp_path / f"layer-{index}.tar.gz"
        subprocess.run(
            ["tar", "-C", source, "--numeric-owner", "--owner=65532", "--group=65532", "--no-recursion", "--null"]
            + ["-T", members, "-czf", archive],
            check=True,
        )
        with tarfile.open(archive) as layer_tar:
            assert {(member.uid, member.gid) for member in layer_tar.getmembers()} == {(65532, 65532)}
        subprocess.run(["tar", "-C", target, "-xzf", archive], check=True)
    assert snapshot(target) == snapshot(source)
    a, c = target / TOP / "uv/archive-v0/a/torch.so", target / TOP / "uv/archive-v0/c/torch.so"
    assert a.stat().st_ino == c.stat().st_ino


def test_offline_image_appends_every_packed_layer() -> None:
    job = yaml.safe_load(WORKFLOW.read_text())["jobs"]["offline-image"]
    scripts = "\n".join(step.get("run", "") for step in job["steps"])
    assert any(step.get("uses", "").startswith("actions/checkout@") for step in job["steps"])
    assert "scripts/ci/pack_cache_layers.py" in scripts
    assert "cache.tar.gz" not in scripts
