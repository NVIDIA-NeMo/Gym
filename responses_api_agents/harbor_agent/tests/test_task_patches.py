# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the opt-in Terminal-Bench 2.1 reference-solution patches."""

import importlib.util
from pathlib import Path

import pytest


_PREPARE_PATH = Path(__file__).parent.parent / "prepare_terminal_bench_2_1.py"


def _load_prepare_module():
    spec = importlib.util.spec_from_file_location("tb21_prepare", _PREPARE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prep = _load_prepare_module()


def _write_patch(patches_dir: Path, name: str, target: str, old: str, new: str) -> None:
    patches_dir.mkdir(parents=True, exist_ok=True)
    (patches_dir / name).write_text(
        f"--- a/{target}\n+++ b/{target}\n@@ -1 +1 @@\n-{old}\n+{new}\n",
    )


def _make_checkout(tmp_path: Path, target: str, content: str) -> Path:
    tasks_dir = tmp_path / "checkout" / "tasks"
    dest = tmp_path / "checkout" / target
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content + "\n")
    tasks_dir.mkdir(parents=True, exist_ok=True)
    return tasks_dir


class TestApplyTaskPatches:
    def test_applies_solution_patch(self, tmp_path, monkeypatch):
        target = "tasks/demo/solution/solve.sh"
        tasks_dir = _make_checkout(tmp_path, target, "pip install foo")
        patches = tmp_path / "patches"
        _write_patch(patches, "demo.patch", target, "pip install foo", "pip install foo 'bar==1.0'")
        monkeypatch.setattr(prep, "TASK_PATCHES_DIR", patches)

        assert prep.apply_task_patches(tasks_dir) == ["demo.patch"]
        assert "bar==1.0" in (tasks_dir.parent / target).read_text()

    @pytest.mark.parametrize("target", ["tasks/demo/tests/test_outputs.py", "tasks/demo/task.toml"])
    def test_refuses_to_patch_graded_files(self, tmp_path, monkeypatch, target):
        # Patching tests/ or task.toml would change what is asked or how it is
        # scored, so it must be rejected before git apply ever runs.
        tasks_dir = _make_checkout(tmp_path, target, "original")
        patches = tmp_path / "patches"
        _write_patch(patches, "bad.patch", target, "original", "tampered")
        monkeypatch.setattr(prep, "TASK_PATCHES_DIR", patches)

        with pytest.raises(RuntimeError, match="may only touch"):
            prep.apply_task_patches(tasks_dir)
        assert (tasks_dir.parent / target).read_text().strip() == "original"

    def test_raises_when_patch_does_not_apply(self, tmp_path, monkeypatch):
        # The checkout is pinned, so a non-applying patch means the pin moved.
        # It must fail loudly rather than be silently skipped.
        target = "tasks/demo/solution/solve.sh"
        tasks_dir = _make_checkout(tmp_path, target, "something else entirely")
        patches = tmp_path / "patches"
        _write_patch(patches, "demo.patch", target, "pip install foo", "pip install bar")
        monkeypatch.setattr(prep, "TASK_PATCHES_DIR", patches)

        with pytest.raises(RuntimeError, match="Failed to apply"):
            prep.apply_task_patches(tasks_dir)

    def test_no_patches_dir_is_a_noop(self, tmp_path, monkeypatch):
        tasks_dir = _make_checkout(tmp_path, "tasks/demo/solution/solve.sh", "noop")
        monkeypatch.setattr(prep, "TASK_PATCHES_DIR", tmp_path / "does-not-exist")
        assert prep.apply_task_patches(tasks_dir) == []


class TestBundledPatches:
    def test_bundled_patches_only_target_solutions(self):
        # Guards the shipped patch set itself, not just the enforcement code.
        patches_dir = _PREPARE_PATH.parent / "task_patches"
        patch_files = sorted(patches_dir.glob("*.patch"))
        assert patch_files, "expected bundled task patches"
        for patch_path in patch_files:
            targets = prep._patch_targets(patch_path.read_text())
            assert targets, f"{patch_path.name} declares no target"
            for target in targets:
                assert target.endswith("/solution/solve.sh"), f"{patch_path.name} targets {target}"
