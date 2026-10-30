# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from benchmarks.gdpval.hsg.aav2.source_copy import copy_tree, inventory, validate_tree


def test_visible_file_and_directory_links_copy_without_target_siblings(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "final.txt").write_text("Final deliverable")
    (assets / "excluded.txt").write_text("Excluded intermediate")
    (assets / "inputs").mkdir()
    (assets / "inputs/reference.txt").write_text("Reference input")
    source = tmp_path / "mirror"
    task = source / "task_one/repeat_0"
    task.mkdir(parents=True)
    (task / "final.txt").symlink_to(assets / "final.txt")
    (task / "reference_files").symlink_to(assets / "inputs", target_is_directory=True)
    (source / "FILTER_MANIFEST.txt").write_text("Selected final and inputs")
    output = tmp_path / "copy"
    manifest = copy_tree(source, output, {"task_one"})
    assert (output / "task_one/repeat_0/final.txt").read_text() == "Final deliverable"
    assert (output / "task_one/repeat_0/reference_files/reference.txt").read_text() == "Reference input"
    assert not (output / "task_one/repeat_0/excluded.txt").exists()
    assert not any(path.is_symlink() for path in output.rglob("*"))
    assert (output / "FILTER_MANIFEST.txt").read_text() == "Selected final and inputs"
    assert {row["resolved_target"] for row in manifest["links"]} == {str(assets / "final.txt"), str(assets / "inputs")}
    assert all(row["source_sha256"] == row["output_sha256"] for row in manifest["files"])
    assert inventory(source, {"task_one"}) == manifest
    validate_tree(source, output, manifest)


def test_old_judge_caches_and_unselected_tasks_are_not_copied(tmp_path):
    source = tmp_path / "source"
    (source / "task_one/cache").mkdir(parents=True)
    (source / "task_one/repeat_0").mkdir()
    (source / "task_one/repeat_0/finish_params.json").write_text("null")
    (source / "task_one/repeat_0_verify_response_0123456789ab.json").write_text("old judge")
    (source / "task_one/cache/repeat_0_verify_response_0123456789abcdef.json").write_text("old judge")
    (source / "task_other").mkdir()
    (source / "task_other/excluded.txt").write_text("Not selected")
    output = tmp_path / "copy"
    manifest = copy_tree(source, output, {"task_one"})
    assert [row["path"] for row in manifest["files"]] == ["task_one/repeat_0/finish_params.json"]
    assert not list(output.rglob("*verify_response*"))
    assert not (output / "task_other").exists()


@pytest.mark.parametrize("kind", ["directory_cycle", "file_cycle", "fifo"])
def test_cycles_and_nonregular_files_fail(tmp_path, kind):
    source = tmp_path / "source"
    task = source / "task_one"
    task.mkdir(parents=True)
    if kind == "directory_cycle":
        (task / "loop").symlink_to(task, target_is_directory=True)
    elif kind == "file_cycle":
        (task / "loop").symlink_to(task / "loop")
    else:
        os.mkfifo(task / "special")
    with pytest.raises(ValueError):
        copy_tree(source, tmp_path / "copy", {"task_one"})


@pytest.mark.parametrize("changed", ["source", "output"])
def test_validation_rejects_source_or_output_drift(tmp_path, changed):
    source = tmp_path / "source"
    (source / "task_one").mkdir(parents=True)
    (source / "task_one/file.txt").write_text("Original")
    output = tmp_path / "copy"
    manifest = copy_tree(source, output, {"task_one"})
    ((source if changed == "source" else output) / "task_one/file.txt").write_text("Changed")
    with pytest.raises(ValueError, match="changed"):
        validate_tree(source, output, manifest)
    with pytest.raises(FileExistsError):
        copy_tree(source, output, {"task_one"})
