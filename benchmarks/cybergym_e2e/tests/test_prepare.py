# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""prepare.py: partial-asset detection fails closed, and row/task-list parsing."""

import prepare
import pytest


def _populate(assets, task="jq/arvo_1"):
    """Create the full set of required assets for `task` under `assets`."""
    proj, name = task.split("/")
    (assets / "scripts").mkdir(parents=True)
    for s in prepare.REPO_SHARED:
        (assets / s).write_text("x")
    pdir = assets / "projects" / proj
    (pdir / name).mkdir(parents=True)
    (pdir / "project.toml").write_text("x")
    for f in prepare.REPO_PER_TASK:
        (pdir / name / f).write_text("x")
    ddir = assets / "data" / "projects" / proj / name
    ddir.mkdir(parents=True)
    for f in prepare.HF_PER_TASK:
        (ddir / f).write_text("x")
    return ddir


def test_missing_assets_flags_empty_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "ASSETS_DIR", tmp_path)
    miss = prepare._missing_assets(["jq/arvo_1"])
    assert any("scripts/validate.py" in m for m in miss)  # repo shared
    assert any("src.tgz" in m for m in miss)              # gated HF
    assert any("config.toml" in m for m in miss)          # per-task repo


def test_complete_tree_has_no_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "ASSETS_DIR", tmp_path)
    _populate(tmp_path)
    assert prepare._missing_assets(["jq/arvo_1"]) == []


def test_partial_hf_download_detected(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "ASSETS_DIR", tmp_path)
    ddir = _populate(tmp_path)
    (ddir / "poc.bin").unlink()  # the ground-truth PoC never arrived
    miss = prepare._missing_assets(["jq/arvo_1"])
    assert miss and all("poc.bin" in m for m in miss)
    # the repo-only check (used to decide cache validity) must not flag the missing HF file
    assert prepare._missing_assets(["jq/arvo_1"], hf=False) == []


def test_partial_repo_download_detected(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "ASSETS_DIR", tmp_path)
    _populate(tmp_path)
    (tmp_path / "projects" / "jq" / "arvo_1" / "compile.sh").unlink()
    assert prepare._missing_assets(["jq/arvo_1"], hf=False)  # repo check catches it too


def test_row_rejects_malformed_id():
    with pytest.raises(ValueError):
        prepare._row("no-slash", "e2e")


def test_row_shape():
    r = prepare._row("jq/arvo_1", "e2e")
    assert r["project"] == "jq" and r["task_id"] == "jq/arvo_1"
    assert r["responses_create_params"] == {"input": []}  # agent builds the real prompt


def test_task_list_skips_blanks_and_comments(tmp_path):
    f = tmp_path / "tasks.txt"
    f.write_text("# header\n\njq/arvo_1\n  yara/arvo_2  \n# trailing\n")
    assert prepare._read_task_list(f) == ["jq/arvo_1", "yara/arvo_2"]
