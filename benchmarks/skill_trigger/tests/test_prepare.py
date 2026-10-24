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
"""Exercise preparation against actual disposable Git source repositories."""

import json
import subprocess
from pathlib import Path

import pytest

from benchmarks.skill_trigger.prepare import (
    Case,
    Catalog,
    Provenance,
    ReleaseBlocker,
    Selection,
    Source,
    digest,
    prepare_release,
    source_path,
    tree_digest,
    tree_entries,
)


def run_git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


@pytest.fixture
def release(tmp_path: Path) -> tuple[Selection, Path]:
    root = tmp_path / "sources"
    repo = root / "example"
    folder = repo / "skills/target"
    (folder / "evals").mkdir(parents=True)
    (folder / "SKILL.md").write_text("---\nname: target\ndescription: Solve the task\n---\nRead these instructions.")
    (folder / "helper.py").write_text("print('helper')\n")
    (folder / "helper.py").chmod(0o755)
    (folder / "NOTICE").write_text("Nested attribution\n")
    (folder / "evals/answers.json").write_text('{"hidden_answer": "do not expose"}')
    (repo / "LICENSE").write_text("Example license for test fixture\n")
    upstream = repo / "cases.json"
    upstream.write_text(json.dumps({"trigger": {"positive": [{"prompt": "Original public prompt"}]}}))
    run_git(repo, "init", "-q")
    run_git(repo, "config", "user.email", "fixture@example.invalid")
    run_git(repo, "config", "user.name", "Fixture")
    run_git(repo, "add", ".")
    run_git(repo, "-c", "commit.gpgsign=false", "commit", "-qm", "fixture")
    tracked = run_git(repo, "ls-files", "-z").split("\0")
    selection = Selection(
        version=1,
        expected_counts={"example": 1},
        sources={
            "example": Source(
                url="https://example.invalid/fixture.git",
                commit=run_git(repo, "rev-parse", "HEAD"),
                notices=["LICENSE"],
            )
        },
        catalogs={
            "example-all": Catalog(
                source="example",
                directories=["skills/target"],
                mounted_sha256={
                    "skills/target": tree_digest(tree_entries(repo, "skills/target", tracked, exclude_labels=True))
                },
            )
        },
        cases=[
            Case(
                id="example-task",
                suite="example",
                skill="target",
                target="skills/target",
                target_tree_sha256=tree_digest(tree_entries(repo, "skills/target", tracked, exclude_labels=False)),
                catalog="example-all",
                prompt="Adapted public prompt",
                prompt_sha256=digest(b"Adapted public prompt"),
                provenance=Provenance(
                    kind="adapted_trigger",
                    path="cases.json",
                    pointer="/trigger/positive/0/prompt",
                    source_sha256=digest(upstream.read_bytes()),
                    reason="Self-contained request",
                ),
            )
        ],
    )
    return selection, root


def test_reproducible_release_keeps_labels_off_catalog_and_notices_intact(release, tmp_path):
    selection, root = release
    first = prepare_release(selection, sources_root=root, output=tmp_path / "one")
    second = prepare_release(selection, sources_root=root, output=tmp_path / "two")
    assert first.read_bytes() == second.read_bytes()
    manifest = json.loads(first.read_text())
    catalog = first.parent / "catalogs/example-all/target"
    assert (catalog / "SKILL.md").read_bytes() == (root / "example/skills/target/SKILL.md").read_bytes()
    assert (catalog / "helper.py").stat().st_mode & 0o111
    assert not (catalog / "evals").exists()
    assert (catalog / "NOTICE").read_text() == "Nested attribution\n"
    assert (first.parent / "notices/example/LICENSE").exists()
    assert manifest["cases"][0]["provenance"]["upstream_prompt"] == "Original public prompt"
    assert manifest["cases"][0]["provenance"]["wording_changed"] is True
    assert manifest["catalogs"]["example-all"]["members"][0]["excluded_files"] == ["evals/answers.json"]
    assert str(tmp_path) not in first.read_text()
    with pytest.raises(FileExistsError):
        prepare_release(selection, sources_root=root, output=first.parent)


@pytest.mark.parametrize("path", ["skills/target/SKILL.md", "skills/target/evals/answers.json", "untracked.txt"])
def test_dirty_source_is_rejected_including_hidden_or_untracked_files(release, tmp_path, path):
    selection, root = release
    (root / "example" / path).write_text("changed")
    with pytest.raises(ValueError, match="Modified or untracked"):
        prepare_release(selection, sources_root=root, output=tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("prompt", "changed prompt", "Prompt changed"),
        ("target_tree_sha256", "0" * 64, "version is missing or changed"),
        ("target", "skills/another-target", "version is missing or changed"),
    ],
)
def test_wrong_release_inputs_fail_without_partial_publication(release, tmp_path, field, value, message):
    selection, root = release
    setattr(selection.cases[0], field, value)
    with pytest.raises(ValueError, match=message):
        prepare_release(selection, sources_root=root, output=tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".skill-trigger-*"))


def test_duplicate_catalog_name_is_rejected(release, tmp_path):
    selection, root = release
    selection.catalogs["example-all"].directories *= 2
    with pytest.raises(ValueError, match="Duplicate skill name"):
        prepare_release(selection, sources_root=root, output=tmp_path / "out")


def test_changed_companion_catalog_hash_is_rejected(release, tmp_path):
    selection, root = release
    selection.catalogs["example-all"].mounted_sha256["skills/target"] = "0" * 64
    with pytest.raises(ValueError, match="Mounted skill tree changed"):
        prepare_release(selection, sources_root=root, output=tmp_path / "out")


def test_mounted_tree_preserves_component_sorting_used_by_reference(tmp_path):
    folder = tmp_path / "skill"
    (folder / "helper").mkdir(parents=True)
    (folder / "helper.py").write_text("module")
    (folder / "helper/data.txt").write_text("data")
    tracked = ["skill/helper.py", "skill/helper/data.txt"]
    complete = tree_entries(tmp_path, "skill", tracked, exclude_labels=False)
    mounted = tree_entries(tmp_path, "skill", tracked, exclude_labels=True)
    assert [p for p, _ in complete] == ["helper.py", "helper/data.txt"]
    assert [p for p, _ in mounted] == ["helper/data.txt", "helper.py"]


def test_source_provenance_cannot_silently_change(release, tmp_path):
    selection, root = release
    selection.cases[0].provenance.source_sha256 = "0" * 64
    with pytest.raises(ValueError, match="Provenance file changed"):
        prepare_release(selection, sources_root=root, output=tmp_path / "out")


def test_missing_case_does_not_become_smaller_benchmark(release, tmp_path):
    selection, root = release
    selection.cases = []
    with pytest.raises(ValueError, match="Case counts"):
        prepare_release(selection, sources_root=root, output=tmp_path / "out")


def test_wrong_revision_is_not_reset(release, tmp_path):
    selection, root = release
    before = run_git(root / "example", "rev-parse", "HEAD")
    selection.sources["example"].commit = "0" * 40
    with pytest.raises(ValueError, match="Wrong revision"):
        prepare_release(selection, sources_root=root, output=tmp_path / "out", fetch=True)
    assert run_git(root / "example", "rev-parse", "HEAD") == before


def test_release_blocker_allows_metadata_audit_but_never_catalog_packaging(release, tmp_path):
    selection, root = release
    selection.release_blockers = [
        ReleaseBlocker(
            source="example",
            path="LICENSE",
            sha256=digest((root / "example/LICENSE").read_bytes()),
            reason="Unresolved distribution terms",
        )
    ]
    with pytest.raises(ValueError, match="Unresolved release blockers"):
        prepare_release(selection, sources_root=root, output=tmp_path / "release")
    assert not (tmp_path / "release").exists()
    audit = prepare_release(selection, sources_root=root, output=tmp_path / "audit", audit_only=True)
    assert list(audit.parent.iterdir()) == [audit]
    manifest = json.loads(audit.read_text())
    assert manifest["artifact_kind"] == "source-audit"
    assert manifest["release_blockers"][0]["reason"] == "Unresolved distribution terms"


def test_non_slug_upstream_skill_names_are_preserved(release, tmp_path):
    selection, root = release
    repo = root / "example"
    skill = repo / "skills/target/SKILL.md"
    skill.write_text(skill.read_text().replace("name: target", "name: Automatic Speech Recognition (ASR)"))
    run_git(repo, "add", ".")
    run_git(repo, "-c", "commit.gpgsign=false", "commit", "-qm", "display name")
    selection.sources["example"].commit = run_git(repo, "rev-parse", "HEAD")
    selection.cases[0].skill = "Automatic Speech Recognition (ASR)"
    tracked = run_git(repo, "ls-files", "-z").split("\0")
    selection.cases[0].target_tree_sha256 = tree_digest(
        tree_entries(repo, "skills/target", tracked, exclude_labels=False)
    )
    selection.catalogs["example-all"].mounted_sha256["skills/target"] = tree_digest(
        tree_entries(repo, "skills/target", tracked, exclude_labels=True)
    )
    manifest = prepare_release(selection, sources_root=root, output=tmp_path / "out")
    assert (manifest.parent / "catalogs/example-all/Automatic Speech Recognition (ASR)/SKILL.md").exists()


@pytest.mark.parametrize("relative", ["../outside", "/etc/passwd", "link", "link/file"])
def test_path_escape_and_symlinks_rejected(tmp_path, relative):
    (tmp_path / "file").write_text("data")
    (tmp_path / "link").symlink_to(tmp_path)
    with pytest.raises(ValueError):
        source_path(tmp_path, relative)
