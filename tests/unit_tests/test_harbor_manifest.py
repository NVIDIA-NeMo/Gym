# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset identity (M33): the derived-only ``manifest.toml``, the prepare check and the recorded digest."""

import errno
import json
import os
import shutil
import subprocess
import threading
import tomllib
from asyncio import Future
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

import nemo_gym.rollouts.rollout_collection
import tests.unit_tests.rollouts.test_rollout_collection as collection_tests
import tests.unit_tests.test_harbor_package_store as test_harbor_package_store
from nemo_gym.data import DIGEST_KEY, load_task
from nemo_gym.data.tasks.harbor.build_options import BuildOptions
from nemo_gym.data.tasks.harbor.dataset_config import GymDatasetConfig, apply_dataset_config, read_dataset_config
from nemo_gym.data.tasks.harbor.digest import DATASET_DIGEST_KEY, content_hash
from nemo_gym.data.tasks.harbor.hub import HubError, RegistryDataset, RegistryTask, fetch_dataset
from nemo_gym.data.tasks.harbor.manifest import (
    MANIFEST_FILE,
    Manifest,
    ManifestError,
    UnsupportedManifestError,
    check_manifest,
    check_task_manifest,
    dataset_digest,
    dataset_files_digest,
    dataset_toml_digest,
    refresh_gym_layer,
    task_differences,
)
from nemo_gym.data.tasks.harbor.materialize import run_config, write_rows
from nemo_gym.data.tasks.harbor.prepare import build_run, prepare_target, resolve_agent
from nemo_gym.data.tasks.harbor.task import MAX_DISCOVERY_DEPTH, iter_task_folders
from nemo_gym.rollouts import RolloutCollectionConfig, RolloutCollectionHelper
from nemo_gym.rollouts.rollout_collection import _CompletedRollout
from tests.unit_tests.test_harbor_tasks import (  # noqa: F401  (fixtures)
    commit_repo,
    fake_registry,
    registry_dataset,
    registry_repo,
    write_task,
)


def manifest_of(folder: Path) -> dict:
    return tomllib.loads((folder / MANIFEST_FILE).read_text())


def write_escaping_task(root: Path) -> Path:
    """A task folder :func:`content_hash` refuses: ``tests/escape`` links to a file outside the task folder."""
    write_task(root)
    outside = root.parent.parent / "outside.txt"
    outside.write_text("not part of the task\n")
    (root / "tests" / "escape").symlink_to(outside)
    return root


def fetched_manifest(folder: Path, names: list[str]) -> Manifest:
    """Write the manifest a registry fetch would have left beside the task folders."""
    pins = {
        name: {"git_url": "https://example.test/tasks.git", "git_commit_id": "a" * 40, "path": f"tasks/{name}"}
        for name in names
    }
    manifest = Manifest.build(folder, source="harbor-registry", pins=pins)
    manifest.write(folder)
    return manifest


class TestManifestFormat:
    def test_write_applies_the_umask_and_keeps_an_existing_mode(self, tmp_path, monkeypatch):
        """Another user prepares a shared dataset by reading its manifest: a rewrite must not demote a readable
        file to owner-only (mkstemp's 0600), and a new file gets the umask-aware mode Path.write_text would."""
        folder = tmp_path / "ds"
        write_task(folder / "a")
        path = folder / MANIFEST_FILE
        previous = os.umask(0o022)
        try:
            Manifest.build(folder, source="local").write(folder)
            assert path.stat().st_mode & 0o777 == 0o644
            # A rewrite keeps whatever mode the file had, group-writable included.
            path.chmod(0o664)
            write_task(folder / "b")
            Manifest.build(folder, source="local").write(folder)
            assert path.stat().st_mode & 0o777 == 0o664
            assert set(Manifest.read(folder).tasks) == {"a", "b"}
            # A replace that fails leaves the previous file, its mode and no temporary behind.
            before = path.read_bytes()
            monkeypatch.setattr(
                "nemo_gym.data.tasks.harbor.manifest.os.replace", lambda *_: (_ for _ in ()).throw(OSError("disk"))
            )
            with pytest.raises(OSError, match="disk"):
                Manifest.build(folder, source="harbor-registry").write(folder)
            assert path.read_bytes() == before and path.stat().st_mode & 0o777 == 0o664
            assert sorted(p.name for p in folder.iterdir()) == ["a", "b", MANIFEST_FILE]
        finally:
            os.umask(previous)

    def test_exclusive_write_creates_once_and_leaves_an_existing_file_untouched(self, tmp_path):
        """A fetcher claims a folder with an exclusive write: the complete file appears where none was, a second
        exclusive write raises FileExistsError without touching the first, and no temporary is left behind."""
        folder = tmp_path / "ds"
        folder.mkdir()
        path = folder / MANIFEST_FILE
        pins = {"a": {"package": "org/a", "content_hash": "sha256:" + "a" * 64}}
        first = Manifest.build(folder, source="harbor-package-store", pins=pins, names=["a"])
        assert first.write(folder, exclusive=True) == path
        assert Manifest.read(folder).tasks == {"a": pins["a"]}
        claim = path.read_bytes()
        other = Manifest.build(folder, source="harbor-package-store", pins={"b": pins["a"]}, names=["b"])
        with pytest.raises(FileExistsError):
            other.write(folder, exclusive=True)
        assert path.read_bytes() == claim
        assert sorted(p.name for p in folder.iterdir()) == [MANIFEST_FILE]
        # A plain write still replaces it.
        other.write(folder)
        assert set(Manifest.read(folder).tasks) == {"b"}

    @pytest.mark.parametrize(
        "error", [PermissionError(errno.EPERM, "Operation not permitted"), OSError(errno.ENOTSUP, "x")]
    )
    def test_exclusive_write_without_hard_links_still_creates_once(self, tmp_path, monkeypatch, error):
        """A datasets directory on a filesystem without hard links (vfat, some FUSE or SMB mounts) refuses os.link:
        the claim is then created with an exclusive open of the target itself, still one winner and no temporary
        left, while any other failure of the link propagates and leaves the folder as it was."""
        import nemo_gym.data.tasks.harbor.manifest as manifest_module

        folder = tmp_path / "ds"
        folder.mkdir()
        path = folder / MANIFEST_FILE
        pins = {"a": {"package": "org/a", "content_hash": "sha256:" + "a" * 64}}
        manifest = Manifest.build(folder, source="harbor-package-store", pins=pins, names=["a"])
        links = []

        def no_links(src, dst, *args, **kwargs):
            links.append((Path(src).name, Path(dst).name))
            raise error

        monkeypatch.setattr(manifest_module.os, "link", no_links)
        assert manifest.write(folder, exclusive=True) == path
        assert Manifest.read(folder) == manifest and len(links) == 1
        assert links[0][0].startswith(".manifest-") and links[0][1] == MANIFEST_FILE
        with pytest.raises(FileExistsError):
            manifest.write(folder, exclusive=True)
        assert sorted(p.name for p in folder.iterdir()) == [MANIFEST_FILE]
        # An unrelated failure of the link is not a missing feature: it propagates, and nothing is created.
        path.unlink()
        monkeypatch.setattr(
            manifest_module.os, "link", lambda *a, **k: (_ for _ in ()).throw(OSError(errno.EIO, "io"))
        )
        with pytest.raises(OSError, match="io"):
            manifest.write(folder, exclusive=True)
        assert list(folder.iterdir()) == []

    def test_temporary_names_are_unique_per_writer(self, tmp_path, monkeypatch):
        """Two writers in one process (or one pid reused across hosts of a shared mount) never share a temporary: the
        name carries the pid, the thread and random bytes, and is opened exclusively."""
        import nemo_gym.data.tasks.harbor.manifest as manifest_module

        folder = tmp_path / "ds"
        folder.mkdir()
        manifest = Manifest.build(folder, source="local")
        names = []
        real_open = os.open

        def recording_open(path, flags, *args, **kwargs):
            if Path(path).name.startswith(".manifest-"):
                names.append(Path(path).name)
                assert flags & os.O_EXCL
            return real_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(manifest_module.os, "open", recording_open)
        manifest.write(folder)
        manifest.write(folder)
        assert len(names) == 2 and names[0] != names[1]
        assert all(name.startswith(f".manifest-{os.getpid()}-{threading.get_ident()}-") for name in names)
        assert sorted(p.name for p in folder.iterdir()) == [MANIFEST_FILE]

    def test_output_is_deterministic_sorted_and_derived_only(self, tmp_path):
        folder = tmp_path / "ds"
        for name in ("zeta", "alpha", "mid"):
            write_task(folder / name)
        (folder / "dataset.toml").write_text('[dataset]\nname = "ds"\nversion = "1.0"\n')
        pins = {
            name: {"path": f"tasks/{name}", "git_commit_id": "b" * 40, "git_url": "https://x.test/r.git"}
            for name in ("mid", "zeta", "alpha")
        }

        first = Manifest.build(folder, source="harbor-registry", pins=pins).text()
        second = Manifest.build(folder, source="harbor-registry", pins=dict(reversed(list(pins.items())))).text()

        assert first == second
        # Fixed key order, sorted tasks, no timestamps, and no copy of the name or version.
        lines = first.splitlines()
        assert lines[0] == "schema_version = 1"
        assert [line for line in lines if line.startswith("[tasks.")] == [
            '[tasks."alpha"]',
            '[tasks."mid"]',
            '[tasks."zeta"]',
        ]
        alpha = lines[lines.index('[tasks."alpha"]') + 1 :][:4]
        assert [line.split(" = ")[0] for line in alpha] == ["digest", "git_url", "git_commit_id", "path"]
        assert not any(line.startswith(("name = ", "version = ")) for line in lines) and "1.0" not in first
        data = tomllib.loads(first)
        assert set(data["dataset"]) == {"source", "digest", "dataset_toml_digest"}
        assert data["tasks"]["alpha"]["digest"] == content_hash(folder / "alpha")

    def test_dataset_digest_covers_tasks_and_dataset_toml(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        base = Manifest.build(folder, source="local")
        assert base.digest == dataset_digest({"a": content_hash(folder / "a")}, base.dataset_toml_digest)
        assert base.dataset_toml_digest == dataset_toml_digest(folder)  # absent: digest of nothing

        (folder / "dataset.toml").write_text("[gym.defaults]\ntimeout_multiplier = 2.0\n")
        with_toml = Manifest.build(folder, source="local")
        # A timeout override changes the score, so it changes the dataset.
        assert with_toml.digest != base.digest and with_toml.dataset_toml_digest != base.dataset_toml_digest
        assert with_toml.tasks == base.tasks

        (folder / "a" / "instruction.md").write_text("edited\n")
        with_edit = Manifest.build(folder, source="local")
        assert with_edit.digest not in (base.digest, with_toml.digest)
        assert with_edit.tasks["a"]["digest"] != base.tasks["a"]["digest"]

    def test_dataset_level_verifier_and_task_data_modules_enter_the_dataset_digest(self, tmp_path, capsys):
        # A dataset-level verifier.py grades every task without tests/ and task_data.py types every row: two runs
        # that differ only in one of them are not one dataset. Without them nothing changes: same digest, same text.
        folder = tmp_path / "ds"
        write_task(folder / "a")
        base = Manifest.build(folder, source="local")
        assert base.dataset_files_digest == dataset_files_digest(folder) == dataset_toml_digest(folder)
        assert "dataset_files_digest" not in base.text()
        assert base.digest == dataset_digest({"a": content_hash(folder / "a")}, base.dataset_toml_digest)

        (folder / "verifier.py").write_text("def verify(workspace):\n    return 1.0\n")
        with_grader = Manifest.build(folder, source="local")
        assert with_grader.digest != base.digest and with_grader.tasks == base.tasks
        assert with_grader.digest == dataset_digest(
            {"a": content_hash(folder / "a")},
            base.dataset_toml_digest,
            with_grader.dataset_files_digest,
        )
        assert f"dataset_files_digest = {json.dumps(with_grader.dataset_files_digest)}" in with_grader.text()
        with_grader.write(folder)
        assert Manifest.read(folder) == with_grader
        assert set(tomllib.loads(with_grader.text())["dataset"]) == {
            "source",
            "digest",
            "dataset_toml_digest",
            "dataset_files_digest",
        }

        (folder / "verifier.py").write_text("def verify(workspace):\n    return 0.0\n")
        edited = Manifest.build(folder, source="local")
        assert edited.digest != with_grader.digest and edited.dataset_files_digest != with_grader.dataset_files_digest
        (folder / "task_data.py").write_text("from pydantic import BaseModel\nclass TaskData(BaseModel):\n    pass\n")
        with_model = Manifest.build(folder, source="local")
        assert with_model.digest not in (base.digest, with_grader.digest, edited.digest)

        # A fetched dataset: the Gym layer moved, the task folders did not, so the manifest is rewritten (not
        # refused, as for dataset.toml), the pins kept, and the rewrite says which file moved.
        before = fetched_manifest(folder, ["a"])
        (folder / "verifier.py").write_text("def verify(workspace):\n    return 0.5\n")
        after = check_manifest(folder)
        assert after.source == "harbor-registry" and after.pins == before.pins and after.digest != before.digest
        assert after.tasks["a"]["digest"] == before.tasks["a"]["digest"]
        assert "a dataset-level task_data.py or verifier.py changed" in capsys.readouterr().out
        assert Manifest.read(folder) == after

        # A single task folder prepared on its own: its identity covers the files beside it the same way.
        prepared = prepare_target(str(folder / "a"), output_root=tmp_path / "out")
        assert prepared.dataset_digest == dataset_digest(
            {"a": prepared.tasks[0].folder_digest},
            dataset_toml_digest(folder),
            dataset_files_digest(folder),
        )

    def test_names_round_trip_through_toml_and_pins_survive_absent_folders(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "task with spaces & $pecial")
        pins = {
            "task with spaces & $pecial": {"package": "org/x", "content_hash": "sha256:" + "c" * 64},
            "pending": {"package": "org/pending", "content_hash": "sha256:" + "d" * 64},
        }
        manifest = Manifest.build(folder, source="harbor-package-store", pins=pins)
        manifest.write(folder)
        again = Manifest.read(folder)
        assert again == manifest
        assert again.tasks["pending"] == pins["pending"]  # pinned, no folder yet, so no digest
        assert again.task_digests["pending"] is None
        assert again.pins["task with spaces & $pecial"] == pins["task with spaces & $pecial"]

    def test_read_is_none_when_absent_and_refuses_a_malformed_file(self, tmp_path):
        assert Manifest.read(tmp_path) is None
        (tmp_path / MANIFEST_FILE).write_text("not = toml =")
        with pytest.raises(ManifestError, match="not a valid manifest"):
            Manifest.read(tmp_path)
        (tmp_path / MANIFEST_FILE).write_text("dataset = 1\n")
        with pytest.raises(ManifestError, match="must be tables"):
            Manifest.read(tmp_path)

    def test_unknown_source_is_refused(self, tmp_path):
        write_task(tmp_path / "a")
        with pytest.raises(ValueError, match="Unknown manifest source"):
            Manifest.build(tmp_path, source="somewhere")

    def test_build_reuses_known_digests_and_extends_the_cache(self, tmp_path, monkeypatch):
        import nemo_gym.data.tasks.harbor.manifest as module

        folder = tmp_path / "ds"
        for name in ("a", "b"):
            write_task(folder / name)
        hashed: list[str] = []
        real = module.content_hash
        monkeypatch.setattr(module, "content_hash", lambda path: hashed.append(path.name) or real(path))
        digests = {"a": real(folder / "a")}
        Manifest.build(folder, source="local", digests=digests)
        assert hashed == ["b"] and set(digests) == {"a", "b"}


class TestCheckManifest:
    def test_local_folder_gets_a_manifest_and_follows_its_own_edits(self, tmp_path, capsys):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        first = check_manifest(folder)
        assert (folder / MANIFEST_FILE).is_file() and manifest_of(folder)["dataset"]["source"] == "local"
        assert "Wrote" in capsys.readouterr().out
        assert check_manifest(folder) == first  # unchanged: nothing rewritten
        assert capsys.readouterr().out == ""

        (folder / "a" / "instruction.md").write_text("the author keeps editing\n")
        second = check_manifest(folder)
        assert second.digest != first.digest and manifest_of(folder)["dataset"]["digest"] == second.digest
        assert "Rewrote" in capsys.readouterr().out

    def test_fetched_folder_edited_by_hand_is_refused_with_the_recovery_step(self, tmp_path):
        folder = tmp_path / "ds"
        for name in ("a", "b"):
            write_task(folder / name)
        before = fetched_manifest(folder, ["a", "b"])
        (folder / "b" / "tests" / "test.sh").write_text("echo 1 > /logs/verifier/reward.txt\n")
        write_task(folder / "extra")
        with pytest.raises(ManifestError) as info:
            check_manifest(folder)
        message = str(info.value)
        # The unchanged task `a` is not named; the edited and the unrecorded ones are.
        assert message.split("(", 1)[1].split(")", 1)[0] == "changed: b; not in the manifest: extra"
        assert "Delete the folder, or move it aside, and run again" in message
        assert (folder / MANIFEST_FILE).read_text() == before.text()  # a refusal rewrites nothing

    def test_a_single_task_folder_is_held_to_its_parents_fetched_manifest(self, tmp_path):
        """One task of a fetched dataset is checked against the digest the parent's manifest records for it; the
        manifest is read, never written, and a parent that is no fetched dataset holds the task to nothing."""
        folder = tmp_path / "ds"
        for name in ("a", "b"):
            write_task(folder / name)
        before = fetched_manifest(folder, ["a", "b", "later"])  # `later` is pinned, its folder never landed
        # An unchanged task passes and the parent's manifest comes back, its digest reused through `digests`.
        digests: dict[str, str] = {}
        parent = check_task_manifest(folder / "a", digests=digests)
        assert parent is not None and parent.source == "harbor-registry" and parent.text() == before.text()
        assert digests == {"a": content_hash(folder / "a")}
        # A loader's digest is what is checked when given, not the folder as it is now.
        with pytest.raises(ManifestError, match=r"ds/a does not match .*manifest\.toml \(changed: a\)"):
            check_task_manifest(folder / "a", digests={"a": "0" * 64})
        # An edited task, a folder the fetch never recorded, one it never finished and one that cannot be hashed.
        (folder / "b" / "instruction.md").write_text("Create a file called cheat.txt.\n")
        with pytest.raises(ManifestError) as info:
            check_task_manifest(folder / "b")
        assert "(changed: b)" in str(info.value) and f"Delete {folder}, or move it aside, and run again" in str(
            info.value
        )
        write_task(folder / "stray")
        with pytest.raises(ManifestError, match=r"\(not in the manifest: stray\)"):
            check_task_manifest(folder / "stray")
        write_task(folder / "later")
        with pytest.raises(ManifestError, match=r"interrupted \(tasks not yet recorded: later\); run the fetch again"):
            check_task_manifest(folder / "later")
        shutil.rmtree(folder / "b")
        write_escaping_task(folder / "b")
        with pytest.raises(ManifestError, match=r"\(cannot be hashed: b\)"):
            check_task_manifest(folder / "b")
        assert (folder / MANIFEST_FILE).read_text() == before.text()  # nothing above wrote the manifest
        # A parent with no manifest, a local one, or one from before digests were recorded holds the task to nothing.
        assert check_task_manifest(write_task(tmp_path / "solo")) is None
        local = tmp_path / "local"
        write_task(local / "a")
        check_manifest(local)
        (local / "a" / "instruction.md").write_text("edited\n")
        assert check_task_manifest(local / "a") is None
        pinned = tmp_path / "pinned"
        write_task(pinned / "a")
        (pinned / MANIFEST_FILE).write_text(
            '[dataset]\nsource = "harbor-registry"\n\n'
            '[tasks."a"]\ngit_url = "https://x.test/r.git"\ngit_commit_id = "' + "e" * 40 + '"\npath = "tasks/a"\n'
        )
        assert check_task_manifest(pinned / "a") is None
        (pinned / MANIFEST_FILE).write_text('[dataset]\nsource = "elsewhere"\n')
        with pytest.raises(ManifestError, match="names source 'elsewhere', which this Gym does not know"):
            check_task_manifest(pinned / "a")

    def test_a_task_moved_deeper_into_a_fetched_dataset_is_held_to_the_nearest_manifest_above_it(self, tmp_path):
        """The dataset folder's check finds a task under a subfolder too and names it by folder name, so the
        single-task check walks up to the nearest manifest instead of reading only the task's parent; a folder in
        between with a manifest of its own holds its tasks itself."""
        folder = tmp_path / "ds"
        for name in ("a", "b"):
            write_task(folder / name)
        before = fetched_manifest(folder, ["a", "b"])
        deep = folder / "group" / "deeper"
        deep.mkdir(parents=True)
        for name in ("a", "b"):
            shutil.move(str(folder / name), str(deep / name))
        digests: dict[str, str] = {}
        held = check_task_manifest(deep / "a", digests=digests)
        assert held is not None and held.text() == before.text() and digests == {"a": content_hash(deep / "a")}
        (deep / "b" / "instruction.md").write_text("Create a file called cheat.txt.\n")
        with pytest.raises(ManifestError) as info:
            check_task_manifest(deep / "b")
        message = str(info.value)
        assert f"{deep / 'b'} does not match {folder / MANIFEST_FILE} (changed: b)" in message
        assert f"Delete {folder}, or move it aside, and run again" in message
        write_task(deep / "stray")
        with pytest.raises(ManifestError, match=r"\(not in the manifest: stray\)"):
            check_task_manifest(deep / "stray")
        assert (folder / MANIFEST_FILE).read_text() == before.text()
        inner = folder / "group" / "inner"
        write_task(inner / "c")
        own = fetched_manifest(inner, ["c"])
        assert check_task_manifest(inner / "c").text() == own.text()
        (inner / "c" / "instruction.md").write_text("edited\n")
        with pytest.raises(ManifestError, match=r"inner/c does not match .*inner/manifest\.toml \(changed: c\)"):
            check_task_manifest(inner / "c")

    def test_the_single_task_check_reaches_exactly_the_tasks_the_dataset_folders_check_discovers(self, tmp_path):
        """The walk to the manifest mirrors task discovery: at most MAX_DISCOVERY_DEPTH levels up, never through a
        hidden folder or a link. A task the dataset folder's check never sees is held to nothing; a manifest.toml
        that is no fetch record this Gym reads (another tool's in a project root, a newer Gym's) claims nothing
        above the task's parent; a local manifest in between is passed over to the fetched one above."""
        folder = tmp_path / "ds"
        write_task(folder / "a")
        before = fetched_manifest(folder, ["a"])
        deepest = folder.joinpath(*[f"l{level}" for level in range(1, MAX_DISCOVERY_DEPTH)])
        deepest.mkdir(parents=True)
        shutil.move(str(folder / "a"), str(deepest / "a"))
        write_task(deepest / "stray")
        write_task(deepest / f"l{MAX_DISCOVERY_DEPTH}" / "deeper")
        write_task(folder / ".scratch" / "mine")
        write_task(folder / ".mine")
        write_task(tmp_path / "elsewhere" / "linked")
        (folder / "link").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
        # What the dataset folder's check sees: the moved task and the stray at the deepest level, nothing deeper,
        # hidden or linked. The single-task check holds exactly those to the manifest.
        assert iter_task_folders(folder) == [deepest / "a", deepest / "stray"]
        held = check_task_manifest(deepest / "a")
        assert held is not None and held.text() == before.text()
        with pytest.raises(ManifestError, match=r"\(not in the manifest: stray\)"):
            check_task_manifest(deepest / "stray")
        for unseen in (
            deepest / f"l{MAX_DISCOVERY_DEPTH}" / "deeper",
            folder / ".scratch" / "mine",
            folder / ".mine",
            folder / "link" / "linked",
        ):
            assert check_task_manifest(unseen) is None, unseen
        # Another tool's manifest.toml, or a newer Gym's, above a standalone task's parent claims nothing: the task
        # and the dataset folder below it are untouched, beyond the discovery range and within it alike. The
        # task's own parent is read as before: its manifest is the task's dataset's, whatever it says.
        within = write_task(tmp_path / "work" / "proj" / "mytask")
        for foreign in ('[package]\nname = "julia-or-whatever"\n', 'schema_version = 99\n[dataset]\nsource = "git"\n'):
            (tmp_path / MANIFEST_FILE).write_text(foreign)
            assert check_task_manifest(within) is None
            assert check_manifest(tmp_path / "work" / "proj").source == "local"
            (tmp_path / "work" / "proj" / MANIFEST_FILE).unlink()
        (tmp_path / "work" / "proj" / MANIFEST_FILE).write_text('[package]\nname = "julia-or-whatever"\n')
        with pytest.raises(ManifestError, match="names source '', which this Gym does not know"):
            check_task_manifest(within)
        # A local manifest between the task and the fetched dataset's is passed over, so an edited task under it
        # is still held to the fetched manifest (and named with it).
        (tmp_path / MANIFEST_FILE).unlink()
        mid = folder / "mid"
        mid.mkdir()
        shutil.move(str(deepest / "a"), str(mid / "a"))
        Manifest.build(mid, source="local").write(mid)
        assert check_task_manifest(mid / "a").text() == before.text()
        (mid / "a" / "instruction.md").write_text("edited\n")
        with pytest.raises(ManifestError, match=r"ds/mid/a does not match .*ds/manifest\.toml \(changed: a\)"):
            check_task_manifest(mid / "a")
        assert (folder / MANIFEST_FILE).read_text() == before.text()

    def test_only_the_gym_layer_moving_rewrites_the_manifest_and_keeps_pins(self, tmp_path, capsys):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        before = fetched_manifest(folder, ["a"])
        (folder / "dataset.toml").write_text('[gym.tasks."a".environment]\nenv = { X = "1" }\n')
        after = check_manifest(folder)
        assert after.source == "harbor-registry" and after.pins == before.pins
        assert after.dataset_toml_digest != before.dataset_toml_digest and after.digest != before.digest
        assert after.tasks["a"]["digest"] == before.tasks["a"]["digest"]
        assert "dataset.toml changed" in capsys.readouterr().out

    def test_refresh_gym_layer_follows_dataset_toml_and_keeps_the_task_record(self, tmp_path, capsys):
        """A pull that fetched nothing brings the manifest's Gym-layer digests up to date without hashing a task: an
        edited task folder keeps its recorded digest (check_manifest, which rehashes, still refuses it), a pin without
        a digest stays one, and a manifest with pins only or none at all is left to check_manifest."""
        folder = tmp_path / "ds"
        write_task(folder / "a")
        pins = {
            name: {"git_url": "https://example.test/tasks.git", "git_commit_id": "a" * 40, "path": f"tasks/{name}"}
            for name in ("a", "b")
        }
        before = Manifest.build(folder, source="harbor-registry", pins=pins, names=["a"])
        before.write(folder)
        assert before.tasks["b"] == pins["b"]  # pinned, not landed: no digest
        assert refresh_gym_layer(tmp_path / "none") is None
        assert refresh_gym_layer(folder) == before and (folder / MANIFEST_FILE).read_text() == before.text()
        (folder / "dataset.toml").write_text('[gym.tasks."a".environment]\nenv = { X = "1" }\n')
        (folder / "a" / "instruction.md").write_text("edited by hand\n")
        after = refresh_gym_layer(folder)
        assert after.dataset_toml_digest == dataset_toml_digest(folder) != before.dataset_toml_digest
        assert after.tasks == before.tasks and after.digest != before.digest and after.source == before.source
        # The rewrite is announced as check_manifest announces it (round 3): the identity changed.
        assert Manifest.read(folder) == after and capsys.readouterr().out == (
            f"Rewrote {folder / MANIFEST_FILE}: dataset.toml changed; dataset digest "
            f"{before.digest[:12]} -> {after.digest[:12]}\n"
        )
        with pytest.raises(ManifestError, match="interrupted"):
            check_manifest(folder)  # the pin without a digest is still the fetch's unfinished business
        (folder / MANIFEST_FILE).write_text('[dataset]\nsource = "harbor-registry"\n[tasks."a"]\npath = "tasks/a"\n')
        text = (folder / MANIFEST_FILE).read_text()
        assert not refresh_gym_layer(folder).records_content and (folder / MANIFEST_FILE).read_text() == text

    def test_a_task_that_cannot_be_hashed_is_recorded_by_name_without_a_digest(self, tmp_path, capsys, caplog):
        folder = tmp_path / "ds"
        write_task(folder / "good")
        write_escaping_task(folder / "bad")
        identity = check_manifest(folder)
        assert identity.tasks == {"bad": {}, "good": {"digest": content_hash(folder / "good")}}
        assert identity.unhashable == {"bad"}
        assert "Task bad cannot be hashed" in caplog.text and "escapes the task folder: tests/escape" in caplog.text
        # The dataset digest covers the hashable tasks only; the unhashable one is named, not blessed.
        assert identity.digest == dataset_digest({"good": content_hash(folder / "good")}, identity.dataset_toml_digest)
        assert "Wrote" in capsys.readouterr().out and Manifest.read(folder).tasks == identity.tasks
        # The next check finds the same manifest: nothing rewritten, no flapping.
        assert check_manifest(folder).tasks == identity.tasks and capsys.readouterr().out == ""

    def test_a_fetched_task_that_can_no_longer_be_hashed_is_refused_by_name(self, tmp_path):
        folder = tmp_path / "ds"
        for name in ("good", "bad"):
            write_task(folder / name)
        before = fetched_manifest(folder, ["good", "bad"])
        write_escaping_task(folder / "bad")
        with pytest.raises(ManifestError) as info:
            check_manifest(folder)
        message = str(info.value)
        assert message.split("(", 1)[1].split(")", 1)[0] == "cannot be hashed: bad"
        assert "Delete the folder, or move it aside, and run again" in message
        assert (folder / MANIFEST_FILE).read_text() == before.text()

    def test_a_manifest_without_digests_is_upgraded_in_place(self, tmp_path, capsys):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        (folder / MANIFEST_FILE).write_text(
            '[dataset]\nname = "ds"\nversion = "1.0"\nsource = "harbor-registry"\n\n'
            '[tasks."a"]\ngit_url = "https://x.test/r.git"\ngit_commit_id = "' + "e" * 40 + '"\npath = "tasks/a"\n'
        )
        upgraded = check_manifest(folder)
        assert upgraded.source == "harbor-registry" and upgraded.tasks["a"]["git_commit_id"] == "e" * 40
        assert upgraded.tasks["a"]["digest"] == content_hash(folder / "a")
        text = (folder / MANIFEST_FILE).read_text()
        assert "name" not in text and "pins only" in capsys.readouterr().out

    def test_a_package_store_manifest_from_before_the_dataset_digest_keeps_its_record(self, tmp_path, capsys):
        """The package-store fetcher of an earlier Gym wrote `[dataset]` with the package's name and no digest, and per
        landed task `digest = "sha256:<hex>"`. Read, the digests are bare and the manifest records content: an
        unchanged folder is upgraded in place with the recorded digests kept, an edited one is refused, and a pin
        without a digest is a task that fetch never finished, not a pin from before digests existed."""
        folder = tmp_path / "ds"
        for name in ("a", "b"):
            write_task(folder / name)
        head = (
            '[dataset]\nname = "org/ds"\nversion = "1.0"\nsource = "harbor-package-store"\n'
            f'content_hash = "sha256:{"d" * 64}"\n\n'
        )
        pin_a = f'[tasks."a"]\npackage = "org/a"\ncontent_hash = "sha256:{"1" * 64}"\n'
        pin_b = f'[tasks."b"]\npackage = "org/b"\ncontent_hash = "sha256:{"2" * 64}"\n'
        landed_a = pin_a + f'digest = "sha256:{content_hash(folder / "a")}"\n\n'
        legacy = head + landed_a + pin_b + f'digest = "sha256:{content_hash(folder / "b")}"\n'
        (folder / MANIFEST_FILE).write_text(legacy)
        previous = Manifest.read(folder)
        assert previous.records_content and not previous.digest and previous.source == "harbor-package-store"
        assert previous.task_digests == {"a": content_hash(folder / "a"), "b": content_hash(folder / "b")}
        # The upstream pin keeps its prefix: only the folder digest reads bare.
        assert previous.pins["a"] == {"package": "org/a", "content_hash": f"sha256:{'1' * 64}"}

        # Edited by hand: refused, the earlier Gym's manifest untouched.
        (folder / "b" / "instruction.md").write_text("edited by hand\n")
        with pytest.raises(ManifestError, match=r"does not match its manifest.toml \(changed: b\)"):
            check_manifest(folder)
        assert (folder / MANIFEST_FILE).read_text() == legacy
        # A pull that fetched nothing brings the record up to date without hashing: the edit stays caught above.
        refreshed = refresh_gym_layer(folder)
        assert refreshed.task_digests == previous.task_digests and refreshed.pins == previous.pins
        assert "it recorded no dataset digest; dataset digest - ->" in capsys.readouterr().out
        (folder / MANIFEST_FILE).write_text(legacy)

        # Unchanged: upgraded in place, the recorded digests and pins kept, and the next check finds it as it is.
        (folder / "b" / "instruction.md").write_text("Create a file called hello.txt.\n")
        upgraded = check_manifest(folder)
        assert upgraded.task_digests == previous.task_digests and upgraded.pins == previous.pins
        assert upgraded.digest == dataset_digest(upgraded.task_digests, upgraded.dataset_toml_digest)
        text = (folder / MANIFEST_FILE).read_text()
        assert text.startswith("schema_version = 1\n") and "name" not in text and "sha256:sha256" not in text
        assert f'digest = "{content_hash(folder / "a")}"' in text
        assert "it recorded no dataset digest; dataset digest - ->" in capsys.readouterr().out
        assert check_manifest(folder) == upgraded and capsys.readouterr().out == ""

        # A pin without a digest in that format is unfinished business, with the other task landed or none:
        # refused until the fetch is run again, nothing rewritten.
        shutil.rmtree(folder / "b")
        for partial, unfinished in ((head + landed_a + pin_b, "b"), (head + pin_a + "\n" + pin_b, "a, b")):
            (folder / MANIFEST_FILE).write_text(partial)
            assert Manifest.read(folder).records_content
            with pytest.raises(ManifestError, match=f"interrupted .*tasks not yet recorded: {unfinished}\\)"):
                check_manifest(folder)
            assert (folder / MANIFEST_FILE).read_text() == partial
        # The registry's manifests from before digests have pins only, as before: upgraded, not refused.
        (folder / MANIFEST_FILE).write_text('[dataset]\nsource = "harbor-registry"\n\n[tasks."a"]\npath = "tasks/a"\n')
        assert not Manifest.read(folder).records_content
        assert check_manifest(folder).tasks["a"]["digest"] == content_hash(folder / "a")


@pytest.mark.usefixtures("fake_registry")  # prepare records base images; never reach Docker Hub
class TestFetchReferenceRecord:
    def test_the_fetch_reference_is_recorded_read_back_and_kept_by_every_rewrite(self, tmp_path, capsys):
        """A fetch records the reference it was made from beside its source (round 32: what a committed entry's
        declared source is checked against without the hub); it stays out of the dataset digest, a local manifest
        has none, and check_manifest and refresh_gym_layer carry it through their rewrites as they carry the pins."""
        folder = tmp_path / "ds"
        write_task(folder / "a")
        pins = {"a": {"package": "org/a", "content_hash": "sha256:" + "a" * 64}}
        ref = "harbor:org/ds@1.0.0"
        fetched = Manifest.build(folder, source="harbor-package-store", pins=pins, ref=ref)
        fetched.write(folder)
        lines = (folder / MANIFEST_FILE).read_text().splitlines()
        assert lines[2:5] == ["[dataset]", 'source = "harbor-package-store"', f'ref = "{ref}"']
        assert Manifest.read(folder) == fetched and fetched.ref == ref
        assert fetched.digest == Manifest.build(folder, source="harbor-package-store", pins=pins).digest
        assert check_manifest(folder) == fetched and capsys.readouterr().out == ""
        (folder / "dataset.toml").write_text('[gym.tasks."a".environment]\nenv = { X = "1" }\n')
        assert refresh_gym_layer(folder).ref == ref and Manifest.read(folder).ref == ref
        (folder / "dataset.toml").write_text('[gym.tasks."a".environment]\nenv = { X = "2" }\n')
        assert check_manifest(folder).ref == ref and Manifest.read(folder).ref == ref
        assert capsys.readouterr().out.count("Rewrote") == 2
        assert Manifest.read(folder).tasks == fetched.tasks
        # A local folder records none, and a value that is not a string reads as none.
        local = tmp_path / "local"
        write_task(local / "a")
        assert check_manifest(local).ref is None and "ref = " not in (local / MANIFEST_FILE).read_text()
        path = folder / MANIFEST_FILE
        path.write_text(path.read_text().replace(f'ref = "{ref}"', "ref = 7"))
        assert Manifest.read(folder).ref is None


class TestPrepareRecordsIdentity:
    def test_rows_and_run_config_carry_the_dataset_digest(self, tmp_path):
        folder = tmp_path / "ds"
        for name in ("a", "b"):
            write_task(folder / name)
        prepared = prepare_target(str(folder), output_root=tmp_path / "out")
        recorded = manifest_of(folder)["dataset"]
        assert (
            prepared.dataset_digest == recorded["digest"]
            and prepared.dataset_toml_digest == recorded["dataset_toml_digest"]
        )
        rows = [json.loads(line) for line in prepared.rows_path.read_text().splitlines()]
        assert {row["task_input"]["task_data"][DATASET_DIGEST_KEY] for row in rows} == {recorded["digest"]}
        assert all(
            row["task_input"]["task_data"][DIGEST_KEY] == recorded_task
            for row, recorded_task in zip(rows, [manifest_of(folder)["tasks"][name]["digest"] for name in ("a", "b")])
        )

        config_path, _ = build_run(prepared, resolve_agent("hermes_agent"), sandbox=None, overrides=[])
        mapping = yaml.safe_load(config_path.read_text())["harbor_ds_environment"]["environment_servers"]["harbor"][
            "tasksets"
        ]["ds"]
        assert mapping["dataset_digest"] == recorded["digest"]
        assert mapping["dataset_toml_digest"] == recorded["dataset_toml_digest"]
        assert mapping["tasks"] == {"a": prepared.tasks[0].digest, "b": prepared.tasks[1].digest}

    def test_selection_does_not_change_the_dataset_digest(self, tmp_path):
        folder = tmp_path / "ds"
        for name in ("a", "b"):
            write_task(folder / name)
        everything = prepare_target(str(folder), output_root=tmp_path / "out")
        subset = prepare_target(str(folder), output_root=tmp_path / "out2", only=["a"])
        assert [task.task_id for task in subset.tasks] == ["a"]
        assert subset.dataset_digest == everything.dataset_digest

    def test_a_single_task_folder_has_an_identity_but_writes_nothing_beside_it(self, tmp_path):
        task = write_task(tmp_path / "solo")
        prepared = prepare_target(str(task), output_root=tmp_path / "out")
        assert prepared.dataset_digest == dataset_digest(
            {"solo": prepared.tasks[0].folder_digest}, dataset_toml_digest(tmp_path)
        )
        assert not (tmp_path / MANIFEST_FILE).exists() and not (task / MANIFEST_FILE).exists()
        row = json.loads(prepared.rows_path.read_text())
        assert row["task_input"]["task_data"][DATASET_DIGEST_KEY] == prepared.dataset_digest

    def test_prepare_skips_a_task_that_cannot_be_hashed_and_selection_still_applies(self, tmp_path, capsys):
        folder = tmp_path / "ds"
        write_task(folder / "good")
        write_escaping_task(folder / "bad")
        # No selection: the rest of the dataset prepares and the unhashable task is reported as skipped.
        prepared = prepare_target(str(folder), output_root=tmp_path / "out")
        assert [task.task_id for task in prepared.tasks] == ["good"]
        assert list(prepared.skipped) == ["bad"] and "escapes the task folder: tests/escape" in prepared.skipped["bad"]
        assert manifest_of(folder)["tasks"] == {"bad": {}, "good": {"digest": prepared.tasks[0].digest}}
        assert prepared.dataset_digest == manifest_of(folder)["dataset"]["digest"]
        capsys.readouterr()
        # Selecting around it: the task is left out, not reported as broken.
        for selection, reason in (
            ({"only": ["good"]}, "not in --only-tasks"),
            ({"exclude": ["bad"]}, "excluded by --exclude-tasks 'bad'"),
        ):
            prepared = prepare_target(str(folder), output_root=tmp_path / "out", **selection)
            assert [task.task_id for task in prepared.tasks] == ["good"] and prepared.skipped == {}
            assert f"Skipping bad: {reason}" in capsys.readouterr().out
        assert json.loads((tmp_path / "out" / "ds" / "skipped.json").read_text()) == {}

    def test_a_dataset_settings_edit_is_not_a_task_file_edit_for_a_fetched_dataset(self, tmp_path, monkeypatch):
        store_tests = test_harbor_package_store.TestPackageStore()
        folders = {name: write_task(tmp_path / "src" / name) for name in ("alpha", "beta")}
        module, digests, _ = store_tests.fake_store(tmp_path, monkeypatch, folders)
        folder = module.fetch_package_dataset(
            module.PackageRef("terminal-bench", "terminal-bench", "4.0.0"), tmp_path / "datasets"
        )
        first = prepare_target(str(folder), output_root=tmp_path / "out1")
        assert first.tasks[0].config.agent.timeout_sec == 120.0

        # The publisher's folders are untouched; only the Gym layer beside them changes.
        with (folder / "dataset.toml").open("a") as handle:
            handle.write("\n[gym.defaults]\ntimeout_multiplier = 2.0\n")
        second = prepare_target(str(folder), output_root=tmp_path / "out2")
        assert second.tasks[0].config.agent.timeout_sec == 240.0
        manifest = manifest_of(folder)
        assert manifest["tasks"]["alpha"]["digest"] == digests["alpha"] == content_hash(folder / "alpha")
        assert manifest["dataset"]["digest"] == second.dataset_digest != first.dataset_digest
        # Rows and the server keep the settings-aware digest: the same folder under other settings is another task.
        assert second.tasks[0].folder_digest == first.tasks[0].folder_digest == digests["alpha"]
        assert second.tasks[0].digest != first.tasks[0].digest == digests["alpha"]
        rows = [json.loads(line) for line in second.rows_path.read_text().splitlines()]
        assert rows[0]["task_input"]["task_data"][DIGEST_KEY] == second.tasks[0].digest

    def test_a_local_dataset_with_settings_records_folder_digests_once(self, tmp_path, capsys):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        (folder / "dataset.toml").write_text("[gym.defaults]\ntimeout_multiplier = 2.0\n")
        prepared = prepare_target(str(folder), output_root=tmp_path / "out")
        out = capsys.readouterr().out
        assert out.count("Wrote") == 1 and "Rewrote" not in out
        assert (
            manifest_of(folder)["tasks"]["a"]["digest"]
            == content_hash(folder / "a")
            == prepared.tasks[0].folder_digest
        )
        assert prepared.tasks[0].digest != prepared.tasks[0].folder_digest
        again = prepare_target(str(folder), output_root=tmp_path / "out")
        out = capsys.readouterr().out
        assert "Wrote" not in out and "Rewrote" not in out
        assert again.dataset_digest == prepared.dataset_digest == manifest_of(folder)["dataset"]["digest"]

    def test_prepare_refuses_a_hand_edited_fetched_dataset(self, tmp_path):
        folder = tmp_path / "ds"
        for name in ("a", "b"):
            write_task(folder / name)
        fetched_manifest(folder, ["a", "b"])
        (folder / "a" / "solution" / "solve.sh").write_text("#!/bin/bash\necho cheat > hello.txt\n")
        with pytest.raises(ManifestError, match="edited by hand"):
            prepare_target(str(folder), output_root=tmp_path / "out")
        assert not (tmp_path / "out" / "ds" / "tasks.jsonl").exists()

    def test_prepare_refuses_an_edited_task_folder_of_a_fetched_dataset(self, tmp_path, monkeypatch, capsys):
        """Preparing one task of a fetched dataset is no way around the manifest check its dataset folder gets:
        the edited task is refused before anything is written, an unchanged sibling prepares with the fetch's
        source, and a folder that changes under the preparation is caught by the loader's digest too."""
        folder = tmp_path / "ds"
        for name in ("copy", "other"):
            write_task(folder / name)
        before = fetched_manifest(folder, ["copy", "other"])
        (folder / "copy" / "instruction.md").write_text("Create a file called cheat.txt.\n")
        with pytest.raises(ManifestError, match=r"ds/copy does not match .*manifest\.toml \(changed: copy\)"):
            prepare_target(str(folder / "copy"), output_root=tmp_path / "out")
        assert not (tmp_path / "out" / "copy").exists()
        assert (folder / MANIFEST_FILE).read_text() == before.text()

        prepared = prepare_target(str(folder / "other"), output_root=tmp_path / "out")
        assert [task.task_id for task in prepared.tasks] == ["other"]
        assert prepared.dataset_digest == dataset_digest(
            {"other": prepared.tasks[0].folder_digest}, dataset_toml_digest(folder)
        )
        # The identity is the one task's, computed in memory as for a standalone task folder, with the fetch's
        # source instead of `local`; the rows carry its digest and nothing is written beside the task.
        assert f"Dataset digest {prepared.dataset_digest} (harbor-registry)" in capsys.readouterr().out
        row = json.loads(prepared.rows_path.read_text())
        assert row["task_input"]["task_data"][DATASET_DIGEST_KEY] == prepared.dataset_digest
        assert (folder / MANIFEST_FILE).read_text() == before.text()
        assert not (folder / "other" / MANIFEST_FILE).exists()

        # An edit that lands after the check and before the load: the loader's digest differs from the checked one,
        # so the folder is held to the manifest once more, as a dataset folder is, and refused.
        discover = nemo_gym.data.tasks.harbor.prepare.discover_tasks

        def edit_then_discover(target, **kwargs):
            (folder / "other" / "instruction.md").write_text("Create a file called cheat.txt.\n")
            return discover(target, **kwargs)

        monkeypatch.setattr(nemo_gym.data.tasks.harbor.prepare, "discover_tasks", edit_then_discover)
        with pytest.raises(ManifestError, match=r"\(changed: other\)"):
            prepare_target(str(folder / "other"), output_root=tmp_path / "out2")
        assert not (tmp_path / "out2" / "other" / "tasks.jsonl").exists()

    def test_prepare_refuses_an_edited_task_moved_into_a_subfolder_of_a_fetched_dataset(self, tmp_path, capsys):
        """One mkdir away from the shape above: the dataset folder is refused for the moved and edited task, and so
        is that task prepared alone, held to the dataset's manifest two levels up; the unchanged moved sibling
        prepares with the fetch's source."""
        folder = tmp_path / "ds"
        for name in ("a", "b"):
            write_task(folder / name)
        before = fetched_manifest(folder, ["a", "b"])
        group = folder / "group"
        group.mkdir()
        for name in ("a", "b"):
            shutil.move(str(folder / name), str(group / name))
        (group / "b" / "instruction.md").write_text("Create a file called cheat.txt.\n")
        with pytest.raises(ManifestError, match=r"ds does not match its manifest\.toml \(changed: b\)"):
            prepare_target(str(folder), output_root=tmp_path / "out")
        with pytest.raises(ManifestError, match=r"ds/group/b does not match .*ds/manifest\.toml \(changed: b\)"):
            prepare_target(str(group / "b"), output_root=tmp_path / "out")
        assert not (tmp_path / "out" / "b").exists()
        capsys.readouterr()
        prepared = prepare_target(str(group / "a"), output_root=tmp_path / "out")
        assert [task.task_id for task in prepared.tasks] == ["a"]
        assert f"Dataset digest {prepared.dataset_digest} (harbor-registry)" in capsys.readouterr().out
        assert (folder / MANIFEST_FILE).read_text() == before.text()
        assert not (group / MANIFEST_FILE).exists() and not (group / "a" / MANIFEST_FILE).exists()

    def test_prepare_refuses_a_subfolder_of_a_fetched_dataset_as_a_dataset_of_its_own(self, tmp_path, capsys):
        """One command after the shape above: preparing ds/group as a dataset target wrote a local manifest there
        and prepared the edited moved task as local, after which the single-task check found that local manifest
        nearest and held the task to nothing. Refused now, nothing written; a stale local manifest an earlier Gym
        left there neither prepares nor shadows; a folder beyond the dataset check's reach is its own dataset."""
        folder = tmp_path / "ds"
        for name in ("a", "b"):
            write_task(folder / name)
        before = fetched_manifest(folder, ["a", "b"])
        group = folder / "group"
        group.mkdir()
        for name in ("a", "b"):
            shutil.move(str(folder / name), str(group / name))
        (group / "b" / "instruction.md").write_text("Create a file called cheat.txt.\n")
        with pytest.raises(ManifestError) as info:
            prepare_target(str(group), output_root=tmp_path / "out")
        message = str(info.value)
        assert f"{group} is no dataset of its own: it sits inside {folder}, whose {MANIFEST_FILE} records" in message
        assert f"Prepare {folder} (selecting tasks with --only-tasks or --exclude-tasks)" in message
        assert not (group / MANIFEST_FILE).exists() and not (tmp_path / "out" / "group").exists()
        # The state the earlier behaviour left behind: a local manifest inside the fetched tree. It is neither
        # prepared nor the one the single task is held to.
        Manifest.build(group, source="local").write(group)
        with pytest.raises(ManifestError, match="is no dataset of its own"):
            prepare_target(str(group), output_root=tmp_path / "out")
        with pytest.raises(ManifestError, match=r"ds/group/b does not match .*ds/manifest\.toml \(changed: b\)"):
            prepare_target(str(group / "b"), output_root=tmp_path / "out")
        assert not (tmp_path / "out" / "b").exists()
        capsys.readouterr()
        prepared = prepare_target(str(group / "a"), output_root=tmp_path / "out")
        assert [task.task_id for task in prepared.tasks] == ["a"]
        assert f"Dataset digest {prepared.dataset_digest} (harbor-registry)" in capsys.readouterr().out
        assert (folder / MANIFEST_FILE).read_text() == before.text()
        # A folder the dataset check never reaches (its tasks would sit deeper than discovery looks, or under a
        # hidden folder) is a dataset of its own, as before: prepared as local, with a manifest of its own.
        deep = folder.joinpath(*[f"l{level}" for level in range(1, MAX_DISCOVERY_DEPTH)])
        for own, name in ((deep / "own", "deep"), (folder / ".own", "hidden")):
            write_task(own / name)
            capsys.readouterr()
            prepared = prepare_target(str(own), output_root=tmp_path / "out")
            assert [task.task_id for task in prepared.tasks] == [name]
            assert f"Dataset digest {prepared.dataset_digest} (local)" in capsys.readouterr().out
            assert Manifest.read(own).source == "local"
        assert [task.name for task in iter_task_folders(folder)] == ["a", "b"]

    def test_write_rows_without_a_digest_keeps_the_old_row_shape(self, tmp_path):
        task = load_task(write_task(tmp_path / "ds" / "a"))
        (plain,) = write_rows([task], "ds", tmp_path / "plain.jsonl")
        (stamped,) = write_rows([task], "ds", tmp_path / "stamped.jsonl", dataset_digest="f" * 64)
        assert plain["task_input"]["task_data"] == {DIGEST_KEY: task.digest}
        assert stamped["task_input"]["task_data"] == {DIGEST_KEY: task.digest, DATASET_DIGEST_KEY: "f" * 64}
        config = run_config(
            taskset="ds",
            folder=tmp_path / "ds",
            tasks=[task],
            rows_path=tmp_path / "plain.jsonl",
            agent_instance_name="x",
            agent_impl="x",
        )
        mapping = config["harbor_ds_environment"]["environment_servers"]["harbor"]["tasksets"]["ds"]
        assert mapping == {"folder": str((tmp_path / "ds").resolve()), "tasks": {"a": task.digest}}


class TestFetchersWriteDigests:
    def test_registry_fetch_records_digests_and_refuses_to_bless_a_hand_edit(self, tmp_path, registry_repo):
        dataset = RegistryDataset(
            "ds",
            "1.0",
            "",
            tuple(RegistryTask(name, registry_repo.as_uri(), "HEAD", f"tasks/{name}") for name in ("a", "b")),
        )
        folder = fetch_dataset(dataset, tmp_path / "datasets")
        manifest = manifest_of(folder)
        assert manifest["dataset"]["source"] == "harbor-registry" and "name" not in manifest["dataset"]
        assert manifest["tasks"]["a"]["digest"] == content_hash(folder / "a")
        assert manifest["tasks"]["a"]["path"] == "tasks/a" and len(manifest["tasks"]["a"]["git_commit_id"]) == 40
        assert check_manifest(folder) == Manifest.read(folder)  # prepare's check agrees with the fetch

        # A rerun with nothing to fetch leaves the manifest alone and hashes nothing.
        before = (folder / MANIFEST_FILE).read_bytes()
        assert fetch_dataset(dataset, tmp_path / "datasets") == folder
        assert (folder / MANIFEST_FILE).read_bytes() == before

        # A hand edit next to a task the registry adds would be recorded as if fetched; it is refused first.
        (folder / "a" / "instruction.md").write_text("edited locally\n")
        write_task(registry_repo / "tasks" / "c")
        commit_repo(registry_repo)
        grown = RegistryDataset(
            "ds", "1.0", "", (*dataset.tasks, RegistryTask("c", registry_repo.as_uri(), "HEAD", "tasks/c"))
        )
        with pytest.raises(HubError, match="edited by hand"):
            fetch_dataset(grown, tmp_path / "datasets")
        assert (folder / MANIFEST_FILE).read_bytes() == before and not (folder / "c").exists()

    def test_package_store_fetch_records_digests_and_refuses_a_stale_latest_folder(self, tmp_path, monkeypatch):
        store_tests = test_harbor_package_store.TestPackageStore()
        folders = {name: write_task(tmp_path / "src" / name) for name in ("alpha", "beta")}
        module, digests, _ = store_tests.fake_store(tmp_path, monkeypatch, folders)
        ref = module.PackageRef("terminal-bench", "terminal-bench", "4.0.0")
        folder = module.fetch_package_dataset(ref, tmp_path / "datasets")
        manifest = manifest_of(folder)
        assert manifest["dataset"]["source"] == "harbor-package-store"
        assert manifest["tasks"]["alpha"] == {
            "digest": digests["alpha"],
            "package": "terminal-bench/alpha",
            "content_hash": f"sha256:{digests['alpha']}",
        }
        assert check_manifest(folder) == Manifest.read(folder)

        # The store's version moved (beta gone, gamma new): the folder is stale, not silently mixed.
        moved = {"alpha": folders["alpha"], "gamma": write_task(tmp_path / "src" / "gamma")}
        store_tests.fake_store(tmp_path, monkeypatch, moved)
        with pytest.raises(module.PackageStoreError, match="another version") as info:
            module.fetch_package_dataset(ref, tmp_path / "datasets")
        assert "added 1, removed 1" in str(info.value) and "move it aside" in str(info.value)
        assert not (folder / "gamma").exists() and (folder / "beta").is_dir()


class TestGymIdentityKeys:
    """`[gym]` gains `license` and `type`; the loader validates them and nothing about a task changes."""

    def test_license_and_type_parse_with_gym_rules(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        (folder / "dataset.toml").write_text('[gym]\nlicense = "Apache 2.0"\ntype = "train"\n')
        config = read_dataset_config(folder)
        assert config.license == "Apache 2.0" and config.type == "train" and config.effective_type == "train"
        assert not config.is_empty and not config.shapes_tasks
        task = load_task(folder / "a")
        assert apply_dataset_config(task.config, "a", config) is task.config

    @pytest.mark.parametrize("split", ["train", "validation"])
    def test_training_splits_require_a_license(self, tmp_path, split):
        folder = tmp_path / "ds"
        folder.mkdir()
        (folder / "dataset.toml").write_text(f'[gym]\ntype = "{split}"\n')
        with pytest.raises(ValueError, match="requires a license"):
            read_dataset_config(folder)

    def test_unknown_license_or_split_is_refused(self, tmp_path):
        folder = tmp_path / "ds"
        folder.mkdir()
        (folder / "dataset.toml").write_text('[gym]\nlicense = "WTFPL"\n')
        with pytest.raises(ValueError, match="license"):
            read_dataset_config(folder)
        (folder / "dataset.toml").write_text('[gym]\ntype = "test"\n')
        with pytest.raises(ValueError, match="type"):
            read_dataset_config(folder)

    def test_defaults_follow_the_rfc(self):
        assert GymDatasetConfig().effective_type == "benchmark"
        assert GymDatasetConfig(license="MIT").effective_type == "validation"
        assert GymDatasetConfig(type="example").effective_type == "example"
        assert GymDatasetConfig().is_empty

    def test_scaffold_documents_the_keys_without_setting_them(self, tmp_path):
        from nemo_gym.data.tasks.harbor.scaffold import init_dataset

        folder = init_dataset(tmp_path, "my-dataset")
        text = (folder / "dataset.toml").read_text()
        assert "[gym]" in text and '# type = "validation"' in text and '# license = "Apache 2.0"' in text
        assert read_dataset_config(folder).is_empty


class TestTrainingMetadataKeys:
    """`[gym]` gains `allowed_agents`, `domain`, `verified` and `[gym.defaults].num_repeats` (M37): each is
    validated by the loader, none shapes a task, and the table is no longer empty with any of them set."""

    def write(self, tmp_path, text: str):
        folder = tmp_path / "ds"
        if not folder.exists():
            write_task(folder / "a")
        (folder / "dataset.toml").write_text(text)
        return folder

    def test_keys_parse_and_leave_tasks_alone(self, tmp_path):
        from nemo_gym.config import Domain

        folder = self.write(
            tmp_path,
            '[gym]\nallowed_agents = ["hermes_agent"]\ndomain = "coding"\nverified = true\n'
            "[gym.defaults]\nnum_repeats = 4\n",
        )
        config = read_dataset_config(folder)
        assert config.allowed_agents == ["hermes_agent"] and config.domain is Domain.CODING
        assert config.verified is True and config.defaults.num_repeats == 4
        assert config.admits("hermes_agent") and not config.admits("terminus_2_sandboxed_agent")
        assert not config.is_empty and not config.shapes_tasks
        task = load_task(folder / "a")
        assert apply_dataset_config(task.config, "a", config) is task.config

    def test_absent_keys_admit_any_agent_and_run_once(self):
        config = GymDatasetConfig()
        assert config.allowed_agents is None and config.admits("anything")
        assert config.domain is None and config.verified is False and config.defaults.num_repeats == 1

    @pytest.mark.parametrize(
        "key",
        ["allowed_agents", "domain", "verified", "num_repeats"],
    )
    def test_each_key_alone_marks_the_table_as_set(self, key):
        value = {"allowed_agents": ["a"], "domain": "math", "verified": True}.get(key)
        data = {"defaults": {"num_repeats": 2}} if key == "num_repeats" else {key: value}
        config = GymDatasetConfig.model_validate(data)
        assert not config.is_empty and not config.shapes_tasks

    @pytest.mark.parametrize(
        ("text", "fragment"),
        [
            ('[gym]\nallowed_agents = "hermes_agent"\n', "allowed_agents"),  # a bare string, not a list
            ("[gym]\nallowed_agents = []\n", "names no agent"),
            ('[gym]\nallowed_agents = ["a", "a"]\n', "more than once"),
            ('[gym]\nallowed_agents = [" a"]\n', "is not an agent name"),
            (
                '[gym]\ndomain = "cooking"\n',
                "domain must be one of Gym's domain values .math, coding, .*not 'cooking'",
            ),
            ('[gym]\nverified = "yes"\n', "verified"),
            ("[gym.defaults]\nnum_repeats = 0\n", "num_repeats"),
            ('[gym.defaults]\nnum_repeats = "2"\n', "num_repeats"),
            ("[gym.defaults]\nnum_repeats = 1.5\n", "num_repeats"),
            ("[gym]\nverfied = true\n", "verfied"),  # misspelt keys stay errors
        ],
    )
    def test_bad_values_are_refused_with_the_file_named(self, tmp_path, text, fragment):
        folder = self.write(tmp_path, text)
        with pytest.raises(ValueError, match=fragment) as exc:
            read_dataset_config(folder)
        assert str(folder / "dataset.toml") in str(exc.value)


def registry_of(repo: Path, names: tuple[str, ...]) -> RegistryDataset:
    return RegistryDataset(
        "ds", "1.0", "", tuple(RegistryTask(name, repo.as_uri(), "HEAD", f"tasks/{name}") for name in names)
    )


class TestFetchersRecordOnlyWhatTheyFetched:
    """A folder nobody fetched is never blessed: the fetch refuses it and `Manifest.build` records named tasks only."""

    def test_build_with_names_records_those_tasks_and_hashes_nothing_it_knows(self, tmp_path, monkeypatch):
        import nemo_gym.data.tasks.harbor.manifest as module

        folder = tmp_path / "ds"
        for name in ("a", "b", "stray"):
            write_task(folder / name)
        hashed: list[str] = []
        real = module.content_hash
        monkeypatch.setattr(module, "content_hash", lambda path: hashed.append(path.name) or real(path))
        digests = {"a": real(folder / "a")}
        pins = {"pending": {"package": "org/pending", "content_hash": "sha256:" + "d" * 64}}
        manifest = Manifest.build(folder, source="harbor-package-store", pins=pins, digests=digests, names=["a", "b"])
        assert set(manifest.tasks) == {"a", "b", "pending"} and "stray" not in manifest.tasks
        assert hashed == ["b"]  # `a` was known, `pending` has no folder, `stray` was not named
        assert manifest.task_digests["pending"] is None
        # Without names, every folder on disk is recorded (prepare recomputing a folder to compare).
        assert set(Manifest.build(folder, source="local").tasks) == {"a", "b", "stray"}

    def test_registry_fetch_refuses_a_stray_folder_instead_of_recording_it(self, tmp_path, registry_repo):
        dataset = registry_of(registry_repo, ("a", "b"))
        folder = fetch_dataset(dataset, tmp_path / "datasets")
        write_task(folder / "evil")
        before = (folder / MANIFEST_FILE).read_bytes()
        with pytest.raises(ManifestError, match="not in the manifest: evil"):
            check_manifest(folder)
        # The registry adds a task: the fetch would rewrite the manifest, so the stray folder stops it first.
        write_task(registry_repo / "tasks" / "c")
        commit_repo(registry_repo)
        with pytest.raises(HubError, match="evil.*not in the registry and not recorded by a previous fetch") as info:
            fetch_dataset(registry_of(registry_repo, ("a", "b", "c")), tmp_path / "datasets")
        assert "move them aside" in str(info.value)
        assert (folder / MANIFEST_FILE).read_bytes() == before and not (folder / "c").exists()
        # Moved aside, the fetch completes and records the registry's tasks only.
        (folder / "evil").rename(tmp_path / "evil-aside")
        fetch_dataset(registry_of(registry_repo, ("a", "b", "c")), tmp_path / "datasets")
        assert set(manifest_of(folder)["tasks"]) == {"a", "b", "c"}
        assert check_manifest(folder) == Manifest.read(folder)

    def test_registry_fetch_scans_once_and_hashes_each_task_once(self, tmp_path, registry_repo, monkeypatch):
        import nemo_gym.data.tasks.harbor.hub as hub
        import nemo_gym.data.tasks.harbor.manifest as module

        scans: list[str] = []
        hashed: list[str] = []
        real_scan, real_hash = module.task_folders, module.content_hash
        monkeypatch.setattr(module, "task_folders", lambda folder: scans.append("build") or real_scan(folder))
        monkeypatch.setattr(hub, "task_folders", lambda folder: scans.append("fetch") or real_scan(folder))
        monkeypatch.setattr(module, "content_hash", lambda path: hashed.append(path.name) or real_hash(path))
        folder = fetch_dataset(registry_of(registry_repo, ("a", "b")), tmp_path / "datasets")
        # One scan for the stray-folder check; the per-task checkpoints neither scan nor rehash.
        assert scans == ["fetch"] and sorted(hashed) == ["a", "b"]
        assert set(manifest_of(folder)["tasks"]) == {"a", "b"}

    def test_package_store_fetch_refuses_a_stray_folder_present_at_the_first_fetch(self, tmp_path, monkeypatch):
        store_tests = test_harbor_package_store.TestPackageStore()
        folders = {name: write_task(tmp_path / "src" / name) for name in ("alpha", "beta")}
        module, _, _ = store_tests.fake_store(tmp_path, monkeypatch, folders)
        ref = module.PackageRef("terminal-bench", "terminal-bench", "4.0.0")
        folder = tmp_path / "datasets" / ref.folder_name(module.PackageStore().resolve_dataset(ref).content_hash)
        write_task(folder / "evil")
        with pytest.raises(module.PackageStoreError, match="evil.*not in the store") as info:
            module.fetch_package_dataset(ref, tmp_path / "datasets")
        assert "move them aside" in str(info.value)
        assert sorted(p.name for p in folder.iterdir()) == ["evil"]  # nothing fetched, nothing written
        (folder / "evil").rename(tmp_path / "evil-aside")
        assert module.fetch_package_dataset(ref, tmp_path / "datasets") == folder
        assert set(manifest_of(folder)["tasks"]) == {"alpha", "beta"}

    def test_an_unpinned_manifest_entry_is_not_a_removed_task(self, tmp_path, monkeypatch):
        store_tests = test_harbor_package_store.TestPackageStore()
        folders = {name: write_task(tmp_path / "src" / name) for name in ("alpha", "beta")}
        module, _, _ = store_tests.fake_store(tmp_path, monkeypatch, folders)
        ref = module.PackageRef("terminal-bench", "terminal-bench", "4.0.0")
        folder = module.fetch_package_dataset(ref, tmp_path / "datasets")
        with (folder / MANIFEST_FILE).open("a") as stream:
            stream.write('\n[tasks."stray"]\ndigest = "' + "9" * 64 + '"\n')
        assert module._stale_reason(Manifest.read(folder), module.PackageStore().resolve_dataset(ref)) is None
        assert module.fetch_package_dataset(ref, tmp_path / "datasets") == folder
        assert set(manifest_of(folder)["tasks"]) == {"alpha", "beta"}


class TestCheckManifestFailsClosed:
    def test_newer_schema_manifest_stops_the_registry_fetch_untouched(self, tmp_path, registry_repo):
        """A manifest a newer Gym wrote is not "unreadable": the registry fetch must not replace it with a
        schema-1 one that drops what it carried, so it stops before it clones anything."""
        folder = fetch_dataset(registry_of(registry_repo, ("a",)), tmp_path / "datasets")
        path = folder / MANIFEST_FILE
        good = path.read_text()
        assert good.startswith("schema_version = 1\n")
        for version in ("2", "1.0"):
            newer = good.replace("schema_version = 1\n", f"schema_version = {version}\n", 1)
            newer += '\n[future]\nkey = "x"\n'
            path.write_text(newer)
            # This fetch would add `b` and rewrite the manifest; it stops at the manifest instead.
            with pytest.raises(UnsupportedManifestError, match=f"schema_version {version}; this Gym reads 1"):
                fetch_dataset(registry_of(registry_repo, ("a", "b")), tmp_path / "datasets")
            assert path.read_text() == newer
            assert not (folder / "b").exists()
            assert sorted(p.name for p in folder.iterdir()) == ["a", MANIFEST_FILE]
        # Restored, the same folder fetches again: the manifest was intact all along.
        path.write_text(good)
        assert fetch_dataset(registry_of(registry_repo, ("a", "b")), tmp_path / "datasets") == folder
        assert set(Manifest.read(folder).tasks) == {"a", "b"}

    def test_unknown_source_is_refused_with_or_without_edits_and_nothing_rewritten(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        fetched_manifest(folder, ["a"])
        path = folder / MANIFEST_FILE
        path.write_text(path.read_text().replace('source = "harbor-registry"', 'source = "gym-hub"'))
        before = path.read_bytes()
        with pytest.raises(ManifestError, match="names source 'gym-hub', which this Gym does not know"):
            check_manifest(folder)
        (folder / "a" / "instruction.md").write_text("edited\n")
        with pytest.raises(ManifestError, match="does not know"):
            check_manifest(folder)
        assert path.read_bytes() == before
        path.write_text(path.read_text().replace('source = "gym-hub"', 'source = ""'))
        with pytest.raises(ManifestError, match="names source ''"):
            check_manifest(folder)

    def test_a_non_table_dataset_or_newer_schema_is_refused(self, tmp_path):
        (tmp_path / MANIFEST_FILE).write_text("dataset = []\n")
        with pytest.raises(ManifestError, match="must be tables"):
            Manifest.read(tmp_path)
        (tmp_path / MANIFEST_FILE).write_text('schema_version = 2\n\n[dataset]\nsource = "local"\n')
        with pytest.raises(UnsupportedManifestError, match="schema_version 2; this Gym reads 1") as info:
            Manifest.read(tmp_path)
        assert isinstance(info.value, ManifestError)  # the CLI's error tuple and the fetchers' catch still see it
        (tmp_path / MANIFEST_FILE).write_text('schema_version = 1.0\n\n[dataset]\nsource = "local"\n')
        with pytest.raises(UnsupportedManifestError, match="schema_version 1.0; this Gym reads 1"):
            Manifest.read(tmp_path)
        # Absent means 1: the pins-only manifests written before the schema was numbered still read.
        (tmp_path / MANIFEST_FILE).write_text('[dataset]\nsource = "local"\n')
        assert Manifest.read(tmp_path).source == "local"

    def test_a_deleted_pinned_folder_is_reported_as_missing(self, tmp_path, capsys):
        folder = tmp_path / "ds"
        for name in ("a", "b"):
            write_task(folder / name)
        fetched_manifest(folder, ["a", "b"])
        shutil.rmtree(folder / "b")
        with pytest.raises(ManifestError, match=r"\(missing: b\)") as info:
            check_manifest(folder)
        assert "changed" not in str(info.value) and "deleted" in str(info.value)
        assert task_differences({"a": "x", "b": "y"}, {"a": "x", "b": None}) == "missing: b"
        assert task_differences({"a": "x", "b": "y"}, {"a": "x"}) == "missing: b"
        assert task_differences({"a": "x", "b": None}, {"a": "x"}) is None  # pinned before its folder existed

        local = tmp_path / "local"
        for name in ("a", "b"):
            write_task(local / name)
        check_manifest(local)
        shutil.rmtree(local / "b")
        check_manifest(local)
        assert "task folders changed (missing: b)" in capsys.readouterr().out


class TestManifestQuoting:
    def test_non_bmp_names_round_trip_through_toml(self, tmp_path, capsys):
        folder = tmp_path / "ds"
        name = "task-\U0001f600-中"
        write_task(folder / name)
        pins = {name: {"git_url": "https://x.test/r.git", "git_commit_id": "a" * 40, "path": "tasks/\U0010ffff/\x7f"}}
        manifest = Manifest.build(folder, source="harbor-registry", pins=pins)
        manifest.write(folder)
        text = (folder / MANIFEST_FILE).read_text(encoding="utf-8")
        assert text.isascii() and "\\U0001f600" in text and "\\U0010ffff" in text and "\\u007f" in text
        assert Manifest.read(folder) == manifest
        assert check_manifest(folder) == manifest and check_manifest(folder) == manifest
        assert capsys.readouterr().out == ""


@pytest.mark.usefixtures("fake_registry")
class TestPrepareOrderAndTrackedFolders:
    def test_prepare_refuses_a_hand_edited_fetched_dataset_before_any_build_or_push(self, tmp_path, monkeypatch):
        import nemo_gym.data.tasks.harbor.prepare as harbor_cli

        calls: list[str] = []
        monkeypatch.setattr(harbor_cli, "record_base_images", lambda *a, **k: calls.append("images"))
        monkeypatch.setattr(harbor_cli, "_build_dockerfiles", lambda *a, **k: calls.append("build"))
        folder = tmp_path / "ds"
        for name in ("a", "b"):
            write_task(folder / name)
        fetched_manifest(folder, ["a", "b"])
        (folder / "a" / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\nRUN echo tampered\n")
        with pytest.raises(ManifestError, match="edited by hand"):
            prepare_target(
                str(folder),
                output_root=tmp_path / "out",
                build=BuildOptions(tool="podman", repository="reg.example/r"),
            )
        assert calls == [] and not (folder / "built-images.json").exists() and not (tmp_path / "out").exists()

    def test_a_git_tracked_folder_gets_its_identity_in_memory_and_no_manifest(self, tmp_path, capsys):
        repo = tmp_path / "repo"
        folder = repo / "data" / "tasks"
        for name in ("a", "b"):
            write_task(folder / name)
        subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
        identity = check_manifest(folder)
        assert identity == Manifest.build(folder, source="local") and not (folder / MANIFEST_FILE).exists()
        assert "computed in memory: git tracks" in capsys.readouterr().out
        prepared = prepare_target(str(folder), output_root=tmp_path / "out")
        assert prepared.dataset_digest == identity.digest and not (folder / MANIFEST_FILE).exists()
        rows = [json.loads(line) for line in prepared.rows_path.read_text().splitlines()]
        assert {row["task_input"]["task_data"][DATASET_DIGEST_KEY] for row in rows} == {identity.digest}
        # An untracked folder in the same repository still gets its manifest.
        untracked = repo / "scratch"
        write_task(untracked / "a")
        check_manifest(untracked)
        assert (untracked / MANIFEST_FILE).is_file()

    @pytest.mark.skipif(os.geteuid() == 0, reason="root writes anywhere")
    def test_a_read_only_local_folder_keeps_its_identity_in_memory(self, tmp_path, caplog):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        folder.chmod(0o555)
        try:
            identity = check_manifest(folder)
        finally:
            folder.chmod(0o755)
        assert identity == Manifest.build(folder, source="local") and not (folder / MANIFEST_FILE).exists()
        assert "Could not write" in caplog.text and identity.digest[:12] in caplog.text


class TestResultsRowsCarryTheDatasetDigest:
    async def test_written_rollout_rows_carry_the_task_and_dataset_digests(self, tmp_path, monkeypatch):
        """The collector copies a materialized row's `_ng_*` task data onto the record it writes, result or failure."""
        monkeypatch.setattr(nemo_gym.rollouts.rollout_collection, "get_global_config_dict", lambda: {})
        task_data = {DIGEST_KEY: "1" * 64, DATASET_DIGEST_KEY: "2" * 64}
        rows = [
            {
                "task_id": {"taskset": "swe_pro", "task_id": name},
                "task_input": {
                    "responses_create_params": {"input": [{"role": "user", "content": "go"}]},
                    "task_data": task_data,
                },
            }
            for name in ("a", "b")
        ]
        input_fpath = tmp_path / "input.jsonl"
        input_fpath.write_text("".join(json.dumps(row) + "\n" for row in rows))
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_fpath),
            output_jsonl_fpath=str(tmp_path / "rollouts.jsonl"),
            environment_routing_mode="taskset",
            environment_server_routes={"swe_pro": "environment"},
            disable_aggregation=True,
            disable_health_check=True,
        )
        mock_client = MagicMock()
        mock_client.global_config_dict = collection_tests._environment_server_config()
        replies = {
            "a": {"result": {"responses_create_params": {"input": []}, "response": {"output": []}, "reward": 1.0}},
            "b": {"failure": {"stage": "agent", "failure_reason": "boom", "terminal": False}},
        }

        class Helper(RolloutCollectionHelper):
            def setup_server_client(self, head_server_config=None):
                return mock_client

            def _run_examples_with_metadata(self, examples, *args, **kwargs):
                futures = []
                for row in examples:
                    identity = {"episode_id": {"rollout": 0}, "task_id": row["task_id"]}
                    future = Future()
                    future.set_result(
                        _CompletedRollout(
                            row=row,
                            result=identity | replies[row["task_id"]["task_id"]],
                            rollout_latency_ms=None,
                            environment_server="environment",
                        )
                    )
                    futures.append(future)
                return futures

        await Helper().run_from_config(config)

        [result] = [json.loads(line) for line in (tmp_path / "rollouts.jsonl").read_text().splitlines()]
        assert result["reward"] == 1.0 and "task_input" not in result
        assert result[DATASET_DIGEST_KEY] == "2" * 64 and result[DIGEST_KEY] == "1" * 64
        failures = Path(str(tmp_path / "rollouts.jsonl").replace(".jsonl", "_failures.jsonl"))
        [failure] = [json.loads(line) for line in failures.read_text().splitlines()]
        assert failure[DATASET_DIGEST_KEY] == "2" * 64 and failure["_ng_failure_message"] == "boom"


def test_a_task_folder_shipped_inside_another_task_is_held_to_nothing(tmp_path):
    """The dataset folder's check never descends into a task folder it found, so a task-shaped fixture a fetched
    task ships under its tests/ (covered by that task's digest) is neither a stray nor held to the fetched manifest
    when prepared on its own, and the folder holding it is a plain local dataset, not a subfolder of the fetched one."""
    folder = tmp_path / "ds"
    write_task(folder / "a")
    write_task(folder / "b")
    inner = write_task(folder / "a" / "tests" / "fixtures" / "inner")
    fetched_manifest(folder, ["a", "b"])
    assert iter_task_folders(folder) == [folder / "a", folder / "b"]
    assert check_task_manifest(inner) is None
    assert check_manifest(folder / "a" / "tests" / "fixtures").source == "local"
    assert check_task_manifest(folder / "b") is not None
