# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests of the Harbor package store client.

Dataset resolution by tag or digest, task download and verification, and the manifest the store
copy writes and checks on the next fetch.
"""

import json
import logging
import shutil
from pathlib import Path

import pytest

from nemo_gym.tasks.harbor import content_hash, discover_tasks
from tests.unit_tests.test_harbor_tasks import (
    no_real_registry,  # noqa: F401 (autouse fixture)
    write_task,
)


class TestPackageStore:
    """The package store is faked at the HTTP layer: one handler per REST path."""

    @staticmethod
    def make_archive(folder: Path) -> bytes:
        import io
        import tarfile

        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            tar.add(folder, arcname=".")
        return buffer.getvalue()

    def fake_store(self, tmp_path, monkeypatch, task_folders: dict[str, Path], tag="4.0.0", publish=content_hash):
        """``publish`` hashes a source folder into the hash the store advertises for it (Harbor's archive hash
        equals Gym's folder digest unless the folder holds a name Gym's digest skips)."""
        from nemo_gym.tasks.harbor import package_store as module

        digests = {name: publish(folder) for name, folder in task_folders.items()}
        calls = []

        def request(self, method, path, *, params=None, body=None):
            calls.append((method, path, params, body))
            if path == "/rest/v1/dataset_version_tag":
                assert params["tag"] == f"eq.{tag}" and params["package.name"] == "eq.terminal-bench"
                return json.dumps(
                    [{"dataset_version": {"id": "dv-1", "content_hash": "d" * 64}, "package": {}}]
                ).encode()
            if path == "/rest/v1/dataset_version":
                return json.dumps([{"id": "dv-1", "content_hash": "d" * 64, "package": {}}]).encode()
            if path == "/rest/v1/dataset_version_task":
                assert params["dataset_version_id"] == "eq.dv-1"
                if params["offset"] != "0":
                    return b"[]"
                rows = [
                    {
                        "task_version_id": f"tv-{name}",
                        "task_version": {
                            "content_hash": digest,
                            "package": {"name": name, "org": {"name": "terminal-bench"}},
                        },
                    }
                    for name, digest in sorted(digests.items())
                ]
                return json.dumps(rows).encode()
            if path == "/rest/v1/rpc/resolve_task_version":
                name = body["p_name"]
                assert body["p_ref"] == f"sha256:{digests[name]}"
                return json.dumps(
                    {"content_hash": f"sha256:{digests[name]}", "archive_path": f"pkgs/{name}.tar.gz"}
                ).encode()
            if path.startswith("/storage/v1/object/packages/pkgs/"):
                name = path.rsplit("/", 1)[1].removesuffix(".tar.gz")
                return self.make_archive_for(name)
            raise AssertionError(f"unexpected request {method} {path}")

        module.PackageStore.make_archive_for = staticmethod(lambda name: self.make_archive(task_folders[name]))
        monkeypatch.setattr(module.PackageStore, "_request", request)
        return module, digests, calls

    def test_fetches_tags_checks_digests_and_writes_manifest(self, tmp_path, monkeypatch):
        import tomllib

        folders = {name: write_task(tmp_path / "src" / name) for name in ("beta", "alpha")}
        module, digests, calls = self.fake_store(tmp_path, monkeypatch, folders)

        folder = module.fetch_package_dataset(
            module.PackageRef("terminal-bench", "terminal-bench", "4.0.0"), tmp_path / "datasets"
        )

        assert folder == tmp_path / "datasets" / "terminal-bench-4.0.0"
        assert sorted(p.name for p in folder.iterdir() if p.is_dir()) == ["alpha", "beta"]
        assert content_hash(folder / "alpha") == digests["alpha"]
        manifest = tomllib.loads((folder / "manifest.toml").read_text())
        assert manifest["dataset"] == {
            "name": "terminal-bench/terminal-bench",
            "version": "4.0.0",
            "source": "harbor-package-store",
            "content_hash": "sha256:" + "d" * 64,
        }
        assert manifest["tasks"]["beta"] == {
            "package": "terminal-bench/beta",
            "content_hash": f"sha256:{digests['beta']}",
            "digest": f"sha256:{digests['beta']}",
        }
        # A second fetch downloads nothing: every folder's digest already matches.
        downloads_before = sum(1 for c in calls if c[1].startswith("/storage/"))
        module.fetch_package_dataset(
            module.PackageRef("terminal-bench", "terminal-bench", "4.0.0"), tmp_path / "datasets"
        )
        assert sum(1 for c in calls if c[1].startswith("/storage/")) == downloads_before
        # The tasks load like any local folder.
        assert [task.task_id for task in discover_tasks(folder)] == ["alpha", "beta"]

    def test_digest_mismatch_is_rejected(self, tmp_path, monkeypatch):
        folders = {"alpha": write_task(tmp_path / "src" / "alpha")}
        module, _, _ = self.fake_store(tmp_path, monkeypatch, folders)
        # The archive the store serves differs from the digest it advertised.
        (folders["alpha"] / "instruction.md").write_text("tampered\n")
        module.PackageStore.make_archive_for = staticmethod(lambda name: self.make_archive(folders[name]))
        with pytest.raises(module.PackageStoreError, match="content hash mismatch"):
            module.fetch_package_dataset(
                module.PackageRef("terminal-bench", "terminal-bench", "4.0.0"), tmp_path / "datasets"
            )
        assert not (tmp_path / "datasets" / "terminal-bench-4.0.0" / "alpha" / "task.toml").exists()

    @pytest.mark.parametrize("body", ["html", "json", "truncated"])
    def test_unreadable_archive_is_a_package_store_error(self, tmp_path, monkeypatch, body):
        """A store reply that is not a gzip tar (an error page, or a truncated stream) must surface through the
        CLI's normal error path, a PackageStoreError (a ValueError), not as a raw tarfile.ReadError or EOFError."""
        folders = {"alpha": write_task(tmp_path / "src" / "alpha")}
        module, _, _ = self.fake_store(tmp_path, monkeypatch, folders)
        real = self.make_archive(folders["alpha"])
        served = {
            "html": b"<html><body>502 Bad Gateway</body></html>",
            "json": b'{"error": "not found"}',
            "truncated": real[: len(real) // 2],
        }[body]
        module.PackageStore.make_archive_for = staticmethod(lambda name: served)
        with pytest.raises(module.PackageStoreError, match="terminal-bench/alpha archive .* not a readable tar.gz"):
            module.fetch_package_dataset(
                module.PackageRef("terminal-bench", "terminal-bench", "4.0.0"), tmp_path / "datasets"
            )
        folder = tmp_path / "datasets" / "terminal-bench-4.0.0"
        assert not (folder / "alpha" / "task.toml").exists()
        # The staging directory went away with the error.
        assert [p.name for p in folder.iterdir() if p.name.startswith(".nemo-gym-package-")] == []

    @pytest.mark.parametrize("command", ["eval run", "dataset validate"])
    def test_unreadable_archive_is_reported_by_eval_run_and_dataset_validate(
        self, tmp_path, monkeypatch, capsys, command
    ):
        """Through ``main()`` with only the store's HTTP replies faked: an error page where the archive should be ends
        `gym eval run` and `gym dataset validate` with parser.error's one-line message and exit status 2, not a
        traceback (fetch_ref raises the store's error as a HubError, which both entry points report)."""
        import sys

        from nemo_gym.cli.main import main

        folders = {"alpha": write_task(tmp_path / "src" / "alpha")}
        module, _, _ = self.fake_store(tmp_path, monkeypatch, folders)
        module.PackageStore.make_archive_for = staticmethod(lambda name: b"<html><body>502 Bad Gateway</body></html>")
        monkeypatch.setenv("NEMO_GYM_DATASETS_DIR", str(tmp_path / "datasets"))
        agent = ["--agent", "hermes_agent"] if command == "eval run" else []
        monkeypatch.setattr(
            sys, "argv", ["gym", *command.split(), "harbor:terminal-bench/terminal-bench@4.0.0", *agent]
        )
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "terminal-bench/alpha archive from the store is not a readable tar.gz package" in err
        assert "Traceback" not in err
        assert not (tmp_path / "datasets" / "terminal-bench-4.0.0" / "alpha").exists()

    def test_spoofed_package_name_is_rejected_before_any_filesystem_use(self, tmp_path, monkeypatch):
        folders = {"../..": write_task(tmp_path / "src" / "evil")}
        module, _, calls = self.fake_store(tmp_path, monkeypatch, folders)
        removed = []
        monkeypatch.setattr(shutil, "rmtree", lambda path, *args, **kwargs: removed.append(Path(path)))
        with pytest.raises(module.PackageStoreError, match=r"'\.\./\.\.'"):
            module.fetch_package_dataset(
                module.PackageRef("terminal-bench", "terminal-bench", "4.0.0"), tmp_path / "datasets"
            )
        assert removed == []
        assert not (tmp_path / "datasets").exists()
        assert not any(c[1].startswith("/storage/") for c in calls)

    def test_never_deletes_outside_the_dataset_folder(self, tmp_path, monkeypatch):
        folders = {"alpha": write_task(tmp_path / "src" / "alpha")}
        module, _, _ = self.fake_store(tmp_path, monkeypatch, folders)
        # `alpha` in the dataset folder is a link to a folder elsewhere that holds no task (no task.toml), which
        # the fetch would otherwise replace with the store's copy.
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "instruction.md").write_text("edited locally\n")
        folder = tmp_path / "datasets" / "terminal-bench-4.0.0"
        folder.mkdir(parents=True)
        (folder / "alpha").symlink_to(outside, target_is_directory=True)
        real_rmtree, removed = shutil.rmtree, []

        def recording_rmtree(path, *args, **kwargs):
            removed.append(Path(path))
            return real_rmtree(path, *args, **kwargs)

        monkeypatch.setattr(shutil, "rmtree", recording_rmtree)
        with pytest.raises(module.PackageStoreError, match="Task destination must not be a symlink"):
            module.fetch_package_dataset(
                module.PackageRef("terminal-bench", "terminal-bench", "4.0.0"), tmp_path / "datasets"
            )
        # `download_task` itself refuses the link too, so a caller that skips the fetch's checks is still safe.
        task = module.PackageTask("terminal-bench", "alpha", content_hash(folders["alpha"]))
        with pytest.raises(module.PackageStoreError, match="outside the dataset folder"):
            module.PackageStore().download_task(task, folder / "alpha")
        # Only the staging directory inside the dataset folder was ever removed; the link and its target were not.
        assert folder / "alpha" not in removed and outside not in removed
        assert all(path.is_relative_to(folder) for path in removed)
        assert (outside / "instruction.md").read_text() == "edited locally\n"
        assert (folder / "alpha").is_symlink()

    def test_linked_dataset_folder_and_manifest_are_refused(self, tmp_path, monkeypatch):
        folders = {"alpha": write_task(tmp_path / "src" / "alpha")}
        module, _, calls = self.fake_store(tmp_path, monkeypatch, folders)
        ref = module.PackageRef("terminal-bench", "terminal-bench", "4.0.0")
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (tmp_path / "datasets").mkdir()
        (tmp_path / "datasets" / "terminal-bench-4.0.0").symlink_to(elsewhere, target_is_directory=True)
        with pytest.raises(module.PackageStoreError, match="Dataset folder must not be a symlink"):
            module.fetch_package_dataset(ref, tmp_path / "datasets")
        assert list(elsewhere.iterdir()) == []

        (tmp_path / "datasets" / "terminal-bench-4.0.0").unlink()
        folder = tmp_path / "datasets" / "terminal-bench-4.0.0"
        folder.mkdir()
        precious = tmp_path / "precious.toml"
        precious.write_text("keep = true\n")
        (folder / "manifest.toml").symlink_to(precious)
        with pytest.raises(module.PackageStoreError, match="Dataset manifest must not be a symlink"):
            module.fetch_package_dataset(ref, tmp_path / "datasets")
        assert precious.read_text() == "keep = true\n"
        assert not any(c[1].startswith("/storage/") for c in calls)

    def test_folder_without_task_toml_is_not_replaced(self, tmp_path, monkeypatch):
        folders = {"alpha": write_task(tmp_path / "src" / "alpha")}
        module, _, calls = self.fake_store(tmp_path, monkeypatch, folders)
        folder = tmp_path / "datasets" / "terminal-bench-4.0.0"
        (folder / "alpha").mkdir(parents=True)
        (folder / "alpha" / "notes.txt").write_text("my notes\n")
        with pytest.raises(module.PackageStoreError, match=r"is not a fetched task \(no task.toml\); move it aside"):
            module.fetch_package_dataset(
                module.PackageRef("terminal-bench", "terminal-bench", "4.0.0"), tmp_path / "datasets"
            )
        assert sorted(p.name for p in (folder / "alpha").iterdir()) == ["notes.txt"]
        assert (folder / "alpha" / "notes.txt").read_text() == "my notes\n"
        assert not (folder / "manifest.toml").exists()
        assert not any(c[1].startswith("/storage/") for c in calls)

    def test_duplicate_task_names_are_refused_before_writing(self, tmp_path, monkeypatch):
        from nemo_gym.tasks.harbor import package_store as module

        ref = module.PackageRef("terminal-bench", "terminal-bench", "4.0.0")
        dataset = module.ResolvedDataset(
            ref=ref,
            version_id="dv-1",
            content_hash="d" * 64,
            tasks=(
                module.PackageTask("alice", "same", "a" * 64),
                module.PackageTask("bob", "same", "b" * 64),
                module.PackageTask("alice", "other", "c" * 64),
            ),
        )
        monkeypatch.setattr(module.PackageStore, "resolve_dataset", lambda self, ref: dataset)
        monkeypatch.setattr(
            module.PackageStore, "download_task", lambda *a, **k: pytest.fail("nothing may be downloaded")
        )
        with pytest.raises(module.PackageStoreError, match="tasks sharing a folder name: same$"):
            module.fetch_package_dataset(ref, tmp_path / "datasets")
        assert not (tmp_path / "datasets").exists()

    def test_matching_concurrent_fetch_result_is_kept(self, tmp_path, monkeypatch):
        import os

        folders = {"alpha": write_task(tmp_path / "src" / "alpha")}
        module, digests, _ = self.fake_store(tmp_path, monkeypatch, folders)
        folder = tmp_path / "datasets" / "terminal-bench-4.0.0"
        target = folder / "alpha"
        real_archive = module.PackageStore.make_archive_for
        inode_before: list[int] = []

        def racing_archive(name):
            # A faster concurrent fetch finishes `alpha` while this one is still downloading the archive.
            shutil.copytree(folders[name], target)
            inode_before.append(os.stat(target).st_ino)
            return real_archive(name)

        module.PackageStore.make_archive_for = staticmethod(racing_archive)
        removed = []
        real_rmtree = shutil.rmtree
        monkeypatch.setattr(
            shutil, "rmtree", lambda path, *a, **k: (removed.append(Path(path)), real_rmtree(path, *a, **k))
        )
        module.fetch_package_dataset(
            module.PackageRef("terminal-bench", "terminal-bench", "4.0.0"), tmp_path / "datasets"
        )
        assert content_hash(target) == digests["alpha"] and (target / "task.toml").is_file()
        # The completed copy is the one that stays: it was neither removed nor swapped out (same inode).
        assert target not in removed and os.stat(target).st_ino == inode_before[0]

    def test_download_task_replaces_only_an_empty_destination(self, tmp_path, monkeypatch):
        """The promotion is one rename of the staging folder: it takes an empty folder's place, and a non-empty one
        that is not the store's copy stops the fetch untouched (nothing is moved away or deleted)."""
        folders = {"alpha": write_task(tmp_path / "src" / "alpha")}
        module, digests, _ = self.fake_store(tmp_path, monkeypatch, folders)
        folder = tmp_path / "datasets" / "terminal-bench-4.0.0"
        target = folder / "alpha"
        task = module.PackageTask("terminal-bench", "alpha", digests["alpha"])
        target.mkdir(parents=True)
        (target / "stale.txt").write_text("half of an earlier fetch\n")
        removed = []
        real_rmtree = shutil.rmtree
        monkeypatch.setattr(shutil, "rmtree", lambda path, *a, **k: (removed.append(Path(path)), real_rmtree(path)))

        with pytest.raises(module.PackageStoreError, match=rf"{target} appeared while fetching .* move it aside"):
            module.PackageStore().download_task(task, target)
        assert (target / "stale.txt").read_text() == "half of an earlier fetch\n"
        assert not (target / "task.toml").exists()
        # Only the staging directory was removed, never the destination.
        assert target not in removed and all(p.is_relative_to(folder) and p != target for p in removed)
        assert sorted(p.name for p in folder.iterdir()) == ["alpha"]

        (target / "stale.txt").unlink()
        module.PackageStore().download_task(task, target)
        assert content_hash(target) == digests["alpha"] and (target / "task.toml").is_file()
        assert sorted(p.name for p in folder.iterdir()) == ["alpha"]

    @pytest.mark.parametrize("exc", [OSError(122, "Disk quota exceeded"), KeyboardInterrupt()])
    def test_failed_rename_leaves_no_partial_destination(self, tmp_path, monkeypatch, exc):
        """A rename that fails (a quota error, an interrupt) propagates and leaves the destination absent: there is
        no copy fallback that could leave a partial folder, and no previous copy to lose."""
        folders = {"alpha": write_task(tmp_path / "src" / "alpha")}
        module, digests, _ = self.fake_store(tmp_path, monkeypatch, folders)
        folder = tmp_path / "datasets" / "terminal-bench-4.0.0"
        target = folder / "alpha"
        calls = []

        def failing_rename(src, dst, *a, **k):
            calls.append((Path(src), Path(dst)))
            raise exc

        monkeypatch.setattr(module.os, "rename", failing_rename)
        with pytest.raises(type(exc)):
            module.PackageStore().download_task(
                module.PackageTask("terminal-bench", "alpha", digests["alpha"]), target
            )
        assert calls == [(calls[0][0], target)] and calls[0][0].parent.parent == folder
        assert not target.exists() and not target.is_symlink()
        # The staging directory went away with the error: no leftovers beside the tasks.
        assert sorted(p.name for p in folder.iterdir()) == []

    def test_peer_copy_promoted_after_check_is_kept(self, tmp_path, monkeypatch):
        """Two complete fetches of the same dataset: the peer's copy of `alpha` lands between this fetch's checks and
        its promotion. The rename refuses the non-empty folder, the peer's copy is recognised and kept (same inode,
        never moved away), and the manifest records the folder's real digest."""
        import os
        import tomllib

        folders = {name: write_task(tmp_path / "src" / name) for name in ("alpha", "beta")}
        module, digests, _ = self.fake_store(tmp_path, monkeypatch, folders)
        folder = tmp_path / "datasets" / "terminal-bench-4.0.0"
        target = folder / "alpha"
        real_rename = os.rename
        peer_inode: list[int] = []

        def racing_rename(src, dst, *a, **k):
            if Path(dst) == target and not peer_inode:
                shutil.copytree(folders["alpha"], target)
                peer_inode.append(os.stat(target).st_ino)
            return real_rename(src, dst, *a, **k)

        monkeypatch.setattr(module.os, "rename", racing_rename)
        module.fetch_package_dataset(
            module.PackageRef("terminal-bench", "terminal-bench", "4.0.0"), tmp_path / "datasets"
        )
        assert peer_inode and os.stat(target).st_ino == peer_inode[0]
        assert content_hash(target) == digests["alpha"] and (target / "task.toml").is_file()
        assert sorted(p.name for p in folder.iterdir() if p.is_dir()) == ["alpha", "beta"]
        manifest = tomllib.loads((folder / "manifest.toml").read_text())
        assert manifest["tasks"]["alpha"]["digest"].removeprefix("sha256:") == digests["alpha"]
        assert [task.task_id for task in discover_tasks(folder)] == ["alpha", "beta"]

    def test_foreign_folder_appearing_during_fetch_stops_it(self, tmp_path, monkeypatch):
        """A folder with a task.toml that is not the store's copy appears at the destination between the checks and
        the promotion: the fetch stops with the path and leaves the foreign files alone."""
        import os

        folders = {"alpha": write_task(tmp_path / "src" / "alpha")}
        module, digests, _ = self.fake_store(tmp_path, monkeypatch, folders)
        folder = tmp_path / "datasets" / "terminal-bench-4.0.0"
        target = folder / "alpha"
        real_rename = os.rename
        landed = []

        def racing_rename(src, dst, *a, **k):
            if Path(dst) == target and not landed:
                write_task(target)
                (target / "instruction.md").write_text("someone else's task\n")
                landed.append(True)
            return real_rename(src, dst, *a, **k)

        monkeypatch.setattr(module.os, "rename", racing_rename)
        with pytest.raises(module.PackageStoreError) as info:
            module.fetch_package_dataset(
                module.PackageRef("terminal-bench", "terminal-bench", "4.0.0"), tmp_path / "datasets"
            )
        assert str(info.value).startswith(f"{target} appeared while fetching and is not the store's copy of ")
        assert "terminal-bench/alpha" in str(info.value) and "move it aside and run again" in str(info.value)
        assert (target / "instruction.md").read_text() == "someone else's task\n"
        assert not (folder / "manifest.toml").exists()
        assert sorted(p.name for p in folder.iterdir()) == ["alpha"]

    def test_edited_folder_stops_the_fetch(self, tmp_path, monkeypatch):
        folders = {"alpha": write_task(tmp_path / "src" / "alpha")}
        module, digests, calls = self.fake_store(tmp_path, monkeypatch, folders)
        ref = module.PackageRef("terminal-bench", "terminal-bench", "4.0.0")
        folder = module.fetch_package_dataset(ref, tmp_path / "datasets")
        (folder / "alpha" / "instruction.md").write_text("edited locally\n")
        edited = content_hash(folder / "alpha")
        downloads_before = sum(1 for c in calls if c[1].startswith("/storage/"))

        with pytest.raises(module.PackageStoreError) as info:
            module.fetch_package_dataset(ref, tmp_path / "datasets")

        message = str(info.value)
        assert str(folder / "alpha") in message
        assert f"sha256:{edited}" in message and f"sha256:{digests['alpha']}" in message
        assert "delete the folder (or move it aside) and run again" in message
        assert (folder / "alpha" / "instruction.md").read_text() == "edited locally\n"
        assert sum(1 for c in calls if c[1].startswith("/storage/")) == downloads_before

    def test_a_deleted_folder_is_fetched_again(self, tmp_path, monkeypatch):
        folders = {"alpha": write_task(tmp_path / "src" / "alpha")}
        module, digests, calls = self.fake_store(tmp_path, monkeypatch, folders)
        ref = module.PackageRef("terminal-bench", "terminal-bench", "4.0.0")
        folder = module.fetch_package_dataset(ref, tmp_path / "datasets")
        (folder / "alpha" / "instruction.md").write_text("edited locally\n")
        # What the error message tells the user to do: move the edited folder aside, then run again.
        (folder / "alpha").rename(tmp_path / "alpha-edited")
        downloads_before = sum(1 for c in calls if c[1].startswith("/storage/"))

        module.fetch_package_dataset(ref, tmp_path / "datasets")

        assert content_hash(folder / "alpha") == digests["alpha"]
        assert (folder / "alpha" / "instruction.md").read_text() != "edited locally\n"
        assert (tmp_path / "alpha-edited" / "instruction.md").read_text() == "edited locally\n"
        assert sum(1 for c in calls if c[1].startswith("/storage/")) == downloads_before + 1

    def test_folder_names_and_digest_refs(self):
        from nemo_gym.tasks.harbor.package_store import PackageRef

        assert PackageRef("o", "n", "4.0.0").folder_name("f" * 64) == "n-4.0.0"
        assert PackageRef("o", "n", "v1/rc 2").folder_name("f" * 64) == "n-v1-rc-2"
        # `latest` is the plain name: the folder name stays readable and the pin lives in manifest.toml.
        assert PackageRef("o", "n").folder_name("f" * 64) == "n"
        pinned = PackageRef("o", "n", "sha256:" + "e" * 64)
        assert pinned.digest == "e" * 64 and pinned.folder_name("e" * 64) == "n-eeeeeeeeeeee"

    def test_fetch_ref_dispatches_package_references(self, tmp_path, monkeypatch):
        from nemo_gym.tasks.harbor import hub, package_store

        seen = {}

        def fake_fetch(ref, root, store=None):
            seen["ref"], seen["root"] = ref, root
            return root / "terminal-bench-4.0.0"

        monkeypatch.setattr(package_store, "fetch_package_dataset", fake_fetch)
        folder = hub.fetch_ref("harbor:terminal-bench/terminal-bench@4.0.0", tmp_path)
        assert folder == tmp_path / "terminal-bench-4.0.0"
        assert seen["ref"] == package_store.PackageRef("terminal-bench", "terminal-bench", "4.0.0")
        assert seen["root"] == tmp_path

    def test_archive_is_checked_without_exclusions_but_a_noisy_folder_still_matches(self, tmp_path, monkeypatch):
        folders = {"alpha": write_task(tmp_path / "src" / "alpha")}
        module, digests, calls = self.fake_store(tmp_path, monkeypatch, folders)
        ref = module.PackageRef("terminal-bench", "terminal-bench", "4.0.0")
        folder = module.fetch_package_dataset(ref, tmp_path / "datasets")
        # Byte-code left behind by a run does not change the fetched folder's identity: no refetch, no refusal.
        (folder / "alpha" / "tests" / "__pycache__").mkdir()
        (folder / "alpha" / "tests" / "__pycache__" / "x.pyc").write_bytes(b"x")
        downloads_before = sum(1 for c in calls if c[1].startswith("/storage/"))
        module.fetch_package_dataset(ref, tmp_path / "datasets")
        assert sum(1 for c in calls if c[1].startswith("/storage/")) == downloads_before
        assert (folder / "alpha" / "tests" / "__pycache__" / "x.pyc").is_file()

        # The store serves an archive with one extra file whose name Gym's folder digest skips. The archive is
        # Harbor's published file set, so that file counts there, and the archive no longer matches the hash
        # the store advertises (which is still the digest of the clean folder).
        (folders["alpha"] / "tests" / "__pycache__").mkdir()
        (folders["alpha"] / "tests" / "__pycache__" / "x.pyc").write_bytes(b"x")
        assert content_hash(folders["alpha"]) == digests["alpha"]
        module.PackageStore.make_archive_for = staticmethod(lambda name: self.make_archive(folders[name]))
        with pytest.raises(module.PackageStoreError) as info:
            module.fetch_package_dataset(ref, tmp_path / "other-datasets")
        message = str(info.value)
        assert "terminal-bench/alpha content hash mismatch" in message
        assert f"advertises sha256:{digests['alpha']}" in message
        assert f"sha256:{module.archive_hash(folders['alpha'])}" in message
        assert not (tmp_path / "other-datasets" / "terminal-bench-4.0.0" / "alpha").exists()

    def test_a_package_that_ships_a_name_the_folder_digest_skips_is_fetched_once_and_kept(self, tmp_path, monkeypatch):
        import os
        import tomllib

        from nemo_gym.tasks.harbor.digest import archive_hash

        # Harbor's packager applies the task's own .gitignore instead of its defaults when there is one, so a
        # published archive can hold a name Gym's folder digest skips (here an editor backup the task's .gitignore
        # does not list). The store's hash covers that file; Gym's digest of the same folder does not.
        source = write_task(tmp_path / "src" / "alpha")
        (source / "tests" / "helper.py~").write_text("shipped on purpose\n")
        folders = {"alpha": source}
        module, digests, calls = self.fake_store(tmp_path, monkeypatch, folders, publish=archive_hash)
        assert digests["alpha"] == archive_hash(source) != content_hash(source)
        ref = module.PackageRef("terminal-bench", "terminal-bench", "4.0.0")

        def downloads():
            return sum(1 for c in calls if c[1].startswith("/storage/"))

        folder = module.fetch_package_dataset(ref, tmp_path / "datasets")
        assert (folder / "alpha" / "tests" / "helper.py~").read_text() == "shipped on purpose\n"
        manifest = tomllib.loads((folder / "manifest.toml").read_text())
        assert manifest["tasks"]["alpha"] == {
            "package": "terminal-bench/alpha",
            "content_hash": f"sha256:{digests['alpha']}",
            "digest": f"sha256:{content_hash(source)}",
        }
        manifest_text = (folder / "manifest.toml").read_text()

        # The rerun compares the folder with the digest the fetch recorded, not with the store's hash: nothing
        # is refused and nothing is downloaded, before and after a run leaves byte-code behind.
        assert downloads() == 1
        assert module.fetch_package_dataset(ref, tmp_path / "datasets") == folder
        (folder / "alpha" / "tests" / "__pycache__").mkdir()
        (folder / "alpha" / "tests" / "__pycache__" / "x.pyc").write_bytes(b"x")
        assert module.fetch_package_dataset(ref, tmp_path / "datasets") == folder
        assert downloads() == 1 and (folder / "manifest.toml").read_text() == manifest_text

        # Interrupted before the manifest was written: the clean folder is still the store's copy by the
        # archive digest, so the rerun keeps it and records it.
        (folder / "manifest.toml").rename(tmp_path / "manifest-aside.toml")
        shutil.rmtree(folder / "alpha" / "tests" / "__pycache__")
        assert module.fetch_package_dataset(ref, tmp_path / "datasets") == folder
        assert downloads() == 1 and (folder / "manifest.toml").read_text() == manifest_text

        # A concurrent fetch that finished this task while the archive was downloading is kept too.
        racing = tmp_path / "racing" / "alpha"
        shutil.copytree(source, racing)
        inode = os.stat(racing).st_ino
        module.PackageStore().download_task(module.PackageTask("terminal-bench", "alpha", digests["alpha"]), racing)
        assert os.stat(racing).st_ino == inode and downloads() == 2

        # An edit is still refused, with the recorded digest named rather than the store's hash.
        (folder / "alpha" / "instruction.md").write_text("edited locally\n")
        edited = content_hash(folder / "alpha")
        with pytest.raises(module.PackageStoreError) as info:
            module.fetch_package_dataset(ref, tmp_path / "datasets")
        message = str(info.value)
        assert message.startswith(f"{folder / 'alpha'} differs from the store: its content hash is sha256:{edited}")
        assert (
            "the copy the previous fetch wrote of terminal-bench/alpha (in terminal-bench/terminal-bench@4.0.0) "
            f"hashed to sha256:{content_hash(source)}"
        ) in message
        assert digests["alpha"] not in message
        assert "delete the folder (or move it aside) and run again" in message
        # Doing what the message says fetches the store's copy, once, and the dataset is whole again.
        (folder / "alpha").rename(tmp_path / "alpha-edited")
        assert module.fetch_package_dataset(ref, tmp_path / "datasets") == folder
        assert downloads() == 3 and (folder / "manifest.toml").read_text() == manifest_text
        assert module.fetch_package_dataset(ref, tmp_path / "datasets") == folder and downloads() == 3

    def test_a_recorded_digest_vouches_only_for_the_task_version_it_pins(self, tmp_path, monkeypatch):
        folders = {"alpha": write_task(tmp_path / "src" / "alpha")}
        module, digests, calls = self.fake_store(tmp_path, monkeypatch, folders)
        ref = module.PackageRef("terminal-bench", "terminal-bench", "4.0.0")
        folder = module.fetch_package_dataset(ref, tmp_path / "datasets")
        (folder / "alpha" / "instruction.md").write_text("edited locally\n")
        edited = content_hash(folder / "alpha")
        head = (folder / "manifest.toml").read_text().split("[tasks.")[0]
        # A hand-edited entry that pins another task version, or no table at all, does not make the edited folder
        # pass: the folder is compared with the store's hash, as a folder without a record is.
        for tasks in (
            f'[tasks."alpha"]\npackage = "terminal-bench/alpha"\ncontent_hash = "sha256:{"f" * 64}"\n'
            f'digest = "sha256:{edited}"\n',
            'tasks = "not a table"\n',
            f'[tasks."alpha"]\npackage = "terminal-bench/alpha"\ncontent_hash = "sha256:{digests["alpha"]}"\n',
        ):
            (folder / "manifest.toml").write_text(head + tasks)
            with pytest.raises(
                module.PackageStoreError, match=f"the store's terminal-bench/alpha .* is sha256:{digests['alpha']}"
            ):
                module.fetch_package_dataset(ref, tmp_path / "datasets")
        assert sum(1 for c in calls if c[1].startswith("/storage/")) == 1

    def test_folder_fetched_for_another_package_is_refused_by_name(self, tmp_path, monkeypatch):
        folders = {"alpha": write_task(tmp_path / "src" / "alpha")}
        module, _, calls = self.fake_store(tmp_path, monkeypatch, folders)
        folder = module.fetch_package_dataset(
            module.PackageRef("terminal-bench", "terminal-bench", "4.0.0"), tmp_path / "datasets"
        )
        manifest_before = (folder / "manifest.toml").read_text()
        downloads_before = sum(1 for c in calls if c[1].startswith("/storage/"))
        # Another org's dataset with the same name and tag would land in the same readable folder.
        with pytest.raises(module.PackageStoreError) as info:
            module.fetch_package_dataset(
                module.PackageRef("someone-else", "terminal-bench", "4.0.0"), tmp_path / "datasets"
            )
        message = str(info.value)
        assert message.startswith(f"{folder} was fetched for terminal-bench/terminal-bench@4.0.0 (sha256:{'d' * 64})")
        assert f"not someone-else/terminal-bench@4.0.0 (sha256:{'d' * 64})" in message
        assert "Move the folder aside (or delete it) and run again" in message
        assert (folder / "manifest.toml").read_text() == manifest_before
        assert content_hash(folder / "alpha") == content_hash(folders["alpha"])
        assert sum(1 for c in calls if c[1].startswith("/storage/")) == downloads_before

        # A folder another fetcher wrote (the registry path) is refused with its source named.
        (folder / "manifest.toml").write_text(
            '[dataset]\nname = "terminal-bench"\nversion = "2.0"\nsource = "harbor-registry"\n'
        )
        with pytest.raises(module.PackageStoreError, match=r"terminal-bench@2\.0 from harbor-registry, not "):
            module.fetch_package_dataset(
                module.PackageRef("terminal-bench", "terminal-bench", "4.0.0"), tmp_path / "datasets"
            )

    def test_moved_latest_is_refused_as_stale_with_both_hashes(self, tmp_path, monkeypatch):
        import dataclasses

        folders = {"alpha": write_task(tmp_path / "src" / "alpha")}
        module, digests, calls = self.fake_store(tmp_path, monkeypatch, folders, tag="latest")
        ref = module.PackageRef("terminal-bench", "terminal-bench")
        folder = module.fetch_package_dataset(ref, tmp_path / "datasets")
        assert folder == tmp_path / "datasets" / "terminal-bench"
        assert f'content_hash = "sha256:{"d" * 64}"' in (folder / "manifest.toml").read_text()
        # `latest` moves on: the store resolves the plain reference to another version.
        resolve = module.PackageStore.resolve_dataset
        monkeypatch.setattr(
            module.PackageStore,
            "resolve_dataset",
            lambda self, ref: dataclasses.replace(resolve(self, ref), content_hash="e" * 64),
        )
        downloads_before = sum(1 for c in calls if c[1].startswith("/storage/"))
        with pytest.raises(module.PackageStoreError) as info:
            module.fetch_package_dataset(ref, tmp_path / "datasets")
        message = str(info.value)
        assert message.startswith(f"{folder} is stale: it was fetched when terminal-bench/terminal-bench@latest")
        assert f"resolved to sha256:{'d' * 64}" in message
        assert f"terminal-bench/terminal-bench@latest now resolves to sha256:{'e' * 64}" in message
        assert "run again to fetch the current version" in message
        assert sum(1 for c in calls if c[1].startswith("/storage/")) == downloads_before
        assert content_hash(folder / "alpha") == digests["alpha"]
        # Moving the folder aside, as the message says, fetches the current version in its place.
        folder.rename(tmp_path / "terminal-bench-old")
        assert module.fetch_package_dataset(ref, tmp_path / "datasets") == folder
        assert f'content_hash = "sha256:{"e" * 64}"' in (folder / "manifest.toml").read_text()

    def test_unreadable_manifest_is_rewritten_after_the_task_checks(self, tmp_path, monkeypatch, caplog):
        folders = {"alpha": write_task(tmp_path / "src" / "alpha")}
        module, _, _ = self.fake_store(tmp_path, monkeypatch, folders)
        ref = module.PackageRef("terminal-bench", "terminal-bench", "4.0.0")
        folder = module.fetch_package_dataset(ref, tmp_path / "datasets")
        good = (folder / "manifest.toml").read_text()
        for broken in ("[dataset\nname = ", 'dataset = "not a table"\n'):
            (folder / "manifest.toml").write_text(broken)
            with caplog.at_level(logging.WARNING, logger=module.__name__):
                assert module.fetch_package_dataset(ref, tmp_path / "datasets") == folder
            assert (folder / "manifest.toml").read_text() == good
        assert "is unreadable and will be rewritten" in caplog.text
