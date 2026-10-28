# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests of the Harbor package store client.

Dataset resolution by tag or digest, task download and verification, and the manifest the store
copy writes and checks on the next fetch.
"""

import http.server
import json
import re
import shutil
import threading
from pathlib import Path

import pytest

import nemo_gym.data.tasks.harbor.manifest as manifest_module
from nemo_gym.data.tasks.harbor import content_hash, discover_tasks
from tests.unit_tests.test_harbor_tasks import (
    no_real_registry,  # noqa: F401 (autouse fixture)
    write_task,
)


@pytest.fixture
def short_reply_server():
    """A local HTTP server that advertises a 4096-byte body, sends 10 bytes and closes the connection."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "4096")
            self.end_headers()
            self.wfile.write(b"0123456789")

        do_POST = do_GET

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


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

    def fake_store(
        self, tmp_path, monkeypatch, task_folders: dict[str, Path], tag="4.0.0", publish=content_hash, task_org=None
    ):
        """``publish`` hashes a source folder into the hash the store advertises for it (Harbor's archive hash
        equals Gym's folder digest unless the folder holds a name Gym's digest skips). ``task_org`` is the org the
        tasks are published under: the requested dataset's own org when ``None``, another org (or a per-task
        mapping) for a dataset that collects tasks across orgs, as Harbor's ``dataset.toml`` lets it."""
        from nemo_gym.data.tasks.harbor import package_store as module

        digests = {name: publish(folder) for name, folder in task_folders.items()}
        calls = []
        requested = {}

        def org_of(name):
            if isinstance(task_org, dict):
                return task_org[name]
            # A test that downloads a task without resolving a dataset first names the org in its PackageTask.
            return task_org or requested.get("org", "terminal-bench")

        def request(self, method, path, *, params=None, body=None):
            calls.append((method, path, params, body))
            if path == "/rest/v1/dataset_version_tag":
                assert params["tag"] == f"eq.{tag}" and params["package.name"] == "eq.terminal-bench"
                requested["org"] = params["package.org.name"].removeprefix("eq.")
                return json.dumps(
                    [{"dataset_version": {"id": "dv-1", "content_hash": "d" * 64}, "package": {}}]
                ).encode()
            if path == "/rest/v1/dataset_version":
                requested["org"] = params["package.org.name"].removeprefix("eq.")
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
                            "package": {"name": name, "org": {"name": org_of(name)}},
                        },
                    }
                    for name, digest in sorted(digests.items())
                ]
                return json.dumps(rows).encode()
            if path == "/rest/v1/rpc/resolve_task_version":
                name = body["p_name"]
                # The archive is resolved under the task's own org, as the store named it, not the dataset's.
                assert body["p_org"] == org_of(name) and body["p_ref"] == f"sha256:{digests[name]}"
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
        # Derived facts only: no copy of the name or version, which live in dataset.toml; the reference the fetch
        # was made from is recorded as the pins are (round 32).
        assert manifest["schema_version"] == 1
        assert set(manifest["dataset"]) == {"source", "ref", "digest", "dataset_toml_digest"}
        assert manifest["dataset"]["source"] == "harbor-package-store"
        assert manifest["dataset"]["ref"] == "harbor:terminal-bench/terminal-bench@4.0.0"
        assert manifest["dataset"]["digest"] == manifest_module.dataset_digest(
            digests, manifest["dataset"]["dataset_toml_digest"]
        )
        assert manifest["tasks"]["beta"] == {
            "digest": digests["beta"],
            "package": "terminal-bench/beta",
            "content_hash": f"sha256:{digests['beta']}",
        }
        assert "name" not in (folder / "manifest.toml").read_text()
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

    def test_short_http_reply_is_a_package_store_error(self, short_reply_server):
        """A server that closes the connection before sending its advertised Content-Length makes
        ``response.read()`` raise ``http.client.IncompleteRead``, which is not an OSError: the store reports it
        as a PackageStoreError naming the request, like any other transport failure."""
        from nemo_gym.data.tasks.harbor import package_store as module

        store = module.PackageStore(base_url=short_reply_server, api_key="public", timeout_s=5)
        with pytest.raises(module.PackageStoreError) as info:
            store._request("GET", "/storage/v1/object/packages/pkgs/alpha.tar.gz")
        message = str(info.value)
        assert message.startswith(
            "Harbor package store closed the connection before the reply to GET "
            "/storage/v1/object/packages/pkgs/alpha.tar.gz was complete: "
        )
        assert "10 bytes read, 4086 more expected" in message
        assert isinstance(info.value.__cause__, http.client.IncompleteRead)
        # The same through the JSON reader and a POST.
        with pytest.raises(module.PackageStoreError, match="closed the connection before the reply to POST"):
            store._request("POST", "/rest/v1/rpc/resolve_task_version", body={"p_org": "o"})

    @pytest.mark.parametrize(
        "reply", [b"<html><body>502 Bad Gateway</body></html>", b'[{"id": "dv-1", "content', b"\xff\xfe"]
    )
    def test_non_json_reply_is_a_package_store_error(self, monkeypatch, reply):
        """An HTML error page, a JSON reply cut short by a close-delimited connection, or bytes that are not UTF-8
        where the REST API's JSON should be: the resolution and archive lookups report a PackageStoreError naming
        the request and the start of the body, not a JSONDecodeError."""
        from nemo_gym.data.tasks.harbor import package_store as module

        monkeypatch.setattr(
            module.PackageStore, "_request", lambda self, method, path, *, params=None, body=None: reply
        )
        store = module.PackageStore()
        with pytest.raises(module.PackageStoreError) as info:
            store.resolve_dataset(module.PackageRef("terminal-bench", "terminal-bench", "4.0.0"))
        assert str(info.value).startswith(
            "Harbor package store reply to GET /rest/v1/dataset_version_tag is not JSON ("
        )
        assert repr(reply.decode(errors="replace")[:200]) in str(info.value)
        assert isinstance(info.value.__cause__, ValueError)
        with pytest.raises(
            module.PackageStoreError, match="reply to POST /rest/v1/rpc/resolve_task_version is not JSON"
        ):
            store.archive_path(module.PackageTask("terminal-bench", "alpha", "a" * 64))

    @pytest.mark.parametrize("command", ["eval run", "dataset validate"])
    @pytest.mark.parametrize("failure", ["error page", "short reply"])
    def test_store_failures_are_reported_by_eval_run_and_dataset_validate(
        self, tmp_path, monkeypatch, capsys, short_reply_server, command, failure
    ):
        """Through ``main()`` with only the store's HTTP replies faked: an error page where the archive should be, or
        a server that closes the connection before the advertised archive bytes arrive (a real local server, read by
        the store's own transport), ends `gym eval run` and `gym dataset validate` with parser.error's one-line
        message and exit status 2, not a traceback (fetch_ref raises the store's error as a HubError, which both
        entry points report)."""
        import sys

        from nemo_gym.cli.main import main

        folders = {"alpha": write_task(tmp_path / "src" / "alpha")}
        from nemo_gym.data.tasks.harbor import package_store as module

        real_request = module.PackageStore._request
        module, _, _ = self.fake_store(tmp_path, monkeypatch, folders)
        if failure == "error page":
            served = staticmethod(lambda name: b"<html><body>502 Bad Gateway</body></html>")
            expected = "terminal-bench/alpha archive from the store is not a readable tar.gz package"
        else:
            short = module.PackageStore(base_url=short_reply_server, api_key="public", timeout_s=5)
            served = staticmethod(lambda name: real_request(short, "GET", f"/storage/v1/object/packages/pkgs/{name}"))
            expected = (
                "Harbor package store closed the connection before the reply to GET "
                "/storage/v1/object/packages/pkgs/alpha was complete: IncompleteRead(10 bytes read, 4086 more expected)"
            )
        module.PackageStore.make_archive_for = served
        monkeypatch.setenv("NEMO_GYM_DATASETS_DIR", str(tmp_path / "datasets"))
        agent = ["--agent", "hermes_agent"] if command == "eval run" else []
        monkeypatch.setattr(
            sys, "argv", ["gym", *command.split(), "harbor:terminal-bench/terminal-bench@4.0.0", *agent]
        )
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert expected in err
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
        from nemo_gym.data.tasks.harbor import package_store as module

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
        import tomllib

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
        # The claim written before the download stays (pins only): the foreign folder is not recorded as fetched.
        assert sorted(p.name for p in folder.iterdir()) == ["alpha", "manifest.toml"]
        assert "digest" not in tomllib.loads((folder / "manifest.toml").read_text())["tasks"]["alpha"]

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

    def test_progress_manifests_never_vouch_for_a_folder_the_fetch_has_not_checked(self, tmp_path, monkeypatch):
        """Delete the earlier task, edit the later one, fetch again: downloading alpha writes a progress manifest
        while beta is still unchecked. That manifest must not record the edited beta's hash, or the next fetch
        would accept the edit as the store's copy and prepare would materialize it with store provenance.
        Fetch after fetch keeps refusing the edit, prepare refuses the folder, and the one download is alpha's."""
        import tomllib

        from nemo_gym.data.tasks.harbor.manifest import ManifestError
        from nemo_gym.data.tasks.harbor.prepare import prepare_target

        folders = {name: write_task(tmp_path / "src" / name) for name in ("alpha", "beta")}
        module, digests, calls = self.fake_store(tmp_path, monkeypatch, folders)
        ref = module.PackageRef("terminal-bench", "terminal-bench", "4.0.0")
        folder = module.fetch_package_dataset(ref, tmp_path / "datasets")
        shutil.rmtree(folder / "alpha")
        (folder / "beta" / "instruction.md").write_text("edited locally\n")
        edited = content_hash(folder / "beta")
        downloads_before = sum(1 for c in calls if c[1].startswith("/storage/"))

        for attempt in range(3):
            with pytest.raises(module.PackageStoreError) as info:
                module.fetch_package_dataset(ref, tmp_path / "datasets")
            message = str(info.value)
            assert message.startswith(f"{folder / 'beta'} differs from the store: its content hash is sha256:{edited}")
            # The previous fetch's digest of beta is what the folder is held to, fetch after fetch.
            assert "the copy the previous fetch wrote of terminal-bench/beta" in message
            assert f"hashed to sha256:{digests['beta']}" in message
            manifest = tomllib.loads((folder / "manifest.toml").read_text())
            assert manifest["tasks"]["beta"]["digest"] == digests["beta"], attempt
            assert manifest["tasks"]["alpha"]["digest"] == digests["alpha"], attempt
            assert edited not in (folder / "manifest.toml").read_text()
            assert (folder / "beta" / "instruction.md").read_text() == "edited locally\n"
        # alpha was downloaded once, on the first rerun; beta never was.
        assert sum(1 for c in calls if c[1].startswith("/storage/")) == downloads_before + 1
        assert content_hash(folder / "alpha") == digests["alpha"]

        # Prepare holds the folder to the same manifest and refuses it, so nothing is materialized with
        # harbor-package-store provenance.
        with pytest.raises(ManifestError, match="does not match its manifest.toml"):
            prepare_target(str(folder), output_root=tmp_path / "out")
        assert not (tmp_path / "out").exists() or not list((tmp_path / "out").rglob("tasks.jsonl"))

        # Doing what the message says fetches the store's beta, and the dataset is whole again.
        (folder / "beta").rename(tmp_path / "beta-edited")
        assert module.fetch_package_dataset(ref, tmp_path / "datasets") == folder
        assert sum(1 for c in calls if c[1].startswith("/storage/")) == downloads_before + 2
        assert content_hash(folder / "beta") == digests["beta"]

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
        from nemo_gym.data.tasks.harbor.package_store import PackageRef

        assert PackageRef("o", "n", "4.0.0").folder_name("f" * 64) == "n-4.0.0"
        assert PackageRef("o", "n", "v1/rc 2").folder_name("f" * 64) == "n-v1-rc-2"
        # `latest` is the plain name: the folder name stays readable and the pin lives in manifest.toml.
        assert PackageRef("o", "n").folder_name("f" * 64) == "n"
        pinned = PackageRef("o", "n", "sha256:" + "e" * 64)
        assert pinned.digest == "e" * 64 and pinned.folder_name("e" * 64) == "n-eeeeeeeeeeee"

    def test_fetch_ref_dispatches_package_references(self, tmp_path, monkeypatch):
        from nemo_gym.data.tasks.harbor import hub, package_store

        seen = {}

        def fake_fetch(ref, root, store=None, *, folder=None):
            seen["ref"], seen["root"], seen["folder"] = ref, root, folder
            return folder or root / "terminal-bench-4.0.0"

        monkeypatch.setattr(package_store, "fetch_package_dataset", fake_fetch)
        folder = hub.fetch_ref("harbor:terminal-bench/terminal-bench@4.0.0", tmp_path)
        assert folder == tmp_path / "terminal-bench-4.0.0"
        assert seen["ref"] == package_store.PackageRef("terminal-bench", "terminal-bench", "4.0.0")
        assert seen["root"] == tmp_path and seen["folder"] is None
        # A caller that names the dataset folder (a committed entry's) passes it through to the store fetch.
        assert hub.fetch_ref("harbor:terminal-bench/terminal-bench@4.0.0", tmp_path, folder=tmp_path / "tb4") == (
            tmp_path / "tb4"
        )
        assert seen["folder"] == tmp_path / "tb4"

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

        from nemo_gym.data.tasks.harbor.digest import archive_hash

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
            "digest": content_hash(source),
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
        # A hand-edited entry that pins another task version, or no entry for the task at all, does not make the
        # edited folder pass: the manifest was written for another version of the dataset, and the fetch stops there.
        for tasks, difference in (
            (
                f'[tasks."alpha"]\npackage = "terminal-bench/alpha"\ncontent_hash = "sha256:{"f" * 64}"\n'
                f'digest = "{edited}"\n',
                "changed 1",
            ),
            ("", "added 1"),
        ):
            (folder / "manifest.toml").write_text(head + tasks)
            with pytest.raises(module.PackageStoreError, match=rf"another version .*\(tasks {difference}\)"):
                module.fetch_package_dataset(ref, tmp_path / "datasets")
        # An entry with the right pin but no digest vouches for nothing: the folder is compared with the store's
        # hash, as a folder without a record is.
        (folder / "manifest.toml").write_text(
            head + f'[tasks."alpha"]\npackage = "terminal-bench/alpha"\ncontent_hash = "sha256:{digests["alpha"]}"\n'
        )
        with pytest.raises(
            module.PackageStoreError, match=f"the store's terminal-bench/alpha .* is sha256:{digests['alpha']}"
        ):
            module.fetch_package_dataset(ref, tmp_path / "datasets")
        assert sum(1 for c in calls if c[1].startswith("/storage/")) == 1

    @pytest.mark.parametrize(
        ("target", "reference"),
        [
            ("harbor:org/ds", "harbor:org/ds@latest"),
            ("harbor:org/ds@latest", "harbor:org/ds@latest"),
            ("harbor:org/ds@4.0.0", "harbor:org/ds@4.0.0"),
            ("harbor:org/ds@sha256:" + "a" * 64, "harbor:org/ds@sha256:" + "a" * 64),
            ("harbor:org/ds@" + "a" * 64, "harbor:org/ds@sha256:" + "a" * 64),
        ],
    )
    def test_the_recorded_reference_reads_a_reference_as_the_store_does(self, target, reference):
        """What a manifest records for a package-store fetch: ``latest`` spelled out, a digest in its ``sha256:`` form,
        so the same dataset declared either way reads as one reference (round 32)."""
        from nemo_gym.data.tasks.harbor.hub import HubRef
        from nemo_gym.data.tasks.harbor.package_store import PackageRef

        ref = PackageRef.of(HubRef.parse(target))
        assert ref.package == "org/ds" and ref.reference == reference

    def test_folder_fetched_for_another_package_is_refused_by_its_task_pins(self, tmp_path, monkeypatch):
        """The manifest holds no copy of the dataset's name, so the per-task `package` pins (`<org>/<task>`) are
        what tell another org's dataset of the same name and tag, which lands in the same readable folder."""
        folders = {"alpha": write_task(tmp_path / "src" / "alpha")}
        module, _, calls = self.fake_store(tmp_path, monkeypatch, folders)
        folder = module.fetch_package_dataset(
            module.PackageRef("terminal-bench", "terminal-bench", "4.0.0"), tmp_path / "datasets"
        )
        manifest_before = (folder / "manifest.toml").read_text()
        # The fetch's reference is recorded, but the check reads the pins, so a manifest written before the reference
        # was recorded (stripped of it here) is told apart the same way.
        recorded = 'ref = "harbor:terminal-bench/terminal-bench@4.0.0"\n'
        assert recorded in manifest_before
        (folder / "manifest.toml").write_text(manifest_before.replace(recorded, ""))
        manifest_before = (folder / "manifest.toml").read_text()
        assert "terminal-bench/terminal-bench" not in manifest_before  # no dataset name to compare by
        downloads_before = sum(1 for c in calls if c[1].startswith("/storage/"))
        with pytest.raises(module.PackageStoreError) as info:
            module.fetch_package_dataset(
                module.PackageRef("someone-else", "terminal-bench", "4.0.0"), tmp_path / "datasets"
            )
        message = str(info.value)
        assert message.startswith(f"{folder} was fetched for another package: its manifest pins terminal-bench/alpha")
        assert "where someone-else/terminal-bench@4.0.0 names someone-else/alpha" in message
        assert "Move the folder aside (or delete it) and run again" in message
        assert (folder / "manifest.toml").read_text() == manifest_before
        assert content_hash(folder / "alpha") == content_hash(folders["alpha"])
        assert sum(1 for c in calls if c[1].startswith("/storage/")) == downloads_before

        # A folder another fetcher wrote (the registry path) is refused with its source named.
        (folder / "manifest.toml").write_text('[dataset]\nsource = "harbor-registry"\n')
        with pytest.raises(module.PackageStoreError, match="written by a harbor-registry fetch, not a package-store"):
            module.fetch_package_dataset(
                module.PackageRef("terminal-bench", "terminal-bench", "4.0.0"), tmp_path / "datasets"
            )
        # The same tasks under the same org, as the store publishes them, are this package: nothing is refused.
        (folder / "manifest.toml").write_text(manifest_before)
        assert (
            module.fetch_package_dataset(
                module.PackageRef("terminal-bench", "terminal-bench", "4.0.0"), tmp_path / "datasets"
            )
            == folder
        )

    def test_a_dataset_that_collects_another_orgs_tasks_is_fetched_once_and_reused(self, tmp_path, monkeypatch):
        """Harbor's dataset.toml pins each task as org/name, so a dataset may collect tasks published under other
        orgs (a curated suite of upstream tasks). Its pins name those packages, not <dataset org>/<task>: the folder
        is this dataset's on every later fetch, with one download per task, prepare takes it, and a same-named
        dataset whose tasks are someone else's packages is still refused before anything is touched."""
        from nemo_gym.data.tasks.harbor.hub import fetch_ref
        from nemo_gym.data.tasks.harbor.prepare import prepare_target

        folders = {name: write_task(tmp_path / "src" / name) for name in ("alpha", "beta")}
        module, digests, calls = self.fake_store(tmp_path, monkeypatch, folders, task_org="upstream")
        ref = module.PackageRef("curator", "terminal-bench", "4.0.0")

        def downloads():
            return sum(1 for c in calls if c[1].startswith("/storage/"))

        folder = module.fetch_package_dataset(ref, tmp_path / "datasets")
        text = (folder / "manifest.toml").read_text()
        # The pins name each task's own package; the dataset's org appears nowhere in them (the manifest may record
        # the fetch's own reference, which names the dataset, beside the pins).
        pins = [line for line in text.splitlines() if line.startswith("package = ")]
        assert 'package = "upstream/alpha"' in pins and not any("curator" in line for line in pins)
        assert downloads() == 2
        assert module.fetch_package_dataset(ref, tmp_path / "datasets") == folder
        assert fetch_ref("harbor:curator/terminal-bench@4.0.0", tmp_path / "datasets") == folder
        assert downloads() == 2 and (folder / "manifest.toml").read_text() == text
        prepared = prepare_target(str(folder), output_root=tmp_path / "out")
        assert [task.task_id for task in prepared.tasks] == ["alpha", "beta"]
        assert (folder / "manifest.toml").read_text() == text and downloads() == 2

        # Tasks of mixed ownership, the dataset's own beside upstream's: the same.
        mixed = self.fake_store(tmp_path, monkeypatch, folders, task_org={"alpha": "curator", "beta": "upstream"})
        mixed_folder = mixed[0].fetch_package_dataset(ref, tmp_path / "mixed")
        mixed_text = (mixed_folder / "manifest.toml").read_text()
        assert 'package = "curator/alpha"' in mixed_text and 'package = "upstream/beta"' in mixed_text
        assert mixed[0].fetch_package_dataset(ref, tmp_path / "mixed") == mixed_folder
        assert (mixed_folder / "manifest.toml").read_text() == mixed_text
        assert sum(1 for c in mixed[2] if c[1].startswith("/storage/")) == 2

        # Another org's dataset of the same name and tag, pinning its own alpha and beta: refused by the pins, the
        # folder and its manifest byte for byte as they were, nothing downloaded.
        other = self.fake_store(tmp_path, monkeypatch, folders, task_org="someone-else")
        with pytest.raises(module.PackageStoreError) as info:
            other[0].fetch_package_dataset(
                module.PackageRef("someone-else", "terminal-bench", "4.0.0"), tmp_path / "datasets"
            )
        assert str(info.value).startswith(
            f"{folder} was fetched for another package: its manifest pins upstream/alpha and 1 more where "
            "someone-else/terminal-bench@4.0.0 names someone-else/alpha"
        )
        assert (folder / "manifest.toml").read_text() == text
        assert content_hash(folder / "alpha") == digests["alpha"] and content_hash(folder / "beta") == digests["beta"]
        assert sum(1 for c in other[2] if c[1].startswith("/storage/")) == 0

    def test_interrupted_fetch_leaves_a_claimed_folder_no_other_dataset_can_complete(self, tmp_path, monkeypatch):
        """A fetch that fails after its first task has landed leaves ``manifest.toml`` with every task pinned,
        written before that first download, and the digest of the task that landed. The folder says whose it is: a fetch of another
        org's same-named dataset into it, and a fetch after ``latest`` moved, are refused before they look at a
        task or download anything, while a rerun of the same dataset completes it."""
        import tomllib

        folders = {name: write_task(tmp_path / "src" / name) for name in ("alpha", "beta")}
        module, digests, calls = self.fake_store(tmp_path, monkeypatch, folders, tag="latest")
        ref = module.PackageRef("terminal-bench", "terminal-bench")
        archives = {name: self.make_archive(folder) for name, folder in folders.items()}

        def serve(name):
            if name == "beta":
                raise module.PackageStoreError("Harbor package store unreachable for GET beta")
            return archives[name]

        module.PackageStore.make_archive_for = staticmethod(serve)
        with pytest.raises(module.PackageStoreError, match="unreachable for GET beta"):
            module.fetch_package_dataset(ref, tmp_path / "datasets")
        folder = tmp_path / "datasets" / "terminal-bench"
        assert (folder / "alpha" / "task.toml").is_file() and not (folder / "beta").exists()
        claim = tomllib.loads((folder / "manifest.toml").read_text())
        assert claim["dataset"]["source"] == "harbor-package-store"
        assert claim["tasks"] == {
            "alpha": {
                "digest": digests["alpha"],
                "package": "terminal-bench/alpha",
                "content_hash": f"sha256:{digests['alpha']}",
            },
            "beta": {"package": "terminal-bench/beta", "content_hash": f"sha256:{digests['beta']}"},
        }
        claim_text = (folder / "manifest.toml").read_text()
        downloads = lambda: sum(1 for c in calls if c[1].startswith("/storage/"))  # noqa: E731
        downloads_before = downloads()

        # Another org's dataset of the same name: refused by the claim, nothing looked at, nothing downloaded.
        with pytest.raises(module.PackageStoreError) as info:
            module.fetch_package_dataset(module.PackageRef("someone-else", "terminal-bench"), tmp_path / "datasets")
        assert str(info.value).startswith(
            f"{folder} was fetched for another package: its manifest pins terminal-bench/alpha"
        )
        assert "where someone-else/terminal-bench@latest names someone-else/alpha" in str(info.value)
        # `latest` moved to a version without alpha: refused as another version, the claim still naming alpha.
        del digests["alpha"]
        with pytest.raises(
            module.PackageStoreError, match=r"another version of terminal-bench/terminal-bench: .*removed 1"
        ):
            module.fetch_package_dataset(ref, tmp_path / "datasets")
        digests["alpha"] = content_hash(folders["alpha"])
        assert (folder / "manifest.toml").read_text() == claim_text
        assert downloads() == downloads_before
        # Prepare does not run the half-fetched folder either: its manifest records pins without digests.
        with pytest.raises(manifest_module.ManifestError):
            manifest_module.check_manifest(folder)
        assert (folder / "manifest.toml").read_text() == claim_text

        # The same dataset again: alpha is kept, beta alone is downloaded and the manifest gains its digests.
        module.PackageStore.make_archive_for = staticmethod(archives.__getitem__)
        assert module.fetch_package_dataset(ref, tmp_path / "datasets") == folder
        assert downloads() == downloads_before + 1
        complete = tomllib.loads((folder / "manifest.toml").read_text())
        assert {name: entry["digest"] for name, entry in complete["tasks"].items()} == digests
        assert [task.task_id for task in discover_tasks(folder)] == ["alpha", "beta"]

    @pytest.mark.parametrize("stage", ["partial", "claim only"])
    def test_dataset_validate_names_an_interrupted_fetch(self, tmp_path, monkeypatch, capsys, stage):
        """`gym dataset validate <folder>` on a folder whose fetch was interrupted, after its first task landed or
        during that first download (the claim alone, which matches its manifest exactly), stops with the tasks the
        fetch never recorded and says to run the fetch again, not that the folders were edited by hand."""
        import sys

        from nemo_gym.cli.main import main

        folders = {name: write_task(tmp_path / "src" / name) for name in ("alpha", "beta")}
        module, _, _ = self.fake_store(tmp_path, monkeypatch, folders)
        archives = {name: self.make_archive(folder) for name, folder in folders.items()}
        failing = "beta" if stage == "partial" else "alpha"

        def serve(name):
            if name == failing:
                raise module.PackageStoreError(f"Harbor package store unreachable for GET {name}")
            return archives[name]

        module.PackageStore.make_archive_for = staticmethod(serve)
        with pytest.raises(module.PackageStoreError, match=f"unreachable for GET {failing}"):
            module.fetch_package_dataset(
                module.PackageRef("terminal-bench", "terminal-bench", "4.0.0"), tmp_path / "datasets"
            )
        folder = tmp_path / "datasets" / "terminal-bench-4.0.0"
        assert (folder / "manifest.toml").is_file()
        assert (folder / "alpha" / "task.toml").is_file() == (stage == "partial")
        monkeypatch.setattr(sys, "argv", ["gym", "dataset", "validate", str(folder)])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        unrecorded = "beta" if stage == "partial" else "alpha, beta"
        assert (
            f"the fetch of {folder} was interrupted (tasks not yet recorded: {unrecorded}); run the fetch again" in err
        )
        assert "edited by hand" not in err and "Traceback" not in err

    def test_claim_that_cannot_be_written_is_a_package_store_error(self, tmp_path, monkeypatch):
        """A folder the claim cannot be written into (here the link fails with an I/O error, which is not the
        missing-hard-links case the manifest writer falls back from) stops the fetch through the normal error path,
        naming the folder, with nothing downloaded."""
        import errno

        folders = {"alpha": write_task(tmp_path / "src" / "alpha")}
        module, _, calls = self.fake_store(tmp_path, monkeypatch, folders)
        monkeypatch.setattr(
            manifest_module.os, "link", lambda *a, **k: (_ for _ in ()).throw(OSError(errno.EIO, "Input/output error"))
        )
        folder = tmp_path / "datasets" / "terminal-bench-4.0.0"
        with pytest.raises(module.PackageStoreError) as info:
            module.fetch_package_dataset(
                module.PackageRef("terminal-bench", "terminal-bench", "4.0.0"), tmp_path / "datasets"
            )
        assert str(info.value).startswith(
            f"Could not record which dataset {folder} holds in {folder / 'manifest.toml'}"
        )
        assert "Input/output error" in str(info.value) and "Nothing was downloaded" in str(info.value)
        assert isinstance(info.value.__cause__, OSError)
        assert sum(1 for c in calls if c[1].startswith("/storage/")) == 0
        assert list(folder.iterdir()) == []

    def test_two_fetches_racing_for_one_folder_never_mix_datasets(self, tmp_path, monkeypatch):
        """A peer claims the folder between this fetch's read of the manifest and its own claim: the exclusive
        create fails, the peer's claim is read back, and the fetch is refused when the peer fetches another
        package, or goes on (and completes the folder) when the peer fetches the same dataset."""
        folders = {"alpha": write_task(tmp_path / "src" / "alpha")}
        module, digests, calls = self.fake_store(tmp_path, monkeypatch, folders)
        ref = module.PackageRef("terminal-bench", "terminal-bench", "4.0.0")
        folder = tmp_path / "datasets" / "terminal-bench-4.0.0"
        real_refuse = module.refuse_unrecorded_folders
        peer_pins = {}

        def peer_claims_after_the_folder_checks(*args, **kwargs):
            real_refuse(*args, **kwargs)
            manifest_module.Manifest.build(
                folder, source=module.PACKAGE_STORE_SOURCE, pins=peer_pins, names=list(peer_pins)
            ).write(folder)

        monkeypatch.setattr(module, "refuse_unrecorded_folders", peer_claims_after_the_folder_checks)

        peer_pins.update({"alpha": {"package": "someone-else/alpha", "content_hash": f"sha256:{'e' * 64}"}})
        with pytest.raises(
            module.PackageStoreError, match="fetched for another package: its manifest pins someone-else/alpha"
        ):
            module.fetch_package_dataset(ref, tmp_path / "datasets")
        assert not (folder / "alpha").exists()
        assert sum(1 for c in calls if c[1].startswith("/storage/")) == 0
        peer_claim = (folder / "manifest.toml").read_text()
        assert "someone-else/alpha" in peer_claim
        # The peer's claim stays as it was: this fetch wrote nothing into a folder that is not its own.
        with pytest.raises(module.PackageStoreError, match="fetched for another package"):
            module.fetch_package_dataset(ref, tmp_path / "datasets")
        assert (folder / "manifest.toml").read_text() == peer_claim
        assert [p.name for p in folder.iterdir() if p.name.startswith(".manifest-")] == []

        shutil.rmtree(folder)
        peer_pins.clear()
        peer_pins.update({"alpha": {"package": "terminal-bench/alpha", "content_hash": f"sha256:{digests['alpha']}"}})
        assert module.fetch_package_dataset(ref, tmp_path / "datasets") == folder
        assert content_hash(folder / "alpha") == digests["alpha"]
        assert f'digest = "{digests["alpha"]}"' in (folder / "manifest.toml").read_text()

    def test_unreadable_manifest_stops_the_fetch_untouched(self, tmp_path, monkeypatch):
        """The manifest is the folder's claim; one this Gym cannot read is refused, never rewritten, so a folder
        whose claim is gone can never be completed with another dataset's tasks."""
        folders = {"alpha": write_task(tmp_path / "src" / "alpha")}
        module, _, calls = self.fake_store(tmp_path, monkeypatch, folders)
        ref = module.PackageRef("terminal-bench", "terminal-bench", "4.0.0")
        folder = module.fetch_package_dataset(ref, tmp_path / "datasets")
        downloads_before = sum(1 for c in calls if c[1].startswith("/storage/"))
        for broken in ("[dataset\nname = ", 'dataset = "not a table"\n'):
            (folder / "manifest.toml").write_text(broken)
            with pytest.raises(
                module.PackageStoreError, match="is not a valid manifest: .*Move the folder aside"
            ) as info:
                module.fetch_package_dataset(ref, tmp_path / "datasets")
            assert isinstance(info.value.__cause__, manifest_module.ManifestError)
            assert (folder / "manifest.toml").read_text() == broken
        assert sum(1 for c in calls if c[1].startswith("/storage/")) == downloads_before

    def test_newer_schema_manifest_stops_the_fetch_untouched(self, tmp_path, monkeypatch):
        """A manifest a newer Gym wrote is not "unreadable": it says more than this Gym reads, so the fetch must
        not replace it with a schema-1 one that drops what it carried, and downloads nothing."""
        folders = {"alpha": write_task(tmp_path / "src" / "alpha")}
        module, digests, calls = self.fake_store(tmp_path, monkeypatch, folders)
        ref = module.PackageRef("terminal-bench", "terminal-bench", "4.0.0")
        folder = module.fetch_package_dataset(ref, tmp_path / "datasets")
        good = (folder / "manifest.toml").read_text()
        assert good.startswith("schema_version = 1\n")
        downloads_before = sum(1 for c in calls if c[1].startswith("/storage/"))
        for version in ("2", "1.0"):
            newer = good.replace("schema_version = 1\n", f"schema_version = {version}\n", 1)
            newer += '\n[future]\nkey = "x"\n'
            (folder / "manifest.toml").write_text(newer)
            with pytest.raises(
                manifest_module.UnsupportedManifestError, match=f"schema_version {version}; this Gym reads 1"
            ) as info:
                module.fetch_package_dataset(ref, tmp_path / "datasets")
            assert isinstance(info.value, manifest_module.ManifestError)
            assert (folder / "manifest.toml").read_text() == newer
            assert content_hash(folder / "alpha") == digests["alpha"]
            assert sum(1 for c in calls if c[1].startswith("/storage/")) == downloads_before
        # Restored, the same folder fetches again with nothing downloaded: the manifest was intact all along.
        (folder / "manifest.toml").write_text(good)
        assert module.fetch_package_dataset(ref, tmp_path / "datasets") == folder
        assert (folder / "manifest.toml").read_text() == good
        assert sum(1 for c in calls if c[1].startswith("/storage/")) == downloads_before

    def test_a_cache_written_before_the_schema_was_numbered_is_reused_and_upgraded(self, tmp_path, monkeypatch):
        """The fetcher of an earlier Gym wrote `[dataset]` with the package's name and no digest, and per landed task
        `digest = "sha256:<hex>"`. Such a cache is the fetch record it always was: an unchanged folder fetches again
        with nothing downloaded and its manifest is upgraded with the recorded digests kept, an edited folder is
        refused by fetch and prepare alike with the manifest untouched, and a task the earlier fetch never landed
        is unfinished business until it is fetched."""
        from nemo_gym.data.tasks.harbor.manifest import ManifestError, check_manifest
        from nemo_gym.data.tasks.harbor.prepare import prepare_target

        folders = {name: write_task(tmp_path / "src" / name) for name in ("alpha", "beta")}
        module, digests, calls = self.fake_store(tmp_path, monkeypatch, folders)
        ref = module.PackageRef("terminal-bench", "terminal-bench", "4.0.0")
        folder = module.fetch_package_dataset(ref, tmp_path / "datasets")
        current = (folder / "manifest.toml").read_text()

        def downloads():
            return sum(1 for c in calls if c[1].startswith("/storage/"))

        def legacy(landed):
            lines = [
                "[dataset]",
                f'name = "{ref.package}"',
                f'version = "{ref.ref}"',
                'source = "harbor-package-store"',
                f'content_hash = "sha256:{"d" * 64}"',
                "",
            ]
            for name in sorted(digests):
                lines += [
                    f'[tasks."{name}"]',
                    f'package = "terminal-bench/{name}"',
                    f'content_hash = "sha256:{digests[name]}"',
                ]
                if name in landed:
                    lines.append(f'digest = "sha256:{digests[name]}"')
                lines.append("")
            return "\n".join(lines)

        # Unchanged: two fetches download nothing, and the first upgrades the manifest to what this Gym writes.
        (folder / "manifest.toml").write_text(legacy({"alpha", "beta"}))
        assert downloads() == 2
        for _ in range(2):
            assert module.fetch_package_dataset(ref, tmp_path / "datasets") == folder
            assert downloads() == 2 and (folder / "manifest.toml").read_text() == current
        (folder / "manifest.toml").write_text(legacy({"alpha", "beta"}))
        # An upgraded legacy manifest holds what the legacy one recorded: the pins and digests, but not a fetch
        # reference, which only a fetch made after the field existed writes.
        upgraded = re.sub(r"^ref = .*\n", "", current, flags=re.M)
        assert check_manifest(folder).text() == upgraded and (folder / "manifest.toml").read_text() == upgraded

        # Edited: refused with the recorded digest named once (not `sha256:sha256:`), the earlier manifest kept.
        (folder / "manifest.toml").write_text(legacy({"alpha", "beta"}))
        (folder / "alpha" / "instruction.md").write_text("edited locally\n")
        edited = content_hash(folder / "alpha")
        with pytest.raises(module.PackageStoreError) as info:
            module.fetch_package_dataset(ref, tmp_path / "datasets")
        message = str(info.value)
        assert message.startswith(f"{folder / 'alpha'} differs from the store: its content hash is sha256:{edited}")
        assert f"hashed to sha256:{digests['alpha']}" in message and "sha256:sha256" not in message
        with pytest.raises(ManifestError, match=r"does not match its manifest.toml \(changed: alpha\)"):
            prepare_target(str(folder), output_root=tmp_path / "out")
        assert (folder / "manifest.toml").read_text() == legacy({"alpha", "beta"}) and downloads() == 2
        assert (folder / "alpha" / "instruction.md").read_text() == "edited locally\n"

        # Interrupted before beta landed: prepare refuses the folder until the fetch completes it, downloading beta alone.
        (folder / "alpha").rename(tmp_path / "alpha-edited")
        shutil.copytree(folders["alpha"], folder / "alpha")
        shutil.rmtree(folder / "beta")
        (folder / "manifest.toml").write_text(legacy({"alpha"}))
        with pytest.raises(ManifestError, match=r"interrupted .*not yet recorded: beta\)"):
            check_manifest(folder)
        assert (folder / "manifest.toml").read_text() == legacy({"alpha"})
        assert module.fetch_package_dataset(ref, tmp_path / "datasets") == folder
        assert downloads() == 3 and (folder / "manifest.toml").read_text() == current
        assert [task.task_id for task in discover_tasks(folder)] == ["alpha", "beta"]
