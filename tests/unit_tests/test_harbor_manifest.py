# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset identity (M33): the derived-only ``manifest.toml``, the prepare check and the recorded digest."""

import json
import os
import shutil
import subprocess
import tomllib
from asyncio import Future
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

import nemo_gym.rollout_collection
import tests.unit_tests.test_harbor_tasks as task_tests
import tests.unit_tests.test_rollout_collection as collection_tests
from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper, _CompletedRollout
from nemo_gym.tasks.harbor.cli import BuildOptions, build_run, prepare_target, resolve_agent
from nemo_gym.tasks.harbor.dataset_config import GymDatasetConfig, apply_dataset_config, read_dataset_config
from nemo_gym.tasks.harbor.digest import DATASET_DIGEST_KEY, DIGEST_KEY, content_hash
from nemo_gym.tasks.harbor.hub import HubError, RegistryDataset, RegistryTask, fetch_dataset
from nemo_gym.tasks.harbor.manifest import (
    MANIFEST_FILE,
    Manifest,
    ManifestError,
    check_manifest,
    dataset_digest,
    dataset_toml_digest,
    downstream_digest,
    task_differences,
)
from nemo_gym.tasks.harbor.materialize import run_config, write_rows
from nemo_gym.tasks.harbor.task import load_task
from tests.unit_tests.test_harbor_tasks import (  # noqa: F401  (fixtures)
    commit_repo,
    fake_registry,
    registry_dataset,
    registry_repo,
    write_task,
)


def manifest_of(folder: Path) -> dict:
    return tomllib.loads((folder / MANIFEST_FILE).read_text())


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
        assert set(data["dataset"]) == {"source", "digest", "dataset_toml_digest", "downstream_digest"}
        assert data["tasks"]["alpha"]["digest"] == content_hash(folder / "alpha")

    def test_dataset_digest_covers_tasks_dataset_toml_and_downstream(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        base = Manifest.build(folder, source="local")
        assert base.digest == dataset_digest(
            {"a": content_hash(folder / "a")}, base.dataset_toml_digest, base.downstream_digest
        )
        assert (
            base.downstream_digest == downstream_digest(folder) == dataset_toml_digest(folder)
        )  # both absent: digest of nothing

        (folder / "dataset.toml").write_text("[gym.defaults]\ntimeout_multiplier = 2.0\n")
        with_toml = Manifest.build(folder, source="local")
        # A timeout override changes the score, so it changes the dataset.
        assert with_toml.digest != base.digest and with_toml.dataset_toml_digest != base.dataset_toml_digest
        assert with_toml.tasks == base.tasks

        (folder / "a" / "instruction.md").write_text("edited\n")
        with_edit = Manifest.build(folder, source="local")
        assert with_edit.digest not in (base.digest, with_toml.digest)
        assert with_edit.tasks["a"]["digest"] != base.tasks["a"]["digest"]

        (folder / "downstream").mkdir()
        (folder / "downstream" / "fix.patch").write_text("--- a\n+++ b\n")
        with_patch = Manifest.build(folder, source="local")
        assert with_patch.downstream_digest != with_edit.downstream_digest and with_patch.digest != with_edit.digest

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
        import nemo_gym.tasks.harbor.manifest as module

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

    def test_only_the_gym_layer_moving_rewrites_the_manifest_and_keeps_pins(self, tmp_path, capsys):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        before = fetched_manifest(folder, ["a"])
        (folder / "dataset.toml").write_text('[gym.tasks."a".environment]\nenv = { X = "1" }\n')
        after = check_manifest(folder)
        assert after.source == "harbor-registry" and after.pins == before.pins
        assert after.dataset_toml_digest != before.dataset_toml_digest and after.digest != before.digest
        assert after.tasks["a"]["digest"] == before.tasks["a"]["digest"]
        assert "dataset.toml or downstream/ changed" in capsys.readouterr().out

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


@pytest.mark.usefixtures("fake_registry")  # prepare records base images; never reach Docker Hub
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
            {"solo": prepared.tasks[0].digest}, dataset_toml_digest(tmp_path), downstream_digest(tmp_path)
        )
        assert not (tmp_path / MANIFEST_FILE).exists() and not (task / MANIFEST_FILE).exists()
        row = json.loads(prepared.rows_path.read_text())
        assert row["task_input"]["task_data"][DATASET_DIGEST_KEY] == prepared.dataset_digest

    def test_prepare_refuses_a_hand_edited_fetched_dataset(self, tmp_path):
        folder = tmp_path / "ds"
        for name in ("a", "b"):
            write_task(folder / name)
        fetched_manifest(folder, ["a", "b"])
        (folder / "a" / "solution" / "solve.sh").write_text("#!/bin/bash\necho cheat > hello.txt\n")
        with pytest.raises(ManifestError, match="edited by hand"):
            prepare_target(str(folder), output_root=tmp_path / "out")
        assert not (tmp_path / "out" / "ds" / "tasks.jsonl").exists()

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

    def test_the_environment_server_accepts_the_recorded_identity(self, tmp_path):
        from environment_servers.harbor.app import HarborTasksetConfig
        from environment_servers.harbor.task_data import TaskData

        config = HarborTasksetConfig.model_validate(
            {
                "folder": str(tmp_path),
                "tasks": {"a": "1" * 64},
                "dataset_digest": "2" * 64,
                "dataset_toml_digest": "3" * 64,
            }
        )
        assert config.dataset_digest == "2" * 64
        assert TaskData.model_fields["ng_dataset_digest"].alias == DATASET_DIGEST_KEY
        data = TaskData.model_validate({DIGEST_KEY: "abc", DATASET_DIGEST_KEY: "def"})
        assert (
            data.ng_dataset_digest == "def" and TaskData.model_validate({DIGEST_KEY: "abc"}).ng_dataset_digest is None
        )


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
        store_tests = task_tests.TestPackageStore()
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
        from nemo_gym.tasks.harbor.scaffold import init_dataset

        folder = init_dataset(tmp_path, "my-dataset")
        text = (folder / "dataset.toml").read_text()
        assert "[gym]" in text and '# type = "validation"' in text and '# license = "Apache 2.0"' in text
        assert read_dataset_config(folder).is_empty


def registry_of(repo: Path, names: tuple[str, ...]) -> RegistryDataset:
    return RegistryDataset(
        "ds", "1.0", "", tuple(RegistryTask(name, repo.as_uri(), "HEAD", f"tasks/{name}") for name in names)
    )


class TestFetchersRecordOnlyWhatTheyFetched:
    """A folder nobody fetched is never blessed: the fetch refuses it and `Manifest.build` records named tasks only."""

    def test_build_with_names_records_those_tasks_and_hashes_nothing_it_knows(self, tmp_path, monkeypatch):
        import nemo_gym.tasks.harbor.manifest as module

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
        import nemo_gym.tasks.harbor.hub as hub
        import nemo_gym.tasks.harbor.manifest as module

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
        store_tests = task_tests.TestPackageStore()
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
        store_tests = task_tests.TestPackageStore()
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
        with pytest.raises(ManifestError, match="schema_version 2; this Gym reads 1"):
            Manifest.read(tmp_path)
        (tmp_path / MANIFEST_FILE).write_text('schema_version = 1.0\n\n[dataset]\nsource = "local"\n')
        with pytest.raises(ManifestError, match="schema_version 1.0; this Gym reads 1"):
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
        import nemo_gym.tasks.harbor.cli as harbor_cli

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
        monkeypatch.setattr(nemo_gym.rollout_collection, "get_global_config_dict", lambda: {})
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
