# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``gym dataset pull`` and committed dataset entries under ``environments/harbor/`` (M38, second slice).

The checkout's own entries never enter these tests: every test runs against a temporary search root with the
datasets dir inside it, so what a pull lands in is what the test wrote.
"""

import json
import sys
from pathlib import Path

import pytest

from nemo_gym.tasks.harbor import cli as harbor_cli
from nemo_gym.tasks.harbor import entries as entries_module
from nemo_gym.tasks.harbor import hub as hub_module
from nemo_gym.tasks.harbor.cli import PulledDataset, entry_folder, prepare_target, pull_dataset
from nemo_gym.tasks.harbor.dataset_config import GymDatasetConfig, read_dataset_config
from nemo_gym.tasks.harbor.digest import content_hash
from nemo_gym.tasks.harbor.entries import (
    DatasetEntry,
    committed_entries,
    entry_for_ref,
    entry_target,
    find_entry,
    plan_pull,
    prepare_destination,
)
from nemo_gym.tasks.harbor.errors import HarborTaskError
from nemo_gym.tasks.harbor.hub import HubError
from nemo_gym.tasks.harbor.manifest import Manifest, dataset_files_digest, dataset_toml_digest
from nemo_gym.tasks.harbor.task_data import load_task_data_schema
from tests.unit_tests.test_harbor_tasks import (
    fake_registry,  # noqa: F401
    no_real_registry,  # noqa: F401 (autouse fixture)
    registry_dataset,
    registry_repo,  # noqa: F401
    write_task,
)


TB4_REF = "harbor:terminal-bench/terminal-bench@4.0.0"


@pytest.fixture
def root(tmp_path, monkeypatch) -> Path:
    """A search root of its own: its ``environments/harbor/`` is both where entries are committed and the datasets
    dir, as in a checkout; the working directory is the root, as on a command line run from the checkout."""
    monkeypatch.setattr(entries_module, "_roots", lambda: [tmp_path])
    monkeypatch.setenv("NEMO_GYM_DATASETS_DIR", str(tmp_path / "environments" / "harbor"))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "environments" / "harbor").mkdir(parents=True)
    return tmp_path


def write_entry(root: Path, name: str, source: str | None, *, extra: str = "") -> Path:
    """A committed entry: README, a YAML overlay and a dataset.toml declaring ``source`` (none when ``None``)."""
    folder = root / "environments" / "harbor" / name
    folder.mkdir(parents=True)
    (folder / "README.md").write_text(f"# {name}\n")
    (folder / f"{name}.yaml").write_text("sandbox: {}\n")
    table = f'source = "{source}"\n' if source is not None else ""
    (folder / "dataset.toml").write_text(f"[gym]\n{table}{extra}")
    return folder


def fake_store(tmp_path, monkeypatch, folders):
    """Harbor's package store, answering for ``terminal-bench/terminal-bench@4.0.0`` with ``folders`` as its tasks."""
    from tests.unit_tests.test_harbor_package_store import TestPackageStore

    return TestPackageStore().fake_store(tmp_path, monkeypatch, folders)


def run_main(monkeypatch, *argv: str) -> None:
    from nemo_gym.cli.main import main

    monkeypatch.setattr(sys, "argv", ["gym", *argv])
    main()


def main_error(monkeypatch, capsys, *argv: str) -> str:
    """The one-line parser error ``gym <argv>`` ends with (exit status 2)."""
    with pytest.raises(SystemExit) as exc:
        run_main(monkeypatch, *argv)
    assert exc.value.code == 2
    return capsys.readouterr().err


class TestSourceKey:
    """``[gym] source`` is the hub reference a committed entry is pulled from; nothing else about the table moves."""

    @pytest.mark.parametrize(
        "source",
        [TB4_REF, "harbor:terminal-bench/terminal-bench-2-1@sha256:" + "7d" * 32, "harbor:ds@1.0", "harbor:ds"],
    )
    def test_accepts_package_store_and_registry_references(self, tmp_path, source):
        (tmp_path / "dataset.toml").write_text(f'[gym]\nsource = "{source}"\n')
        config = read_dataset_config(tmp_path)
        assert config.source == source
        assert not config.is_empty and not config.shapes_tasks  # provenance, not a task-shaping setting

    @pytest.mark.parametrize("source", ["terminal-bench/terminal-bench@4.0.0", "harbor:", "harbor:a b", 3])
    def test_refuses_what_is_not_a_hub_reference_naming_the_key(self, tmp_path, source):
        value = json.dumps(source)
        (tmp_path / "dataset.toml").write_text(f"[gym]\nsource = {value}\n")
        with pytest.raises(ValueError, match="source"):
            read_dataset_config(tmp_path)

    def test_default_is_unset_and_the_scaffold_documents_it_without_setting_it(self, tmp_path):
        from nemo_gym.tasks.harbor.scaffold import init_dataset

        assert GymDatasetConfig().source is None and GymDatasetConfig().is_empty
        folder = init_dataset(tmp_path, "my-dataset")
        assert '# source = "harbor:<org>/<name>@<tag>"' in (folder / "dataset.toml").read_text()
        assert read_dataset_config(folder).is_empty

    def test_committed_entries_declare_their_sources(self):
        """The two committed entries pin the datasets their READMEs describe, and nothing else in their tables."""
        from nemo_gym import PARENT_DIR

        tb4 = read_dataset_config(PARENT_DIR / "environments" / "harbor" / "terminal_bench_4")
        tb21 = read_dataset_config(PARENT_DIR / "environments" / "harbor" / "terminal_bench_2_1")
        assert tb4.source == TB4_REF
        assert tb21.source == (
            "harbor:terminal-bench/terminal-bench-2-1@sha256:"
            "7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a"  # pragma: allowlist secret
        )
        for config in (tb4, tb21):
            assert config.model_copy(update={"source": None}).is_empty


class TestEntryLookup:
    def test_find_entry_reads_the_source_and_resolves_without_one(self, root):
        write_entry(root, "tb4", TB4_REF)
        write_entry(root, "swe", None)
        assert find_entry("tb4") == DatasetEntry("tb4", root / "environments" / "harbor" / "tb4", TB4_REF)
        assert find_entry("swe") == DatasetEntry("swe", root / "environments" / "harbor" / "swe", None)
        assert find_entry("missing") is None
        assert find_entry("../tb4") is None and find_entry("tb4/") is None

    def test_find_entry_reports_an_unreadable_table_but_the_mapping_skips_it(self, root):
        write_entry(root, "tb4", TB4_REF)
        broken = root / "environments" / "harbor" / "broken"
        broken.mkdir()
        (broken / "dataset.toml").write_text("[gym]\nsource = 3\n")
        with pytest.raises(ValueError, match="source"):
            find_entry("broken")
        assert [entry.name for entry in committed_entries()] == ["tb4"]

    def test_committed_entries_need_a_dataset_toml_with_a_source_and_skip_pulled_hub_folders(self, root):
        write_entry(root, "tb4", TB4_REF)
        write_entry(root, "swe", None)  # an entry without a source is not a pull destination
        pulled = root / "environments" / "harbor" / "terminal-bench-4.0.0"  # a hub folder: no dataset.toml
        write_task(pulled / "a")
        (root / "environments" / "harbor" / ".harbor").mkdir()  # the registry cache
        (root / "environments" / "harbor" / "notes.md").write_text("not a folder\n")
        assert [entry.name for entry in committed_entries()] == ["tb4"]

    def test_an_earlier_root_wins_a_name(self, root, tmp_path, monkeypatch):
        other = tmp_path / "other"
        write_entry(root, "tb4", TB4_REF)
        write_entry(other, "tb4", "harbor:ds@1.0")
        monkeypatch.setattr(entries_module, "_roots", lambda: [other, root])
        assert find_entry("tb4").source == "harbor:ds@1.0"
        assert [entry.folder for entry in committed_entries()] == [other / "environments" / "harbor" / "tb4"]

    def test_an_entry_in_another_root_is_pulled_into_its_own_folder(self, root, tmp_path, monkeypatch):
        """Without NEMO_GYM_DATASETS_DIR an entry lands in its own folder wherever the root that commits it lives (a
        NEMO_GYM_EXTRA_ROOTS root here), never in the Gym checkout's environments/harbor/; set, the variable moves
        the entry's pull as it moves every other."""
        extra = tmp_path / "extra"
        monkeypatch.setattr(entries_module, "_roots", lambda: [extra, root])
        monkeypatch.delenv("NEMO_GYM_DATASETS_DIR")
        folder = write_entry(extra, "tb4", TB4_REF).resolve()
        entry = find_entry("tb4")
        assert entry.folder == folder and entry.destination == folder
        assert plan_pull(TB4_REF).folder == folder and plan_pull("tb4").folder == folder
        assert prepare_destination(entry) == folder and sorted(p.name for p in folder.iterdir()) == [
            "README.md",
            "dataset.toml",
            "tb4.yaml",
        ]
        assert hub_module.datasets_dir() != folder.parent  # the checkout's datasets dir is elsewhere, untouched
        fake_store(root, monkeypatch, {"alpha": write_task(root / "src" / "alpha")})
        run_main(monkeypatch, "dataset", "pull", "tb4")
        assert (folder / "alpha" / "task.toml").is_file() and Manifest.read(folder) is not None
        assert not (hub_module.datasets_dir() / "tb4").exists()
        monkeypatch.setenv("NEMO_GYM_DATASETS_DIR", str(tmp_path / "shared"))
        assert find_entry("tb4").destination == (tmp_path / "shared" / "tb4").resolve()

    def test_entry_for_ref_matches_the_same_dataset_and_version_only(self, root):
        write_entry(root, "tb4", TB4_REF)
        assert entry_for_ref(TB4_REF).name == "tb4"
        # `latest` and another tag are other references; a plain reference never lands in a pinned entry.
        assert entry_for_ref("harbor:terminal-bench/terminal-bench") is None
        assert entry_for_ref("harbor:terminal-bench/terminal-bench@3.0.0") is None
        assert entry_for_ref("harbor:terminal-bench@4.0.0") is None  # a registry dataset of that name

    def test_two_entries_with_one_source_are_refused_with_both_named(self, root):
        write_entry(root, "tb4", TB4_REF)
        write_entry(root, "tb4_copy", TB4_REF)
        with pytest.raises(HubError, match=r"2 entries .*\(tb4, tb4_copy\)"):
            entry_for_ref(TB4_REF)
        with pytest.raises(HubError, match="tb4_copy"):
            plan_pull(TB4_REF)

    def test_entry_target_is_a_bare_name_that_is_not_a_path(self, root):
        write_entry(root, "tb4", TB4_REF)
        assert entry_target("tb4").name == "tb4"
        assert entry_target("environments/harbor/tb4") is None  # a path, resolved as a path
        assert entry_target("./tb4") is None and entry_target(TB4_REF) is None
        assert entry_target(".") is None and entry_target("..") is None
        # An existing folder of the entry's name in the working directory keeps meaning that folder.
        (root / "tb4").mkdir()
        assert entry_target("tb4") is None


class TestPlanPull:
    def test_a_reference_an_entry_declares_lands_in_the_entry(self, root):
        folder = write_entry(root, "tb4", TB4_REF)
        plan = plan_pull(TB4_REF)
        assert plan.ref == TB4_REF and plan.entry.name == "tb4" and plan.folder == folder

    def test_a_reference_no_entry_declares_lands_in_the_hub_folder(self, root):
        write_entry(root, "tb4", TB4_REF)
        plan = plan_pull("harbor:terminal-bench/terminal-bench")
        assert plan.entry is None and plan.folder is None

    def test_an_entry_name_pulls_its_source_into_its_folder(self, root):
        folder = write_entry(root, "tb4", TB4_REF)
        plan = plan_pull("tb4")
        assert plan.ref == TB4_REF and plan.folder == folder

    def test_an_entry_without_a_source_has_nothing_to_pull(self, root):
        write_entry(root, "swe", None)
        with pytest.raises(HubError, match=r"declares no \[gym\] source.*entry swe"):
            plan_pull("swe")

    def test_a_path_is_not_a_pull(self, root):
        write_entry(root, "tb4", TB4_REF)
        assert plan_pull("some/folder") is None and plan_pull("nonsense") is None
        assert plan_pull("./tb4") is None and plan_pull("environments/harbor/tb4") is None


class TestPrepareDestination:
    def test_the_entry_folder_itself_needs_nothing(self, root):
        folder = write_entry(root, "tb4", TB4_REF)
        before = sorted(path.name for path in folder.iterdir())
        assert prepare_destination(find_entry("tb4")) == folder
        assert sorted(path.name for path in folder.iterdir()) == before

    def test_a_moved_datasets_dir_gets_a_copy_of_the_entry_table(self, root, tmp_path, monkeypatch):
        folder = write_entry(root, "tb4", TB4_REF, extra="verified = true\n")
        monkeypatch.setenv("NEMO_GYM_DATASETS_DIR", str(tmp_path / "shared"))
        entry = find_entry("tb4")
        destination = prepare_destination(entry)
        assert destination == (tmp_path / "shared" / "tb4").resolve()
        assert (destination / "dataset.toml").read_bytes() == (folder / "dataset.toml").read_bytes()
        # The same copy again is fine. With nothing pulled yet (no manifest, no task folder: a first fetch that
        # failed before claiming the folder) the copy is this function's own and follows the entry's file.
        assert prepare_destination(entry) == destination
        (destination / "dataset.toml").write_text("[gym]\nverified = false\n")
        assert prepare_destination(entry) == destination
        assert (destination / "dataset.toml").read_bytes() == (folder / "dataset.toml").read_bytes()
        # Beside task folders no pull recorded, a differing copy may be the user's own table: refused, never
        # overwritten.
        write_task(destination / "alpha")
        (destination / "dataset.toml").write_text("[gym]\nverified = false\n")
        with pytest.raises(HubError, match="was edited in place and differs from the entry's"):
            prepare_destination(entry)
        assert (destination / "dataset.toml").read_text() == "[gym]\nverified = false\n"

    def test_a_copy_the_last_pull_recorded_follows_the_entry_table(self, root, tmp_path, monkeypatch):
        """The copy is replaced when the entry's file changed and the copy is still the one the manifest hashed;
        a prepare (``create=False``) does the same and leaves a folder that was never pulled missing."""
        entry = write_entry(root, "tb4", TB4_REF)
        monkeypatch.setenv("NEMO_GYM_DATASETS_DIR", str(tmp_path / "shared"))
        destination = (tmp_path / "shared" / "tb4").resolve()
        assert prepare_destination(find_entry("tb4"), create=False) == destination and not destination.exists()
        prepare_destination(find_entry("tb4"))
        write_task(destination / "alpha")
        Manifest.build(destination, source="local").write(destination)  # what a pull leaves: the copy's digest
        (entry / "dataset.toml").write_text(f'[gym]\nsource = "{TB4_REF}"\nverified = true\n')
        for create in (False, True):
            assert prepare_destination(find_entry("tb4"), create=create) == destination
            assert (destination / "dataset.toml").read_bytes() == (entry / "dataset.toml").read_bytes()
        # The record follows the copy: the manifest's dataset_toml_digest is the new copy's, the task record is kept
        # as it was, and a second edit of the entry's file, with nothing rewriting the manifest in between, follows
        # as well instead of reading as an in-place edit (round 2).
        recorded = Manifest.read(destination)
        assert recorded.dataset_toml_digest == dataset_toml_digest(destination) == dataset_toml_digest(entry)
        assert recorded.tasks["alpha"]["digest"] == content_hash(destination / "alpha")
        (entry / "dataset.toml").write_text(f'[gym]\nsource = "{TB4_REF}"\nverified = false\n')
        assert prepare_destination(find_entry("tb4"), create=False) == destination
        assert (destination / "dataset.toml").read_bytes() == (entry / "dataset.toml").read_bytes()
        assert Manifest.read(destination).dataset_toml_digest == dataset_toml_digest(entry)
        # Edited in place after that pull: refused by both, and kept.
        (destination / "dataset.toml").write_text("[gym]\nverified = false\n")
        for create in (False, True):
            with pytest.raises(HubError, match="was edited in place and differs from the entry's"):
                prepare_destination(find_entry("tb4"), create=create)
        assert (destination / "dataset.toml").read_text() == "[gym]\nverified = false\n"


class TestPullDataset:
    """The pull itself, through the package store (faked at its HTTP boundary) and the registry (a local git repo)."""

    def test_pulls_an_entry_into_its_folder_and_reports_tasks_and_digest(self, root, monkeypatch, capsys):
        folders = {name: write_task(root / "src" / name) for name in ("beta", "alpha")}
        module, digests, calls = fake_store(root, monkeypatch, folders)
        folder = write_entry(root, "tb4", TB4_REF)
        committed = {path.name: path.read_bytes() for path in folder.iterdir()}

        run_main(monkeypatch, "dataset", "pull", "tb4")

        out = capsys.readouterr().out.splitlines()
        manifest = Manifest.read(folder)
        assert sorted(path.name for path in folder.iterdir() if path.is_dir()) == ["alpha", "beta"]
        assert content_hash(folder / "alpha") == digests["alpha"]
        assert manifest.source == "harbor-package-store" and set(manifest.tasks) == {"alpha", "beta"}
        assert out == [
            f"Pulled {TB4_REF} into {folder} (entry tb4)",
            f"2 task(s), dataset digest {manifest.digest} (harbor-package-store)",
            "Next: gym dataset validate tb4 --sandbox <provider>",
        ]
        # The committed files are untouched, and nothing was prepared: no rows, records or results.
        assert {name: (folder / name).read_bytes() for name in committed} == committed
        assert not (folder / "compose-images.json").exists() and not (root / "results").exists()
        # The dataset's identity covers the entry's table (dataset.toml is hashed into the manifest).
        assert manifest.dataset_toml_digest != Manifest.build(root / "src", source="local").dataset_toml_digest

    def test_a_second_pull_downloads_nothing_and_says_so(self, root, monkeypatch, capsys):
        folders = {"alpha": write_task(root / "src" / "alpha")}
        _, _, calls = fake_store(root, monkeypatch, folders)
        folder = write_entry(root, "tb4", TB4_REF)
        run_main(monkeypatch, "dataset", "pull", "tb4")
        first = capsys.readouterr().out
        downloads = sum(1 for call in calls if call[1].startswith("/storage/"))
        manifest = (folder / "manifest.toml").read_bytes()

        run_main(monkeypatch, "dataset", "pull", "tb4")

        second = capsys.readouterr().out
        assert sum(1 for call in calls if call[1].startswith("/storage/")) == downloads
        assert (folder / "manifest.toml").read_bytes() == manifest
        assert "unchanged since the last pull" in second and "unchanged" not in first
        assert second.splitlines()[0] == first.splitlines()[0]

    def test_the_reference_and_the_entry_name_are_one_folder(self, root, monkeypatch, capsys):
        """Pulling the reference an entry declares lands in the entry, so the dataset never has a sibling copy."""
        folders = {"alpha": write_task(root / "src" / "alpha")}
        fake_store(root, monkeypatch, folders)
        folder = write_entry(root, "tb4", TB4_REF)

        run_main(monkeypatch, "dataset", "pull", TB4_REF)

        assert capsys.readouterr().out.splitlines()[0] == f"Pulled {TB4_REF} into {folder} (entry tb4)"
        harbor = root / "environments" / "harbor"
        assert sorted(path.name for path in harbor.iterdir()) == ["tb4"]
        run_main(monkeypatch, "dataset", "pull", "tb4")
        assert "unchanged since the last pull" in capsys.readouterr().out
        assert sorted(path.name for path in harbor.iterdir()) == ["tb4"]

    def test_a_reference_without_an_entry_lands_in_the_hub_folder_as_before(self, root, monkeypatch, capsys):
        folders = {"alpha": write_task(root / "src" / "alpha")}
        fake_store(root, monkeypatch, folders)
        write_entry(root, "other", "harbor:ds@1.0")  # an entry for another dataset does not attract this one

        run_main(monkeypatch, "dataset", "pull", TB4_REF)

        folder = (root / "environments" / "harbor" / "terminal-bench-4.0.0").resolve()
        out = capsys.readouterr().out.splitlines()
        assert out[0] == f"Pulled {TB4_REF} into {folder}"
        assert out[2] == f"Next: gym dataset validate {folder} --sandbox <provider>"
        assert (folder / "alpha" / "task.toml").is_file() and Manifest.read(folder).tasks.keys() == {"alpha"}
        # The hub folder's name follows from the reference, so a second pull of it reports unchanged too.
        run_main(monkeypatch, "dataset", "pull", TB4_REF)
        assert "unchanged since the last pull" in capsys.readouterr().out

    def test_expected_folder_follows_from_the_reference_alone(self, root):
        from nemo_gym.tasks.harbor.hub import expected_folder

        harbor = (root / "environments" / "harbor").resolve()
        assert expected_folder(TB4_REF) == harbor / "terminal-bench-4.0.0"
        assert expected_folder("harbor:terminal-bench/terminal-bench") == harbor / "terminal-bench"
        assert expected_folder("harbor:o/n@sha256:" + "7d" * 32) == harbor / ("n-" + "7d" * 6)
        assert expected_folder("harbor:ds@1.0") == harbor / "ds-1.0"
        assert expected_folder("harbor:ds") is None  # the registry picks the version
        assert expected_folder("harbor:ds@1.0", root / "elsewhere") == root / "elsewhere" / "ds-1.0"

    def test_resolved_folder_asks_the_registry_only_when_the_reference_leaves_it_the_version(
        self, root, monkeypatch, tmp_path
    ):
        from nemo_gym.tasks.harbor.hub import RegistryDataset, resolved_folder

        harbor = (root / "environments" / "harbor").resolve()
        datasets = [RegistryDataset("ds", version, "", ()) for version in ("1.0", "2.0")]
        loads = []

        def load_registry(cache_dir, *, refresh=False):
            loads.append((cache_dir, refresh))
            return datasets

        monkeypatch.setattr(hub_module, "load_registry", load_registry)
        assert resolved_folder(TB4_REF) == harbor / "terminal-bench-4.0.0" and loads == []
        assert resolved_folder("harbor:ds@1.0") == harbor / "ds-1.0" and loads == []
        assert resolved_folder("harbor:ds") == harbor / "ds-2.0"  # the registry's newest
        assert (
            resolved_folder("harbor:ds", tmp_path / "elsewhere", refresh_registry=True)
            == tmp_path / "elsewhere" / "ds-2.0"
        )
        assert loads == [(harbor / ".harbor", False), (tmp_path / "elsewhere" / ".harbor", True)]
        with pytest.raises(HubError, match="not in the Harbor registry"):
            resolved_folder("harbor:other")

    def test_a_registry_reference_without_a_version_reports_unchanged_on_the_second_pull(
        self, root, monkeypatch, registry_repo, capsys
    ):  # noqa: F811
        """`harbor:ds` leaves the version to the registry, so the folder is known only after resolution; the pull
        resolves first and the second pull says it downloaded nothing, as the versioned form does (round 3)."""
        dataset = registry_dataset(registry_repo)
        loads = []

        def load_registry(cache_dir, *, refresh=False):
            loads.append(refresh)
            return [dataset]

        monkeypatch.setattr(hub_module, "load_registry", load_registry)
        folder = (root / "environments" / "harbor" / "ds-1.0").resolve()
        run_main(monkeypatch, "dataset", "pull", "harbor:ds")
        first = capsys.readouterr().out.splitlines()
        assert first[0] == f"Pulled harbor:ds into {folder}" and "unchanged" not in first[1]
        run_main(monkeypatch, "dataset", "pull", "harbor:ds")
        second = capsys.readouterr().out.splitlines()
        assert second[0] == first[0] and second[1] == first[1] + ", unchanged since the last pull"
        assert loads == [False] * 4  # the cache is read, never downloaded twice for one pull
        # A refresh asked for by prepare_target's kwarg downloads the registry once, for the resolution and the fetch.
        pull_dataset("harbor:ds", refresh_registry=True)
        assert loads[4:] == [True, False]

    def test_a_registry_reference_pulls_through_git_into_the_entry(self, root, monkeypatch, registry_repo, capsys):  # noqa: F811
        dataset = registry_dataset(registry_repo)
        monkeypatch.setattr(hub_module, "load_registry", lambda cache_dir, *, refresh=False: [dataset])
        folder = write_entry(root, "ds_entry", "harbor:ds@1.0")

        run_main(monkeypatch, "dataset", "pull", "ds_entry")

        manifest = Manifest.read(folder)
        out = capsys.readouterr().out.splitlines()
        assert (folder / "a" / "task.toml").is_file()
        assert manifest.source == "harbor-registry" and manifest.pins["a"]["path"] == "tasks/a"
        assert out[0] == f"Pulled harbor:ds@1.0 into {folder} (entry ds_entry)"
        assert out[1] == f"1 task(s), dataset digest {manifest.digest} (harbor-registry)"
        # A registry pull with every folder in place writes nothing; the report says so.
        run_main(monkeypatch, "dataset", "pull", "ds_entry")
        assert "unchanged since the last pull" in capsys.readouterr().out

    def test_refresh_registry_is_prepare_targets_pass_through_not_a_flag(
        self, root, monkeypatch, capsys, registry_repo
    ):  # noqa: F811
        """``prepare_target(refresh_registry=True)`` reaches the registry loader through the pull; the command takes
        no such flag (none of the three commands does; deleting ``.harbor/registry.json`` refreshes the cache)."""
        seen = []
        dataset = registry_dataset(registry_repo)

        def load(cache_dir, *, refresh=False):
            seen.append((Path(cache_dir), refresh))
            return [dataset]

        monkeypatch.setattr(hub_module, "load_registry", load)
        write_entry(root, "ds_entry", "harbor:ds@1.0")
        pull_dataset("ds_entry", refresh_registry=True)
        run_main(monkeypatch, "dataset", "pull", "ds_entry")
        cache = (root / "environments" / "harbor").resolve() / ".harbor"
        assert seen == [(cache, True), (cache, False)]
        err = main_error(monkeypatch, capsys, "dataset", "pull", "ds_entry", "--refresh-registry")
        assert "unrecognized arguments: --refresh-registry" in err

    def test_a_moved_datasets_dir_holds_the_tasks_and_the_entry_table(self, root, tmp_path, monkeypatch, capsys):
        folders = {"alpha": write_task(root / "src" / "alpha")}
        fake_store(root, monkeypatch, folders)
        entry = write_entry(root, "tb4", TB4_REF, extra="verified = true\n")
        monkeypatch.setenv("NEMO_GYM_DATASETS_DIR", str(tmp_path / "shared"))

        run_main(monkeypatch, "dataset", "pull", "tb4")

        destination = (tmp_path / "shared" / "tb4").resolve()
        assert capsys.readouterr().out.splitlines()[0] == f"Pulled {TB4_REF} into {destination} (entry tb4)"
        assert (destination / "alpha" / "task.toml").is_file()
        assert (destination / "dataset.toml").read_bytes() == (entry / "dataset.toml").read_bytes()
        assert not (entry / "alpha").exists() and not (entry / "manifest.toml").exists()
        assert (
            Manifest.read(destination).dataset_toml_digest == Manifest.build(entry, source="local").dataset_toml_digest
        )

    def test_a_moved_datasets_dir_gets_the_entry_s_dataset_level_files_too(self, root, tmp_path, monkeypatch, capsys):
        """An entry's task_data.py and verifier.py enter the dataset digest and shape loading as its dataset.toml
        does, so they travel with it to a moved destination, follow the entry's files (a file the entry dropped goes
        too) and are refused when a copy was edited in place (round 3)."""
        fake_store(root, monkeypatch, {"alpha": write_task(root / "src" / "alpha")})
        entry = write_entry(root, "tb4", TB4_REF)
        model = "from pydantic import BaseModel\n\n\nclass TaskData(BaseModel):\n    answer: str = 'x'\n"
        (entry / "task_data.py").write_text(model)
        (entry / "verifier.py").write_text("def verify(workspace):\n    return 1.0\n")
        monkeypatch.setenv("NEMO_GYM_DATASETS_DIR", str(tmp_path / "shared"))
        destination = (tmp_path / "shared" / "tb4").resolve()

        run_main(monkeypatch, "dataset", "pull", "tb4")

        assert sorted(p.name for p in destination.iterdir()) == [
            "alpha",
            "dataset.toml",
            "manifest.toml",
            "task_data.py",
            "verifier.py",
        ]
        assert (destination / "verifier.py").read_text() == "def verify(workspace):\n    return 1.0\n"
        assert load_task_data_schema(destination) is not None
        manifest = Manifest.read(destination)
        for digest in (dataset_toml_digest, dataset_files_digest):
            assert digest(destination) == digest(entry)
        assert manifest.dataset_files_digest == dataset_files_digest(entry)
        # The entry drops its model and edits the grader: a prepare alone (no pull) brings the copies along, the record
        # follows, and the run reads them from the destination.
        (entry / "task_data.py").unlink()
        (entry / "verifier.py").write_text("def verify(workspace):\n    return 0.5\n")
        capsys.readouterr()
        prepared = prepare_target("tb4", output_root=tmp_path / "out")
        out = capsys.readouterr().out
        assert prepared.folder == destination and not (destination / "task_data.py").exists()
        assert (destination / "verifier.py").read_text() == "def verify(workspace):\n    return 0.5\n"
        second = Manifest.read(destination)
        assert prepared.dataset_digest == second.digest != manifest.digest and second.tasks == manifest.tasks
        assert second.dataset_files_digest == dataset_files_digest(entry) and out.count("Rewrote") == 1
        assert "a dataset-level task_data.py or verifier.py changed" in out
        # A file the entry no longer has goes, and one it gains arrives.
        (entry / "verifier.py").unlink()
        (entry / "task_data.py").write_text(model)
        run_main(monkeypatch, "dataset", "pull", "tb4")
        assert (destination / "task_data.py").exists() and not (destination / "verifier.py").exists()
        assert Manifest.read(destination).dataset_files_digest == dataset_files_digest(entry)
        # A copy edited in place is refused by pull, validate and run alike, and kept.
        (destination / "task_data.py").write_text(model + "# edited here\n")
        (destination / "verifier.py").write_text("# placed here\n")
        expected = (
            f"{destination / 'task_data.py'}, {destination / 'verifier.py'} were edited in place and "
            f"differ from the entry's files under {entry}"
        )
        for argv in (
            ("dataset", "pull", "tb4"),
            ("dataset", "validate", "tb4"),
            ("eval", "run", "tb4", "--agent", "hermes_agent"),
        ):
            err = main_error(monkeypatch, capsys, *argv)
            assert expected in err and "Traceback" not in err, argv
        assert (destination / "task_data.py").read_text() == model + "# edited here\n"
        (destination / "verifier.py").unlink()
        err = main_error(monkeypatch, capsys, "dataset", "pull", "tb4")
        assert f"{destination / 'task_data.py'} was edited in place and differs from the entry's" in err

    def test_an_entry_table_edited_after_the_pull_reaches_pull_and_prepare_alike(
        self, root, tmp_path, monkeypatch, capsys
    ):
        """Under a moved datasets dir the copy beside the tasks follows the entry's file, so `gym dataset pull <entry>`
        and `gym eval run <entry>` read one table: the TB4 README's `nextjs-performance` override, written into the
        entry's dataset.toml after the pull, shapes the next run, and the digest follows."""
        folders = {"alpha": write_task(root / "src" / "alpha")}
        _, _, calls = fake_store(root, monkeypatch, folders)
        entry = write_entry(root, "tb4", TB4_REF)
        monkeypatch.setenv("NEMO_GYM_DATASETS_DIR", str(tmp_path / "shared"))
        destination = (tmp_path / "shared" / "tb4").resolve()
        run_main(monkeypatch, "dataset", "pull", "tb4")
        first = Manifest.read(destination)
        downloads = sum(1 for call in calls if call[1].startswith("/storage/"))
        capsys.readouterr()

        def override(total: str) -> None:
            (entry / "dataset.toml").write_text(
                f'[gym]\nsource = "{TB4_REF}"\n[gym.tasks."alpha".environment]\nenv = {{ CIRCLE_NODE_TOTAL = "{total}" }}\n'
            )

        override("3")
        run_main(monkeypatch, "dataset", "pull", "tb4")
        out = capsys.readouterr().out
        second = Manifest.read(destination)
        assert (destination / "dataset.toml").read_bytes() == (entry / "dataset.toml").read_bytes()
        assert second.digest != first.digest and f"dataset digest {second.digest}" in out and "unchanged" not in out
        # Announced once (round 3): by the copy's refresh, not again by the store fetch that then rewrote the same text.
        assert out.count("Rewrote") == 1 and f"dataset digest {first.digest[:12]} -> {second.digest[:12]}" in out
        assert second.dataset_toml_digest == Manifest.build(entry, source="local").dataset_toml_digest
        assert sum(1 for call in calls if call[1].startswith("/storage/")) == downloads  # nothing fetched again
        # Edited again, a prepare alone (no pull in between) sees the entry's table too.
        override("5")
        prepared = prepare_target("tb4", output_root=tmp_path / "out")
        assert prepared.folder == destination and prepared.tasks[0].env == {"CIRCLE_NODE_TOTAL": "5"}
        assert prepared.dataset_toml_digest == Manifest.build(entry, source="local").dataset_toml_digest
        assert prepared.dataset_digest == Manifest.read(destination).digest != second.digest
        run_main(monkeypatch, "dataset", "pull", "tb4")
        assert "unchanged since the last pull" in capsys.readouterr().out

    def test_a_copy_edited_in_place_is_refused_by_pull_and_prepare_alike(self, root, tmp_path, monkeypatch, capsys):
        folders = {"alpha": write_task(root / "src" / "alpha")}
        fake_store(root, monkeypatch, folders)
        entry = write_entry(root, "tb4", TB4_REF)
        monkeypatch.setenv("NEMO_GYM_DATASETS_DIR", str(tmp_path / "shared"))
        destination = (tmp_path / "shared" / "tb4").resolve()
        run_main(monkeypatch, "dataset", "pull", "tb4")
        capsys.readouterr()
        (destination / "dataset.toml").write_text("[gym]\nverified = true\n")
        expected = (
            f"{destination / 'dataset.toml'} was edited in place and differs from the entry's {entry / 'dataset.toml'}"
        )
        for argv in (
            ("dataset", "pull", "tb4"),
            ("dataset", "validate", "tb4"),
            ("eval", "run", "tb4", "--agent", "hermes_agent"),
        ):
            err = main_error(monkeypatch, capsys, *argv)
            assert expected in err and "Traceback" not in err, argv
        assert (destination / "dataset.toml").read_text() == "[gym]\nverified = true\n"  # never overwritten
        assert not (tmp_path / "out").exists() and not (root / "results").exists()

    def test_an_earlier_hub_folder_beside_the_entry_is_named_and_left_alone(self, root, tmp_path, monkeypatch, capsys):
        """A checkout that fetched the reference before the entry declared it holds `terminal-bench-4.0.0/`; the
        reference now lands in the entry, and the pull, and a run of the reference, say so (finding of round 24)."""
        folders = {"alpha": write_task(root / "src" / "alpha")}
        fake_store(root, monkeypatch, folders)
        folder = write_entry(root, "tb4", TB4_REF)
        old = (root / "environments" / "harbor" / "terminal-bench-4.0.0").resolve()
        write_task(old / "alpha")
        (old / "alpha" / "instruction.md").write_text("the old fetch\n")
        note = (
            f"Note: {old} holds an earlier fetch of {TB4_REF}; fetches of it now land in {folder} and nothing reads "
            "the old folder, so it can be moved aside or deleted"
        )

        run_main(monkeypatch, "dataset", "pull", TB4_REF)

        out = capsys.readouterr().out.splitlines()
        assert out[0] == f"Pulled {TB4_REF} into {folder} (entry tb4)" and out[2] == note
        assert out[3] == "Next: gym dataset validate tb4 --sandbox <provider>"
        assert (old / "alpha" / "instruction.md").read_text() == "the old fetch\n"
        prepare_target(TB4_REF, output_root=tmp_path / "out")
        assert note in capsys.readouterr().out
        # Once the old folder is moved aside the note is gone.
        old.rename(tmp_path / "moved-aside")
        run_main(monkeypatch, "dataset", "pull", "tb4")
        assert "Note:" not in capsys.readouterr().out

    def test_an_edited_task_folder_is_refused_as_a_fetch_inside_eval_run_refuses_it(self, root, monkeypatch, capsys):
        folders = {"alpha": write_task(root / "src" / "alpha")}
        fake_store(root, monkeypatch, folders)
        folder = write_entry(root, "tb4", TB4_REF)
        run_main(monkeypatch, "dataset", "pull", "tb4")
        capsys.readouterr()
        (folder / "alpha" / "instruction.md").write_text("edited by hand\n")

        err = main_error(monkeypatch, capsys, "dataset", "pull", "tb4")

        assert f"{folder / 'alpha'} differs from the store" in err and "Traceback" not in err
        assert (folder / "alpha" / "instruction.md").read_text() == "edited by hand\n"  # never discarded
        # `gym eval run` on the same reference stops in the same words, before anything is prepared.
        err = main_error(monkeypatch, capsys, "eval", "run", TB4_REF, "--agent", "hermes_agent")
        assert f"{folder / 'alpha'} differs from the store" in err

    def test_a_foreign_folder_is_refused(self, root, monkeypatch, capsys):
        folders = {"alpha": write_task(root / "src" / "alpha")}
        fake_store(root, monkeypatch, folders)
        folder = write_entry(root, "tb4", TB4_REF)
        write_task(folder / "stray")
        (folder / "manifest.toml").write_text(
            '[dataset]\nsource = "harbor-registry"\ndigest = "x"\n[tasks]\n[tasks.stray]\ndigest = "x"\n'
        )
        err = main_error(monkeypatch, capsys, "dataset", "pull", "tb4")
        assert "was written by a harbor-registry fetch, not a package-store fetch" in err
        assert (folder / "stray" / "task.toml").is_file()

    @pytest.mark.parametrize(
        ("argv", "expected"),
        [
            (("nonsense",), "'nonsense' is neither a `harbor:` reference nor a committed entry"),
            (("harbor:bad ref",), "Malformed Harbor hub reference"),
            (("swe",), "declares no [gym] source, so there is nothing to pull for entry swe"),
            (("tb4", "+num_repeats=2"), "takes no configuration overrides: +num_repeats=2"),
        ],
    )
    def test_refusals_are_one_line_parser_errors(self, root, monkeypatch, capsys, argv, expected):
        write_entry(root, "tb4", TB4_REF)
        write_entry(root, "swe", None)
        err = main_error(monkeypatch, capsys, "dataset", "pull", *argv)
        assert expected in err and "Traceback" not in err

    def test_pulled_dataset_report(self, tmp_path):
        manifest = Manifest(source="harbor-package-store", digest="d" * 64, dataset_toml_digest="")
        plan = entries_module.PullPlan(ref=TB4_REF, entry=None)
        pulled = PulledDataset(TB4_REF, plan, tmp_path / "a b", manifest, unchanged=False)
        assert pulled.task_count == 0 and pulled.entry is None
        assert pulled.text().splitlines()[2] == f"Next: gym dataset validate '{tmp_path / 'a b'}' --sandbox <provider>"
        assert pull_dataset.__doc__  # the function the command and prepare share

    def test_a_registry_entry_under_a_moved_datasets_dir_follows_two_successive_table_edits(
        self, root, tmp_path, monkeypatch, registry_repo, capsys
    ):  # noqa: F811
        """A registry fetch with every folder in place rewrites no manifest, so the record of the copy beside the tasks
        is the pull's own to keep: two edits of the entry's file with a pull between are both taken as such, each pull
        reports the digest the table now has, and the task record is never touched (round 2)."""
        dataset = registry_dataset(registry_repo)
        monkeypatch.setattr(hub_module, "load_registry", lambda cache_dir, *, refresh=False: [dataset])
        entry = write_entry(root, "ds_entry", "harbor:ds@1.0")
        monkeypatch.setenv("NEMO_GYM_DATASETS_DIR", str(tmp_path / "shared"))
        destination = (tmp_path / "shared" / "ds_entry").resolve()
        run_main(monkeypatch, "dataset", "pull", "ds_entry")
        first = Manifest.read(destination)
        digests = []
        for verified in ("true", "false"):
            (entry / "dataset.toml").write_text(f'[gym]\nsource = "harbor:ds@1.0"\nverified = {verified}\n')
            capsys.readouterr()
            run_main(monkeypatch, "dataset", "pull", "ds_entry")
            out = capsys.readouterr().out
            manifest = Manifest.read(destination)
            assert (destination / "dataset.toml").read_bytes() == (entry / "dataset.toml").read_bytes()
            assert manifest.dataset_toml_digest == dataset_toml_digest(entry)
            assert manifest.tasks == first.tasks and manifest.source == "harbor-registry"
            assert f"dataset digest {manifest.digest}" in out and "unchanged" not in out
            digests.append(manifest.digest)
        assert len({first.digest, *digests}) == 3
        run_main(monkeypatch, "dataset", "pull", "ds_entry")
        assert "unchanged since the last pull" in capsys.readouterr().out

    def test_an_entry_table_edited_around_a_failed_fetch_is_not_an_in_place_edit(
        self, root, tmp_path, monkeypatch, capsys
    ):
        """The copy is refreshed before the fetch; a fetch that then fails (store unreachable) rewrites nothing, and the
        next edit of the entry's file is still taken as such (round 2). A first fetch that fails leaves only the copy
        behind, no manifest, and that copy follows the entry's file too."""
        fake_store(root, monkeypatch, {"alpha": write_task(root / "src" / "alpha")})
        entry = write_entry(root, "tb4", TB4_REF)
        monkeypatch.setenv("NEMO_GYM_DATASETS_DIR", str(tmp_path / "shared"))
        destination = (tmp_path / "shared" / "tb4").resolve()
        real = harbor_cli.fetch_ref

        def unreachable(*args, **kwargs):
            raise HubError("store unreachable")

        def table(verified: str) -> None:
            (entry / "dataset.toml").write_text(f'[gym]\nsource = "{TB4_REF}"\nverified = {verified}\n')

        monkeypatch.setattr(harbor_cli, "fetch_ref", unreachable)
        err = main_error(monkeypatch, capsys, "dataset", "pull", "tb4")
        assert err.rstrip().endswith("error: store unreachable")  # no entry hint: the message names no folder
        assert (destination / "dataset.toml").is_file() and Manifest.read(destination) is None
        table("true")
        monkeypatch.setattr(harbor_cli, "fetch_ref", real)
        run_main(monkeypatch, "dataset", "pull", "tb4")
        assert (destination / "dataset.toml").read_bytes() == (entry / "dataset.toml").read_bytes()
        assert (destination / "alpha" / "task.toml").is_file()
        table("false")
        monkeypatch.setattr(harbor_cli, "fetch_ref", unreachable)
        main_error(monkeypatch, capsys, "dataset", "pull", "tb4")
        assert (destination / "dataset.toml").read_bytes() == (entry / "dataset.toml").read_bytes()
        table("true")
        monkeypatch.setattr(harbor_cli, "fetch_ref", real)
        run_main(monkeypatch, "dataset", "pull", "tb4")
        out = capsys.readouterr().out
        manifest = Manifest.read(destination)
        assert "unchanged" not in out and f"dataset digest {manifest.digest}" in out
        assert manifest.dataset_toml_digest == dataset_toml_digest(entry)

    def test_an_entry_table_fixed_after_a_refused_run_is_not_an_in_place_edit(
        self, root, tmp_path, monkeypatch, capsys
    ):
        """`gym eval run <entry>` refreshes the copy, then refuses for the agent before its manifest check rewrites
        anything; the fix of the entry's table that follows pulls and prepares (round 2)."""
        fake_store(root, monkeypatch, {"alpha": write_task(root / "src" / "alpha")})
        entry = write_entry(root, "tb4", TB4_REF)
        monkeypatch.setenv("NEMO_GYM_DATASETS_DIR", str(tmp_path / "shared"))
        destination = (tmp_path / "shared" / "tb4").resolve()
        run_main(monkeypatch, "dataset", "pull", "tb4")
        (entry / "dataset.toml").write_text(f'[gym]\nsource = "{TB4_REF}"\nallowed_agents = ["other_agent"]\n')
        err = main_error(monkeypatch, capsys, "eval", "run", "tb4", "--agent", "hermes_agent")
        assert "allowed_agents admits other_agent" in err
        assert (destination / "dataset.toml").read_bytes() == (entry / "dataset.toml").read_bytes()
        (entry / "dataset.toml").write_text(f'[gym]\nsource = "{TB4_REF}"\nallowed_agents = ["hermes_agent"]\n')
        run_main(monkeypatch, "dataset", "pull", "tb4")
        out = capsys.readouterr().out
        assert "unchanged" not in out and f"dataset digest {Manifest.read(destination).digest}" in out
        assert (destination / "dataset.toml").read_bytes() == (entry / "dataset.toml").read_bytes()
        prepared = prepare_target("tb4", output_root=tmp_path / "out")
        assert prepared.dataset_digest == Manifest.read(destination).digest
        assert "Rewrote" not in capsys.readouterr().out

    def test_a_registry_entry_pulled_in_place_reports_the_digest_of_the_edited_table(
        self, root, monkeypatch, registry_repo, capsys
    ):  # noqa: F811
        """A registry fetch with every folder in place writes nothing, so after an edit of the entry's dataset.toml the
        pull brings the manifest's Gym layer up to date itself: it reports the new digest, not "unchanged", and the
        prepare that follows finds nothing to rewrite (round 2)."""
        dataset = registry_dataset(registry_repo)
        monkeypatch.setattr(hub_module, "load_registry", lambda cache_dir, *, refresh=False: [dataset])
        folder = write_entry(root, "ds_entry", "harbor:ds@1.0")
        run_main(monkeypatch, "dataset", "pull", "ds_entry")
        first = Manifest.read(folder)
        (folder / "dataset.toml").write_text('[gym]\nsource = "harbor:ds@1.0"\nverified = true\n')
        capsys.readouterr()
        run_main(monkeypatch, "dataset", "pull", "ds_entry")
        out = capsys.readouterr().out
        second = Manifest.read(folder)
        assert second.digest != first.digest and second.tasks == first.tasks
        assert second.dataset_toml_digest == dataset_toml_digest(folder)
        assert f"dataset digest {second.digest}" in out and "unchanged" not in out
        # The identity change is announced by the pull, in the manifest check's words (round 3).
        assert out.splitlines()[0] == (
            f"Rewrote {folder / 'manifest.toml'}: dataset.toml changed; dataset digest "
            f"{first.digest[:12]} -> {second.digest[:12]}"
        )
        prepared = prepare_target("ds_entry", output_root=root / "out")
        assert prepared.dataset_digest == second.digest and "Rewrote" not in capsys.readouterr().out

    def test_a_pull_whose_fetch_rewrote_the_gym_layer_announces_the_digest_change(
        self, root, tmp_path, monkeypatch, capsys
    ):
        """The package-store fetch rewrites the manifest without a word when every folder is in place; a dataset.toml
        written by hand into a hub folder (no entry) changed the dataset's identity, and the pull, and a run of the
        reference, say so as the manifest check used to (round 3)."""
        fake_store(root, monkeypatch, {"alpha": write_task(root / "src" / "alpha")})
        run_main(monkeypatch, "dataset", "pull", TB4_REF)
        folder = (root / "environments" / "harbor" / "terminal-bench-4.0.0").resolve()
        before = Manifest.read(folder)
        (folder / "dataset.toml").write_text('[gym.tasks."alpha".environment]\nenv = { X = "1" }\n')
        capsys.readouterr()
        run_main(monkeypatch, "dataset", "pull", TB4_REF)
        out = capsys.readouterr().out.splitlines()
        after = Manifest.read(folder)
        assert after.digest != before.digest and after.tasks == before.tasks
        assert out[0] == (
            f"Rewrote {folder / 'manifest.toml'}: dataset.toml changed; dataset digest "
            f"{before.digest[:12]} -> {after.digest[:12]}"
        )
        assert out[2] == f"1 task(s), dataset digest {after.digest} (harbor-package-store)"
        (folder / "dataset.toml").write_text('[gym.tasks."alpha".environment]\nenv = { X = "2" }\n')
        prepared = prepare_target(TB4_REF, output_root=tmp_path / "out")
        out = capsys.readouterr().out
        assert prepared.dataset_digest != after.digest and prepared.tasks[0].env == {"X": "2"}
        assert (
            out.count("Rewrote") == 1
            and f"dataset digest {after.digest[:12]} -> {prepared.dataset_digest[:12]}" in out
        )
        # Unchanged since: nothing announced, nothing rewritten.
        run_main(monkeypatch, "dataset", "pull", TB4_REF)
        out = capsys.readouterr().out
        assert "Rewrote" not in out and "unchanged since the last pull" in out

    def test_a_refusal_naming_the_entry_folder_says_what_to_delete_there(self, root, tmp_path, monkeypatch, capsys):
        """The fetchers' "delete the folder, or move it aside" would delete the committed README, YAML and dataset.toml
        when the folder is the entry's own; the pull, a run of the reference and the manifest check of a run by entry
        name add what to delete (round 2). Under a moved datasets dir the folder holds nothing git tracks and the
        hint is not added."""
        fake_store(root, monkeypatch, {"alpha": write_task(root / "src" / "alpha")})
        folder = write_entry(root, "tb4", TB4_REF)
        run_main(monkeypatch, "dataset", "pull", "tb4")
        hint = (
            f"{folder} is the committed entry tb4's own folder: delete, or move aside, only the pulled task folders and "
            "manifest.toml in it; its README, YAML files and dataset.toml are the entry's and stay (to pull another "
            "version, change `source` in that dataset.toml)"
        )
        # An interrupted fetch is finished by the pull, not by deleting what it landed: that refusal names the pull to
        # run again and carries no deletion hint (round 3).
        manifest = Manifest.read(folder)
        text = (folder / "manifest.toml").read_text()
        Manifest(
            source=manifest.source,
            digest=manifest.digest,
            dataset_toml_digest=manifest.dataset_toml_digest,
            tasks={**manifest.tasks, "zzz": {"package": "terminal-bench/zzz", "content_hash": "sha256:" + "0" * 64}},
        ).write(folder)
        err = main_error(monkeypatch, capsys, "eval", "run", "tb4", "--agent", "hermes_agent")
        assert (
            f"the fetch of {folder} was interrupted (tasks not yet recorded: zzz); run `gym dataset pull tb4` again"
            in err
        )
        assert err.rstrip().endswith("run `gym dataset pull tb4` again") and "delete, or move aside" not in err
        (folder / "manifest.toml").write_text(text)
        # The store's 4.0.0 now has another task set (or the entry's source was bumped to a tag that does).
        fake_store(root, monkeypatch, {"alpha": root / "src" / "alpha", "beta": write_task(root / "src" / "beta")})
        for argv in (
            ("dataset", "pull", "tb4"),
            ("dataset", "pull", TB4_REF),
            ("eval", "run", TB4_REF, "--agent", "hermes_agent"),
        ):
            err = main_error(monkeypatch, capsys, *argv)
            assert "was written for another version" in err and err.rstrip().endswith(hint), argv
            assert "Traceback" not in err
        # The manifest check of a run by entry name (no fetch) adds the same hint to its refusal.
        (folder / "alpha" / "instruction.md").write_text("edited by hand\n")
        err = main_error(monkeypatch, capsys, "eval", "run", "tb4", "--agent", "hermes_agent")
        assert "does not match its manifest.toml" in err and err.rstrip().endswith(hint)
        assert sorted(path.name for path in folder.iterdir()) == [
            "README.md",
            "alpha",
            "dataset.toml",
            "manifest.toml",
            "tb4.yaml",
        ]
        # Under a moved datasets dir the entry's folder there holds the pull alone: the fetcher's words suffice.
        monkeypatch.setenv("NEMO_GYM_DATASETS_DIR", str(tmp_path / "shared"))
        run_main(monkeypatch, "dataset", "pull", "tb4")
        fake_store(root, monkeypatch, {"alpha": root / "src" / "alpha"})
        err = main_error(monkeypatch, capsys, "dataset", "pull", "tb4")
        assert "was written for another version" in err and "committed entry" not in err

    def test_a_folder_of_the_entry_name_in_the_working_directory_does_not_hide_it_from_pull(
        self, root, monkeypatch, capsys
    ):
        """The pull takes no path, so `gym dataset pull tb4` run from a folder holding a tb4/ (benchmarks/ holds a
        terminal_bench_4/) still pulls the entry; validate and run keep meaning the path there (round 2)."""
        fake_store(root, monkeypatch, {"alpha": write_task(root / "src" / "alpha")})
        folder = write_entry(root, "tb4", TB4_REF)
        (root / "tb4").mkdir()
        assert entry_target("tb4") is None and plan_pull("tb4").folder == folder
        run_main(monkeypatch, "dataset", "pull", "tb4")
        assert capsys.readouterr().out.splitlines()[0] == f"Pulled {TB4_REF} into {folder} (entry tb4)"
        assert (folder / "alpha" / "task.toml").is_file() and not any((root / "tb4").iterdir())


@pytest.mark.usefixtures("fake_registry")
class TestEntryTargets:
    """`gym dataset validate <entry>` and `gym eval run <entry>` resolve the name to the pulled folder, no fetch."""

    def test_prepare_resolves_an_entry_to_its_pulled_folder(self, root, tmp_path, monkeypatch):
        fake_store(root, monkeypatch, {"a": write_task(root / "src" / "a")})
        folder = write_entry(root, "tb4", TB4_REF)
        pull_dataset("tb4")
        prepared = prepare_target("tb4", output_root=tmp_path / "out")
        assert prepared.folder == folder and prepared.taskset == "tb4" and prepared.task_source == "tb4"
        assert [task.task_id for task in prepared.tasks] == ["a"] and prepared.dataset_source == "harbor-package-store"

    @pytest.mark.parametrize("moved", [False, True])
    def test_a_pull_that_lost_its_manifest_is_refused_not_run_as_local(
        self, root, tmp_path, monkeypatch, capsys, moved
    ):
        """Task folders in a sourced entry's folder without the pull's record (the manifest deleted, as the deletion
        hint's first step, or never written) are not the entry's pull: validate and run by entry name refuse them with
        the pull to run, write nothing (no `local` manifest), and the pull repairs the folder (round 3)."""
        _, _, calls = fake_store(root, monkeypatch, {"alpha": write_task(root / "src" / "alpha")})
        entry = write_entry(root, "tb4", TB4_REF)
        if moved:
            monkeypatch.setenv("NEMO_GYM_DATASETS_DIR", str(tmp_path / "shared"))
        folder = (tmp_path / "shared" / "tb4").resolve() if moved else entry
        pull_dataset("tb4")
        (folder / "manifest.toml").unlink()
        downloads = sum(1 for call in calls if call[1].startswith("/storage/"))
        capsys.readouterr()
        refusal = (
            f"Entry tb4's folder {folder} holds task folders but no fetch record (manifest.toml), so they cannot be "
            f"run as its pull of {TB4_REF}; run `gym dataset pull tb4` to fetch it"
        )
        with pytest.raises(HarborTaskError) as exc:
            prepare_target("tb4", output_root=tmp_path / "out")
        assert str(exc.value) == refusal
        for argv in (("dataset", "validate", "tb4"), ("eval", "run", "tb4", "--agent", "hermes_agent")):
            err = main_error(monkeypatch, capsys, *argv)
            assert refusal in err and "Traceback" not in err, argv
        assert not (folder / "manifest.toml").exists() and not (tmp_path / "out").exists()
        assert not (root / "results").exists() and capsys.readouterr().out == ""
        # The pull records the folders that still match the store without downloading them again, and the entry runs.
        run_main(monkeypatch, "dataset", "pull", "tb4")
        out = capsys.readouterr().out
        assert "unchanged" not in out and sum(1 for call in calls if call[1].startswith("/storage/")) == downloads
        assert Manifest.read(folder).source == "harbor-package-store"
        prepared = prepare_target("tb4", output_root=tmp_path / "out")
        assert prepared.dataset_source == "harbor-package-store" and prepared.task_source == "tb4"
        # A manifest of another source (one a run wrote before this check existed, or the registry fetcher's) is no
        # record of the entry's pull either, and says what to clear.
        Manifest.build(folder, source="local").write(folder)
        err = main_error(monkeypatch, capsys, "dataset", "validate", "tb4")
        assert (
            f"Entry tb4's folder {folder} holds task folders whose manifest.toml records source 'local', not a fetch "
            f"of {TB4_REF} (harbor-package-store); delete, or move aside, the task folders and manifest.toml there "
            "and run `gym dataset pull tb4`"
        ) in err
        assert Manifest.read(folder).source == "local"  # nothing rewritten

    def test_an_existing_folder_of_the_same_name_wins(self, root, tmp_path):
        entry = write_entry(root, "tb4", TB4_REF)
        write_task(entry / "a")
        local = write_task(root / "tb4" / "b").parent
        prepared = prepare_target("tb4", output_root=tmp_path / "out")
        assert prepared.folder == local and [task.task_id for task in prepared.tasks] == ["b"]

    def test_an_unpulled_entry_names_the_pull_to_run(self, root, tmp_path, monkeypatch, capsys):
        write_entry(root, "tb4", TB4_REF)
        with pytest.raises(
            HarborTaskError, match=r"run `gym dataset pull tb4` to fetch " + TB4_REF.replace(".", r"\.")
        ):
            prepare_target("tb4", output_root=tmp_path / "out")
        assert not (tmp_path / "out").exists()
        err = main_error(monkeypatch, capsys, "dataset", "validate", "tb4")
        assert "run `gym dataset pull tb4`" in err
        err = main_error(monkeypatch, capsys, "eval", "run", "tb4", "--agent", "hermes_agent")
        assert "run `gym dataset pull tb4`" in err

    def test_an_entry_without_a_source_asks_for_the_task_folder(self, root, tmp_path):
        write_entry(root, "swe", None)
        with pytest.raises(HarborTaskError, match=r"declares no \[gym\] source .* pass the folder"):
            prepare_target("swe", output_root=tmp_path / "out")
        # Once its tasks are in place (by whatever means), the name resolves like any entry's.
        write_task(root / "environments" / "harbor" / "swe" / "a")
        assert prepare_target("swe", output_root=tmp_path / "out").taskset == "swe"

    def test_entry_folder_follows_a_moved_datasets_dir(self, root, tmp_path, monkeypatch):
        fake_store(root, monkeypatch, {"a": write_task(root / "src" / "a")})
        entry = write_entry(root, "tb4", TB4_REF)
        monkeypatch.setenv("NEMO_GYM_DATASETS_DIR", str(tmp_path / "shared"))
        pull_dataset("tb4")
        folder = entry_folder(find_entry("tb4"))
        assert folder == (tmp_path / "shared" / "tb4").resolve() and (folder / "a" / "task.toml").is_file()
        # The copy of the entry's table beside the tasks follows the entry's file, as the pull keeps it.
        (entry / "dataset.toml").write_text(f'[gym]\nsource = "{TB4_REF}"\nverified = true\n')
        assert entry_folder(find_entry("tb4")) == folder
        assert (folder / "dataset.toml").read_bytes() == (entry / "dataset.toml").read_bytes()

    def test_a_pulled_entry_that_was_edited_is_refused_at_prepare(self, root, tmp_path, monkeypatch):
        """The path form and the entry form check the pull's manifest alike: an edited folder never prepares."""
        folders = {"alpha": write_task(root / "src" / "alpha")}
        fake_store(root, monkeypatch, folders)
        folder = write_entry(root, "tb4", TB4_REF)
        pull_dataset("tb4")
        assert prepare_target("tb4", output_root=tmp_path / "out").dataset_source == "harbor-package-store"
        (folder / "alpha" / "instruction.md").write_text("edited by hand\n")
        with pytest.raises(Exception, match="does not match its manifest.toml"):
            prepare_target("tb4", output_root=tmp_path / "out")
