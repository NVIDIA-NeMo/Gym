# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`--benchmark` and `--environment` select what the catalog lists under a name, across search roots."""

import logging
import shutil
from pathlib import Path

import pytest

from nemo_gym.environment.assets import asset_config_path
from nemo_gym.environment.registry import RegistryError, resolve_catalog_entry
from tests.unit_tests.environment.test_registry import _manifest, _write_manifest, _write_taskset


def _use_roots(monkeypatch: pytest.MonkeyPatch, *roots: Path) -> None:
    for module in ("nemo_gym.environment.assets", "nemo_gym.environment.registry"):
        monkeypatch.setattr(f"{module}.component_search_roots", lambda: list(roots))


@pytest.fixture
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """An earlier and a later search root, as `--search-dir` and the Gym checkout would be."""
    earlier, later = tmp_path / "earlier", tmp_path / "later"
    _use_roots(monkeypatch, earlier, later)
    return earlier, later


def _selected(name: str) -> Path:
    return Path(asset_config_path("environment", name))


def _listed(name: str) -> Path:
    return resolve_catalog_entry(name, "environment").config_path.resolve()


def test_a_moved_entry_in_an_earlier_root_wins_for_listing_and_selection(
    roots: tuple[Path, Path], caplog: pytest.LogCaptureFixture
) -> None:
    earlier, later = roots
    moved = _write_manifest(earlier, "environments", "group/moved", _manifest("demo"))
    _write_manifest(later, "environments", "demo", _manifest("demo"))

    with caplog.at_level(logging.WARNING):
        selected = _selected("demo")

    assert selected == _listed("demo") == moved.with_name("config.yaml").resolve()
    assert "matches multiple configs" in caplog.text


def test_a_folder_claimed_under_another_name_does_not_shadow_that_name(roots: tuple[Path, Path]) -> None:
    # The earlier root's folder `x` holds the entry named `z`; the later root's folder `y` holds the entry named `x`.
    earlier, later = roots
    z = _write_manifest(earlier, "environments", "x", _manifest("z"))
    x = _write_manifest(later, "environments", "y", _manifest("x"))

    assert _selected("x") == _listed("x") == x.with_name("config.yaml").resolve()
    assert _selected("z") == _listed("z") == z.with_name("config.yaml").resolve()


def test_an_old_folder_path_still_selects_when_no_entry_has_that_name(roots: tuple[Path, Path]) -> None:
    # As with `livecodebench/v5_2408_2502`: the manifest in `suite/v1/config/` names the config one folder up.
    earlier, _ = roots
    manifest = _write_manifest(
        earlier,
        "environments",
        "suite/v1/config",
        _manifest("suite/v1/config", config_path="../config.yaml"),
        with_config=False,
    )
    config = manifest.parent.parent / "config.yaml"
    config.write_text("{}\n", encoding="utf-8")

    assert _selected("suite/v1/config") == _listed("suite/v1/config") == config.resolve()
    assert _selected("suite/v1") == config.resolve()
    with pytest.raises(RegistryError, match="Unknown catalog entry"):
        _listed("suite/v1")


def test_a_shadowed_manifests_folder_does_not_take_that_name_from_a_later_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The middle root's folder `moved` holds a `demo` that the first root shadows; the last root declares `moved`.
    first, middle, last = tmp_path / "first", tmp_path / "middle", tmp_path / "last"
    _use_roots(monkeypatch, first, middle, last)
    _write_manifest(first, "environments", "demo", _manifest("demo"))
    _write_manifest(middle, "environments", "moved", _manifest("demo"))
    target = _write_manifest(last, "environments", "target", _manifest("moved"))

    assert _selected("moved") == _listed("moved") == target.with_name("config.yaml").resolve()


def test_a_flavor_path_in_an_earlier_root_does_not_shadow_a_declared_name(roots: tuple[Path, Path]) -> None:
    # The catalog lists no environment flavor file, so `suite/fast` here is a path, not a catalog entry.
    earlier, later = roots
    flavor = earlier / "environments/suite/fast.yaml"
    flavor.parent.mkdir(parents=True)
    flavor.write_text("{}\n", encoding="utf-8")
    declared = _write_manifest(later, "environments", "moved", _manifest("suite/fast"))

    assert _selected("suite/fast") == _listed("suite/fast") == declared.with_name("config.yaml").resolve()
    shutil.rmtree(declared.parent)
    assert _selected("suite/fast") == flavor.resolve()


def test_a_nested_benchmark_path_in_an_earlier_root_does_not_shadow_a_declared_name(roots: tuple[Path, Path]) -> None:
    # The catalog lists the earlier root's config as `suite/v1/config`; `suite/v1` reaches it only as a path.
    earlier, later = roots
    config = earlier / "benchmarks/suite/v1/config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("agent:\n  datasets:\n  - {type: benchmark}\n", encoding="utf-8")
    declared = _write_manifest(later, "benchmarks", "moved", _manifest("suite/v1", "benchmark"))

    def selected(name: str) -> Path:
        return Path(asset_config_path("benchmark", name))

    def listed(name: str) -> Path:
        return resolve_catalog_entry(name, "benchmark").config_path.resolve()

    assert selected("suite/v1") == listed("suite/v1") == declared.with_name("config.yaml").resolve()
    assert selected("suite/v1/config") == listed("suite/v1/config") == config.resolve()
    shutil.rmtree(declared.parent)
    assert selected("suite/v1") == config.resolve()


@pytest.mark.parametrize("excluded", ["harbor/dataset/task", ".staging/task"])
def test_a_manifest_discovery_skips_does_not_shadow_a_declared_name(roots: tuple[Path, Path], excluded: str) -> None:
    # Discovery stops at a pulled Harbor dataset (its `manifest.toml`) and skips hidden folders.
    earlier, later = roots
    skipped = _write_manifest(earlier, "environments", excluded, _manifest(excluded))
    if excluded.startswith("harbor/"):
        skipped.parent.parent.joinpath("manifest.toml").write_text("", encoding="utf-8")
    declared = _write_manifest(later, "environments", "moved", _manifest(excluded))

    assert _selected(excluded) == _listed(excluded) == declared.with_name("config.yaml").resolve()


def test_typo_suggestions_include_environment_flavors(roots: tuple[Path, Path]) -> None:
    earlier, _ = roots
    folder = earlier / "environments/demo"
    folder.mkdir(parents=True)
    for name in ("config.yaml", "fast.yaml"):
        folder.joinpath(name).write_text("{}\n", encoding="utf-8")

    assert _selected("demo/fast") == folder.joinpath("fast.yaml").resolve()
    with pytest.raises(ValueError, match=r"Did you mean `demo/fast`\?"):
        asset_config_path("environment", "demo/fats")


@pytest.mark.parametrize("kind", ["benchmark", "environment"])
def test_a_task_set_is_selected_by_its_name_with_its_kinds_flag(roots: tuple[Path, Path], kind: str) -> None:
    _, later = roots
    manifest = _write_taskset(later, "folder", _manifest("stable_id", kind))
    config = (manifest.parent / "configs/folder.yaml").resolve()

    assert Path(asset_config_path(kind, "stable_id")) == resolve_catalog_entry("stable_id", kind).config_path == config
    other_kind = "environment" if kind == "benchmark" else "benchmark"
    with pytest.raises(ValueError, match="does not exist"):
        asset_config_path(other_kind, "stable_id")


def test_a_task_set_in_an_earlier_root_wins_over_a_benchmark_in_a_later_one(
    roots: tuple[Path, Path], caplog: pytest.LogCaptureFixture
) -> None:
    earlier, later = roots
    moved = _write_taskset(earlier, "shared", _manifest("shared", "benchmark"))
    _write_manifest(later, "benchmarks", "shared", _manifest("shared", "benchmark"))

    with caplog.at_level(logging.WARNING):
        selected = Path(asset_config_path("benchmark", "shared"))

    assert selected == resolve_catalog_entry("shared", "benchmark").config_path
    assert selected == (moved.parent / "configs/shared.yaml").resolve()
    assert "matches multiple configs" in caplog.text


@pytest.mark.parametrize("kind", ["benchmark", "environment"])
@pytest.mark.parametrize("name", ["stable_id", "suite/stable_id"])
@pytest.mark.parametrize("in_tasksets", [False, True])
def test_typo_suggestions_include_declared_ids(
    roots: tuple[Path, Path], kind: str, name: str, in_tasksets: bool
) -> None:
    _, later = roots
    if in_tasksets:
        _write_taskset(later, "moved", _manifest(name, kind))
    else:
        _write_manifest(later, f"{kind}s", "group/moved", _manifest(name, kind))

    with pytest.raises(ValueError) as error:
        asset_config_path(kind, name.removesuffix("id") + "if")

    assert f"Did you mean `{name}`?" in str(error.value)


@pytest.mark.parametrize("kind", ["benchmark", "environment"])
def test_typo_suggestions_preserve_folder_aliases(roots: tuple[Path, Path], kind: str) -> None:
    earlier, _ = roots
    manifest = _write_manifest(earlier, f"{kind}s", "folder_alias", _manifest("stable_id", kind))
    # Benchmark path discovery recognizes configs by their dataset declaration.
    dataset_kind = "benchmark" if kind == "benchmark" else "example"
    manifest.with_name("config.yaml").write_text(f"dataset:\n  type: {dataset_kind}\n", encoding="utf-8")

    with pytest.raises(ValueError) as error:
        asset_config_path(kind, "folder_aliaz")

    assert "Did you mean `folder_alias`?" in str(error.value)
