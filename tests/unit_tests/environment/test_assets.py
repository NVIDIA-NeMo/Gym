# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`--benchmark` and `--environment` select what the catalog lists under a name, across search roots."""

import logging
import shutil
from pathlib import Path

import pytest

from nemo_gym.environment.assets import asset_config_path
from nemo_gym.environment.registry import RegistryError, resolve_catalog_entry
from tests.unit_tests.environment.test_registry import _manifest, _write_manifest


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


@pytest.mark.parametrize("first_suite", ["manifest", "config without a manifest"])
def test_a_setup_of_a_shadowed_manifest_does_not_take_its_name_from_a_later_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first_suite: str
) -> None:
    # The first root's `suite` shadows the middle root's `suite` and the setup `suite/fast` it lists, so the catalog
    # lists the last root's `suite/fast`.
    first, middle, last = tmp_path / "first", tmp_path / "middle", tmp_path / "last"
    _use_roots(monkeypatch, first, middle, last)
    if first_suite == "manifest":
        _write_manifest(first, "environments", "suite", _manifest("suite"))
    else:
        (first / "environments/suite").mkdir(parents=True)
        (first / "environments/suite/config.yaml").write_text("{}\n", encoding="utf-8")
    listing = {"setups": [{"name": "suite/fast", "config_path": "fast.yaml"}]}
    hidden = _write_manifest(middle, "environments", "suite", _manifest("suite", **listing)).with_name("fast.yaml")
    hidden.write_text("{}\n", encoding="utf-8")
    declared = _write_manifest(last, "environments", "fast", _manifest("suite/fast"))

    assert _selected("suite/fast") == _listed("suite/fast") == declared.with_name("config.yaml").resolve()


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


def test_a_listed_setup_selects_by_its_id_and_its_config_path_still_selects(roots: tuple[Path, Path]) -> None:
    # The setup `codex_id` runs `suite/codex.yaml`; the catalog lists that config only under the setup's ID.
    _, later = roots
    listing = {"setups": [{"name": "codex_id", "config_path": "codex.yaml"}]}
    manifest = _write_manifest(later, "benchmarks", "suite", _manifest("suite", "benchmark", **listing))
    codex = manifest.with_name("codex.yaml")
    codex.write_text("agent:\n  datasets:\n  - {type: benchmark}\n", encoding="utf-8")

    def selected(name: str) -> Path:
        return Path(asset_config_path("benchmark", name))

    assert selected("codex_id") == resolve_catalog_entry("codex_id", "benchmark").config_path.resolve()
    assert selected("codex_id") == codex.resolve()
    assert selected("suite/codex") == codex.resolve()
    with pytest.raises(RegistryError, match="Unknown catalog entry"):
        resolve_catalog_entry("suite/codex", "benchmark")


def test_a_listed_name_that_is_another_configs_path_selects_what_the_catalog_lists(roots: tuple[Path, Path]) -> None:
    # Validation rejects the name `other` here, because it hides benchmarks/other/config.yaml; listing and launching
    # still pick the same config.
    _, later = roots
    listing = {"setups": [{"name": "other", "config_path": "variant.yaml"}]}
    manifest = _write_manifest(later, "benchmarks", "demo", _manifest("demo", "benchmark", **listing))
    config = "agent:\n  datasets:\n  - {type: benchmark}\n"
    manifest.with_name("variant.yaml").write_text(config, encoding="utf-8")
    other = later / "benchmarks/other/config.yaml"
    other.parent.mkdir()
    other.write_text(config, encoding="utf-8")

    selected = Path(asset_config_path("benchmark", "other"))
    assert selected == resolve_catalog_entry("other", "benchmark").config_path.resolve()
    assert selected == manifest.with_name("variant.yaml").resolve()


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
@pytest.mark.parametrize("name", ["stable_id", "suite/stable_id"])
def test_typo_suggestions_include_declared_ids(roots: tuple[Path, Path], kind: str, name: str) -> None:
    _, later = roots
    _write_manifest(later, f"{kind}s", "group/moved", _manifest(name, kind))

    with pytest.raises(ValueError) as error:
        asset_config_path(kind, name.removesuffix("id") + "if")

    assert f"Did you mean `{name}`?" in str(error.value)


@pytest.mark.parametrize("listed_as", ["setups", "protocols"])
def test_typo_suggestions_include_listed_ids(roots: tuple[Path, Path], listed_as: str) -> None:
    _, later = roots
    listed = {"name": "codex_id", "config_path": "codex.yaml", "description": "Codex"}
    _write_manifest(later, "benchmarks", "suite", _manifest("suite", "benchmark", **{listed_as: [listed]}))

    with pytest.raises(ValueError) as error:
        asset_config_path("benchmark", "codex_if")

    assert "Did you mean `codex_id`?" in str(error.value)


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
