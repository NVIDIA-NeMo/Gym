# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`--benchmark` and `--environment` select what the catalog lists under a name, across search roots."""

import logging
from pathlib import Path

import pytest

from nemo_gym.environment.assets import asset_config_path
from nemo_gym.environment.registry import RegistryError, resolve_catalog_entry
from tests.unit_tests.environment.test_registry import _manifest, _write_manifest


@pytest.fixture
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """An earlier and a later search root, as `--search-dir` and the Gym checkout would be."""
    earlier, later = tmp_path / "earlier", tmp_path / "later"
    for module in ("nemo_gym.environment.assets", "nemo_gym.environment.registry"):
        monkeypatch.setattr(f"{module}.component_search_roots", lambda: [earlier, later])
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


@pytest.mark.parametrize("kind", ["benchmark", "environment"])
@pytest.mark.parametrize("name", ["stable_id", "suite/stable_id"])
def test_typo_suggestions_include_declared_ids(roots: tuple[Path, Path], kind: str, name: str) -> None:
    _, later = roots
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
