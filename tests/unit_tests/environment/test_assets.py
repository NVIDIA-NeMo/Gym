# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
from pathlib import Path

import pytest

from nemo_gym.environment.assets import asset_config_path
from nemo_gym.environment.registry import RegistryError, discover_environment_catalog
from tests.unit_tests.environment.test_registry import _manifest, _write_manifest


_BENCHMARK_CONFIG = "agent:\n  responses_api_agents:\n    a:\n      datasets:\n      - {type: benchmark}\n"


@pytest.fixture
def roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[Path]:
    """Two search roots, highest priority first, that replace the repository's."""
    search_roots = [tmp_path / "high", tmp_path / "low"]
    for root in search_roots:
        root.mkdir()
    monkeypatch.setattr("nemo_gym.environment.assets.component_search_roots", lambda: search_roots)
    return search_roots


def _write_config(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_BENCHMARK_CONFIG, encoding="utf-8")
    return path


class TestManifestSelection:
    def test_a_manifest_selects_a_config_outside_its_folder(self, roots: list[Path]) -> None:
        high, _low = roots
        config = _write_config(high / "benchmarks" / "suite" / "opencode.yaml")
        data = _manifest("suite/verified", "benchmark", config_path="../opencode.yaml")
        _write_manifest(high, "benchmarks", "suite/verified", data, with_config=False)

        assert asset_config_path("benchmark", "suite/verified") == str(config.resolve())

    def test_a_manifest_selects_a_named_config_beside_it(self, roots: list[Path]) -> None:
        high, _low = roots
        data = _manifest("suite/verified", "benchmark", config_path="opencode.yaml")
        manifest_path = _write_manifest(high, "benchmarks", "suite/verified", data, with_config=False)
        config = _write_config(manifest_path.with_name("opencode.yaml"))

        assert asset_config_path("benchmark", "suite/verified") == str(config.resolve())

    def test_the_manifest_wins_over_a_config_yaml_in_its_folder(self, roots: list[Path]) -> None:
        # The catalog lists the manifest's config, so the selector must load that one, not the folder's config.yaml.
        high, _low = roots
        data = _manifest("workload", config_path="example.yaml")
        manifest_path = _write_manifest(high, "environments", "workload", data)
        example = _write_config(manifest_path.with_name("example.yaml"))

        assert asset_config_path("environment", "workload") == str(example.resolve())

    def test_a_manifest_that_does_not_load_falls_back_to_the_config_path(
        self, roots: list[Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        high, _low = roots
        folder = high / "benchmarks" / "draft"
        config = _write_config(folder / "config.yaml")
        (folder / "manifest.yaml").write_text("name: [invalid\n", encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            assert asset_config_path("benchmark", "draft") == str(config.resolve())
        assert "does not load" in caplog.text

    def test_a_config_path_that_escapes_the_catalog_tree_is_never_selected(
        self, roots: list[Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        high, _low = roots
        _write_config(high / "outside.yaml")
        data = _manifest("escape", "benchmark", config_path="../../outside.yaml")
        _write_manifest(high, "benchmarks", "escape", data, with_config=False)

        with caplog.at_level(logging.WARNING), pytest.raises(ValueError, match="does not exist"):
            asset_config_path("benchmark", "escape")
        assert "escapes catalog tree" in caplog.text

    def test_a_manifest_whose_name_disagrees_with_its_folder_is_refused(self, roots: list[Path]) -> None:
        high, _low = roots
        _write_manifest(high, "benchmarks", "suite/verified", _manifest("suite/other", "benchmark"))

        with pytest.raises(RegistryError, match="catalog path requires 'suite/verified'"):
            asset_config_path("benchmark", "suite/verified")


class TestRootOrder:
    def test_a_higher_root_config_wins_over_a_lower_root_manifest(
        self, roots: list[Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        # Mirrors the catalog: a higher-priority root claims the name before a lower root's manifest.
        high, low = roots
        config = _write_config(high / "benchmarks" / "suite" / "verified.yaml")
        data = _manifest("suite/verified", "benchmark", config_path="../opencode.yaml")
        _write_manifest(low, "benchmarks", "suite/verified", data, with_config=False)
        _write_config(low / "benchmarks" / "suite" / "opencode.yaml")

        with caplog.at_level(logging.WARNING):
            assert asset_config_path("benchmark", "suite/verified") == str(config.resolve())
        assert "matches multiple configs" in caplog.text

    def test_a_higher_root_manifest_wins_over_a_lower_root_config(self, roots: list[Path]) -> None:
        high, low = roots
        data = _manifest("suite/verified", "benchmark", config_path="../opencode.yaml")
        _write_manifest(high, "benchmarks", "suite/verified", data, with_config=False)
        config = _write_config(high / "benchmarks" / "suite" / "opencode.yaml")
        _write_config(low / "benchmarks" / "suite" / "verified" / "config.yaml")

        assert asset_config_path("benchmark", "suite/verified") == str(config.resolve())

    @pytest.mark.parametrize("flag", ["benchmark", "environment"])
    @pytest.mark.parametrize("defect", ["wrong name", "wrong kind", "missing config"])
    def test_a_lower_root_manifest_the_catalog_refuses_cannot_veto_a_higher_root_config(
        self, roots: list[Path], monkeypatch: pytest.MonkeyPatch, flag: str, defect: str
    ) -> None:
        # The catalog skips a name a higher root already claims without reading the lower root's manifest.
        high, low = roots
        monkeypatch.setattr("nemo_gym.environment.registry.component_search_roots", lambda: roots)
        tree = f"{flag}s"
        config = _write_config(high / tree / "suite" / "config.yaml")
        other_kind = "environment" if flag == "benchmark" else "benchmark"
        data = {
            "wrong name": _manifest("other", flag),
            "wrong kind": _manifest("suite", other_kind),
            "missing config": _manifest("suite", flag, config_path="missing.yaml"),
        }[defect]
        _write_manifest(low, tree, "suite", data, with_config=defect != "missing config")

        catalog = {(entry.kind, entry.name): entry for entry in discover_environment_catalog()}
        assert catalog[(flag, "suite")].config_path.resolve() == config.resolve()
        assert asset_config_path(flag, "suite") == str(config.resolve())

    @pytest.mark.parametrize("flag", ["benchmark", "environment"])
    def test_a_higher_root_manifest_the_catalog_refuses_is_refused_over_a_lower_root_config(
        self, roots: list[Path], flag: str
    ) -> None:
        high, low = roots
        _write_manifest(high, f"{flag}s", "suite", _manifest("other", flag))
        _write_config(low / f"{flag}s" / "suite" / "config.yaml")

        with pytest.raises(RegistryError, match="catalog path requires 'suite'"):
            asset_config_path(flag, "suite")


class TestConfigPaths:
    @pytest.mark.parametrize(
        "value, relative",
        [
            ("suite", "suite/config.yaml"),  # `<name>/config.yaml`
            ("suite/opencode", "suite/opencode.yaml"),  # `<server>/<flavor>.yaml`
            ("suite/nested", "suite/nested/config.yaml"),  # `<name>/config.yaml` for a nested name
        ],
    )
    def test_names_without_a_manifest_still_select_by_path(self, roots: list[Path], value: str, relative: str) -> None:
        high, _low = roots
        for path in ("suite/config.yaml", "suite/opencode.yaml", "suite/nested/config.yaml"):
            _write_config(high / "benchmarks" / path)

        assert asset_config_path("benchmark", value) == str((high / "benchmarks" / relative).resolve())

    def test_an_unknown_name_suggests_a_catalog_name(self, roots: list[Path]) -> None:
        high, _low = roots
        data = _manifest("suite/verified", "benchmark", config_path="opencode.yaml")
        manifest_path = _write_manifest(high, "benchmarks", "suite/verified", data, with_config=False)
        _write_config(manifest_path.with_name("opencode.yaml"))

        with pytest.raises(ValueError, match="Did you mean `suite/verified`"):
            asset_config_path("benchmark", "suite/verifed")


def test_every_listed_catalog_name_selects_its_own_config() -> None:
    # Every benchmark and environment `gym list` prints must load, through the selector, the config it lists.
    for entry in discover_environment_catalog():
        if entry.resources_server_selector is not None:
            continue  # resources-server workloads are selected with `--resources-server`
        resolved = Path(asset_config_path(entry.kind, entry.name))
        assert resolved == entry.config_path.resolve(), f"`--{entry.kind} {entry.name}` does not select its config"
