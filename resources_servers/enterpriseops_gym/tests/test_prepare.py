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
"""Offline tests for the tool-snapshot prepare script. No test touches the network."""

from pathlib import Path

import pytest

from benchmarks.enterpriseops import prepare as benchmark_prepare
from resources_servers.enterpriseops_gym import prepare as prepare_module


def _write_flat_directory(directory: Path, files: dict[str, str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for filename, content in files.items():
        (directory / filename).write_text(content, encoding="utf-8")


def _snapshot_files() -> dict[str, str]:
    return {name: f'{{"gym_name": "{Path(name).stem}", "tools": []}}\n' for name in prepare_module.SNAPSHOT_FILENAMES}


def _pinned_bundle(tmp_path: Path, source: Path, monkeypatch: pytest.MonkeyPatch) -> prepare_module.AssetBundle:
    """Repoint the module's single bundle at a fixture dir, pinned to that dir's real hash."""
    file_count, tree_sha256 = prepare_module.tree_hash(source)
    bundle = prepare_module.AssetBundle(
        remote_dir=prepare_module.REMOTE_TOOLS_DIR,
        local_dir=tmp_path / "prepared/tools",
        filenames=prepare_module.SNAPSHOT_FILENAMES,
        file_count=file_count,
        tree_sha256=tree_sha256,
    )
    monkeypatch.setattr(prepare_module, "_tools_bundle", lambda _root: bundle)
    monkeypatch.setattr(prepare_module, "_runtime_bundles", lambda _root: (bundle,))
    return bundle


def test_default_hf_source_is_explicit_and_immutable() -> None:
    assert prepare_module.DEFAULT_REPO_ID == "nvidia/NeMo-Gym-EnterpriseOps-Assets"
    assert prepare_module.DEFAULT_REVISION != "main"
    assert len(prepare_module.DEFAULT_REVISION) == 40


def test_snapshot_filenames_cover_every_benchmark_domain() -> None:
    """The benchmark's per-domain map is derived from SNAPSHOT_FILENAMES; pin that contract."""
    single_domains = {Path(name).stem for name in prepare_module.SNAPSHOT_FILENAMES}
    assert set(benchmark_prepare.DOMAIN_SNAPSHOTS) == single_domains | {"hybrid"}
    assert benchmark_prepare.DOMAIN_SNAPSHOTS["hybrid"] == list(prepare_module.SNAPSHOT_FILENAMES)
    assert set(benchmark_prepare.DOMAINS) == set(benchmark_prepare.DOMAIN_SNAPSHOTS)


def test_prepare_downloads_validates_and_materializes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "snapshot" / prepare_module.REMOTE_TOOLS_DIR
    _write_flat_directory(source, _snapshot_files())
    bundle = _pinned_bundle(tmp_path, source, monkeypatch)
    calls = []

    def fake_snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(tmp_path / "snapshot")

    destinations = prepare_module.prepare(
        repo_id="test/assets",
        revision="immutable-revision",
        repo_root=tmp_path,
        snapshot_download=fake_snapshot_download,
    )

    assert calls == [
        {
            "repo_id": "test/assets",
            "repo_type": "dataset",
            "revision": "immutable-revision",
            "allow_patterns": ["enterpriseops_gym/tools/*"],
        }
    ]
    assert destinations == (bundle.local_dir,)
    for name in prepare_module.SNAPSHOT_FILENAMES:
        assert (bundle.local_dir / name).is_file()


def test_prepare_rejects_checksum_mismatch_without_touching_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "snapshot" / prepare_module.REMOTE_TOOLS_DIR
    _write_flat_directory(source, _snapshot_files())
    destination = tmp_path / "prepared/tools"
    _write_flat_directory(destination, {"csm.json": "existing\n"})
    bad_bundle = prepare_module.AssetBundle(
        remote_dir=prepare_module.REMOTE_TOOLS_DIR,
        local_dir=destination,
        filenames=prepare_module.SNAPSHOT_FILENAMES,
        file_count=len(prepare_module.SNAPSHOT_FILENAMES),
        tree_sha256="0" * 64,
    )
    monkeypatch.setattr(prepare_module, "_runtime_bundles", lambda _root: (bad_bundle,))

    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare_module.prepare(repo_root=tmp_path, snapshot_download=lambda **_kwargs: str(tmp_path / "snapshot"))

    assert (destination / "csm.json").read_text() == "existing\n"


def test_prepare_rejects_a_snapshot_with_unexpected_filenames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "snapshot" / prepare_module.REMOTE_TOOLS_DIR
    files = _snapshot_files()
    files["unexpected.json"] = "{}\n"
    _write_flat_directory(source, files)
    _pinned_bundle(tmp_path, source, monkeypatch)

    with pytest.raises(ValueError, match="Invalid filenames"):
        prepare_module.prepare(repo_root=tmp_path, snapshot_download=lambda **_kwargs: str(tmp_path / "snapshot"))


def test_ensure_tool_snapshots_skips_the_download_when_already_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "snapshot" / prepare_module.REMOTE_TOOLS_DIR
    _write_flat_directory(source, _snapshot_files())
    bundle = _pinned_bundle(tmp_path, source, monkeypatch)
    calls = []

    def counting_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(tmp_path / "snapshot")

    first = prepare_module.ensure_tool_snapshots(repo_root=tmp_path, snapshot_download=counting_download)
    second = prepare_module.ensure_tool_snapshots(repo_root=tmp_path, snapshot_download=counting_download)

    assert first == second == bundle.local_dir
    assert len(calls) == 1, "a prepared directory must not be re-downloaded"


def test_ensure_tool_snapshots_stays_current_despite_an_unrelated_stray_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """tools/ is gitignored, so a stray file there must not force a permanent re-download."""
    source = tmp_path / "snapshot" / prepare_module.REMOTE_TOOLS_DIR
    _write_flat_directory(source, _snapshot_files())
    bundle = _pinned_bundle(tmp_path, source, monkeypatch)
    calls = []

    prepare_module.ensure_tool_snapshots(
        repo_root=tmp_path, snapshot_download=lambda **k: calls.append(k) or str(tmp_path / "snapshot")
    )
    (bundle.local_dir / "scratch.json").write_text("{}\n", encoding="utf-8")

    prepare_module.ensure_tool_snapshots(
        repo_root=tmp_path, snapshot_download=lambda **k: calls.append(k) or str(tmp_path / "snapshot")
    )
    assert len(calls) == 1


def test_ensure_tool_snapshots_accepts_a_validated_local_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = tmp_path / "air-gapped"
    _write_flat_directory(override, _snapshot_files())
    file_count, tree_sha256 = prepare_module.tree_hash(override)
    monkeypatch.setattr(prepare_module, "TOOLS_FILE_COUNT", file_count)
    monkeypatch.setattr(prepare_module, "TOOLS_TREE_SHA256", tree_sha256)
    monkeypatch.setenv(prepare_module.TOOLS_DIR_ENV_VAR, str(override))

    def forbidden_download(**_kwargs: object) -> str:
        raise AssertionError("the override must not trigger a download")

    assert prepare_module.ensure_tool_snapshots(repo_root=tmp_path, snapshot_download=forbidden_download) == override


def test_ensure_tool_snapshots_rejects_an_override_that_does_not_match_the_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = tmp_path / "air-gapped"
    _write_flat_directory(override, _snapshot_files())
    monkeypatch.setenv(prepare_module.TOOLS_DIR_ENV_VAR, str(override))

    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare_module.ensure_tool_snapshots(repo_root=tmp_path, snapshot_download=lambda **_k: "")


def test_ensure_tool_snapshots_reports_an_incomplete_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    override = tmp_path / "air-gapped"
    _write_flat_directory(override, {"csm.json": "{}\n"})
    monkeypatch.setenv(prepare_module.TOOLS_DIR_ENV_VAR, str(override))

    with pytest.raises(ValueError, match="missing snapshots"):
        prepare_module.ensure_tool_snapshots(repo_root=tmp_path, snapshot_download=lambda **_k: "")


def test_benchmark_prepare_fetches_snapshots_before_truncating_its_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A snapshot-download failure must not destroy an already-built benchmark JSONL."""
    output = tmp_path / "enterpriseops_oracle_benchmark.jsonl"
    output.write_text('{"existing": "rows"}\n', encoding="utf-8")
    monkeypatch.setattr(benchmark_prepare, "DATA_DIR", tmp_path)
    monkeypatch.setattr(benchmark_prepare, "OUTPUT_FPATH", output)

    def failing_ensure(**_kwargs: object) -> Path:
        raise RuntimeError("hub unreachable")

    monkeypatch.setattr(benchmark_prepare, "ensure_tool_snapshots", failing_ensure)

    with pytest.raises(RuntimeError, match="hub unreachable"):
        benchmark_prepare.prepare()

    assert output.read_text() == '{"existing": "rows"}\n'


@pytest.mark.skipif(not prepare_module.TOOLS_DIR.is_dir(), reason="snapshots not prepared")
def test_prepared_snapshots_match_the_pinned_hash() -> None:
    """When the snapshots are present locally, they must match the committed pin."""
    actual = prepare_module.tree_hash(prepare_module.TOOLS_DIR, prepare_module.SNAPSHOT_FILENAMES)
    assert actual == (prepare_module.TOOLS_FILE_COUNT, prepare_module.TOOLS_TREE_SHA256)
