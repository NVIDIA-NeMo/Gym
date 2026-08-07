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

from pathlib import Path

import pytest

from resources_servers.conversational_tool_use_simulation import prepare as prepare_module
from responses_api_agents.conversational_tool_use.domain_generation import assets as domain_assets
from responses_api_agents.conversational_tool_use.policy_tool_generation import assets as policy_tool_assets
from responses_api_agents.conversational_tool_use.scenario_generation import assets as scenario_assets


def _write_flat_directory(directory: Path, files: dict[str, str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for filename, content in files.items():
        (directory / filename).write_text(content, encoding="utf-8")


def _bundle(remote_dir: str, local_dir: Path, source: Path) -> prepare_module.AssetBundle:
    filenames = tuple(sorted(path.name for path in source.iterdir()))
    count, tree_sha256 = prepare_module.tree_hash(source)
    return prepare_module.AssetBundle(
        remote_dir=Path(remote_dir),
        local_dir=local_dir,
        filenames=filenames,
        file_count=count,
        tree_sha256=tree_sha256,
    )


def test_default_hf_source_is_explicit_and_immutable() -> None:
    assert prepare_module.DEFAULT_REPO_ID == "nvidia/NeMo-Gym-Conversational-Tool-Use-Assets"
    assert prepare_module.DEFAULT_REVISION != "main"
    assert len(prepare_module.DEFAULT_REVISION) == 40


def test_runtime_bundle_filenames_match_component_loaders(tmp_path: Path) -> None:
    bundles = {bundle.remote_dir.as_posix(): bundle for bundle in prepare_module._runtime_bundles(tmp_path)}

    assert bundles["conversational_tool_use_domain_generation/prompts"].filenames == domain_assets.PROMPT_FILENAMES
    assert (
        bundles["conversational_tool_use_policy_tool_generation/prompts"].filenames
        == policy_tool_assets.PROMPT_FILENAMES
    )
    assert bundles["conversational_tool_use_scenario_generation/prompts"].filenames == scenario_assets.PROMPT_FILENAMES


def test_prepare_downloads_all_runtime_assets_and_optional_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    runtime_source = snapshot / "component/prompts"
    history_source = snapshot / "component/prompt_history"
    _write_flat_directory(runtime_source, {"prompt.txt": "runtime prompt\n"})
    _write_flat_directory(history_source, {"old.txt": "historical prompt\n"})
    runtime_destination = tmp_path / "prepared/prompts"
    history_destination = tmp_path / "prepared/prompts/archive"
    runtime_bundle = _bundle("component/prompts", runtime_destination, runtime_source)
    history_bundle = _bundle("component/prompt_history", history_destination, history_source)
    monkeypatch.setattr(prepare_module, "_runtime_bundles", lambda _root: (runtime_bundle,))
    monkeypatch.setattr(prepare_module, "_history_bundles", lambda _root: (history_bundle,))
    calls = []

    def fake_snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(snapshot)

    destinations = prepare_module.prepare(
        repo_id="test/assets",
        revision="immutable-revision",
        repo_root=tmp_path,
        include_prompt_history=True,
        snapshot_download=fake_snapshot_download,
    )

    assert calls == [
        {
            "repo_id": "test/assets",
            "repo_type": "dataset",
            "revision": "immutable-revision",
            "allow_patterns": ["component/prompts/*", "component/prompt_history/*"],
        }
    ]
    assert destinations == (runtime_destination, history_destination)
    assert (runtime_destination / "prompt.txt").read_text() == "runtime prompt\n"
    assert (history_destination / "old.txt").read_text() == "historical prompt\n"


def test_prepare_preserves_non_asset_files_in_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    source = snapshot / "component/prompts"
    _write_flat_directory(source, {"prompt.txt": "new prompt\n"})
    destination = tmp_path / "prepared/prompts"
    _write_flat_directory(destination, {"schema.json": "{}\n", "prompt.txt": "old prompt\n"})
    bundle = _bundle("component/prompts", destination, source)
    monkeypatch.setattr(prepare_module, "_runtime_bundles", lambda _root: (bundle,))

    prepare_module.prepare(repo_root=tmp_path, snapshot_download=lambda **_kwargs: str(snapshot))

    assert (destination / "prompt.txt").read_text() == "new prompt\n"
    assert (destination / "schema.json").read_text() == "{}\n"


def test_prepare_validates_every_bundle_before_replacing_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    first_source = snapshot / "first/prompts"
    second_source = snapshot / "second/prompts"
    _write_flat_directory(first_source, {"first.txt": "new first\n"})
    _write_flat_directory(second_source, {"second.txt": "changed\n"})
    first_destination = tmp_path / "prepared/first"
    second_destination = tmp_path / "prepared/second"
    _write_flat_directory(first_destination, {"first.txt": "old first\n"})
    first_bundle = _bundle("first/prompts", first_destination, first_source)
    invalid_bundle = prepare_module.AssetBundle(
        remote_dir=Path("second/prompts"),
        local_dir=second_destination,
        filenames=("second.txt",),
        file_count=1,
        tree_sha256="0" * 64,
    )
    monkeypatch.setattr(prepare_module, "_runtime_bundles", lambda _root: (first_bundle, invalid_bundle))

    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare_module.prepare(repo_root=tmp_path, snapshot_download=lambda **_kwargs: str(snapshot))

    assert (first_destination / "first.txt").read_text() == "old first\n"
