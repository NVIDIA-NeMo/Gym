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

from responses_api_agents.conversational_tool_use.policy_tool_generation import assets
from responses_api_agents.conversational_tool_use.policy_tool_generation import prepare as prepare_module


def _write_flat_directory(directory: Path, files: dict[str, str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for filename, content in files.items():
        (directory / filename).write_text(content, encoding="utf-8")


def test_default_hf_source_is_explicit_and_immutable() -> None:
    assert prepare_module.DEFAULT_REPO_ID == "nvidia/NeMo-Gym-Conversational-Tool-Use-Assets"
    assert prepare_module.DEFAULT_REVISION == "090dadd53f838150cc566d71cd2d6ff47729fdbe"  # pragma: allowlist secret


def test_prepare_downloads_validated_references_and_optional_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    golden_files = {"policy-1.md": "policy\n", "tools_1.jsonl": "{}\n"}
    history_files = {"old.txt": "old prompt\n"}
    _write_flat_directory(snapshot / prepare_module.REMOTE_GOLDENS_DIR, golden_files)
    _write_flat_directory(snapshot / prepare_module.REMOTE_PROMPT_HISTORY_DIR, history_files)

    monkeypatch.setattr(prepare_module, "GOLDEN_FILENAMES", tuple(golden_files))
    monkeypatch.setattr(
        prepare_module,
        "GOLDEN_TREE_SHA256",
        prepare_module.tree_hash(snapshot / prepare_module.REMOTE_GOLDENS_DIR)[1],
    )
    monkeypatch.setattr(prepare_module, "PROMPT_HISTORY_COUNT", len(history_files))
    monkeypatch.setattr(
        prepare_module,
        "PROMPT_HISTORY_TREE_SHA256",
        prepare_module.tree_hash(snapshot / prepare_module.REMOTE_PROMPT_HISTORY_DIR)[1],
    )
    calls = []

    def fake_snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(snapshot)

    output_dir = tmp_path / "prepared" / "golden_policies"
    history_dir = tmp_path / "prepared" / "prompt_history"
    result, history_result = prepare_module.prepare(
        repo_id="test/assets",
        revision="immutable-revision",
        output_dir=output_dir,
        include_prompt_history=True,
        prompt_history_dir=history_dir,
        snapshot_download=fake_snapshot_download,
    )

    assert calls == [
        {
            "repo_id": "test/assets",
            "repo_type": "dataset",
            "revision": "immutable-revision",
            "allow_patterns": [
                "conversational_tool_use_policy_tool_generation/golden_policies/*",
                "conversational_tool_use_policy_tool_generation/prompt_history/*",
            ],
        }
    ]
    assert result == output_dir
    assert history_result == history_dir
    assert {path.name: path.read_text() for path in output_dir.iterdir()} == golden_files
    assert {path.name: path.read_text() for path in history_dir.iterdir()} == history_files


def test_prepare_rejects_checksum_mismatch_without_replacing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    _write_flat_directory(snapshot / prepare_module.REMOTE_GOLDENS_DIR, {"policy-1.md": "changed\n"})
    monkeypatch.setattr(prepare_module, "GOLDEN_FILENAMES", ("policy-1.md",))
    monkeypatch.setattr(prepare_module, "GOLDEN_TREE_SHA256", "0" * 64)
    output_dir = tmp_path / "golden_policies"
    _write_flat_directory(output_dir, {"existing.txt": "keep\n"})

    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare_module.prepare(
            output_dir=output_dir,
            snapshot_download=lambda **_kwargs: str(snapshot),
        )

    assert (output_dir / "existing.txt").read_text() == "keep\n"


def test_load_assets_explains_how_to_prepare_missing_references(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(assets, "GOLDENS_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="policy_tool_generation.prepare"):
        assets.load_assets("general")
