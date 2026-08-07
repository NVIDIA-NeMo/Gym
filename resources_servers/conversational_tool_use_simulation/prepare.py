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

"""Prepare all conversational tool-use prompts and references from Hugging Face."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


DEFAULT_REPO_ID = "nvidia/NeMo-Gym-Conversational-Tool-Use-Assets"
DEFAULT_REVISION = "b835033a2288a991a7efc9b7a1e358dcfa2423da"  # pragma: allowlist secret
REPO_ROOT = Path(__file__).resolve().parents[2]
PREPARE_COMMAND = "python -m resources_servers.conversational_tool_use_simulation.prepare"

POLICY_TOOL_REFERENCE_FILENAMES = tuple(
    filename for index in range(1, 9) for filename in (f"policy-{index}.md", f"tools_{index}.jsonl")
)

SnapshotDownload = Callable[..., str]


@dataclass(frozen=True)
class AssetBundle:
    remote_dir: Path
    local_dir: Path
    file_count: int
    tree_sha256: str
    filenames: tuple[str, ...] | None = None


def _runtime_bundles(repo_root: Path) -> tuple[AssetBundle, ...]:
    return (
        AssetBundle(
            remote_dir=Path("conversational_tool_use_domain_generation/prompts"),
            local_dir=repo_root / "responses_api_agents/conversational_tool_use/domain_generation/prompts",
            filenames=("domain_followup.txt", "domain_generation.txt"),
            file_count=2,
            tree_sha256="1b71b9f7fb0d2f7fe487b24ecf09014f2ed75f8833508cc4eef503fdd8ee2b9d",  # pragma: allowlist secret
        ),
        AssetBundle(
            remote_dir=Path("conversational_tool_use_policy_tool_generation/prompts"),
            local_dir=repo_root / "responses_api_agents/conversational_tool_use/policy_tool_generation/prompts",
            filenames=(
                "cohesion_judge.txt",
                "general_policy.txt",
                "general_policy_refine.txt",
                "general_tools.txt",
                "golden_judge.txt",
                "proactive_policy.txt",
                "proactive_policy_refine.txt",
                "proactive_tools.txt",
                "tools_refine.txt",
            ),
            file_count=9,
            tree_sha256="d602f69b263e124468e0e952a39f473ba83b0278fb46462e44b4d3ed2e6f412e",  # pragma: allowlist secret
        ),
        AssetBundle(
            remote_dir=Path("conversational_tool_use_policy_tool_generation/golden_policies"),
            local_dir=repo_root
            / "responses_api_agents/conversational_tool_use/policy_tool_generation/references/golden_policies",
            filenames=POLICY_TOOL_REFERENCE_FILENAMES,
            file_count=16,
            tree_sha256="c1c621e88f763dab8fa23e6721180376d65b1386b99e662d32c652dcf28e1cd6",  # pragma: allowlist secret
        ),
        AssetBundle(
            remote_dir=Path("conversational_tool_use_scenario_generation/prompts"),
            local_dir=repo_root / "responses_api_agents/conversational_tool_use/scenario_generation/prompts",
            filenames=("scenario_system.txt", "scenario_user.txt"),
            file_count=2,
            tree_sha256="684e433926cee22beb71f34412578c26ff8e1589bd903ddae82021e613af03fb",  # pragma: allowlist secret
        ),
        AssetBundle(
            remote_dir=Path("conversational_tool_use_agent/prompts"),
            local_dir=repo_root / "responses_api_agents/conversational_tool_use/simulation/prompts",
            filenames=("agent_parallel_system.txt", "agent_system.txt"),
            file_count=2,
            tree_sha256="1a1c11c25d409113b378c92e8af778f223db21f4b56866518b32a33a79645a0d",  # pragma: allowlist secret
        ),
        AssetBundle(
            remote_dir=Path("conversational_tool_use_simulation/prompts"),
            local_dir=repo_root / "resources_servers/conversational_tool_use_simulation/prompts",
            filenames=(
                "agent_conversation_evaluation_system.txt",
                "agent_message_evaluation_system.txt",
                "complete_conversation.txt",
                "environment_conversation.txt",
                "environment_conversation_message.txt",
                "environment_message_evaluation_system.txt",
                "environment_simulator_system.txt",
                "environment_user_model_message.txt",
                "message_conversation.txt",
                "message_system_prefix.txt",
                "text_message.txt",
                "tool_call_message.txt",
                "tool_definition.txt",
                "tool_execution_message.txt",
                "user_agent_environment_conversation_evaluation_system.txt",
                "user_message_evaluation_system.txt",
                "user_simulator_system.txt",
            ),
            file_count=17,
            tree_sha256="7eb48bbdf4b6e01b4bd2f6214a1892cb06df21f534de8e7dbf1e3ed9bee28ee0",  # pragma: allowlist secret
        ),
    )


def _history_bundles(repo_root: Path) -> tuple[AssetBundle, ...]:
    return (
        AssetBundle(
            remote_dir=Path("conversational_tool_use_domain_generation/prompt_history"),
            local_dir=repo_root / "responses_api_agents/conversational_tool_use/domain_generation/prompts/archive",
            file_count=2,
            tree_sha256="27e7dab1d6a3a9766929766fe192e35ead23876eab419692d11822f27bfba770",  # pragma: allowlist secret
        ),
        AssetBundle(
            remote_dir=Path("conversational_tool_use_policy_tool_generation/prompt_history"),
            local_dir=repo_root
            / "responses_api_agents/conversational_tool_use/policy_tool_generation/prompts/archive",
            file_count=42,
            tree_sha256="fd4d674dea96fee4d258daef1defa55b4aa42dfe6bc7720ac2b0e44aa41c5d90",  # pragma: allowlist secret
        ),
    )


def tree_hash(directory: Path) -> tuple[int, str]:
    """Return a stable filename-and-content hash for one flat asset directory."""
    digest = hashlib.sha256()
    paths = sorted(path for path in directory.iterdir() if path.is_file())
    for path in paths:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return len(paths), digest.hexdigest()


def _snapshot_download() -> SnapshotDownload:
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise RuntimeError("Preparing conversational tool-use assets requires `huggingface-hub`.") from exc
    return snapshot_download


def _validate_bundle(snapshot: Path, bundle: AssetBundle) -> Path:
    source = snapshot / bundle.remote_dir
    if not source.is_dir():
        raise ValueError(f"Missing asset directory in snapshot: {bundle.remote_dir}")
    actual_names = tuple(sorted(path.name for path in source.iterdir() if path.is_file()))
    if bundle.filenames is not None and actual_names != tuple(sorted(bundle.filenames)):
        raise ValueError(
            f"Invalid filenames for {bundle.remote_dir}: expected={sorted(bundle.filenames)}, actual={actual_names}"
        )
    actual_hash = tree_hash(source)
    expected_hash = (bundle.file_count, bundle.tree_sha256)
    if actual_hash != expected_hash:
        raise ValueError(
            f"Asset checksum mismatch for {bundle.remote_dir}: expected={expected_hash}, actual={actual_hash}"
        )
    return source


def _materialize_bundle(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".conversational-tool-use-prepare-", dir=destination) as temp_dir:
        staging = Path(temp_dir)
        for source_file in source.iterdir():
            if source_file.is_file():
                shutil.copy2(source_file, staging / source_file.name)
        for staged_file in staging.iterdir():
            staged_file.replace(destination / staged_file.name)


def prepare(
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
    repo_root: Path = REPO_ROOT,
    include_prompt_history: bool = False,
    snapshot_download: SnapshotDownload | None = None,
) -> tuple[Path, ...]:
    """Download, validate, and materialize all runtime assets and optional prompt history."""
    bundles = _runtime_bundles(repo_root)
    if include_prompt_history:
        bundles += _history_bundles(repo_root)
    download = snapshot_download or _snapshot_download()
    snapshot = Path(
        download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            allow_patterns=[f"{bundle.remote_dir.as_posix()}/*" for bundle in bundles],
        )
    )

    sources = tuple(_validate_bundle(snapshot, bundle) for bundle in bundles)
    for source, bundle in zip(sources, bundles, strict=True):
        _materialize_bundle(source, bundle.local_dir)
    return tuple(bundle.local_dir for bundle in bundles)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--include-prompt-history",
        action="store_true",
        help="Also materialize historical prompt revisions; they are not required at runtime.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    destinations = prepare(
        repo_id=args.repo_id,
        revision=args.revision,
        repo_root=args.repo_root,
        include_prompt_history=args.include_prompt_history,
    )
    for destination in destinations:
        print(f"Prepared conversational tool-use assets in {destination}")


if __name__ == "__main__":
    main()
