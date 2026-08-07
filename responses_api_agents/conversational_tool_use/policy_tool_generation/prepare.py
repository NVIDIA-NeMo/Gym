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

"""Download policy/tool generation reference assets from Hugging Face."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

from responses_api_agents.conversational_tool_use.policy_tool_generation.assets import (
    GOLDEN_FILENAMES,
    GOLDEN_TREE_SHA256,
    GOLDENS_DIR,
    PROMPTS_DIR,
)


DEFAULT_REPO_ID = "nvidia/NeMo-Gym-Conversational-Tool-Use-Assets"
DEFAULT_REVISION = "090dadd53f838150cc566d71cd2d6ff47729fdbe"  # pragma: allowlist secret
REMOTE_AGENT_DIR = Path("conversational_tool_use_policy_tool_generation")
REMOTE_GOLDENS_DIR = REMOTE_AGENT_DIR / "golden_policies"
REMOTE_PROMPT_HISTORY_DIR = REMOTE_AGENT_DIR / "prompt_history"
PROMPT_HISTORY_COUNT = 42
PROMPT_HISTORY_TREE_SHA256 = (  # pragma: allowlist secret
    "fd4d674dea96fee4d258daef1defa55b4aa42dfe6bc7720ac2b0e44aa41c5d90"
)

SnapshotDownload = Callable[..., str]


def tree_hash(directory: Path) -> tuple[int, str]:
    """Return a stable filename-and-content hash for one flat asset directory."""
    digest = hashlib.sha256()
    paths = sorted(path for path in directory.iterdir() if path.is_file())
    for path in paths:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return len(paths), digest.hexdigest()


def _replace_directory(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def _snapshot_download() -> SnapshotDownload:
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Preparing conversational tool-use references requires `huggingface-hub`. "
            "Install the repository development environment first."
        ) from exc
    return snapshot_download


def prepare(
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
    output_dir: Path = GOLDENS_DIR,
    include_prompt_history: bool = False,
    prompt_history_dir: Path | None = None,
    snapshot_download: SnapshotDownload | None = None,
) -> tuple[Path, Path | None]:
    """Download, validate, and materialize the runtime references and optional prompt history."""
    download = snapshot_download or _snapshot_download()
    patterns = [f"{REMOTE_GOLDENS_DIR.as_posix()}/*"]
    if include_prompt_history:
        patterns.append(f"{REMOTE_PROMPT_HISTORY_DIR.as_posix()}/*")
    snapshot = Path(
        download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            allow_patterns=patterns,
        )
    )

    source_goldens = snapshot / REMOTE_GOLDENS_DIR
    expected_names = set(GOLDEN_FILENAMES)
    actual_names = {path.name for path in source_goldens.iterdir() if path.is_file()}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ValueError(f"Invalid golden reference filenames: missing={missing}, extra={extra}")
    if tree_hash(source_goldens) != (len(GOLDEN_FILENAMES), GOLDEN_TREE_SHA256):
        raise ValueError("Golden policy/tool reference checksum mismatch")

    materialized_history: Path | None = None
    source_history: Path | None = None
    if include_prompt_history:
        source_history = snapshot / REMOTE_PROMPT_HISTORY_DIR
        if tree_hash(source_history) != (PROMPT_HISTORY_COUNT, PROMPT_HISTORY_TREE_SHA256):
            raise ValueError("Policy/tool prompt-history checksum mismatch")
        materialized_history = prompt_history_dir or PROMPTS_DIR / "archive"

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".conversational-tool-use-prepare-", dir=output_dir.parent) as temp_dir:
        staged_goldens = Path(temp_dir) / "golden_policies"
        shutil.copytree(source_goldens, staged_goldens)
        _replace_directory(staged_goldens, output_dir)

    if source_history is not None and materialized_history is not None:
        _replace_directory(source_history, materialized_history)

    return output_dir, materialized_history


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output-dir", type=Path, default=GOLDENS_DIR)
    parser.add_argument(
        "--include-prompt-history",
        action="store_true",
        help="Also materialize historical prompt revisions; they are not required at runtime.",
    )
    parser.add_argument("--prompt-history-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir, history_dir = prepare(
        repo_id=args.repo_id,
        revision=args.revision,
        output_dir=args.output_dir,
        include_prompt_history=args.include_prompt_history,
        prompt_history_dir=args.prompt_history_dir,
    )
    print(f"Prepared policy/tool references in {output_dir}")
    if history_dir is not None:
        print(f"Prepared prompt history in {history_dir}")


if __name__ == "__main__":
    main()
