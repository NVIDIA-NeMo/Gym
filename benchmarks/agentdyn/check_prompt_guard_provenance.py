# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare the pinned PromptGuard2 mirror against Meta's canonical gated repository.

The PromptGuard2 treatment currently scores with a public mirror, because
`meta-llama/Llama-Prompt-Guard-2-86M` is gated behind manual approval. That provenance
caveat can only be retired by a byte-level comparison against the canonical weights, which
needs a Hugging Face account that has been granted access.

    HF_TOKEN=hf_... python benchmarks/agentdyn/check_prompt_guard_provenance.py

Exit codes: 0 every shared file matches, 1 a file differs or is missing, 2 access is still
blocked (nothing was compared, and no provenance claim may be made either way).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, LocalTokenNotFoundError


CANONICAL_REPO = "meta-llama/Llama-Prompt-Guard-2-86M"
MIRROR_REPO = "project-free-llama/Llama-Prompt-Guard-2-86M"
MIRROR_REVISION = "43882965632dcb7b20299530f6436ac759d07fd9"  # pragma: allowlist secret
# Metadata that legitimately differs between a mirror and its source; a mismatch here says
# nothing about the weights.
IGNORED_NAMES = frozenset({"README.md", ".gitattributes", "checklist.chk", "LICENSE", "USE_POLICY.md"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_files(root: Path) -> dict[str, Path]:
    return {
        str(path.relative_to(root)): path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in IGNORED_NAMES and ".cache" not in path.parts
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-repo", default=CANONICAL_REPO)
    parser.add_argument("--mirror-repo", default=MIRROR_REPO)
    parser.add_argument("--mirror-revision", default=MIRROR_REVISION)
    args = parser.parse_args()

    try:
        canonical_root = Path(snapshot_download(repo_id=args.canonical_repo))
    except (GatedRepoError, LocalTokenNotFoundError, HfHubHTTPError) as error:
        print(f"BLOCKED: cannot read {args.canonical_repo}: {type(error).__name__}")
        print()
        print("The canonical repository is gated behind manual approval. To run the comparison:")
        print(f"  1. Accept the license at https://huggingface.co/{args.canonical_repo}")
        print("  2. Wait for Meta to grant access, then create a read token at")
        print("     https://huggingface.co/settings/tokens")
        print("  3. Export it as HF_TOKEN and re-run this script")
        return 2

    mirror_root = Path(snapshot_download(repo_id=args.mirror_repo, revision=args.mirror_revision))
    canonical_files = _snapshot_files(canonical_root)
    mirror_files = _snapshot_files(mirror_root)

    mismatched = 0
    for name in sorted(canonical_files):
        canonical_digest = _sha256(canonical_files[name])
        mirror_path = mirror_files.get(name)
        if mirror_path is None:
            print(f"MISSING  {name}: absent from the mirror")
            mismatched += 1
            continue
        mirror_digest = _sha256(mirror_path)
        verdict = "MATCH   " if canonical_digest == mirror_digest else "MISMATCH"
        mismatched += canonical_digest != mirror_digest
        print(f"{verdict} {name}  canonical={canonical_digest[:16]} mirror={mirror_digest[:16]}")

    extra = sorted(set(mirror_files) - set(canonical_files))
    for name in extra:
        print(f"EXTRA    {name}: present only in the mirror (not a weight-level mismatch by itself)")

    print()
    print(f"Compared {len(canonical_files)} files; {mismatched} differ or are missing.")
    return 1 if mismatched else 0


if __name__ == "__main__":
    sys.exit(main())
