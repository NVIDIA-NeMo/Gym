# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Obtain the pinned upstream scorer and its released reference data at startup."""

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import snapshot_download


REPOSITORY = "https://github.com/fresnellll/ChemCoTBench-V2.git"
REPO_REVISION = "dcd35470de4096a1b10ee9ed6f072bcee983a9cc"  # pragma: allowlist secret
DATA_REVISION = "f0bb2fb00c97cb3257294a639e28f960f2da157e"  # pragma: allowlist secret


def ensure_repository(repo_path: str | None, revision: str = REPO_REVISION) -> Path:
    if repo_path:
        path = Path(repo_path).expanduser().resolve()
    else:
        path = Path(__file__).parent / ".upstream" / revision
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(dir=path.parent) as temporary:
                checkout = Path(temporary) / "repo"
                subprocess.run(["git", "clone", "--quiet", "--no-checkout", REPOSITORY, str(checkout)], check=True)
                subprocess.run(["git", "-C", str(checkout), "checkout", "--quiet", revision], check=True)
                if not path.exists():
                    checkout.rename(path)
    if not all((path / name).is_dir() for name in ("formal_cot", "evaluation", "baselines")):
        raise ValueError(f"Not a ChemCoTBench-V2 checkout: {path}")
    return path


def ensure_data(data_dir: str | None, revision: str = DATA_REVISION) -> Path:
    path = (
        Path(data_dir).expanduser().resolve()
        if data_dir
        else Path(
            snapshot_download(
                "fresnellll/ChemCoTBench-V2",
                repo_type="dataset",
                revision=revision,
            )
        )
    )
    if not (path / "manifest.json").is_file():
        raise ValueError(f"ChemCoTBench data directory must contain manifest.json: {path}")
    return path
