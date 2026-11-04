# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Expose declared checkout metadata inside each native pytest sandbox."""

import json
import shutil
from pathlib import Path


def pytest_load_initial_conftests(early_config, parser, args):
    root = Path(__file__).resolve().parents[3]
    metadata = root / "scripts/pants/native_core/git-state"
    checkout_git = root / ".git"
    if checkout_git.exists():
        raise RuntimeError("Native test sandbox unexpectedly contains Git metadata")
    # Pants materializes files, so Git's required empty refs directory is recreated.
    (metadata / "refs").mkdir(exist_ok=True)
    checkout_git.symlink_to(metadata, target_is_directory=True)
    distributions = list((metadata.parent / "distribution").glob("*.dist-info"))
    if len(distributions) != 1:
        raise RuntimeError("Native sandbox is missing the sampled editable distribution")
    installed = root / distributions[0].name
    shutil.copytree(distributions[0], installed)
    # Relocate the original editable metadata alongside the relocated source.
    (installed / "direct_url.json").write_text(json.dumps({"dir_info": {"editable": True}, "url": root.as_uri()}))
