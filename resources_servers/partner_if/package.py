# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Package paths that configs name, and the native task bundles staged from them."""

from pathlib import Path
from typing import Annotated

from pydantic import AfterValidator, Field

from nemo_gym import _resolve_under_cwd_or_install
from resources_servers.partner_if.grading_contract import GradingContract, StrictModel


def resolve_repo_path(path: Path) -> Path:
    """Servers start inside their own directories, so resolve relative package paths like Gym's other
    read paths: under NEMO_GYM_EXTRA_ROOTS (a benchmark plugin root), the cwd, then the Gym root."""
    return _resolve_under_cwd_or_install(path)


RepoPath = Annotated[Path, AfterValidator(resolve_repo_path)]


class TaskBundle(StrictModel):
    """A native task's packaged contract and the documents staged for its actor."""

    contract: GradingContract
    documents: dict[str, str] = Field(default_factory=dict)
