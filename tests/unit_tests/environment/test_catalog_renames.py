# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

from nemo_gym.environment.assets import asset_config_path
from nemo_gym.environment.catalog_renames import RENAMED_CATALOG_NAMES
from nemo_gym.environment.registry import discover_environment_catalog


REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def catalog_kinds() -> dict[str, str]:
    """The kind of every manifest in the repository's catalog, by catalog name."""
    return {entry.name: entry.kind for entry in discover_environment_catalog() if entry.manifest_path is not None}


def test_every_new_name_is_a_manifest_and_no_old_name_is(catalog_kinds: dict[str, str]) -> None:
    for old, new in RENAMED_CATALOG_NAMES.items():
        assert new in catalog_kinds, f"`{old}` was renamed to `{new}`, which no manifest names"
        assert old not in catalog_kinds, f"`{old}` was renamed but a manifest still names it"


def test_renames_do_not_chain() -> None:
    # Each old name maps straight to a current name, so one lookup is enough and a cycle cannot form.
    assert not set(RENAMED_CATALOG_NAMES.values()) & set(RENAMED_CATALOG_NAMES)


def test_every_old_name_still_selects_a_config(catalog_kinds: dict[str, str]) -> None:
    for old, new in RENAMED_CATALOG_NAMES.items():
        assert Path(asset_config_path(catalog_kinds[new], old)).is_file(), f"`{old}` no longer selects a config"
