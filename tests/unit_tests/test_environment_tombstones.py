# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import runpy
import sys
from types import ModuleType

import pytest
from pytest import LogCaptureFixture, MonkeyPatch

from nemo_gym import WORKING_DIR
from nemo_gym._config_aliases import LEGACY_ENVIRONMENT_ALIASES


@pytest.mark.parametrize(("legacy", "canonical"), LEGACY_ENVIRONMENT_ALIASES.items())
def test_legacy_prepare_entry_point_forwards_to_canonical_environment(
    monkeypatch: MonkeyPatch, caplog: LogCaptureFixture, legacy: str, canonical: str
) -> None:
    called = False

    def fake_main() -> None:
        nonlocal called
        called = True

    canonical_prepare = ModuleType(f"environments.{canonical}.prepare")
    canonical_prepare.main = fake_main
    monkeypatch.setitem(sys.modules, canonical_prepare.__name__, canonical_prepare)
    with caplog.at_level(logging.WARNING):
        runpy.run_path(str(WORKING_DIR / "environments" / legacy / "prepare.py"), run_name="__main__")

    assert called
    assert f"`environments/{legacy}/prepare.py` is deprecated" in caplog.text
