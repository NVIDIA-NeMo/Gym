# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A Chromium build counts as installed only once Playwright finished installing it."""

from pathlib import Path

import pytest
from browser import _chromium


def test_a_build_still_being_unpacked_is_not_installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
    build = tmp_path / "chromium-1234"
    executable = build / "chrome-linux64" / "chrome"
    executable.parent.mkdir(parents=True)
    executable.write_text("")

    # Another worker's install has unpacked the executable but not the rest of the build.
    assert not _chromium._executable_present()

    (build / "INSTALLATION_COMPLETE").write_text("")
    assert _chromium._executable_present()
