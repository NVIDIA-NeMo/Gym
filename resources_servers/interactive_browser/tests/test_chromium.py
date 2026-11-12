# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Chromium cache location and installation-completion checks."""

from pathlib import Path

import pytest
from browser import _chromium


@pytest.mark.parametrize(
    ("xdg_cache_home", "expected_parent"),
    [("/xdg/cache", Path("/xdg/cache")), ("", None), (None, None)],
    ids=["set", "empty", "unset"],
)
def test_browsers_dir_follows_xdg_cache_home_on_linux(monkeypatch, tmp_path, xdg_cache_home, expected_parent):
    """Playwright installs under $XDG_CACHE_HOME when it is non-empty, else under ~/.cache."""
    monkeypatch.setattr(_chromium.sys, "platform", "linux")
    monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    if xdg_cache_home is None:
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    else:
        monkeypatch.setenv("XDG_CACHE_HOME", xdg_cache_home)

    assert _chromium._browsers_dir() == (expected_parent or tmp_path / ".cache") / "ms-playwright"


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
