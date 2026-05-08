# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Idempotent host-side libreoffice install for GDPVal preconvert.

The deployment container (where the gdpval resources server runs) does
not ship libreoffice. We install on first server start so Office → PDF
preconversion in ``preconvert.py`` actually produces sibling PDFs.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys


LOGGER = logging.getLogger(__name__)

_APT_PACKAGES = (
    "libreoffice",
    "fonts-liberation",
    # libreoffice's chart/formula rendering needs Java; without a JRE it
    # logs `Warning: failed to launch javaldx` and silently exits rc=0
    # without producing the expected PDF for any doc with charts,
    # complex formulas, embedded objects, or pivot tables. Headless JRE
    # is enough — we never display a GUI.
    "default-jre-headless",
)

_APT_INSTALL_TIMEOUT_S = 600


def _run(cmd: list[str], *, timeout: int) -> tuple[int, str, str]:
    p = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return p.returncode, p.stdout, p.stderr


def _java_runs() -> bool:
    """Return True iff `java -version` runs successfully (rc=0).

    `shutil.which("java")` is necessary but not sufficient — the
    deployment image may have a `java` binary on PATH that's
    non-functional for libreoffice's `javaldx` helper (e.g. partially-
    installed openjdk, broken symlink, missing libjvm.so). The only way
    to know if the JRE is *usable* is to actually try to invoke it.
    """
    if not shutil.which("java"):
        return False
    try:
        rc, _, _ = _run(["java", "-version"], timeout=15)
    except Exception:
        return False
    return rc == 0


def ensure_libreoffice() -> bool:
    """Make sure ``libreoffice`` *and* a *functional* JRE are present; apt-install on Linux if missing.

    Returns True if libreoffice is available after the call, False otherwise.
    Idempotent: when libreoffice is on PATH AND `java -version` runs
    cleanly, returns immediately. When libreoffice is present but Java is
    missing or broken (e.g. the deployment image bakes libreoffice in
    without a usable JRE), still runs apt-install — without a working
    JRE, libreoffice silently exits rc=0 with `Warning: failed to launch
    javaldx` for any chart/formula/embedded-object doc, producing
    filename-only stubs in comparison mode.

    A previous version of this function gated the early-exit on
    `shutil.which("java")` alone; that wasn't enough — the deployment
    image had *some* `java` binary on PATH that satisfied `which()` but
    couldn't actually execute (`java -version` failed), so apt-install
    was skipped and the JRE was never landed. The functional probe in
    `_java_runs()` catches that case.

    Logs a WARNING (not raise) on install failure so the server still boots
    and rubric-mode tasks keep working; comparison-mode preconvert will then
    surface its own per-file errors via ``preconvert.py``.
    """
    if shutil.which("libreoffice") and _java_runs():
        return True

    if not sys.platform.startswith("linux"):
        LOGGER.warning(
            "libreoffice not on PATH and auto-install only supports Linux (sys.platform=%s); "
            "GDPVal preconvert will be a no-op.",
            sys.platform,
        )
        return False

    if not shutil.which("apt-get"):
        LOGGER.warning(
            "libreoffice or java not on PATH and apt-get is unavailable; GDPVal preconvert will be a no-op."
        )
        return False

    LOGGER.info(
        "Installing %s via apt-get (one-time, ~500 MB if libreoffice is missing)...",
        ", ".join(_APT_PACKAGES),
    )

    try:
        rc, _, err = _run(["apt-get", "update", "-qq"], timeout=_APT_INSTALL_TIMEOUT_S)
        if rc != 0:
            LOGGER.warning("apt-get update failed (rc=%d): %s", rc, (err or "").strip()[:500])
            return False
        rc, _, err = _run(
            ["apt-get", "install", "-y", "--no-install-recommends", *_APT_PACKAGES],
            timeout=_APT_INSTALL_TIMEOUT_S,
        )
        if rc != 0:
            LOGGER.warning("apt-get install libreoffice failed (rc=%d): %s", rc, (err or "").strip()[:500])
            return False
    except subprocess.TimeoutExpired:
        LOGGER.warning("apt-get timed out after %ds while installing libreoffice", _APT_INSTALL_TIMEOUT_S)
        return False
    except Exception as exc:
        LOGGER.warning("Unexpected error installing libreoffice: %r", exc)
        return False

    if not shutil.which("libreoffice"):
        LOGGER.warning("apt-get install reported success but libreoffice still not on PATH")
        return False
    if not _java_runs():
        LOGGER.warning(
            "apt-get install reported success but `java -version` still fails "
            "(libreoffice will fail on chart/formula docs)"
        )
        return False

    rc, out, err = _run(["libreoffice", "--version"], timeout=30)
    if rc != 0:
        LOGGER.warning("libreoffice --version failed after install (rc=%d): %s", rc, (err or "").strip()[:200])
        return False

    LOGGER.info("libreoffice ready: %s", out.strip())
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ok = ensure_libreoffice()
    sys.exit(0 if ok else 1)
