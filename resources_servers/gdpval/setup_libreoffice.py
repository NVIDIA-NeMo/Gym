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
    # `javaldx` (the helper libreoffice uses to locate the JRE via JNI)
    # ships in `libreoffice-java-common`, which is NOT a dependency of
    # the `libreoffice` metapackage on Ubuntu 24.04. Without it,
    # `/usr/lib/libreoffice/program/javaldx` simply doesn't exist —
    # libreoffice can't discover the JRE no matter how many JRE
    # packages are installed. Without this entry, every attempt at
    # the JRE-prebaked fix (v1–v4) silently failed because the bridge
    # between libreoffice and the JRE was missing.
    "libreoffice-java-common",
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
    """Make sure libreoffice + a *functional* JRE are present. ALWAYS run apt-install on Linux.

    Returns True if libreoffice + a usable JRE are available after the call.

    History (so we don't regress again):
      v1 — early-exit on `shutil.which("libreoffice")`. The deployment image
           bakes libreoffice in *without* a JRE; apt-install was skipped and
           libreoffice's `javaldx` helper failed for every chart/formula doc.
      v2 — early-exit on `shutil.which("libreoffice") and shutil.which("java")`.
           Same regression — the image has a `java` binary that satisfies
           `which()` but isn't functional.
      v3 — early-exit on `shutil.which("libreoffice") and _java_runs()` (i.e.
           `java -version` rc=0). STILL not enough — `java -version` succeeds
           but libreoffice's `javaldx` (which loads libjvm.so via JNI) still
           can't find a usable JRE. Concrete signal: 10 `failed to launch
           javaldx` errors at 35 rollouts on slurm 2621556.
      v4 (this) — drop the early-exit entirely. ALWAYS run apt-update +
           apt-install of the full package list. `apt-get install` is
           idempotent on already-installed packages (a few seconds when
           everything is present), and the only configuration that
           reliably gets `javaldx` working is having `default-jre-headless`
           installed via the same dpkg DB that libreoffice's wrapper
           consults. Rather than try to detect that out-of-band, just
           install.

    Logs a WARNING on apt failure so the server still boots and rubric-mode
    tasks keep working; comparison-mode preconvert will surface its own
    per-file errors via ``preconvert.py``.
    """
    if not sys.platform.startswith("linux"):
        LOGGER.warning(
            "auto-install only supports Linux (sys.platform=%s); GDPVal preconvert will be a no-op.",
            sys.platform,
        )
        return False

    if not shutil.which("apt-get"):
        LOGGER.warning("apt-get is unavailable; GDPVal preconvert will be a no-op.")
        return False

    LOGGER.info(
        "Ensuring %s via apt-get (idempotent; first call adds ~500 MB if libreoffice is missing, "
        "subsequent calls only verify the packages)...",
        ", ".join(_APT_PACKAGES),
    )

    try:
        rc, _, err = _run(["apt-get", "update", "-qq"], timeout=_APT_INSTALL_TIMEOUT_S)
        if rc != 0:
            # Non-fatal: stale apt index can still satisfy install if the packages are cached.
            LOGGER.warning("apt-get update failed (rc=%d): %s", rc, (err or "").strip()[:500])
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
