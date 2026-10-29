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
preconversion in ``preconvert.py`` actually produces sibling PDFs. With a
configured ``libreoffice_command``, ``ensure_libreoffice`` checks that command
instead and installs nothing.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence


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
    # packages are installed, because the bridge between libreoffice
    # and the JRE is missing.
    "libreoffice-java-common",
)

# A cold install pulls ~500 MB. When many servers start at once (e.g. several
# evaluation jobs launched together), each one pulls it concurrently from the
# same mirror, which can take well over 10 minutes. Leave generous headroom:
# in comparison mode a failed install aborts resources-server startup.
_APT_INSTALL_TIMEOUT_S = 1800
# A cold container start (e.g. an image on a shared filesystem) is slower than a host binary.
_COMMAND_CHECK_TIMEOUT_S = 120


def _run(cmd: list[str], *, timeout: int) -> tuple[int, str, str]:
    p = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        errors="replace",
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


def _javaldx_works() -> bool:
    """Return True iff libreoffice can actually find a JRE through `javaldx`.

    Cheaper proxies are not enough: `which("libreoffice")`, `which("java")`
    and `java -version` can all pass on an image where `javaldx` still cannot
    load a JVM, so chart and formula documents produce no PDF.

    `javaldx` is the bridge itself: it loads libjvm.so via JNI and prints the
    JRE path it resolved. Running it answers the real question instead of an
    adjacent one. On failure it prints "failed to launch" / "Could not find a
    Java Runtime" and yields no path.
    """
    exe = "/usr/lib/libreoffice/program/javaldx"
    if not os.path.exists(exe):
        # Ships in libreoffice-java-common, which the libreoffice metapackage
        # does not depend on. Absent means the bridge was never installed.
        return False
    try:
        rc, out, err = _run([exe], timeout=30)
    except Exception:
        return False
    blob = f"{out}\n{err}".lower()
    if "failed to launch" in blob or "could not find a java runtime" in blob:
        return False
    return rc == 0 and bool(out.strip())


def _write_probe_workbook(path: Path) -> None:
    """Write the probe workbook: one sheet with a formula and a chart."""
    from openpyxl import Workbook
    from openpyxl.chart import BarChart, Reference

    workbook = Workbook()
    sheet = workbook.active
    for row in (("Quarter", "Revenue"), ("Q1", 10), ("Q2", 30)):
        sheet.append(row)
    sheet["C2"] = "=SUM(B2:B3)"
    chart = BarChart()
    chart.add_data(Reference(sheet, min_col=2, min_row=1, max_row=3), titles_from_data=True)
    sheet.add_chart(chart, "E2")
    workbook.save(path)


def _probe_converts(libreoffice_command: Sequence[str], root: Path | None) -> bool:
    """Convert a probe workbook in place, in a fresh directory under *root* (the temp directory when None).

    This is one LibreOffice run, like the first attempt of every real conversion. ``convert_to_pdf`` would retry
    a workbook it could not convert from a copy in the temp directory, which hides a command that cannot see *root*.
    The probe also fails when the server cannot delete what the run left behind, such as the root-owned profile of a
    Docker command without ``--user``; every real conversion would leave the same files.
    """
    from resources_servers.gdpval.preconvert import libreoffice_pdf_argv

    probe_dir = profile_dir = None
    try:
        try:
            if root is not None:
                root.mkdir(parents=True, exist_ok=True)
            probe_dir = Path(tempfile.mkdtemp(prefix="gdpval-lo-probe-", dir=root))
            profile_dir = Path(tempfile.mkdtemp(prefix="lo-profile-"))
            probe = probe_dir / "probe.xlsx"
            _write_probe_workbook(probe)
        except OSError as exc:
            LOGGER.warning(
                "could not write the LibreOffice probe workbook in %s: %r", root or tempfile.gettempdir(), exc
            )
            return False
        argv = libreoffice_pdf_argv(probe, probe_dir, profile_dir, libreoffice_command)
        try:
            rc, _, err = _run(argv, timeout=_COMMAND_CHECK_TIMEOUT_S)
        except (OSError, subprocess.TimeoutExpired) as exc:
            rc, err = None, repr(exc)
        pdf = probe.with_suffix(".pdf")
        # A run that timed out fails like a real conversion, even if it wrote the PDF first.
        converted = rc is not None and pdf.is_file() and pdf.read_bytes().startswith(b"%PDF")
    finally:
        for directory in (probe_dir, profile_dir):
            if directory is not None:
                shutil.rmtree(directory, ignore_errors=True)
    if not converted:
        LOGGER.warning(
            "LibreOffice command %s did not convert a probe workbook in %s (rc=%s): %s",
            libreoffice_command,
            probe_dir,
            rc,
            # The end of stderr: container startup noise, such as font-cache warnings, comes first.
            (err or "").strip()[-300:],
        )
        return False
    left = [directory for directory in (probe_dir, profile_dir) if directory.exists()]
    if left:
        owner = _other_owner_uid(left)
        if owner is None:
            # Not an ownership problem: e.g. a container still running after a timeout, or NFS `.nfs*` files.
            LOGGER.warning(
                "LibreOffice command %s left files that the server cannot delete in %s",
                libreoffice_command,
                ", ".join(map(str, left)),
            )
        else:
            LOGGER.warning(
                "LibreOffice command %s left files owned by uid %d, which the server (uid %d) cannot delete, in %s; "
                "run it as the server's user, e.g. with Docker's --user %d:%d",
                libreoffice_command,
                owner,
                os.getuid(),
                ", ".join(map(str, left)),
                os.getuid(),
                os.getgid(),
            )
        return False
    return True


def _other_owner_uid(directories: Sequence[Path]) -> int | None:
    """Return the uid of a file under *directories* that the server's user does not own, or None."""
    uid = os.getuid()
    for directory in directories:
        for dirpath, dirnames, filenames in os.walk(directory):
            for name in (*dirnames, *filenames):
                try:
                    owner = os.lstat(os.path.join(dirpath, name)).st_uid
                except OSError:
                    continue
                if owner != uid:
                    return owner
    return None


def _command_works(libreoffice_command: Sequence[str], probe_dirs: Sequence[Path] = ()) -> bool:
    """Return True iff *libreoffice_command* runs, converts a probe workbook to PDF and leaves nothing the server
    cannot delete.

    The probe converts a workbook in the temp directory that real conversions
    use and under each of *probe_dirs*, so a container command that cannot see
    one of them fails here instead of on every task.
    """
    try:
        rc, out, err = _run([*libreoffice_command, "--version"], timeout=_COMMAND_CHECK_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as exc:
        LOGGER.warning("libreoffice_command %s could not run: %r", libreoffice_command, exc)
        return False
    if rc != 0:
        LOGGER.warning(
            "libreoffice_command %s --version failed (rc=%d): %s",
            libreoffice_command,
            rc,
            (err or "").strip()[-300:],
        )
        return False

    if not all(_probe_converts(libreoffice_command, root) for root in (None, *probe_dirs)):
        return False
    LOGGER.info("libreoffice_command ready: %s", out.strip())
    return True


def ensure_libreoffice(libreoffice_command: Sequence[str] | None = None, *, probe_dirs: Sequence[Path] = ()) -> bool:
    """Make sure libreoffice + a *functional* JRE are present on Linux.

    With *libreoffice_command* (e.g. a container prefix), only that command is
    checked, by ``--version`` and a real conversion in the temp directory and
    under each of *probe_dirs*; the host packages, javaldx and apt below are not
    used. *probe_dirs* is ignored for the host ``libreoffice``, which sees every path.

    Returns True if libreoffice + a usable JRE are available after the call.

    apt is skipped only when libreoffice's own `javaldx` bridge resolves a JRE
    (see ``_javaldx_works``) and a probe workbook with a formula and a chart
    converts; the same probe runs after an install. Weaker checks are not safe
    early-exits: an image can ship libreoffice without a JRE, or with a `java`
    binary that passes `which()` and even `java -version` while `javaldx` (which loads libjvm.so
    via JNI) still can't find a usable JRE, and then every chart/formula
    document fails to convert. Otherwise this runs apt-update + apt-install of
    the full package list: `apt-get install` is idempotent on already-installed
    packages (a few seconds when everything is present), and installing
    `default-jre-headless` through the same dpkg DB that libreoffice's wrapper
    consults is what reliably gets `javaldx` working.

    Logs a WARNING on apt failure so the server still boots and rubric-mode
    tasks keep working; comparison-mode preconvert will surface its own
    per-file errors via ``preconvert.py``.
    """
    if libreoffice_command is not None:
        return _command_works(libreoffice_command, probe_dirs)

    if not sys.platform.startswith("linux"):
        LOGGER.warning(
            "auto-install only supports Linux (sys.platform=%s); GDPVal preconvert will be a no-op.",
            sys.platform,
        )
        return False

    if shutil.which("libreoffice") and _java_runs() and _javaldx_works() and _probe_converts(["libreoffice"], None):
        LOGGER.info("libreoffice, a working JRE and javaldx are already present; skipping apt-get.")
        return True

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
    if not _probe_converts(["libreoffice"], None):
        return False

    LOGGER.info("libreoffice ready: %s", out.strip())
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Optional argv: check a configured command, e.g. `apptainer exec ... libreoffice`.
    ok = ensure_libreoffice(sys.argv[1:] or None)
    sys.exit(0 if ok else 1)
