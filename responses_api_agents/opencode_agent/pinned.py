# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage files that the sandbox owner pins by SHA-256 into OpenCode's private session storage.

The sandbox needs only a POSIX shell and coreutils: it checks the SHA-256 of a private copy, so the verified bytes are
the ones used, and runs an executable's ``--version``. An archive's members are checked here, on the agent server, in
a download of the verified copy, before the sandbox unpacks it with tar.
"""

import hashlib
import posixpath
import tarfile
from pathlib import Path


# Exit status of a step whose pinned bytes are unusable (a SHA-256 mismatch, a failed health check): retrying cannot
# help. A missing file or tool exits 1 instead, so the setup stays retryable.
REJECTED = 3

# Copies the pinned file "$1" to "$3" and keeps the copy only if its SHA-256 is "$2".
COPY_PINNED = f"""[ -f "$1" ] || {{ echo "Pinned file $1 is missing" >&2; exit 1; }}
command -v sha256sum >/dev/null 2>&1 || {{ echo 'Checking a pinned file requires sha256sum' >&2; exit 1; }}
mkdir -p -- "${{3%/*}}" && cp -- "$1" "$3" || exit 1
digest=$(sha256sum <"$3") || exit 1
digest=${{digest%% *}}
if [ "$digest" != "$2" ]; then
    rm -f -- "$3"
    echo "SHA-256 mismatch: expected $2, found $digest in $1" >&2
    exit {REJECTED}
fi
"""

# After COPY_PINNED, installs the copy "$3" as the executable "$4" once it answers --version. Only a verdict on the bytes
# is rejected: bytes the kernel refuses to run (126) or a failing probe. A missing loader (127) or a signal (above 128)
# may pass on a retry.
STAGE_EXECUTABLE = (
    COPY_PINNED
    + f"""chmod 755 "$3" || exit 1
output=$("$3" --version 2>&1 </dev/null)
status=$?
if [ "$status" -ne 0 ]; then
    rm -f -- "$3"
    printf '%s --version exited %s: %.1000s\\n' "$1" "$status" "$output" >&2
    if [ "$status" -eq 127 ] || [ "$status" -gt 128 ]; then exit 1; fi
    exit {REJECTED}
fi
mv -f -- "$3" "$4"
"""
)

# Unpacks the checked archive copy "$1" into the folder "$2" without the archive's owners, then removes the copy.
UNPACK_ARCHIVE = """command -v tar >/dev/null 2>&1 || { echo 'Unpacking a pinned archive requires tar' >&2; exit 1; }
mkdir -p -- "$2" && tar -x -o -f "$1" -C "$2" && rm -f -- "$1"
"""


def _archive_path(folder: str, path: str, links: frozenset[str]) -> str:
    """Where ``path``, read from the archive folder ``folder``, leads in the extraction directory.

    The walk is lexical, so it is exact only while no folder on the way is a link: a path that passes through one of
    ``links`` goes wherever that link points. Raises ValueError for such a path and for one that leaves.
    """
    if posixpath.isabs(path):
        raise ValueError("leaves the target directory")
    parts = [part for part in folder.split("/") if part]
    for component in path.split("/"):
        if component in ("", "."):
            continue
        if "/".join(parts) in links:
            raise ValueError(f"passes through the link {'/'.join(parts)!r}")
        if component != "..":
            parts.append(component)
        elif parts:
            parts.pop()
        else:
            raise ValueError("leaves the target directory")
    return "/".join(parts)


def check_archive(path: Path, digest: str) -> None:
    """Raise ValueError unless the tar archive at ``path`` unpacks only files, folders and links inside its target.

    Nothing is unpacked here, so no member path or link target may pass through a link member, wherever that link
    points, and no member may carry setuid, setgid or sticky bits, which tar keeps. ``path`` is a download of the copy
    the sandbox verified: a SHA-256 other than ``digest`` faults the download (RuntimeError), not the pinned file.
    """
    with path.open("rb") as file:
        actual = hashlib.file_digest(file, "sha256").hexdigest()
    if actual != digest:
        raise RuntimeError(f"Downloaded archive has SHA-256 {actual}, not the pinned {digest}")
    try:
        with tarfile.open(path) as archive:
            members = archive.getmembers()
    except tarfile.TarError as error:
        raise ValueError(f"not a usable tar archive: {error}") from error
    links = frozenset(posixpath.normpath(member.name) for member in members if member.issym() or member.islnk())
    for member in members:
        if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
            raise ValueError(f"unsupported archive member {member.name!r}")
        if member.mode & 0o7000:
            raise ValueError(f"archive member {member.name!r} has setuid, setgid or sticky bits")
        try:
            name = _archive_path("", member.name, links)
        except ValueError as error:
            raise ValueError(f"archive member {member.name!r} {error}") from None
        if member.issym() or member.islnk():
            # A symbolic link is relative to its own folder; a hard link names another member.
            folder = posixpath.dirname(name) if member.issym() else ""
            try:
                _archive_path(folder, member.linkname, links)
            except ValueError as error:
                raise ValueError(f"archive member {member.name!r} -> {member.linkname!r} {error}") from None
