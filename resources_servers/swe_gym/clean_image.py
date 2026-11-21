# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline, fail-closed SWE agent rootfs construction. Never execute the input image."""

from __future__ import annotations

import argparse
import bz2
import gzip
import hashlib
import io
import json
import lzma
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
import zlib
from pathlib import Path, PurePosixPath


SCANNER_VERSION = 2
MAX_FILE_BYTES = 256 * 1024 * 1024
BAD_PARTS = {
    ".git",
    ".hg",
    ".svn",
    ".cache",
    ".npm",
    ".cargo",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".gradle",
    ".m2",
    "go-build",
    ".ruff_cache",
    "node_modules",
    ".ssh",
    ".aws",
    ".kube",
}
BAD_NAMES = {".netrc", ".git-credentials", ".pypirc", ".npmrc", "credentials.json"}
SECRET = re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|(?:gh[pousr]_[A-Za-z0-9]{30,})")


def sha256_file(path: Path) -> str:
    """Hash an artifact without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(name: str) -> str:
    """Canonical relative tar path; no ambiguous duplicate or traversal spelling."""
    if name.startswith("/") or ".." in PurePosixPath(name).parts:
        raise ValueError("unsafe archive path")
    return str(PurePosixPath(name))


def check_path(name: str) -> None:
    parts = PurePosixPath(name).parts
    if (
        set(parts) & BAD_PARTS
        or PurePosixPath(name).name in BAD_NAMES
        or PurePosixPath(name).name.startswith(".env")
        or name.startswith(("tmp/", "var/tmp/", "var/cache/", "go/pkg/mod/", "root/go/"))
    ):
        raise ValueError(f"forbidden artifact path: {name}")


def reference_needles(paths: list[Path], *, repo: Path, base_commit: str) -> list[bytes]:
    """Require distinctive added lines from each trusted reference/hidden-test input."""
    needles = set()
    for path in paths:
        lines = path.read_bytes().splitlines()
        patch = any(line.startswith(b"diff --git ") for line in lines)
        selected = set()
        for line in lines:
            if patch:
                if not line.startswith(b"+") or line.startswith(b"+++"):
                    continue
                line = line[1:]
            line = line.strip()
            if len(line) >= 40:
                selected.add(line)
        if not selected:
            raise ValueError("reference input has no distinctive lines of at least 40 bytes")
        needles.update(selected)
    if not needles:
        raise ValueError("reference and hidden-test material are required")
    # Existing base lines are public task context, including code moved by the reference.
    # Compare complete stripped lines, not substrings, so a novel longer answer still trips the scan.
    seen = set()
    for entry in git(repo, "ls-tree", "-rz", "--full-tree", base_commit).split(b"\0"):
        if not entry:
            continue
        metadata, _ = entry.split(b"\t", 1)
        _, kind, oid = metadata.split()
        if kind != b"blob" or oid in seen:
            continue
        seen.add(oid)
        if int(git(repo, "cat-file", "-s", oid.decode())) > MAX_FILE_BYTES:
            raise ValueError("source file exceeds scanner limit")
        data = git(repo, "cat-file", "blob", oid.decode())
        needles.difference_update(line.strip() for line in data.splitlines())
    return sorted(needles)


def scan_bytes(data: bytes, needles: list[bytes], *, depth: int = 0, _budget: list[int] | None = None) -> None:
    """Scan plain bytes and bounded common compression, including loose Git objects."""
    if _budget is None:
        _budget = [MAX_FILE_BYTES]
    _budget[0] -= len(data)
    if _budget[0] < 0 or depth > 3:
        raise ValueError("scanner limit exceeded")
    if data.startswith(b"PACK"):
        raise ValueError("Git pack data is forbidden in runtime inputs")
    if SECRET.search(data) or any(needle in data for needle in needles):
        raise ValueError("solution or credential fragment detected")
    if data.startswith((b"BZh", b"\xfd7zXZ\x00")):
        opener = bz2.open if data.startswith(b"BZh") else lzma.open
        with opener(io.BytesIO(data), "rb") as stream:
            scan_bytes(stream.read(_budget[0] + 1), needles, depth=depth + 1, _budget=_budget)
    elif data.startswith(b"\x1f\x8b"):
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
            scan_bytes(stream.read(_budget[0] + 1), needles, depth=depth + 1, _budget=_budget)
    elif data.startswith(b"PK\x03\x04"):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            total = 0
            for member in archive.infolist():
                total += member.file_size
                if total > MAX_FILE_BYTES or member.file_size > _budget[0]:
                    raise ValueError("scanner archive limit exceeded")
                scan_bytes(archive.read(member), needles, depth=depth + 1, _budget=_budget)
    elif data[:1] == b"x":
        # zlib headers used by loose Git objects; ordinary text beginning with x is allowed.
        if len(data) > 1 and (data[0] * 256 + data[1]) % 31 == 0:
            decoder = zlib.decompressobj()
            decoded = decoder.decompress(data, _budget[0] + 1)
            if not decoder.eof:
                raise ValueError("scanner zlib limit exceeded")
            scan_bytes(decoded, needles, depth=depth + 1, _budget=_budget)


def git(repo: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    """Run trusted Git with no inherited config, hooks, alternates or object directories."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env |= {
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_AUTHOR_NAME": "Task base",
        "GIT_AUTHOR_EMAIL": "base@nemo-gym.local",
        "GIT_COMMITTER_NAME": "Task base",
        "GIT_COMMITTER_EMAIL": "base@nemo-gym.local",
        "GIT_AUTHOR_DATE": "2000-01-01T00:00:00Z",
        "GIT_COMMITTER_DATE": "2000-01-01T00:00:00Z",
    }
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "core.hooksPath=/dev/null", *args],
        env=env,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        timeout=120,
    ).stdout


def reconstruct(repo: Path, base_commit: str, target: Path, needles: list[bytes]) -> tuple[str, str]:
    """Export exactly one commit and create a fresh one-commit repository; never copy .git."""
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", base_commit):
        raise ValueError("base_commit must be a full commit object ID")
    resolved = git(repo, "rev-parse", "--verify", base_commit + "^{commit}").decode().strip()
    target.mkdir()
    # Read tree/blob objects directly. git archive's export-subst/info attributes can disclose
    # later ref names; index, worktree, attributes, filters and LFS smudges must have no effect.
    for entry in git(repo, "ls-tree", "-rz", "--full-tree", resolved).split(b"\0"):
        if not entry:
            continue
        metadata, raw_name = entry.split(b"\t", 1)
        mode, kind, oid = metadata.split()
        name = safe_path(raw_name.decode("utf-8"))
        check_path(name)
        scan_bytes(raw_name, needles)
        if kind != b"blob" or mode not in (b"100644", b"100755"):
            raise ValueError("source export supports only regular files (no submodules or symlinks)")
        size = int(git(repo, "cat-file", "-s", oid.decode()))
        if size > MAX_FILE_BYTES:
            raise ValueError("source file exceeds scanner limit")
        data = git(repo, "cat-file", "blob", oid.decode())
        scan_bytes(data, needles)
        dest = target / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        dest.chmod(0o755 if mode == b"100755" else 0o644)
    git(
        target,
        "init",
        "--quiet",
        "--initial-branch=main",
        "--template=",
        "--object-format=" + ("sha256" if len(resolved) == 64 else "sha1"),
    )
    # info/attributes overrides repository attributes for both staging and later agent diffs.
    # Keep it in the reconstructed repo so CRLF and encoded files remain byte-exact.
    (target / ".git/info").mkdir(exist_ok=True)
    (target / ".git/info/attributes").write_text("* -text -filter -ident -working-tree-encoding\n")
    git(target, "config", "core.autocrlf", "false")
    git(target, "config", "core.fileMode", "true")
    git(target, "add", "--all", "--force")
    if git(target, "write-tree").strip() != git(repo, "rev-parse", resolved + "^{tree}").strip():
        raise ValueError("reconstructed tree differs from the task base")
    git(target, "-c", "commit.gpgsign=false", "commit", "--quiet", "--allow-empty", "-m", "base")
    # Remove local timing/log/index artifacts; HEAD/main plus the new reachable objects suffice.
    shutil.rmtree(target / ".git/logs", ignore_errors=True)
    (target / ".git/index").unlink()
    git(target, "read-tree", "HEAD")
    return resolved, git(target, "rev-parse", "HEAD").decode().strip()


def build_rootfs(
    *,
    runtime: Path,
    runtime_sha256: str,
    repo: Path,
    base_commit: str,
    references: list[Path],
    task_id: str,
    output: Path,
) -> dict:
    """Build a flattened import tar from a separately approved, task-free runtime export.

    The caller owns the runtime digest approval and reference completeness. No runtime is executed.
    """
    if output.exists() or output.is_symlink():
        raise ValueError("output already exists")
    if sha256_file(runtime) != runtime_sha256:
        raise ValueError("runtime digest mismatch")
    needles = reference_needles(references, repo=repo, base_commit=base_commit)
    with tempfile.TemporaryDirectory() as scratch:
        clean = Path(scratch) / "repo"
        source, agent_base = reconstruct(repo, base_commit, clean, needles)
        seen = set()
        symlinks = set()
        created = False
        try:
            output_stream = output.open("xb")
            created = True
            with (
                output_stream,
                tarfile.open(runtime, "r:") as src,
                tarfile.open(fileobj=output_stream, mode="w", format=tarfile.PAX_FORMAT) as dst,
            ):
                for member in src:
                    name = safe_path(member.name)
                    if name == ".":
                        continue
                    if name in seen or name == "testbed" or name.startswith("testbed/"):
                        raise ValueError("duplicate runtime path or runtime contains task repository")
                    if any(str(parent) in symlinks for parent in PurePosixPath(name).parents):
                        raise ValueError("runtime member traverses a symlink")
                    if member.issym():
                        if any(existing.startswith(name + "/") for existing in seen):
                            raise ValueError("runtime symlink shadows existing members")
                        symlinks.add(name)
                    seen.add(name)
                    check_path(name)
                    scan_bytes(name.encode(), needles)
                    if not (member.isfile() or member.isdir() or member.issym()):
                        raise ValueError("runtime must be flattened (no devices, hardlinks or special files)")
                    if member.issym():
                        # No input link is traversed on the host. Absolute runtime library links are normal.
                        scan_bytes(member.linkname.encode(), needles)
                    data = b""
                    if member.isfile():
                        if member.size > MAX_FILE_BYTES:
                            raise ValueError("runtime file exceeds scanner limit")
                        data = src.extractfile(member).read()
                        scan_bytes(data, needles)
                    member.name = name
                    member.uid = member.gid = 0
                    member.uname = member.gname = ""
                    member.mtime = 0
                    member.pax_headers = {}
                    dst.addfile(member, io.BytesIO(data) if member.isfile() else None)
                for path in [clean, *sorted(clean.rglob("*"))]:
                    name = "testbed" + ("/" + path.relative_to(clean).as_posix() if path != clean else "")
                    member = dst.gettarinfo(str(path), arcname=name)
                    member.uid = member.gid = 0
                    member.uname = member.gname = ""
                    member.mtime = 0
                    member.pax_headers = {}
                    data = path.read_bytes() if path.is_file() else b""
                    scan_bytes(data, needles)
                    dst.addfile(member, io.BytesIO(data) if member.isfile() else None)
        except BaseException:
            if created:
                output.unlink(missing_ok=True)
            raise
    return {
        "scanner_version": SCANNER_VERSION,
        "task_id": task_id,
        "base_commit": source,
        "agent_base_commit": agent_base,
        "runtime_sha256": runtime_sha256,
        "rootfs_sha256": sha256_file(output),
        "reference_needles": len(needles),
        "reference_sha256": [sha256_file(path) for path in references],
        "checks": ["base-export", "fresh-git", "artifact-paths", "reference-fragments", "mtime-zero", "flat-rootfs"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "repo", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("runtime-sha256", "base-commit", "task-id"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument(
        "--import-docker",
        action="store_true",
        help="import the tar and bind its immutable image ID; this manifest is valid only on this Docker daemon",
    )
    parser.add_argument(
        "--reference",
        required=True,
        action="append",
        type=Path,
        help="trusted reference patch AND hidden-test files; repeat for each input",
    )
    args = parser.parse_args()
    manifest_path = args.output.with_suffix(".json")
    if manifest_path.exists() or manifest_path.is_symlink():
        parser.error("manifest already exists")
    manifest = build_rootfs(
        runtime=args.runtime,
        runtime_sha256=args.runtime_sha256,
        repo=args.repo,
        base_commit=args.base_commit,
        references=args.reference,
        task_id=args.task_id,
        output=args.output,
    )
    if args.import_docker:
        result = subprocess.run(
            ["docker", "import", str(args.output)], check=True, capture_output=True, text=True, timeout=300
        )
        image_id = result.stdout.strip()
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
            raise ValueError("docker import did not return an immutable image ID")
        manifest["image_id"] = image_id
    with manifest_path.open("x") as stream:
        stream.write(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
