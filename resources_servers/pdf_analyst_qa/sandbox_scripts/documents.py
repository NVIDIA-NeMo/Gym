# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Copy selected PDFs and restrict access to the source bucket inside a sandbox."""

import argparse
import hashlib
import json
import os
import pathlib
import shutil


def materialize_documents(
    manifest_path: pathlib.Path,
    *,
    docs: pathlib.Path = pathlib.Path("/workspace/docs"),
    task: pathlib.Path = pathlib.Path("/workspace/task"),
) -> None:
    """Copy and verify selected PDFs, then make them read-only for agent tools."""
    docs.mkdir(parents=True, exist_ok=True)
    task.mkdir(parents=True, exist_ok=True)
    if any(docs.iterdir()):
        raise RuntimeError("document directory must start empty")
    manifest = json.loads(manifest_path.read_text())
    manifest_path.unlink()
    for record in manifest:
        source = pathlib.Path(record["source"])
        target = docs / record["target"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        with target.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != record["sha256"]:
            raise RuntimeError("mounted PDF differs from the dataset PDF on the server")
        target.chmod(0o444)
    for directory in sorted(docs.rglob("*"), reverse=True):
        if directory.is_dir():
            directory.chmod(0o555)
    docs.chmod(0o555)
    os.chown(task, 65534, 65534)
    task.chmod(0o700)


def hide_document_store(source_mount: pathlib.Path) -> None:
    """Restrict the shared bucket's parent directory to root."""
    if not os.statvfs(source_mount).f_flag & os.ST_RDONLY:
        raise RuntimeError("shared bucket mount must be read-only")
    private_parent = source_mount
    while not private_parent.is_mount():
        private_parent = private_parent.parent
    while private_parent.is_mount() and private_parent != pathlib.Path("/"):
        private_parent = private_parent.parent
    if private_parent == pathlib.Path("/"):
        raise RuntimeError("cannot hide the document store through the filesystem root")
    private_parent.chmod(0o700)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=pathlib.Path)
    parser.add_argument("source_mount", type=pathlib.Path)
    parser.add_argument("workspace", nargs="?")
    args = parser.parse_args()
    if args.workspace:
        from document_cache import check_document_cache, initialize_cache

        print(json.dumps(initialize_cache(args.manifest, args.source_mount, args.workspace)), flush=True)
        check_document_cache(args.manifest)
    else:
        materialize_documents(args.manifest)
    hide_document_store(args.source_mount)
