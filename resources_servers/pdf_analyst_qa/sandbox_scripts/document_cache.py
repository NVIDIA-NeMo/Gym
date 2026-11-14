# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cache selected PDFs on a worker and check their read-only sandbox mount."""

import argparse
import concurrent.futures
import contextlib
import errno
import fcntl
import hashlib
import json
import os
import pathlib
import random
import shutil
import time
import uuid


def check_document_cache(
    manifest_path: pathlib.Path,
    *,
    docs: pathlib.Path = pathlib.Path("/workspace/docs"),
    task: pathlib.Path = pathlib.Path("/workspace/task"),
) -> None:
    """Check that the mount contains exactly the selected read-only PDFs."""
    manifest = json.loads(manifest_path.read_text())
    manifest_path.unlink()
    if not os.statvfs(docs).f_flag & os.ST_RDONLY:
        raise RuntimeError("cached PDF view must be a read-only mount")
    expected = {item["target"]: item["bytes"] for item in manifest}
    actual = {str(path.relative_to(docs)): path.stat().st_size for path in docs.rglob("*") if path.is_file()}
    if actual != expected or any(path.is_symlink() for path in docs.rglob("*")):
        raise RuntimeError("cached PDF view does not match the permitted documents")
    task.mkdir(parents=True, exist_ok=True)
    os.chown(task, 65534, 65534)
    task.chmod(0o700)


def initialize_cache(
    manifest_path: pathlib.Path,
    source_mount: pathlib.Path,
    workspace: str,
    *,
    cache: pathlib.Path = pathlib.Path("/var/lib/pdf-analyst-qa-cache"),
    docs: pathlib.Path = pathlib.Path("/workspace/docs"),
    private: pathlib.Path = pathlib.Path("/tmp/pdf-analyst-qa-private"),
    boot_id_path: pathlib.Path = pathlib.Path("/proc/sys/kernel/random/boot_id"),
) -> dict[str, str | int | float | bool]:
    """Copy missing PDFs under file locks and create a reusable workspace view.

    The cache and read-only view are separate mounts of the same filesystem.
    Copies are checked against the dataset source directory and file sizes.
    """
    started = time.monotonic()
    payload = manifest_path.read_bytes()
    records = json.loads(payload)
    if pathlib.Path(workspace).name != workspace or not workspace.startswith("workspace_"):
        raise RuntimeError("invalid cache workspace identity")
    cache.chmod(0o700)
    store = cache / "store"
    locks = cache / "locks"
    receipts = cache / "view-receipts"
    for directory in (store, locks, receipts):
        directory.mkdir(exist_ok=True)
    with (locks / ".cache-id.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        identity = cache / ".cache-id"
        if not identity.exists():
            identity.write_text(str(uuid.uuid4()))
        cache_id = identity.read_text()
        dataset_directory = str(pathlib.Path(records[0]["source"]).parent)
        if any(str(pathlib.Path(record["source"]).parent) != dataset_directory for record in records):
            raise RuntimeError("cache initialization mixes dataset directories")
        origin = cache / ".publication"
        if origin.exists() and origin.read_text() != dataset_directory:
            raise RuntimeError("node cache belongs to a different dataset directory")
        if not origin.exists():
            origin.write_text(dataset_directory)

    @contextlib.contextmanager
    def download_slot():
        # Bound cache-fill transfers across every sandbox using this worker cache.
        while True:
            for index in range(4):
                handle = (locks / ("download-slot-" + str(index))).open("a")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    continue
                try:
                    yield
                finally:
                    fcntl.flock(handle, fcntl.LOCK_UN)
                    handle.close()
                return
            time.sleep(0.02 + random.uniform(0, 0.02))

    def prepare_pdf(record):
        source = pathlib.Path(record["source"])
        name = source.name
        if source.suffix.lower() != ".pdf" or not source.is_relative_to(source_mount):
            raise RuntimeError("PDF source is outside the mounted dataset")
        final = store / name
        retries = 0
        waited = time.monotonic()
        with (locks / (name + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            waited = time.monotonic() - waited
            copied = False
            if final.is_symlink():
                raise RuntimeError("cached PDF must not be a symlink")
            if not final.exists():
                temporary = store / ("." + name + ".partial")
                with download_slot():
                    for attempt in range(4):
                        try:
                            shutil.copyfile(source, temporary)
                            if temporary.stat().st_size != record["bytes"]:
                                raise OSError(errno.EIO, "copied PDF size differs from the dataset PDF")
                            break
                        except OSError as error:
                            if (
                                error.errno
                                not in {errno.EIO, errno.ESTALE, errno.EAGAIN, errno.ETIMEDOUT, errno.ECONNRESET}
                                or attempt == 3
                            ):
                                raise
                            retries += 1
                            time.sleep(0.5 * 2**attempt + random.uniform(0, 0.25))
                temporary.chmod(0o444)
                os.link(temporary, final)
                temporary.unlink()
                copied = True
            if final.stat().st_size != record["bytes"] or final.stat().st_mode & 0o222:
                raise RuntimeError("cached PDF differs from the published file size or permissions")
        return {
            "copied": copied,
            "bytes": record["bytes"] if copied else 0,
            "lock_wait_s": waited,
            "copy_retries": retries,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        transfers = list(pool.map(prepare_pdf, records))
    pdfs_done = time.monotonic()
    view = cache / "views" / workspace
    view.mkdir(parents=True, exist_ok=True)
    view_hash = hashlib.sha256(payload).hexdigest()
    receipt = receipts / (workspace + ".json")
    with (locks / (workspace + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        view_hit = receipt.exists()
        if view_hit:
            if receipt.read_text() != view_hash:
                raise RuntimeError("cached view belongs to a different permitted document set")
        else:
            for record in records:
                relative = pathlib.PurePosixPath(record["target"])
                if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() != ".pdf":
                    raise RuntimeError("invalid permitted PDF path")
                target = view / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                source = store / pathlib.Path(record["source"]).name
                if target.exists():
                    if target.is_symlink() or not target.samefile(source):
                        raise RuntimeError("cached view refers to the wrong PDF")
                else:
                    os.link(source, target)
            for directory in sorted(view.rglob("*"), reverse=True):
                if directory.is_dir():
                    directory.chmod(0o555)
            view.chmod(0o555)
            temporary = receipts / (workspace + ".partial")
            temporary.write_text(view_hash)
            os.replace(temporary, receipt)
    if not view.samefile(docs):
        raise RuntimeError("document mount is not this worker cache view")
    metrics = {
        "cache_id": cache_id,
        "worker_boot_id": boot_id_path.read_text().strip(),
        "files": len(records),
        "cache_hits": sum(not item["copied"] for item in transfers),
        "copied_files": sum(item["copied"] for item in transfers),
        "copied_bytes": sum(item["bytes"] for item in transfers),
        "copy_retries": sum(item["copy_retries"] for item in transfers),
        "lock_wait_s": sum(item["lock_wait_s"] for item in transfers),
        "pdf_prepare_s": pdfs_done - started,
        "view_prepare_s": time.monotonic() - pdfs_done,
        "view_cache_hit": view_hit,
    }
    private.mkdir(exist_ok=True)
    private.chmod(0o700)
    (private / "cache-metrics.json").write_text(json.dumps(metrics))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_path", type=pathlib.Path)
    parser.add_argument("source_mount", type=pathlib.Path)
    parser.add_argument("workspace")
    parser.add_argument("--cache", type=pathlib.Path, default=pathlib.Path("/var/lib/pdf-analyst-qa-cache"))
    parser.add_argument("--docs", type=pathlib.Path, default=pathlib.Path("/workspace/docs"))
    parser.add_argument("--private", type=pathlib.Path, default=pathlib.Path("/tmp/pdf-analyst-qa-private"))
    parser.add_argument("--boot-id-path", type=pathlib.Path, default=pathlib.Path("/proc/sys/kernel/random/boot_id"))
    print(json.dumps(initialize_cache(**vars(parser.parse_args()))), flush=True)
