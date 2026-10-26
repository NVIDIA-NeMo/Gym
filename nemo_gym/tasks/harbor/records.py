# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The two record files prepare writes next to the task folders, and how they are read.

compose-images.json records image configurations, built-images.json records the images built
per context digest and service. This module names the files and the service keys, takes the
sidecar lock, replaces a file atomically, reads it with one retry against a torn write, and
reads the per-service entries with the legacy main-only fallback. The loader, the recorders,
the build modules and the environment server all read through it.
"""

import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from nemo_gym.tasks.harbor.errors import HarborTaskError


# The OCI configurations recorded at prepare next to the task folders, keyed by image reference as the
# tasks wrote it (see ``nemo_gym.tasks.harbor.image_configs``): Compose sidecar images and the base image
# of every Dockerfile the loader resolves.
IMAGE_CONFIGS_FILE = "compose-images.json"
# The images prepare built, next to the task folders, keyed by Harbor's content hash of the task's ``environment/``
# build context (see ``nemo_gym.tasks.harbor.image_build``). An entry records one image per built Compose service
# under ``services`` (``main`` for the Dockerfile outside pull and overlay mode, each sidecar with a ``build:``
# section under its own name) and keeps ``main``'s image at the top level as ``image`` for older readers; an entry
# with ``image`` and no ``services`` is a record of ``main`` alone.
BUILT_IMAGES_FILE = "built-images.json"
# The Compose service name of the task's own image, in Harbor's templates and the task's Compose file.
MAIN_SERVICE = "main"
# The service name the separate verifier's image is recorded under in ``built-images.json``, in the entry keyed by
# Harbor's content hash of ``tests/`` (its own build context, so it never shares an entry with ``environment/``'s
# services and a Compose sidecar of that name, kumo's ``verifier``, keeps its own record).
VERIFIER_SERVICE = "verifier"

# The sidecar lock file next to a record file (``<file>.lock``), see :func:`record_lock`.
RECORD_LOCK_SUFFIX = ".lock"
# A reader that finds a record file unparsable reads it once more after this pause before giving up.
RECORD_READ_RETRY_S = 0.05


@contextmanager
def record_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive lock on ``path``'s sidecar lock file (``<file>.lock``) for a read-modify-write of it.

    Two Gym processes can share one datasets folder (``gym eval run`` and ``gym dataset validate`` of one
    ``harbor:`` ref both record into it), so every read, merge and replace of a record file happens under
    this lock or a concurrent writer's entries are lost. ``flock`` is per open file description, so threads
    of one process are serialised as well. The lock file stays next to the record; it carries no content.
    """
    path = Path(path)
    lock_path = path.with_name(path.name + RECORD_LOCK_SUFFIX)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def write_json_record(path: Path, data: dict[str, Any]) -> None:
    """Replace the record file at ``path`` with ``data`` atomically: written sorted and indented to a temporary
    file in the same directory, fsynced, then moved over ``path`` with ``os.replace``, so a reader opens either
    the old file or the new one and never a truncated one. Callers merging into an existing record hold
    :func:`record_lock` around the read, the merge and this call."""
    path = Path(path)
    text = json.dumps(dict(sorted(data.items())), indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    # os.open with 0o666 applies the umask like Path.write_text did; mkstemp would make the record 0600.
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o666)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def read_json_record(path: Path) -> dict[str, Any]:
    """The JSON object in the record file at ``path``, or ``{}`` when the file is missing or not an object.

    Writers replace the file atomically (:func:`write_json_record`), so a torn read should not happen; as
    belt and braces a file that does not parse is read once more after a short pause (a record an older
    writer is still truncating and rewriting) and :class:`HarborTaskError` names it if it stays invalid.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    for attempt in (1, 2):
        try:
            recorded = json.loads(path.read_text())
            break
        except FileNotFoundError:
            return {}
        except (ValueError, UnicodeDecodeError) as exc:
            if attempt == 1:
                time.sleep(RECORD_READ_RETRY_S)
                continue
            raise HarborTaskError(
                f"{path} is not valid JSON (read twice, {RECORD_READ_RETRY_S:g}s apart): {exc}"
            ) from exc
    return recorded if isinstance(recorded, dict) else {}


def read_image_configs(folder: Path) -> dict[str, Any]:
    """The image configurations recorded in ``folder`` (the tasks' parent), or ``{}`` when none were."""
    return read_json_record(Path(folder) / IMAGE_CONFIGS_FILE)


def read_built_images(folder: Path) -> dict[str, Any]:
    """The built images recorded in ``folder`` (the tasks' parent), or ``{}`` when none were."""
    return read_json_record(Path(folder) / BUILT_IMAGES_FILE)


def recorded_service_records(entry: Any) -> dict[str, dict[str, Any]]:
    """The per-service records of one ``built-images.json`` entry, by Compose service name: ``image`` plus ``tool``
    and ``pushed`` as recorded for that service.

    A ``services`` map is read as written; a service record without the two flags (written before they were kept
    per service) takes the entry's own, as does the single ``main`` record of an entry with ``image`` and no
    ``services``, written before sidecars were built. Anything else records nothing.
    """
    if not isinstance(entry, dict):
        return {}
    flags = {key: entry[key] for key in ("tool", "pushed") if key in entry}
    services = entry.get("services")
    if isinstance(services, dict):
        records: dict[str, dict[str, Any]] = {}
        for name, record in services.items():
            fields = dict(record) if isinstance(record, dict) else {"image": record}
            if fields.get("image"):
                records[str(name)] = {**flags, **fields, "image": str(fields["image"])}
        return records
    image = entry.get("image")
    return {MAIN_SERVICE: {**flags, "image": str(image)}} if image else {}


def recorded_service_images(entry: Any) -> dict[str, str]:
    """The images one ``built-images.json`` entry records, by Compose service name (:func:`recorded_service_records`
    without the flags)."""
    return {name: record["image"] for name, record in recorded_service_records(entry).items()}
