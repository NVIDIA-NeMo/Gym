# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan the volumes of a Compose document for a sandbox that has no Docker volumes.

Parses each service's volume entries in their short and long forms, checks the target path and
the bind source against the task folder, refuses options the sandbox cannot honour, resolves
links before anything is read, and returns the plan the adapter uploads and mounts: bind mounts
as uploads, named volumes as shared storage, the logs tree kept apart. Every symbol is M22 and
the module reads only the standard library.
"""

from pathlib import Path, PurePosixPath
from typing import Any


class ComposeVolumeError(ValueError):
    """A Compose ``volumes`` entry prepare cannot run; ``name`` is the volume, source or field it names."""

    def __init__(self, name: str, message: str) -> None:
        self.name = name
        super().__init__(message)


# Short-syntax mount options: the first group is honoured, the second has no meaning without a Docker host.
_VOLUME_OPTIONS = {"ro": {"read_only": True}, "rw": {"read_only": False}, "nocopy": {"nocopy": True}}
_VOLUME_NOOP_OPTIONS = frozenset({"z", "Z", "cached", "delegated", "consistent"})
# In-sandbox tree Harbor's ``${HOST_*_PATH}`` log mounts resolve to (see ``image_configs.LOG_MOUNT_PATHS``).
LOGS_TREE = "/logs"


def parse_volume(entry: Any, service: str) -> dict[str, Any]:
    """One ``volumes`` entry as a long-syntax mapping: ``type``, ``source``, ``target``, ``read_only``
    and, for a named volume, ``volume.nocopy``.

    Short syntax ``SOURCE:TARGET[:OPTIONS]`` is parsed the way Compose does: a source starting with
    ``.``, ``/`` or ``~`` is a bind mount, anything else a named volume. A bare ``TARGET`` (an anonymous
    volume), a ``tmpfs``/``npipe``/``cluster`` type, a bind ``propagation`` or an unknown option raises
    :class:`ComposeVolumeError` naming the field: nothing in a sandbox group stands in for it.
    """
    where = f"service {service!r} volumes"
    if isinstance(entry, str):
        parts = entry.split(":")
        if len(parts) == 1 or not parts[0]:
            raise ComposeVolumeError(entry, f"{where}: anonymous volume {entry!r} is not supported; name a source")
        if len(parts) > 3:
            raise ComposeVolumeError(entry, f"{where}: cannot parse {entry!r}")
        source, target = parts[0], parts[1]
        volume: dict[str, Any] = {"type": "bind" if source[0] in "./~" else "volume", "source": source}
        volume["target"] = _checked_target(target, where)
        volume["read_only"] = False
        for option in parts[2].split(",") if len(parts) == 3 else []:
            if option in _VOLUME_NOOP_OPTIONS or not option:
                continue
            if option not in _VOLUME_OPTIONS:
                raise ComposeVolumeError(option, f"{where}: unsupported mount option {option!r} in {entry!r}")
            if option == "nocopy":
                volume.setdefault("volume", {})["nocopy"] = True
            else:
                volume.update(_VOLUME_OPTIONS[option])
        return volume
    if not isinstance(entry, dict):
        raise ComposeVolumeError("volumes", f"{where}: entries must be strings or mappings, got {entry!r}")
    unknown = set(entry) - {"type", "source", "target", "read_only", "bind", "volume", "consistency"}
    if unknown:
        raise ComposeVolumeError(sorted(unknown)[0], f"{where}: unsupported field(s) {sorted(unknown)}")
    kind = entry.get("type", "volume")
    if kind not in ("bind", "volume"):
        raise ComposeVolumeError("type", f"{where}: volume type {kind!r} is not supported")
    source, target = entry.get("source"), entry.get("target")
    if kind == "bind" and not source:
        raise ComposeVolumeError("source", f"{where}: a bind mount needs a source")
    if not source:
        raise ComposeVolumeError("source", f"{where}: anonymous volume for {target!r} is not supported; name a source")
    target = _checked_target(target, where)
    # ``bind: 123`` or ``volume: [nocopy]`` would be a ``TypeError`` out of ``set(...)`` or an ``AttributeError``
    # out of ``.get`` below, past prepare's per-task boundary: the planner's own error, naming the field, instead.
    for field in ("bind", "volume"):
        value = entry.get(field)
        if value is not None and not isinstance(value, dict):
            raise ComposeVolumeError(field, f"{where}: {field} options must be a mapping, got {value!r}")
    bind = entry.get("bind") or {}
    if set(bind) - {"create_host_path"}:
        raise ComposeVolumeError(
            sorted(set(bind) - {"create_host_path"})[0], f"{where}: unsupported bind options {bind}"
        )
    options = entry.get("volume") or {}
    if set(options) - {"nocopy"}:
        raise ComposeVolumeError(
            sorted(set(options) - {"nocopy"})[0], f"{where}: unsupported volume options {options}"
        )
    volume = {"type": kind, "source": str(source), "target": target, "read_only": bool(entry.get("read_only", False))}
    if options.get("nocopy"):
        volume["volume"] = {"nocopy": True}
    return volume


def _checked_target(target: Any, where: str) -> str:
    """A mount target the group can serve: an absolute container path below ``/`` with no ``..`` segment.

    The same rule for both syntaxes, and the one the Compose adapter applies to uploads, so a target it
    would refuse fails at prepare naming the field (``target``).
    """
    if not isinstance(target, str) or not target.startswith("/") or ".." in target.split("/"):
        raise ComposeVolumeError("target", f"{where}: target must be an absolute container path, got {target!r}")
    # ``/.`` and ``/././`` name the root as ``/`` and ``//`` do (M22 review): no last component below it.
    if not PurePosixPath(target).name:
        raise ComposeVolumeError("target", f"{where}: target {target!r} is the container root; mount below it")
    return target


def lexical_target(target: str) -> str:
    """A mount target in its lexical form: trailing and repeated separators and ``.`` segments dropped, so
    ``/app/output/``, ``/app/output/.`` and ``//app/./output`` are all ``/app/output``.

    A target is recorded as written (``/data/.`` stays ``/data/.`` in the planned document) and compared in
    this form, by the planner's rule for a ``main`` logs bind and by the Compose adapter's, which also names the
    link in it: a target that nests its source only once spelled out (``/logs/verifier:/logs/.``) is refused at
    prepare like ``/logs/verifier:/logs``, and never linked over the tree it names (M22 review).
    """
    path = str(PurePosixPath(target))
    # POSIX leaves a path of exactly two leading separators to the implementation and ``PurePosixPath`` keeps
    # them; the container's filesystem does not tell ``//logs`` from ``/logs``.
    return path[1:] if path.startswith("//") else path


def _bind_source(source: str, base_dir: Path, where: str) -> Path:
    """A bind source as a host path: relative to ``base_dir`` (the Compose file's folder), ``~`` expanded.

    ``~user`` for a user the host does not know cannot be resolved; it raises :class:`ComposeVolumeError`
    naming the source, like any other bind source the task does not contain.
    """
    try:
        path = Path(source).expanduser()
    except RuntimeError as exc:
        raise ComposeVolumeError(
            source,
            f"{where}: cannot resolve the home directory in bind source {source!r}; map it under `volume_sources`",
        ) from exc
    return path if path.is_absolute() else base_dir / path


def first_link(folder: Path) -> Path | None:
    """The first symbolic link anywhere under ``folder`` (files or folders), or ``None``.

    A folder a Compose file binds is uploaded entry by entry (a copy, not a mount); a link inside it would
    be read through, reaching whatever it points at on the host, so the planner and the Compose adapter both
    refuse one.
    """
    return next((path for path in sorted(folder.rglob("*")) if path.is_symlink()), None)


def absolute_link(path: Path, root: Path) -> Path | None:
    """The first component of ``path`` below ``root`` that is a link with an absolute target, or ``None``.

    The server plans a task on a copy of its folder (links copied as links), where an absolute target still
    points into the original folder: a bind source reached through one resolves inside the task at prepare and
    outside the copy at seed. The planner refuses it, as the hub refuses such a link in a fetched task
    (``_validate_task_tree``), so the two agree. A relative link that stays inside the task is fine at both.
    """
    if not path.is_relative_to(root):
        return None
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        if current.is_symlink() and current.readlink().is_absolute():
            return current
    return None
