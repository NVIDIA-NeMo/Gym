# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decide when a recorded image can be reused and record a finished build.

Two record files describe built images: compose-images.json holds each image's configuration
with the inputs digest of what was built, built-images.json maps a context digest to the
images of its services. This module computes the inputs digest over the settled plan entry,
compares a recorded image against the platform and repository of the current build, reports
which services need rebuilding, and merges a build result into both files under the record
lock. It knows nothing about the build tools.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from nemo_gym.tasks.harbor.build_compose import (
    compose_build_env,
    image_name,
    service_build_settings,
    validate_task_compose,
)
from nemo_gym.tasks.harbor.build_plan import BuildError, BuildPlanEntry, BuildResult
from nemo_gym.tasks.harbor.digest import inputs_digest
from nemo_gym.tasks.harbor.records import (
    BUILT_IMAGES_FILE,
    IMAGE_CONFIGS_FILE,
    MAIN_SERVICE,
    read_built_images,
    read_image_configs,
    record_lock,
    recorded_service_records,
    write_json_record,
)


def settle_entry(entry: BuildPlanEntry, *, repository: str | None, platform: str | None) -> BuildPlanEntry:
    """``entry`` with its service's ``build`` section of the task's Compose file applied, as
    :meth:`ImageBuilder.build_many` applies it before the build command: the file is checked and interpolated
    against the build's own environment (:func:`compose_build_env`, for the image name and platform a builder with
    ``repository`` and ``platform`` uses), and its settings fill the entry (:func:`service_build_settings`). Raises
    :class:`BuildError` where the build would."""
    env = compose_build_env(entry, image_name(entry.digest, repository), platform)
    document = validate_task_compose(
        entry.context, env, tasks=entry.tasks, builds_main=entry.compose_service == MAIN_SERVICE
    )
    return replace(entry, **service_build_settings(entry, document, env))


def build_inputs_digest(entry: BuildPlanEntry) -> str:
    """The inputs digest of a settled entry (:func:`settle_entry`): :func:`~nemo_gym.tasks.harbor.digest.inputs_digest`
    of its context (every entry of the tree with its mode, content or link target, nothing skipped) plus what the
    build reads outside the tree: the service's ``build`` section of the task's Compose file as a whole, as it
    stands after interpolation (Compose interpolates every string of the file against the build's environment, so
    an inline Dockerfile, a label or a cache setting changes the image while no file does), its two paths by their
    resolved forms relative to the context (the text may spell them through ``CONTEXT_DIR``, an absolute path that
    differs between hosts holding the same folder) and the build arguments as resolved (a ``null`` or bare argument
    is read from the environment, which the section does not show). The services of one context share the tree and
    differ in the section, so each image carries its own. Recorded in the built image's configuration record and
    compared before a recorded image is reused (:func:`build_inputs_mismatch`)."""
    section = {
        key: value for key, value in (entry.compose_build or {}).items() if key not in ("context", "dockerfile")
    }
    settings = {
        "build_context": os.path.relpath(entry.build_dir, entry.context),
        "dockerfile": os.path.relpath(entry.dockerfile_path, entry.context),
        "build": _canonical_json(section),
        **{f"arg:{key}": value for key, value in entry.build_args.items()},
    }
    return inputs_digest(entry.context, settings)


def _canonical_json(value: Any) -> str:
    """``value`` (a parsed YAML section) as one line of JSON that is the same for the same section: keys sorted and
    read as text (YAML allows other scalars as keys), and a scalar JSON has no form for, a date say, as its text."""

    def plain(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): plain(inner) for key, inner in item.items()}
        if isinstance(item, (list, tuple)):
            return [plain(inner) for inner in item]
        if item is None or isinstance(item, (bool, int, float, str)):
            return item
        return str(item)

    return json.dumps(plain(value), sort_keys=True, separators=(",", ":"))


def build_inputs_mismatch(
    entry: BuildPlanEntry,
    record: dict[str, Any],
    image_configs: dict[str, Any],
    *,
    repository: str | None,
    platform: str | None,
) -> str | None:
    """Why the image the ``built-images.json`` entry ``record`` holds for ``entry``'s service was built from other
    inputs than the context holds now, or ``None`` when the inputs match and the recorded image may be reused.

    The context's Harbor hash names the image and keys the record, but it covers file names and bytes only; the
    ``inputs_digest`` in the image's ``compose-images.json`` record (``image_configs``, written by the build with
    :func:`build_inputs_digest`) covers the rest, so a context whose name is unchanged is still rebuilt under that
    name when its inputs are not. An image whose record has no digest (written before it was recorded, or
    missing) counts as changed. Settling the entry can fail as the build would (:class:`BuildError`: a Compose file
    that does not interpolate or is refused); that is reported as the reason too, so the caller rebuilds, and the
    rebuild reports the error in full.
    """
    try:
        current = build_inputs_digest(settle_entry(entry, repository=repository, platform=platform))
    except BuildError as exc:
        return f"its build inputs cannot be checked: {exc}"
    image = recorded_service_records(record).get(entry.service, {}).get("image")
    recorded = (image_configs.get(image or "") or {}).get("inputs_digest")
    if recorded is None:
        return "its configuration records no inputs digest (it was written before build inputs were recorded)"
    if recorded != current:
        return (
            "its build inputs changed since it was built (a file's content or mode, a symbolic link, a folder, "
            "or a build setting) while the context's hash did not"
        )
    return None


def platform_key(platform: str | None) -> str:
    """``os/arch`` of a platform string, with a variant such as ``linux/arm64/v8`` dropped."""
    parts = (platform or "").split("/")
    return "/".join(parts[:2])


def recorded_image_mismatch(
    entry: dict[str, Any],
    image_configs: dict[str, Any],
    *,
    repository: str | None,
    platform: str,
    service: str = MAIN_SERVICE,
) -> str | None:
    """Why the image a ``built-images.json`` entry records for ``service`` cannot serve this builder, or ``None``
    when it can.

    A service the entry does not record is missing. A local (unpushed) image serves only local builds; a pushed
    one serves both, judged by that service's own ``pushed`` flag (:func:`recorded_service_records`: the entry's
    flag stands in for a record written before the flags were kept per service), since the services of one
    context may have been built by differently configured runs. The recorded platform has to be the one asked for.
    """
    service_record = recorded_service_records(entry).get(service)
    if service_record is None:
        return f"no image is recorded for service {service}"
    if not service_record.get("pushed") and repository is not None:
        return "it was built locally and this run pushes to a repository"
    image = service_record["image"]
    record = image_configs.get(image)
    if record is None:
        return "its configuration is missing from compose-images.json"
    recorded = f"{record.get('os')}/{record.get('architecture')}"
    if platform_key(recorded) != platform_key(platform):
        return f"it is {recorded}, not {platform}"
    return None


def record_built_image(folder: Path, result: BuildResult, *, overwrite: bool = False) -> Path:
    """Record one built image next to the task folders: under its context digest and service in
    ``built-images.json`` and by image in ``compose-images.json``. Existing entries are kept unless ``overwrite``.

    The digest's entry is merged, not replaced: ``services[<service>]`` is set to this build's ``image``, ``tool``
    and ``pushed`` (the other services keep theirs: one context's services may be built by differently configured
    runs, and a pushed ``main`` must not vouch for a local sidecar or the reverse), ``tasks`` unioned, and the
    top-level keys older readers use kept equal to ``main``'s: ``image`` (absent while ``main`` is not recorded)
    with ``tool`` and ``pushed`` (the first recorded service's while ``main`` is not recorded). A service recorded
    once keeps its image unless ``overwrite``, like ``compose-images.json`` keeps its entries.

    Both files are merged and replaced under their sidecar locks (:func:`record_lock`), the built-images
    lock first and the compose-images lock inside it, always in that order; :func:`record_image_configs`
    takes only the compose-images lock, so the two writers cannot deadlock and neither loses the other's
    entries. The configuration lands before the build entry, as before, so a reader that finds the entry
    finds its configuration.
    """
    folder = Path(folder)
    built_path = folder / BUILT_IMAGES_FILE
    configs_path = folder / IMAGE_CONFIGS_FILE
    with record_lock(built_path):
        built = read_built_images(folder)
        existing = built.get(result.digest) if isinstance(built.get(result.digest), dict) else {}
        services = recorded_service_records(existing)
        if result.service in services and not overwrite:
            # Pinned: a service recorded once keeps its image, like compose-images.json keeps its entries.
            return built_path
        services[result.service] = {"image": result.image, "tool": result.tool, "pushed": result.pushed}
        tasks = sorted(set(result.tasks) | {str(task) for task in existing.get("tasks") or []})
        lead = services.get(MAIN_SERVICE) or next(iter(services.values()))
        entry: dict[str, Any] = {
            "tasks": tasks,
            "tool": lead.get("tool", result.tool),
            "pushed": lead.get("pushed", result.pushed),
            "services": services,
        }
        if MAIN_SERVICE in services:
            entry["image"] = services[MAIN_SERVICE]["image"]
        built[result.digest] = entry
        with record_lock(configs_path):
            configs = read_image_configs(folder)
            if overwrite:
                configs[result.image] = result.record
            else:
                configs.setdefault(result.image, result.record)
            write_json_record(configs_path, configs)
        write_json_record(built_path, built)
    return built_path
