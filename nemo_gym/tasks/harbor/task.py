# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load one task folder, or a folder of task folders, from disk."""

import fcntl
import json
import logging
import os
import re
import time
import tomllib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from nemo_gym.tasks.harbor.dataset_config import apply_dataset_config, read_dataset_config
from nemo_gym.tasks.harbor.digest import content_hash, context_digest
from nemo_gym.tasks.harbor.dockerfile import (
    BaseImageConfig,
    DockerfileNeedsBuild,
    ImageConfigRequired,
    OverlayRun,
    parse_dockerfile,
    resolve_dockerfile,
)
from nemo_gym.tasks.harbor.models import HarborTaskConfig


logger = logging.getLogger(__name__)

TASK_FILE = "task.toml"
INSTRUCTION_FILE = "instruction.md"
DOCKERFILE = Path("environment") / "Dockerfile"
# The OCI configurations recorded at prepare next to the task folders, keyed by image reference as the
# tasks wrote it (see ``nemo_gym.tasks.harbor.image_configs``): Compose sidecar images and the base image
# of every Dockerfile the loader resolves.
IMAGE_CONFIGS_FILE = "compose-images.json"
# The images prepare built for Dockerfiles outside pull and overlay mode, next to the task folders, keyed by
# Harbor's content hash of the task's ``environment/`` build context (see ``nemo_gym.tasks.harbor.image_build``).
BUILT_IMAGES_FILE = "built-images.json"


class HarborTaskError(ValueError):
    """The folder is not a runnable Harbor task."""


@dataclass(frozen=True)
class HarborTask:
    """A task folder read into memory.

    ``task_id`` is the folder name: ``[task].name`` is an ``org/name`` label that is
    not unique across a dataset. ``image`` is the prebuilt image when the task
    declares one, else the base image of a pull-mode or overlay-mode Dockerfile, else
    ``None`` (a build is required). ``workdir``, ``env`` and ``user`` merge ``task.toml``
    with what the Dockerfile recorded; ``task.toml`` wins.

    Dockerfile ``ENV``, ``WORKDIR`` and ``USER`` are resolved to literals as ``docker build``
    resolves them, starting from the base image's configuration recorded in
    :data:`IMAGE_CONFIGS_FILE` next to the task folders (see
    :mod:`nemo_gym.tasks.harbor.dockerfile`).

    ``overlay`` is Gym-derived data, not a Harbor field: the Dockerfile's ``RUN`` lines
    (with the ``WORKDIR``, ``ENV`` and ``USER`` in effect for each) that the server runs
    in the pulled base image at seed. Empty for a prebuilt image or a pull-mode Dockerfile.

    ``sandbox_user`` is the user the sandbox runs commands as when no user is asked for: the
    image the sandbox starts from is the Dockerfile's base image, so it is that image's recorded
    ``User`` (``None`` is root), not the Dockerfile's final ``USER`` or ``[agent].user``. When the
    base image's configuration is not known (a prebuilt image, or a Dockerfile resolved without
    it), the task's ``user`` stands in for it.

    ``built_image`` is the image prepare built from the task's ``environment/`` folder when the
    Dockerfile needed a build (``built-images.json``); ``image`` is then that image and the task
    runs in pull mode from it. ``None`` in the other two modes and for a prebuilt image.

    ``entrypoint`` is the image's ``Entrypoint`` as Docker would start it: the built image's recorded
    one, or the base image's for a Dockerfile the loader resolves unless that Dockerfile wrote
    ``ENTRYPOINT []``. ``None`` when the image records none or no configuration is recorded (a prebuilt
    ``docker_image``). ``entrypoint_user`` and ``entrypoint_workdir`` are the ``User`` and ``WorkingDir``
    the image starts with (``None`` is root and ``/``): what Docker gives its ``ENTRYPOINT`` and its
    ``HEALTHCHECK``, independent of ``[agent].user`` and ``[environment].workdir``, which only shape the
    commands run beside them. They are known whenever the image's configuration is, with or without an
    entrypoint. ``image_healthcheck`` is the recorded Dockerfile ``HEALTHCHECK`` block (``Test``,
    ``Interval``, ... as Docker records them), or ``None``.
    """

    path: Path
    task_id: str
    config: HarborTaskConfig
    instruction: str
    digest: str
    image: str | None
    workdir: str | None
    env: dict[str, str]
    user: str | None
    overlay: tuple[OverlayRun, ...] = ()
    sandbox_user: str | None = None
    built_image: str | None = None
    entrypoint: tuple[str, ...] | None = None
    entrypoint_user: str | None = None
    entrypoint_workdir: str | None = None
    image_healthcheck: dict[str, Any] | None = None

    @property
    def needs_sandbox(self) -> bool:
        """A task runs in a sandbox exactly when it declares an image or a Dockerfile."""
        return self.image is not None or (self.path / DOCKERFILE).is_file()

    @property
    def needs_compose(self) -> bool:
        """The task's environment is a Compose group: ``main`` plus the sidecars its overlay declares."""
        return any(
            (self.path / "environment" / name).is_file()
            for name in ("docker-compose.yaml", "docker-compose.yml", "compose.yaml", "compose.yml")
        )

    @property
    def has_solution(self) -> bool:
        return (self.path / "solution" / "solve.sh").is_file()


def is_task_folder(path: Path) -> bool:
    return (Path(path) / TASK_FILE).is_file()


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


def built_image(task_folder: Path) -> str | None:
    """The image prepare built for the task's Dockerfile, when ``built-images.json`` records one."""
    task_folder = Path(task_folder)
    context = task_folder / DOCKERFILE.parent
    if not context.is_dir():
        return None
    entry = read_built_images(task_folder.parent).get(context_digest(context))
    image = entry.get("image") if isinstance(entry, dict) else None
    return str(image) if image else None


def _build_required(dockerfile: Path, reason: Exception) -> HarborTaskError:
    return HarborTaskError(
        f"{dockerfile} needs a build ({reason}) and no built image is recorded in "
        f"{dockerfile.parent.parent.parent / BUILT_IMAGES_FILE}. `gym eval run` and `gym dataset validate` build it "
        "when buildah, podman or docker is installed (configured by the `builds:` block, see "
        "environment_servers/harbor/README.md), or declare [environment].docker_image with a prebuilt image."
    )


def dockerfile_base_image(task_folder: Path, declared_image: str | None) -> str | None:
    """The image whose recorded configuration the task's Dockerfile resolves against, if it has one.

    A prebuilt ``docker_image`` already holds whatever its Dockerfile's ``RUN`` lines did, so only a
    pull-mode Dockerfile contributes settings next to a declared image. Dockerfiles that need a build
    contribute nothing (``load_task`` reports them).
    """
    dockerfile = Path(task_folder) / DOCKERFILE
    if not dockerfile.is_file():
        return None
    try:
        parsed = parse_dockerfile(dockerfile.read_text())
    except (DockerfileNeedsBuild, UnicodeDecodeError):
        return None
    if declared_image is not None and parsed.has_runs:
        return None
    return parsed.image


_CANARY_LINE = re.compile(r"^(<!--.*canary.*-->|#.*canary.*)$", re.IGNORECASE)


def read_instruction(path: Path) -> str:
    """``instruction.md`` as the agent sees it: leading canary marker lines and blank lines dropped.

    Harbor tasks open with an HTML comment or heading carrying a canary GUID for contamination
    tracking. It is not part of the task, so it is removed the way Gym's Terminal Bench servers
    always did; everything after it is kept byte for byte.
    """
    lines = path.read_text().split("\n")
    removed_canary = False
    while lines and _CANARY_LINE.match(lines[0].strip()):
        lines.pop(0)
        removed_canary = True
    # Only the blank lines that separated the canary from the task go with it. A file with no canary is
    # the author's prompt byte for byte, leading newline included, as the legacy servers passed it.
    while removed_canary and lines and not lines[0].strip():
        lines.pop(0)
    return "\n".join(lines)


def load_task(path: Path) -> HarborTask:
    """Read ``path`` as one task. Raises :class:`HarborTaskError` when it is not one."""
    path = Path(path)
    if path.is_symlink():
        raise HarborTaskError(f"Task folder must not be a symlink: {path}")
    path = path.resolve()
    if not is_task_folder(path):
        raise HarborTaskError(f"{path} has no {TASK_FILE}")
    try:
        data = tomllib.loads((path / TASK_FILE).read_text())
        for key in HarborTaskConfig.unknown_keys(data):
            logger.warning("%s: ignoring unknown key `%s`", path / TASK_FILE, key)
        config = HarborTaskConfig.model_validate(data)
    except (OSError, ValueError) as exc:
        raise HarborTaskError(f"{path / TASK_FILE}: {exc}") from exc
    # The dataset's own settings sit beside the task folders and shape the effective task.toml.
    try:
        config = apply_dataset_config(config, path.name, read_dataset_config(path.parent))
    except ValueError as exc:
        raise HarborTaskError(str(exc)) from exc

    phases = {"environment": config.environment, "agent": config.agent, "verifier": config.verifier}
    if config.verifier.environment is not None:
        phases["verifier.environment"] = config.verifier.environment
    for name, phase in phases.items():
        if phase.network_mode not in (None, "public") or phase.allowed_hosts:
            raise HarborTaskError(f"{path}: {name} network restrictions are not supported yet")
    for name, environment in (
        ("environment", config.environment),
        ("verifier.environment", config.verifier.environment),
    ):
        if environment is not None and environment.os != "linux":
            raise HarborTaskError(f"{path}: {name} Windows tasks are not supported yet")

    instruction_path = path / INSTRUCTION_FILE
    if not instruction_path.is_file():
        raise HarborTaskError(f"{path} has no {INSTRUCTION_FILE}")
    if not (path / "tests").is_dir():
        raise HarborTaskError(f"{path} has no tests/ folder")
    try:
        instruction = read_instruction(instruction_path)
    except (OSError, UnicodeDecodeError) as exc:
        raise HarborTaskError(f"{instruction_path}: {exc}") from exc
    if not config.is_shared_verifier and not config.artifacts:
        # Harbor accepts this shape: the separate verifier still receives /logs/artifacts.
        logger.warning(
            "%s uses separate verification but declares no artifacts; the verifier grades a fresh container "
            "that receives only what the agent writes under /logs/artifacts",
            path / TASK_FILE,
        )

    environment = config.environment
    image = environment.docker_image
    workdir = environment.workdir
    env = dict(environment.env)
    user: str | None = None
    overlay: tuple[OverlayRun, ...] = ()
    base_known = False
    base_user: str | None = None
    dockerfile = path / DOCKERFILE
    built: str | None = None
    # What Docker starts the image with: the record of the image the sandbox runs (built image), or the
    # base image's record carried through the Dockerfile's final USER and WORKDIR (pull and overlay mode).
    entrypoint: tuple[str, ...] | None = None
    entrypoint_user: str | None = None
    entrypoint_workdir: str | None = None
    image_healthcheck: dict[str, Any] | None = None
    if dockerfile.is_file():
        try:
            dockerfile_text = dockerfile.read_text()
        except (OSError, UnicodeDecodeError) as exc:
            raise HarborTaskError(f"{dockerfile}: {exc}") from exc
        try:
            parsed = parse_dockerfile(dockerfile_text)
        except DockerfileNeedsBuild as exc:
            if image is None:
                # Build mode: prepare built the context once and recorded the image next to the tasks.
                built = built_image(path)
                if built is None:
                    raise _build_required(dockerfile, exc) from exc
            parsed = None
        if built is not None:
            # The built image is pulled like a prebuilt one; its recorded configuration gives the
            # Dockerfile's final WORKDIR, ENV and USER.
            record = read_image_configs(path.parent).get(built)
            base = BaseImageConfig.from_record(record) if record is not None else None
            image = built
            if base is not None:
                base_known = True
                base_user = base.user
                workdir = workdir or base.workdir
                env = base.env | env
                user = base.user
                entrypoint, entrypoint_user, entrypoint_workdir = base.entrypoint, base.user, base.workdir
                image_healthcheck = base.healthcheck
        if parsed is not None and (image is None or not parsed.has_runs):
            # A prebuilt image already holds whatever its Dockerfile's RUN lines did; only a
            # pull-mode Dockerfile contributes settings next to a declared image.
            record = read_image_configs(path.parent).get(parsed.image)
            base = BaseImageConfig.from_record(record) if record is not None else None
            base_known = base is not None
            base_user = base.user if base is not None else None
            try:
                resolved = resolve_dockerfile(parsed, base)
            except ImageConfigRequired as exc:
                raise HarborTaskError(
                    f"{dockerfile}: {exc}, and the configuration of base image {parsed.image!r} is not recorded "
                    f"in {path.parent / IMAGE_CONFIGS_FILE}. Prepare the dataset again (`gym eval run` or "
                    "`gym dataset validate`) to record it."
                ) from exc
            except DockerfileNeedsBuild as exc:
                # An expansion the overlay cannot resolve: a build, unless prepare already built it.
                built = built_image(path)
                if built is None:
                    raise _build_required(dockerfile, exc) from exc
                record = read_image_configs(path.parent).get(built)
                base = BaseImageConfig.from_record(record) if record is not None else None
                image = built
                base_known = base is not None
                base_user = base.user if base is not None else None
                if base is not None:
                    workdir = workdir or base.workdir
                    env = base.env | env
                    user = base.user
                    entrypoint, entrypoint_user, entrypoint_workdir = base.entrypoint, base.user, base.workdir
                    image_healthcheck = base.healthcheck
                resolved = None
            if resolved is not None:
                image = image or resolved.image
                workdir = workdir or resolved.workdir
                env = resolved.env | env
                user = resolved.user
                overlay = resolved.runs
                if base is not None:
                    # The image Docker would build from this Dockerfile keeps the base's entrypoint (unless
                    # the Dockerfile cleared it) and healthcheck, started with its final USER and WORKDIR.
                    entrypoint = None if resolved.entrypoint_cleared else base.entrypoint
                    entrypoint_user, entrypoint_workdir = resolved.user, resolved.workdir
                    image_healthcheck = base.healthcheck
    if config.agent.user is not None:
        user = str(config.agent.user)

    try:
        digest = content_hash(path)
    except (OSError, ValueError) as exc:
        raise HarborTaskError(f"{path}: {exc}") from exc

    return HarborTask(
        path=path,
        task_id=path.name,
        config=config,
        instruction=instruction,
        digest=digest,
        image=image,
        workdir=workdir,
        env=env,
        user=user,
        overlay=overlay,
        sandbox_user=base_user if base_known else user,
        built_image=built,
        entrypoint=entrypoint,
        entrypoint_user=entrypoint_user,
        entrypoint_workdir=entrypoint_workdir,
        image_healthcheck=image_healthcheck,
    )


def discover_tasks(root: Path, *, skipped: dict[str, HarborTaskError] | None = None) -> list[HarborTask]:
    """A folder with ``task.toml`` is one task; otherwise its direct children are the tasks.

    With ``skipped``, a child that fails to load is recorded there under its folder name
    and the others still load; without it the first failure raises. A root that is
    itself a task always raises.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise HarborTaskError(f"{root} is not a directory")
    if is_task_folder(root):
        return [load_task(root)]
    children = sorted(child for child in root.iterdir() if child.is_dir() and is_task_folder(child))
    if not children:
        raise HarborTaskError(f"{root} is neither a task folder nor a folder of task folders (no {TASK_FILE} found)")
    if skipped is None:
        return [load_task(child) for child in children]
    tasks: list[HarborTask] = []
    for child in children:
        try:
            tasks.append(load_task(child))
        except HarborTaskError as exc:
            skipped[child.name] = exc
    if not tasks:
        raise HarborTaskError(f"No task under {root} loads; first error: {next(iter(skipped.values()))}")
    return tasks
