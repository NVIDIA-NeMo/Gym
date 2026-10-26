# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drive docker, buildah or podman for one planned build.

ImageBuilder owns the per-tool argv for build, compose build, push and inspect, the generated
containers.conf and registries.conf environment, the compose plugin and heredoc preflights,
the pushed digest and the inspect normalisation that turns tool output into the recorded
configuration. It is the only build module that runs subprocesses, through the Runner seam.
ImageBuilder resolves detect_tool from this module's globals, which is where tests patch it.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from nemo_gym.tasks.harbor.build_compose import (
    build_section,
    compose_build_env,
    task_compose_file,
    validate_task_compose,
    write_compose_files,
)
from nemo_gym.tasks.harbor.build_plan import (
    BUILD_TOOLS,
    DEFAULT_PLATFORM,
    HARBOR_COMPOSE_FILE,
    BuildError,
    BuildPlanEntry,
    BuildResult,
    Runner,
    default_storage_root,
    detect_tool,
)
from nemo_gym.tasks.harbor.build_records import build_inputs_digest, platform_key, settle_with_document
from nemo_gym.tasks.harbor.dockerfile import RECORD_VERSION, dockerfile_has_heredoc
from nemo_gym.tasks.harbor.harbor_compat import sanitize_docker_image_name
from nemo_gym.tasks.harbor.records import MAIN_SERVICE
from nemo_gym.tasks.harbor.registry import RECORDED_CONFIG_KEYS, ImageConfigError, platform_pair
from nemo_gym.tasks.harbor.verifier_env import (
    verifier_compose_main_overrides,
    verifier_compose_refusal,
    verifier_compose_sidecars,
)


LOGGER = logging.getLogger(__name__)

# A 403 in a tool's output, not the digits inside a hex digest.
HTTP_403 = re.compile(r"(?<![0-9a-fA-F])403(?![0-9a-fA-F])")
# The Compose plugin the docker backend needs (`docker compose`, Compose v2), by its package name.
COMPOSE_PLUGIN = "docker-compose-plugin"

# buildah and podman read this variable as a containers.conf layered over the host's; see ImageBuilder.tool_env.
CONTAINERS_CONF_OVERRIDE = "CONTAINERS_CONF_OVERRIDE"
CONTAINERS_CONF_FILE = "containers.conf"
# No default volumes: the host's containers.conf may bind-mount files such as /etc/localtime into every container,
# build steps included, and a Dockerfile that installs tzdata then fails replacing the mounted file.
CONTAINERS_CONF_TEXT = "[containers]\nvolumes = []\n"
# buildah and podman read this variable as their registries.conf (it replaces the host's main file; the drop-ins
# under registries.conf.d still load); see ImageBuilder.tool_env.
CONTAINERS_REGISTRIES_CONF = "CONTAINERS_REGISTRIES_CONF"
REGISTRIES_CONF_FILE = "registries.conf"
# An unqualified FROM (``FROM oven/bun:1``) resolves to docker.io, as it does under ``docker compose build``. Debian
# and Ubuntu ship a registries.conf with no unqualified-search registry, so buildah and podman otherwise refuse it.
REGISTRIES_CONF_TEXT = 'unqualified-search-registries = ["docker.io"]\n'
# The image format buildah and podman build with: docker, the format BuildKit builds. Under their default OCI
# format SHELL is ignored by the build's RUN steps and HEALTHCHECK and ONBUILD are dropped from the image.
IMAGE_FORMAT = "docker"
# The heredoc preflight of buildah and podman: a two-line build that pulls no base image and parses only when the
# tool was built with openshift/imagebuilder 1.2.6 or later (Ubuntu 24.04's packages were not).
HEREDOC_PROBE_DOCKERFILE = "FROM scratch\nCOPY <<EOF /probe\nprobe\nEOF\n"
HEREDOC_PROBE_TAG = "localhost/nemo-gym/heredoc-probe"
HEREDOC_REMEDY = (
    "Dockerfile heredocs need a buildah or podman built with openshift/imagebuilder 1.2.6 or later (upstream buildah "
    "1.33.0 or later; Ubuntu 24.04's packaged buildah 1.33.7 and podman 4.9.3 do not qualify), named with "
    "builds.tool, or builds.tool: docker."
)
# Configuration keys recorded for a built image: the pulled-image keys plus the docker-format fields a build sets.
BUILT_CONFIG_KEYS = (*RECORDED_CONFIG_KEYS, "OnBuild", "Shell")


def _run_default(argv: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, errors="replace", env=env)


class ImageBuilder:
    """Build the services of one context into named images with docker, buildah or podman, and push them when asked.

    ``main``'s image is named as Harbor names it, ``hb__<hash>``, a sidecar's ``hb__<hash>-<service>`` (see
    :meth:`tag_for`). ``repository`` is the registry path the names go under (``registry/path/hb__<hash>``) and
    the images are pushed there. Without a repository an image stays local under its name in the Docker daemon,
    the only local store a sandbox provider starts images from, so buildah and podman need a repository. ``push``
    defaults to whether a repository was named. ``platform`` (``linux/amd64`` by default, the platform sandboxes
    run) is passed to the tool, and an image that comes out for another platform is refused before it is pushed
    or recorded; a platform the records could not tell from the baseline (an amd64 micro-architecture level above
    it, see ``platform_pair``) is refused before anything is built. ``run`` is the command seam.

    docker builds the way Harbor's Docker environment does: one ``docker compose build`` over Harbor's build
    Compose file, the task's own and Gym's images override (:meth:`compose_build_argv`), never ``docker build``.
    buildah and podman build each service's Dockerfile directly (:meth:`build_argv`).
    """

    def __init__(
        self,
        *,
        tool: str | None = None,
        repository: str | None = None,
        push: bool | None = None,
        platform: str | None = None,
        storage_root: Path | None = None,
        run: Runner | None = None,
    ) -> None:
        self.repository = repository.rstrip("/") if repository else None
        self.tool = tool or detect_tool(self.repository)
        if self.tool is None:
            wanted = (
                ", ".join(BUILD_TOOLS) if self.repository else "docker (buildah and podman need builds.repository)"
            )
            raise BuildError(f"No image builder found: install {wanted}, or name one with builds.tool")
        if self.tool not in BUILD_TOOLS:
            raise BuildError(f"Unknown build tool {self.tool!r}; expected one of " + ", ".join(BUILD_TOOLS))
        if self.tool != "docker" and not self.repository:
            raise BuildError(
                f"{self.tool} keeps built images in a store no sandbox provider reads; set builds.repository "
                "so the image is pushed, or build with docker to keep it in the local daemon"
            )
        self.push = bool(self.repository) if push is None else push
        if self.push and not self.repository:
            raise BuildError("Pushing a built image needs builds.repository")
        self.platform = platform or DEFAULT_PLATFORM
        # builds.platform's check (BuildOptions.from_config), repeated for a BuildOptions made in code: every build
        # passes through here, and an image built for linux/amd64/v3 would be pushed and recorded under os/arch
        # alone, for a later linux/amd64 run to reuse.
        try:
            platform_pair(self.platform)
        except ImageConfigError as exc:
            raise BuildError(f"builds.platform: {exc}") from exc
        self.storage_root = Path(storage_root) if storage_root is not None else default_storage_root()
        self._run = run or _run_default
        self._compose_checked = False
        self._tool_env: dict[str, str] | None = None
        # The heredoc preflight's outcome: None until it ran, "" when the tool builds heredocs, else why it does not.
        self._heredoc_failure: str | None = None

    # -- commands ------------------------------------------------------------------------------

    def _storage_flags(self) -> list[str]:
        if self.tool != "buildah":
            return []
        return [
            "--storage-driver",
            "vfs",
            "--root",
            str(self.storage_root / "storage"),
            "--runroot",
            str(self.storage_root / "run"),
        ]

    def tag_for(self, entry: BuildPlanEntry) -> str:
        """The image's name under the repository when there is one: ``hb__<hash>`` for ``main``,
        ``hb__<hash>-<service>`` for a sidecar.

        The hash is the entry's digest, Harbor's environment id of the context, and ``hb__<hash>`` is the
        value Harbor exports as ``MAIN_IMAGE_NAME`` for it (the name a task Compose file that sets
        ``image: ${MAIN_IMAGE_NAME}`` builds under; Harbor's own template leaves the tag to Compose). A sidecar
        takes Compose's default form for a service without ``image:``, ``<project>-<service>``, under the project
        name Gym passes to ``docker compose build`` (``hb__<hash>``; Harbor's is the session id, so its sidecar tags
        change per run and are never published). The separate verifier's image takes the sidecar form too,
        ``hb__<hash>-verifier`` with Harbor's hash of ``tests/``: the hash is the one Harbor exports as
        ``MAIN_IMAGE_NAME`` for its verifier environment, and the suffix names the role where Harbor's own tag is the
        per-session ``<session>-main``. One flat name per image, sanitised like Harbor's, stays within the
        one repository path ``builds.repository`` promises; a slash would open a path per hash.
        """
        return self.tag_name(entry.digest, entry.service)

    def tag_name(self, digest: str, service: str = MAIN_SERVICE) -> str:
        """:meth:`tag_for` by digest and service."""
        name = f"hb__{digest}" if service == MAIN_SERVICE else f"hb__{digest}-{service}"
        name = sanitize_docker_image_name(name)
        return f"{self.repository}/{name}" if self.repository else name

    def build_argv(self, entry: BuildPlanEntry, tag: str) -> list[str]:
        """The direct Dockerfile build of buildah and podman (docker builds through :meth:`compose_build_argv`).

        ``--format docker`` builds the image format BuildKit builds, so ``SHELL`` applies to the ``RUN`` steps and
        ``HEALTHCHECK`` and ``ONBUILD`` reach the image (the tools' default OCI format has none of the three).
        The service's ``build`` section of the task's Compose file, which ``docker compose build`` would apply (over
        Harbor's template for ``main``), is applied the same way: its context and Dockerfile, ``--build-arg`` per arg
        and ``--target`` (:func:`service_build_settings`).
        """
        platform = ["--platform", self.platform] if self.platform else []
        options = [*platform]
        for key, value in entry.build_args.items():
            options += ["--build-arg", f"{key}={value}"]
        if entry.target:
            options += ["--target", entry.target]
        tail = ["-t", tag, "-f", str(entry.dockerfile_path), str(entry.build_dir)]
        if self.tool == "buildah":
            storage = self._storage_flags()
            return ["buildah", *storage, "bud", "--format", IMAGE_FORMAT, "--isolation", "chroot", *options, *tail]
        return [self.tool, "build", "--format", IMAGE_FORMAT, *options, *tail]

    def compose_build_argv(
        self,
        entry: BuildPlanEntry,
        compose_file: Path,
        images_file: Path | None = None,
        services: tuple[str, ...] = (),
    ) -> list[str]:
        """Harbor's ``docker compose build`` line for the context (``_run_docker_compose_command``,
        https://github.com/harbor-framework/harbor/blob/v0.22.0/src/harbor/environments/docker/docker.py#L606).

        The project name stands in for Harbor's session id (Compose's default image name is
        ``<project>-<service>``; ``main`` is named explicitly, see :func:`harbor_compose_document`); the
        project directory is the context, so relative paths in the task's Compose file resolve as Harbor
        resolves them; the files are Harbor's build template, then the task's ``docker-compose.yaml`` when
        it has one, in Harbor's order, then Gym's images override (:func:`images_compose_document`), last so its
        names win. ``services`` names the services to build; Harbor passes none, and so does a full build, so every
        service with a ``build`` section is built. A partial build (some services of the context recorded, or
        ``main`` not planned) names the ones it builds.
        """
        argv = [
            "docker",
            "compose",
            "--project-name",
            f"hb__{entry.digest}",
            "--project-directory",
            str(entry.context),
            "-f",
            str(compose_file),
        ]
        task_compose = task_compose_file(entry.context)
        if task_compose is not None:
            argv += ["-f", str(task_compose)]
        if images_file is not None:
            argv += ["-f", str(images_file)]
        return [*argv, "build", *services]

    def push_argv(self, tag: str, digest_file: Path) -> list[str]:
        if self.tool == "buildah":
            return [
                "buildah",
                *self._storage_flags(),
                "push",
                "--digestfile",
                str(digest_file),
                tag,
                f"docker://{tag}",
            ]
        if self.tool == "podman":
            return ["podman", "push", "--digestfile", str(digest_file), tag]
        return ["docker", "push", tag]

    def inspect_argv(self, tag: str) -> list[str]:
        if self.tool == "buildah":
            return ["buildah", *self._storage_flags(), "inspect", "--type", "image", tag]
        return [self.tool, "image", "inspect", tag]

    def tool_env(self) -> dict[str, str] | None:
        """The environment buildah and podman commands run with, or ``None`` (inherit) for docker.

        The host's ``containers.conf`` may give every container a default bind mount (this deployment mounts
        ``/etc/localtime`` read-only into each one through ``containers.conf.d``), and buildah and podman apply
        those mounts to build steps too, so a Dockerfile that installs ``tzdata`` (Ubuntu with ``python3`` pulls
        it in) fails with ``mv: cannot move '/etc/localtime.dpkg-new' to '/etc/localtime': Device or resource
        busy``. The builder writes a ``containers.conf`` with no default volumes under its storage root, once, and
        hands it to the tool as ``CONTAINERS_CONF_OVERRIDE``, the layer containers-common applies over the host
        files. A ``CONTAINERS_CONF_OVERRIDE`` the caller already set is left as it is.

        Both tools resolve an unqualified ``FROM`` (``FROM oven/bun:1``) through the host's ``registries.conf``, and
        Debian's and Ubuntu's stock file defines no unqualified-search registry, so the build fails with
        ``short-name "oven/bun:1" did not resolve to an alias`` where ``docker compose build`` pulls from docker.io.
        The builder writes a ``registries.conf`` naming docker.io as the one unqualified-search registry under its
        storage root and hands it to the tool as ``CONTAINERS_REGISTRIES_CONF``. That file replaces the host's main
        ``registries.conf`` (the drop-ins under ``registries.conf.d``, short-name aliases included, still load), so a
        site whose main file carries mirrors or blocked registries exports its own ``CONTAINERS_REGISTRIES_CONF``,
        which is left as it is. docker needs neither file and inherits the process environment.
        """
        if self.tool == "docker":
            return None
        if self._tool_env is None:
            env = dict(os.environ)
            self.storage_root.mkdir(parents=True, exist_ok=True)
            for variable, name, text in (
                (CONTAINERS_CONF_OVERRIDE, CONTAINERS_CONF_FILE, CONTAINERS_CONF_TEXT),
                (CONTAINERS_REGISTRIES_CONF, REGISTRIES_CONF_FILE, REGISTRIES_CONF_TEXT),
            ):
                if variable in os.environ:
                    continue
                path = self.storage_root / name
                if not path.is_file() or path.read_text() != text:
                    path.write_text(text)
                env[variable] = str(path)
            self._tool_env = env
        return self._tool_env

    # -- the pipeline --------------------------------------------------------------------------

    def _check(self, argv: list[str], what: str, *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
        try:
            completed = self._run(argv) if env is None else self._run(argv, env=env)
        except OSError as exc:
            raise BuildError(f"{what} failed to start ({argv[0]}): {exc}") from exc
        if completed.returncode != 0:
            tail = "\n".join((completed.stderr or completed.stdout or "").strip().splitlines()[-15:])
            message = f"{what} failed (exit {completed.returncode}): {' '.join(argv)}\n{tail}"
            note = _proxy_note(tail, env if env is not None else dict(os.environ))
            if note:
                LOGGER.warning("%s: %s", what, note)
                message += "\n" + note
            raise BuildError(message)
        return completed

    def build(self, entry: BuildPlanEntry) -> BuildResult:
        """:meth:`build_many` for one service, returning its result."""
        (result,) = list(self.build_many([entry]))
        return result

    def build_many(self, entries: list[BuildPlanEntry], *, partial: bool = False) -> Iterator[BuildResult]:
        """Build the services ``entries`` of one context, push when configured, inspect, and yield each result to
        record as soon as it is done, in order. Raises :class:`BuildError`; what was yielded before stays built.

        The task's Compose file is read here, not at planning: it is checked once for the context
        (:func:`validate_task_compose`) and each entry's service's ``build`` section fills it
        (:func:`service_build_settings`), so a file that does not interpolate, escapes ``environment/`` or carries a
        refused key fails this context's build and the caller leaves its tasks out, the way it leaves out a task
        whose build command fails. The build's environment is Harbor's (:func:`compose_build_env`) with
        ``MAIN_IMAGE_NAME`` the ``main`` tag, whether or not ``main`` is among the entries, as Harbor always exports
        it.

        docker runs one ``docker compose build`` for the context, Harbor's build step, and then inspects and pushes
        each image; ``partial`` says ``entries`` are not every service the context builds (the others are
        recorded), so the build names them rather than building the whole project. buildah and podman run one build
        per service (:meth:`build_argv`); on that path ``dockerfile_inline`` and ``additional_contexts`` (even
        ``service:`` references, which need ``--build-context`` and a build order) are refused naming the service,
        before the context's first build, since the checked file already says so, and ``no_cache`` and ``pull`` are
        ignored, as the direct tools have no Compose to honour them.
        """
        if not entries:
            return
        first = entries[0]
        tasks = ", ".join(first.tasks[:5])
        if first.verifier:
            # A tests/docker-compose.yaml the loader refuses as written (a top-level key other than services that
            # Harbor's compose up applies to the verifier project, a file that is not read as a services mapping) is
            # refused here too, before any command: validate_task_compose below refuses include and invalid YAML but
            # reads nothing of networks or volumes, so the verifier would be built, pushed and recorded for a task
            # the loader then refuses.
            refusal = verifier_compose_refusal(first.context.parent)
            if refusal:
                raise BuildError(f"task(s) {tasks}: {refusal}")
            # Harbor builds and starts every service of tests/docker-compose.yaml as the verifier's Compose project;
            # the server starts one verifier sandbox, so the loader refuses such a task. Refused here too, before
            # any command, so the docker backend builds no sidecar that nothing records or runs.
            sidecars = verifier_compose_sidecars(first.context.parent)
            if sidecars:
                raise BuildError(
                    f"{first.context / HARBOR_COMPOSE_FILE} of task(s) {tasks} declares service(s) "
                    f"{', '.join(sidecars)} beside main; Harbor starts them as the separate verifier's Compose "
                    "project and this server starts a single verifier sandbox"
                )
            # A runtime setting on main (environment, env_file, volumes...) reaches Harbor's verifier container and
            # never this server's verifier sandbox: refused like a sidecar, before an image nothing will run is built.
            # A verifier entry is always a build (a prebuilt verifier plans none), so main's image is read, pinned to
            # MAIN_IMAGE_NAME by validate_task_compose below.
            overrides = verifier_compose_main_overrides(first.context.parent, prebuilt=False)
            if overrides:
                raise BuildError(
                    f"{first.context / HARBOR_COMPOSE_FILE} of task(s) {tasks} sets {', '.join(overrides)} on main; "
                    "Harbor applies the file to the separate verifier's container and this server starts that "
                    "verifier from task.toml's environment and the verifier image alone"
                )
        main_tag = self.tag_name(first.digest)
        if self.tool == "docker":
            self._ensure_compose_plugin()
        env = compose_build_env(first, main_tag, self.platform)
        # Checked as docker compose will read it: interpolated against the build's own environment. On every tool:
        # a context the loader refuses is never built and pushed under the hb__ name.
        document = validate_task_compose(
            first.context,
            env,
            tasks=first.tasks,
            builds_main=any(entry.compose_service == MAIN_SERVICE for entry in entries),
        )
        # Settled as build_inputs_mismatch settles an entry it checks: the section's settings and the inputs digests
        # of the services the entry builds from, so the digest recorded below is the one a later run compares.
        settled = [settle_with_document(entry, document, env) for entry in entries]
        # Image names are lower-case (sanitize_docker_image_name, as Harbor sanitises its), so two services that
        # differ by case only (`Db` and `db`, both valid Compose names) would build under one tag, the second build
        # silently replacing the first on either tool path and both records naming one image. Refused before any
        # command, naming both services; Harbor itself never lower-cases a sidecar's tag, so docker rejects the upper-
        # case one there.
        tagged: dict[str, str] = {}
        for entry in settled:
            tag = self.tag_for(entry)
            if tag in tagged:
                raise BuildError(
                    f"{first.context / HARBOR_COMPOSE_FILE} of task(s) {tasks} declares services {tagged[tag]!r} and "
                    f"{entry.service!r}, which both build as image {tag}; image names are lower-case, so rename one"
                )
            tagged[tag] = entry.service
        for entry in settled:
            if not entry.dockerfile_inline and not entry.dockerfile_path.is_file():
                where = "" if entry.compose_service == MAIN_SERVICE else f" for services.{entry.service}.build"
                raise BuildError(f"{entry.build_dir} has no {entry.dockerfile}{where}")
            if self.tool != "docker":
                # Knowable from the checked document: refused before this context builds or pushes anything.
                self._refuse_on_direct_path(entry, document, tasks)
        if self.tool == "buildah":
            self.storage_root.mkdir(parents=True, exist_ok=True)
        if self.tool == "docker":
            # By Compose service: the verifier entry is Harbor's `main` of the tests/ context, tagged with Gym's name
            # for it through the images override, which is passed last and wins over the template's hb__<hash>.
            images = {entry.compose_service: self.tag_for(entry) for entry in settled}
            # Harbor builds the whole project; a build of some of its services names them.
            services = () if not partial and MAIN_SERVICE in images else tuple(images)
            what = f"Building {', '.join(images.values())} for task(s) {tasks}"
            LOGGER.info("%s: docker compose build of service(s) %s", what, ", ".join(images))
            with tempfile.TemporaryDirectory(prefix="nemo-gym-compose-") as tmp:
                build_file, images_file = write_compose_files(Path(tmp), first.context, main_tag, images)
                self._check(self.compose_build_argv(first, build_file, images_file, services), what, env=env)
            for entry in settled:
                yield self._finish(entry, images[entry.compose_service])
            return
        for entry in settled:
            tag = self.tag_for(entry)
            what = f"Building {tag} for task(s) {tasks}"
            if dockerfile_has_heredoc(entry.dockerfile_path.read_text()):
                self._ensure_heredoc_support(entry.dockerfile_path)
            self._check(self.build_argv(entry, tag), what, env=self.tool_env())
            yield self._finish(entry, tag)

    def _refuse_on_direct_path(self, entry: BuildPlanEntry, document: dict[str, Any] | None, tasks: str) -> None:
        """The build section keys only ``docker compose build`` can honour, refused or ignored by buildah and podman."""
        path = entry.context / HARBOR_COMPOSE_FILE
        service = entry.compose_service
        if entry.dockerfile_inline:
            raise BuildError(
                f"{path}: services.{service}.build.dockerfile_inline is not supported by the direct "
                f"{self.tool} build of task(s) {tasks}; docker compose build reads it, so build with "
                "builds.tool: docker, or move the Dockerfile text into a file under environment/"
            )
        section = build_section(document, service) or {}
        if section.get("additional_contexts"):
            raise BuildError(
                f"{path}: services.{service}.build.additional_contexts is not supported by the direct "
                f"{self.tool} build of task(s) {tasks} (it needs --build-context and a build order); build with "
                "builds.tool: docker"
            )
        for key in ("no_cache", "pull"):
            if key in section:
                LOGGER.debug(
                    "%s: services.%s.build.%s is ignored by the direct %s build", path, service, key, self.tool
                )

    def _finish(self, entry: BuildPlanEntry, tag: str) -> BuildResult:
        """Inspect the built ``tag``, refuse another platform, push when configured, and return the result."""
        tasks = ", ".join(entry.tasks[:5])
        record = self.inspect(tag)
        built_platform = f"{record.get('os')}/{record.get('architecture')}"
        if platform_key(built_platform) != platform_key(self.platform):
            service = "" if entry.is_main else f" (service {entry.service})"
            raise BuildError(
                f"Building {tag} for task(s) {tasks}{service}: the image came out as {built_platform}, not "
                f"{self.platform}. Build on a host of that platform, or one with qemu/binfmt emulation; "
                "builds.platform overrides the target."
            )
        image = tag
        if self.push:
            with tempfile.TemporaryDirectory(prefix="nemo-gym-push-") as tmp:
                digest_file = Path(tmp) / "digest"
                completed = self._check(self.push_argv(tag, digest_file), f"Pushing {tag}", env=self.tool_env())
                digest = _pushed_digest(self.tool, digest_file, completed, self._run, tag)
            image = f"{tag}@{digest}"
        # The image's record says what it was built from, beside the config_digest of what came out.
        record = {**record, "image": image, "inputs_digest": build_inputs_digest(entry)}
        return BuildResult(
            digest=entry.digest,
            tasks=entry.tasks,
            tag=tag,
            image=image,
            tool=self.tool,
            pushed=self.push,
            record=record,
            service=entry.service,
        )

    def _ensure_compose_plugin(self) -> None:
        """Fail before anything is built when ``docker compose`` (the Compose v2 plugin) is missing."""
        if self._compose_checked:
            return
        argv = ["docker", "compose", "version"]
        try:
            completed = self._run(argv)
        except OSError as exc:
            raise BuildError(
                f"docker failed to start ({exc}); the docker backend needs docker and {COMPOSE_PLUGIN}"
            ) from exc
        if completed.returncode != 0:
            detail = [line for line in (completed.stderr or completed.stdout or "").splitlines() if line.strip()]
            raise BuildError(
                f"The docker backend builds with `docker compose build`, as Harbor does, and the Compose plugin "
                f"is not available: `{' '.join(argv)}` exited {completed.returncode}"
                + (f" ({detail[0].strip()})" if detail else "")
                + f". Install {COMPOSE_PLUGIN} (Docker Compose v2), or build with buildah or podman and a "
                "builds.repository."
            )
        self._compose_checked = True

    def _ensure_heredoc_support(self, dockerfile: Path) -> None:
        """Fail before a Dockerfile with a heredoc is built when this host's buildah or podman cannot parse one.

        Ubuntu 24.04's packaged buildah 1.33.7 and podman 4.9.3 are linked against openshift/imagebuilder 1.2.5,
        which reads the body of ``RUN <<EOF`` as instructions and fails with ``Unknown instruction: "ECHO"``. The
        preflight builds :data:`HEREDOC_PROBE_DOCKERFILE` (``FROM scratch`` plus a ``COPY`` heredoc; no base image
        is pulled) once per builder with the build's own flags and environment, and a failure is one
        :class:`BuildError` naming the tool, its version, the Dockerfile and the remedy (:data:`HEREDOC_REMEDY`)
        instead of one ``Unknown instruction`` per task. The outcome is cached either way.
        """
        if self._heredoc_failure is None:
            with tempfile.TemporaryDirectory(prefix="nemo-gym-heredoc-probe-") as tmp:
                probe = BuildPlanEntry("heredoc-probe", Path(tmp), ())
                probe.dockerfile_path.write_text(HEREDOC_PROBE_DOCKERFILE)
                argv = self.build_argv(probe, HEREDOC_PROBE_TAG)
                try:
                    completed = self._run(argv, env=self.tool_env())
                except OSError as exc:
                    self._heredoc_failure = f"{argv[0]} failed to start: {exc}"
                else:
                    tail = "\n".join((completed.stderr or completed.stdout or "").strip().splitlines()[-3:])
                    self._heredoc_failure = (
                        "" if completed.returncode == 0 else f"exit {completed.returncode}: {tail or 'no output'}"
                    )
            if self._heredoc_failure:
                self._heredoc_failure = f"{self._tool_version()}: {self._heredoc_failure}"
        if self._heredoc_failure:
            raise BuildError(
                f"{dockerfile} uses a Dockerfile heredoc (<<EOF) and this host's {self.tool} cannot build one: the "
                f"preflight build of `FROM scratch` + `COPY <<EOF /probe` failed ({self._heredoc_failure}). "
                + HEREDOC_REMEDY
            )

    def _tool_version(self) -> str:
        """``<tool> --version``'s first line, or the tool's name when it will not say."""
        try:
            completed = self._run([self.tool, "--version"])
        except (OSError, ValueError, StopIteration):
            return self.tool
        line = ((completed.stdout or "").strip() or (completed.stderr or "").strip()).splitlines()
        return line[0].strip() if line else self.tool

    def inspect(self, tag: str) -> dict[str, Any]:
        """The image's OCI configuration in the ``compose-images.json`` record shape (``image`` unset)."""
        completed = self._check(self.inspect_argv(tag), f"Inspecting {tag}", env=self.tool_env())
        try:
            data = json.loads(completed.stdout)
        except ValueError as exc:
            raise BuildError(f"Inspecting {tag}: {self.tool} returned no JSON ({exc})") from exc
        return normalize_inspect(self.tool, data)


def proxy_variable_names(env: dict[str, str]) -> list[str]:
    """The proxy-related variables in ``env`` (``http_proxy``, ``HTTPS_PROXY``, ``no_proxy``, ``all_proxy`` and the
    like), names only: their values can carry credentials."""
    return sorted(name for name in env if name.lower().endswith("_proxy"))


def _proxy_note(output: str, env: dict[str, str]) -> str:
    """Why a registry step may have answered 403, when the tool's output says it did: a proxy the build tool
    inherits (from its environment, or from its own configuration when the environment names none) can answer
    403 to the registry ping unless ``no_proxy`` carries the registry host. Names only, never values."""
    if not HTTP_403.search(output):
        return ""
    names = proxy_variable_names(env)
    if names:
        return (
            f"The registry answered 403 and the build tool's environment carries proxy variables {', '.join(names)}: "
            "a proxy inherited by the build tool can answer 403 to the registry ping; add the registry host to "
            "no_proxy (and NO_PROXY) or clear the proxy variables for the build."
        )
    return (
        "The registry answered 403 and the build tool's environment carries no proxy variables; a proxy the tool reads "
        "from its own configuration (containers.conf [engine].env, ~/.docker/config.json proxies, the Docker daemon) "
        "can still answer 403 to the registry ping unless its no_proxy carries the registry host."
    )


def _pushed_digest(tool: str, digest_file: Path, completed: subprocess.CompletedProcess, run: Runner, tag: str) -> str:
    """The manifest digest the registry assigned to the pushed image."""
    if tool in ("buildah", "podman"):
        if digest_file.is_file():
            digest = digest_file.read_text().strip()
            if digest.startswith("sha256:"):
                return digest
        raise BuildError(f"Pushing {tag}: {tool} wrote no digest file")
    # docker: the push output ends with `latest: digest: sha256:... size: ...`; RepoDigests holds it afterwards.
    for line in (completed.stdout or "").splitlines():
        if "digest: sha256:" in line:
            return "sha256:" + line.split("digest: sha256:", 1)[1].split()[0]
    inspected = run(["docker", "image", "inspect", "--format", "{{json .RepoDigests}}", tag])
    if inspected.returncode == 0:
        for repo_digest in json.loads(inspected.stdout or "[]"):
            if "@sha256:" in repo_digest:
                return repo_digest.split("@", 1)[1]
    raise BuildError(f"Pushing {tag}: docker reported no digest")


def normalize_inspect(tool: str, data: Any) -> dict[str, Any]:
    """Turn a tool's inspect JSON into the ``compose-images.json`` record shape (without ``image``).

    buildah reports ``{"OCIv1": {"architecture", "os", "config"}, "Docker": {"config"}, "FromImageID"}``; podman and
    docker report a list with ``Architecture``, ``Os``, ``Config`` and ``Id``. The image is built in docker format
    (:data:`IMAGE_FORMAT`), whose ``Shell``, ``Healthcheck`` and ``OnBuild`` the OCI image configuration cannot
    hold: buildah reports them under ``Docker.config``, read here over the OCI config; docker reports them in
    ``Config``; podman's ``Config`` is the OCI image configuration (its inspect does not expose ``Shell`` and
    ``OnBuild``, although the image carries them) and ``Healthcheck`` sits beside ``Config`` as a top-level key,
    read here when ``Config`` has none.
    """
    if tool == "buildah":
        if not isinstance(data, dict) or "OCIv1" not in data:
            raise BuildError("buildah inspect returned no OCIv1 block")
        oci = data["OCIv1"]
        config = {**(oci.get("config") or {}), **_docker_config(data)}
        config_digest = data.get("FromImageID") or ""
        architecture, os_name = oci.get("architecture"), oci.get("os")
    else:
        entry = data[0] if isinstance(data, list) and data else data
        if not isinstance(entry, dict) or "Config" not in entry:
            raise BuildError(f"{tool} image inspect returned no Config block")
        config = {**(entry.get("Config") or {}), **_docker_config(entry)}
        # podman: the docker-format healthcheck is a top-level field of the inspect entry.
        config.setdefault("Healthcheck", entry.get("Healthcheck"))
        config_digest = str(entry.get("Id") or "")
        architecture, os_name = entry.get("Architecture"), entry.get("Os")
    if config_digest and not config_digest.startswith("sha256:"):
        config_digest = f"sha256:{config_digest}"
    if (os_name, architecture) != ("linux", "amd64"):
        LOGGER.warning(
            "Built image is %s/%s, not linux/amd64; it runs only on a sandbox host of that platform",
            os_name,
            architecture,
        )
    return {
        "architecture": architecture,
        "config": {key: config[key] for key in BUILT_CONFIG_KEYS if config.get(key)},
        "config_digest": config_digest,
        "os": os_name,
        "record_version": RECORD_VERSION,
    }


def _docker_config(data: dict[str, Any]) -> dict[str, Any]:
    """The set fields of an inspect's ``Docker.config`` section (the docker-format image configuration), or ``{}``."""
    docker = data.get("Docker")
    config = docker.get("config") if isinstance(docker, dict) else None
    return {key: value for key, value in config.items() if value} if isinstance(config, dict) else {}
