# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The job-level Compose overlay: extra Compose files, MCP entries and a prompt template from the run config.

Harbor's job config names ``extra_docker_compose`` files, appended to every task's ``docker compose -f ...``
command after the task's own file, ``agents[].mcp_servers`` entries, merged by name with the task's
``[[environment.mcp_servers]]`` (``trial.py`` ``_init_agent`` at v0.22.0: the task's first, the job's after,
so a job entry replaces a task entry of the same name), and a ``prompt_template_path`` the agent renders
around the instruction (``harbor/utils/templating.py``). Gym reads the same three things from one block of
the environment server's configuration, ``compose_overlay`` (:class:`HarborComposeOverlay`), and changes no
task file.

The overlay files are merged onto the task's Compose document the way Compose merges ``-f`` files
(:func:`merge_compose`): mappings key by key, sequences appended (``ports``, ``volumes``, ``expose`` and the
other keyed sequences unique on their key, the later entry winning), ``environment``, ``labels`` and
``extra_hosts`` as mappings whatever their form, ``depends_on`` and ``networks`` as mappings, ``command``,
``entrypoint`` and ``healthcheck.test`` replaced whole, every scalar by the last file, a key written with no
value keeping the earlier file's value. The ``!reset`` tag removes the attribute the earlier files set
(``ports: !reset []``, ``networks.x.name: !reset null``, ``environment: {DISABLE_AUTH: !reset}``) and
``!override`` replaces a value instead of merging into it, as Compose defines both. Each file is
interpolated with the same variables as the task's file (Harbor's one ``docker compose`` command sees one
environment). Its relative paths resolve against the project directory, the task's ``environment`` folder, never
against the overlay file's own folder: Compose resolves the relative paths of every ``-f`` file against the project
directory (compose-go's ``ResolveRelativePaths`` with the one ``WorkingDir``; "Paths in override files are not
resolved relative to the override file's own location"), and Harbor runs ``docker compose --project-directory
<task>/environment -f <task file> -f <extra> ...``, so an extra file's ``./x.env`` or ``./conf`` is the task's
``environment/x.env`` or ``environment/conf`` under Harbor and here alike (the memory team's
``${GRAPHITI_CONFIG_PATH:-./config.yaml}`` and ``${MAMS_HARBOR_MCP_BUILD_CONTEXT:-./mcp-server}`` defaults land in the
task folder under both; set the variable to an absolute path, as the Harbor job must). An overlay file's
``env_file`` entries are resolved to absolute paths before the merge (:func:`absolute_env_files`) and the env files of
the merged document are read once after it, the merged ``environment`` winning over them, as Compose reads them (so
a later file's ``env_file: !reset []`` drops the earlier files' lists, the task's included). An overlay file is the
operator's, so its ``env_file`` may name any readable file (an absolute path anywhere, a relative one wherever it
lands), where a task's must stay under the task; its bind sources are left for the volume planner, which resolves a
relative source against the same folder for every file of the group.
A task that declares only an image gets a one-service group, ``main``, for the overlay to extend
(:func:`synthesized_group`).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from nemo_gym.data.tasks.harbor.compose_variables import EnvFileResolver
from nemo_gym.data.tasks.harbor.models import HarborMCPServer
from nemo_gym.data.tasks.harbor.records import MAIN_SERVICE


if TYPE_CHECKING:
    from nemo_gym.data.tasks.harbor.task import HarborTask


# The Jinja variable Harbor's prompt template renders the instruction into (``render_prompt_template``).
PROMPT_TEMPLATE_VARIABLE = "instruction"
# The Compose tags, as Compose spells them (https://docs.docker.com/reference/compose-file/merge/).
RESET_TAG = "!reset"
OVERRIDE_TAG = "!override"


class ComposeOverlayError(ValueError):
    """An overlay file, its merge or the prompt template cannot be applied as written."""


class HarborComposeOverlay(BaseModel):
    """The ``compose_overlay`` block of the Harbor environment server's configuration.

    ``files`` are Compose files merged, in order, onto every task's Compose document after the task's own
    file (Harbor's ``extra_docker_compose``); ``mcp_servers`` are MCP entries merged by name with the task's
    ``[[environment.mcp_servers]]``, the overlay's winning on a name (Harbor's ``agents[].mcp_servers``);
    ``prompt_template`` is a Jinja template with an ``{{ instruction }}`` variable, rendered around the
    instruction of the episode's first turn (Harbor's ``prompt_template_path``). Every path must exist when
    the block is read: Harbor refuses a missing extra Compose file the same way (``FileNotFoundError`` in
    ``_normalize_extra_docker_compose_paths``), and a template that does not name ``instruction`` or does
    not parse is refused before any episode, as Harbor's ``render_prompt_template`` refuses it. The template
    is read and compiled once, when the block is read (:func:`compile_prompt_template`); every episode renders
    the compiled template (:meth:`render_instruction`), so the file is not re-read per episode and an edit to it
    under a running server is not seen until the server is started again.
    """

    model_config = ConfigDict(extra="forbid")

    files: list[Path] = Field(default_factory=list)
    mcp_servers: list[HarborMCPServer] = Field(default_factory=list)
    prompt_template: Path | None = None
    # The compiled ``prompt_template`` (a ``jinja2.Template`` with ``StrictUndefined``), ``None`` without one.
    _compiled_template: Any = PrivateAttr(default=None)

    @field_validator("files")
    @classmethod
    def existing_files(cls, files: list[Path]) -> list[Path]:
        normalized: list[Path] = []
        for raw in files:
            path = Path(raw).expanduser()
            if not path.is_file():
                raise ValueError(f"Compose overlay file not found: {path}")
            normalized.append(path.resolve())
        return normalized

    @field_validator("prompt_template")
    @classmethod
    def readable_template(cls, path: Path | None) -> Path | None:
        if path is None:
            return None
        path = Path(path).expanduser()
        if not path.is_file():
            raise ValueError(f"Prompt template not found: {path}")
        return path.resolve()

    @model_validator(mode="after")
    def compile_template(self) -> "HarborComposeOverlay":
        """Read, check and compile the template once, when the block is read: a template that does not parse or
        does not name ``instruction`` is refused here, before any episode."""
        if self.prompt_template is None:
            return self
        try:
            self._compiled_template = compile_prompt_template(self.prompt_template)
        except ComposeOverlayError as exc:
            raise ValueError(str(exc)) from exc
        return self

    @property
    def configured(self) -> bool:
        return bool(self.files or self.mcp_servers or self.prompt_template)

    def render_instruction(self, instruction: str) -> str:
        """``instruction`` rendered into the compiled ``prompt_template`` (:func:`render_compiled_template`); the
        instruction as given when the block names no template."""
        if self.prompt_template is None:
            return instruction
        if self._compiled_template is None:  # a block built without validation (``model_construct``)
            self._compiled_template = compile_prompt_template(self.prompt_template)
        return render_compiled_template(self._compiled_template, self.prompt_template, instruction)


def overlay_files(overlay: HarborComposeOverlay | None) -> list[Path]:
    """The overlay's Compose files, ``[]`` without an overlay."""
    return list(overlay.files) if overlay is not None else []


def merged_mcp_servers(
    task_servers: Iterable[HarborMCPServer], overlay: HarborComposeOverlay | None
) -> list[HarborMCPServer]:
    """The task's MCP entries with the overlay's merged in by name, in Harbor's order and with Harbor's rule.

    ``trial.py`` ``_init_agent`` (v0.22.0) builds ``{server.name: server}`` over the task's
    ``environment.mcp_servers`` followed by the job's ``agent.mcp_servers`` and hands the agent the values:
    a job entry replaces a task entry of the same name, in the task entry's position; a new name goes last.
    """
    servers = {server.name: server for server in task_servers}
    if overlay is not None:
        for server in overlay.mcp_servers:
            servers[server.name] = server
    return list(servers.values())


# -- Compose documents ---------------------------------------------------------------------------------------


class _Reset:
    """A ``!reset`` value: the attribute the earlier files set is removed; the value written after the tag is
    not read (Compose ignores it too)."""

    def __repr__(self) -> str:
        return RESET_TAG


RESET = _Reset()


class OverrideMapping(dict):
    """A mapping tagged ``!override``: it replaces the earlier files' value instead of merging into it."""


class OverrideSequence(list):
    """A sequence tagged ``!override``: it replaces the earlier files' value instead of appending to it."""


class OverrideScalar(str):
    """A scalar tagged ``!override``. A scalar replaces the earlier value whether tagged or not, so the tag changes
    nothing where the value stays a scalar; it is kept for the fields whose scalar form is a one-entry sequence
    (``env_file: !override /path/x.env``, ``tmpfs``, ``dns``), which the normalization before the merge turns into
    an :class:`OverrideSequence`, so the one entry replaces the earlier files' lists instead of appending to them,
    as the tagged list form does. :func:`_plain` leaves a plain ``str`` in the merged document."""


class _OverlayLoader(yaml.SafeLoader):
    """``yaml.SafeLoader`` that reads Compose's ``!reset`` and ``!override`` tags."""


def _construct_reset(loader: yaml.SafeLoader, node: yaml.Node) -> _Reset:
    return RESET


def _construct_override(loader: yaml.SafeLoader, node: yaml.Node) -> Any:
    if isinstance(node, yaml.MappingNode):
        return OverrideMapping(loader.construct_mapping(node, deep=True))
    if isinstance(node, yaml.SequenceNode):
        return OverrideSequence(loader.construct_sequence(node, deep=True))
    return OverrideScalar(loader.construct_scalar(node))


_OverlayLoader.add_constructor(RESET_TAG, _construct_reset)
_OverlayLoader.add_constructor(OVERRIDE_TAG, _construct_override)


def load_overlay_text(text: str, *, source: str = "<overlay>") -> dict[str, Any]:
    """Parse one overlay Compose document with its ``!reset`` and ``!override`` tags, as a mapping.

    Top-level extension fields (``x-*``, the anchors a file keeps its shared blocks under) are dropped, as
    Compose ignores them; their aliases were expanded at parse time. Anything that is not a mapping, or does
    not parse, raises :class:`ComposeOverlayError` naming ``source``.
    """
    try:
        document = yaml.load(text, Loader=_OverlayLoader) or {}  # noqa: S506 (a SafeLoader subclass)
    except yaml.YAMLError as exc:
        raise ComposeOverlayError(f"{source} is not valid YAML: {_yaml_problem(exc)}") from exc
    if not isinstance(document, dict):
        raise ComposeOverlayError(f"{source} is not a mapping")
    return {str(key): value for key, value in document.items() if not str(key).startswith("x-")}


def _yaml_problem(exc: yaml.YAMLError) -> str:
    """What went wrong and where, without PyYAML's snippet of the offending line: an overlay file commonly carries the
    deployment's keys, and the message reaches prepare's ``skipped.json`` and the seed's failure reason. A
    ``MarkedYAMLError`` gives its context and problem (PyYAML's own phrases, no source text) with the line and column;
    any other error only its kind."""
    if isinstance(exc, yaml.MarkedYAMLError):
        parts = [part for part in (exc.context, exc.problem) if part]
        mark = exc.problem_mark or exc.context_mark
        where = f" (line {mark.line + 1}, column {mark.column + 1})" if mark is not None else ""
        return f"{'; '.join(parts) or type(exc).__name__}{where}"
    return type(exc).__name__


def load_overlay_file(path: Path) -> dict[str, Any]:
    """Read one overlay Compose file (:func:`load_overlay_text`); a file that cannot be read raises
    :class:`ComposeOverlayError` naming it."""
    try:
        text = Path(path).read_text()
    except OSError as exc:
        raise ComposeOverlayError(f"{path} cannot be read: {exc}") from exc
    return load_overlay_text(text, source=str(path))


def synthesized_group(task: HarborTask | None = None) -> dict[str, Any]:
    """The Compose document of a task that declares no Compose file: Harbor's own ``main`` service, nothing else.

    Harbor starts every task through Compose (its base file plus the build or prebuilt file define ``main``),
    so an extra Compose file extends a plain image task the same way it extends a Compose task. The sandbox
    adapter fills ``main``'s image and command from the task, as it does for a task file that leaves them out.

    ``main`` starts as the task's single sandbox would: when the loader read the task image's configuration
    (``task.image_shell`` is set whenever it did) it resolved how Docker starts the image the task's Dockerfile
    describes, the recorded ``ENTRYPOINT`` or none after ``ENTRYPOINT []`` (``HarborTask.entrypoint``, what the
    single sandbox runs beside its keepalive) in the Dockerfile's final ``WORKDIR`` (``HarborTask.entrypoint_workdir``,
    the single sandbox's ``cwd`` for it). The image resolution otherwise fills ``main``'s ``entrypoint`` and
    ``working_dir`` from the record of ``task.image``, the base image's for a Dockerfile the loader resolved, so the
    entrypoint is written here always: the resolution would start the base's on a task whose Dockerfile cleared it
    merely because an overlay was attached. ``working_dir`` is the entrypoint's directory, the ``WORKDIR``: a
    Compose service's ``working_dir`` is the directory its command runs in, and Harbor's built image carries the
    ``WORKDIR`` as that (its ``docker-compose-build.yaml`` writes no ``working_dir`` for ``main``), so the adapter
    starts ``main``'s command, entrypoint included, and runs its recorded healthcheck there, as the single sandbox
    runs both (``_start_entrypoint`` names ``task.entrypoint_workdir``, ``_wait_healthy`` gives the image check the
    same). The adapter makes a missing ``working_dir`` as root at create, where the single sandbox makes the
    ``WORKDIR`` the entrypoint user's before its entrypoint starts (``_start_entrypoint``, ``overlay_workdir_command``),
    so the server hands it to that user between the create and ``main``'s command (``_start_compose_group``'s
    setup hook on ``main``), and the entrypoint starts in its own directory in the group too. An
    ``[environment].workdir`` is not the container's directory in Harbor either: it is the ``cwd`` of
    Harbor's execs (harbor 0.22.0 ``environments/docker/docker.py`` 1104), and the server passes it per exec the same
    way (the agent's commands, the step scripts and checks, the verifier's run), so ``main``'s ``working_dir`` is
    never that. Written for a task without an overlay (a pull-mode Dockerfile, no ``RUN`` lines), never unset for
    a Dockerfile the loader resolved (``/`` at least); a built image without a recorded ``WorkingDir`` leaves the
    resolution nothing to fill either.

    An overlay task's ``main`` carries no ``working_dir``, as its single sandbox is created without the task workdir
    (``_create_sandbox_within_deadline``): the Compose adapter creates a service's ``working_dir`` as root before
    the overlay's ``RUN`` lines run, and the overlay's ``WORKDIR`` step (``overlay_workdir_command``) leaves an
    existing directory as it is, so a final ``WORKDIR`` under a non-root ``USER`` would stay root's and a ``RUN``
    line in it fail; left to the overlay, the directory is created and owned by the user in effect at that
    ``WORKDIR``, as in the build. No ``user`` is written either: the adapter runs ``main``'s command as the
    sandbox's default user. For a task without an overlay that is the task's resolved user (``HarborTask.user``,
    installed at creation by ``_sandbox_spec``: the Dockerfile's final ``USER`` unless ``[agent].user`` names
    another), the user the single sandbox gives the entrypoint (``task.entrypoint_user``) unless ``[agent].user``
    differs. For an overlay task the sandbox has no default user until ``_set_default_user`` installs the task's
    after the group started, and the resolution fills ``working_dir`` from the base image's record, so ``main``'s
    entrypoint starts as the base image's own user in the base image's own ``WorkingDir``, not as the Dockerfile's
    final ``USER`` in its final ``WORKDIR``, which is how the single sandbox runs it (``_start_entrypoint`` names
    ``task.entrypoint_user`` and ``task.entrypoint_workdir``) and how Harbor's built image starts it. That gap is
    left open here: the directory cannot be written (above), and the user would need the name mapping the single
    sandbox's ``_image_user_name`` does for a numeric ``USER``. The agent's commands are unaffected: they run as
    ``task.user`` in ``task.workdir`` once the seed has installed the user and prepared the directory. A task whose
    image the loader has no configuration for (a prebuilt ``docker_image`` with no Dockerfile) is left to the
    resolution, as a task file's ``main`` is; without a ``task`` the service is empty.
    """
    main: dict[str, Any] = {}
    if task is not None and task.image_shell is not None:
        main["entrypoint"] = list(task.entrypoint or [])
        if task.entrypoint_workdir and not task.overlay:
            # The entrypoint's directory, the Dockerfile's final WORKDIR, not [environment].workdir, which is the
            # cwd of the execs (above). An overlay task's final WORKDIR is the overlay's to create and own.
            main["working_dir"] = task.entrypoint_workdir
    return {"services": {MAIN_SERVICE: main}}


def interpolate_overlay(document: Any, substitute: Callable[[str], str]) -> Any:
    """``document`` with ``substitute`` applied to every string, the ``!reset`` and ``!override`` markers kept.

    Compose interpolates each ``-f`` file against the command's environment before merging, so the overlay
    files take the same ``substitute`` as the task's file (the task's resolver, with the deployment's values
    when the server has them).
    """
    if document is RESET:
        return document
    if isinstance(document, dict):
        return type(document)((key, interpolate_overlay(value, substitute)) for key, value in document.items())
    if isinstance(document, list):
        return type(document)(interpolate_overlay(value, substitute) for value in document)
    if isinstance(document, OverrideScalar):
        return OverrideScalar(substitute(str(document)))
    if isinstance(document, str):
        return substitute(document)
    return document


def resolve_passthrough_environment(document: Any, variables: Mapping[str, str]) -> Any:
    """``document`` with each overlay service's pass-through ``environment`` entries resolved against ``variables``:
    a bare ``NAME`` in the list form becomes ``NAME=value`` and a ``NAME:`` with no value in the mapping form becomes
    ``NAME: value`` for every name ``variables`` sets; a name it does not set stays bare, and the markers are kept (a
    ``!reset`` entry, an ``!override`` list or mapping, a service tagged either).

    Compose resolves the pass-through spelling against the command's environment, the one every ``-f`` file is
    interpolated with, so under Harbor it follows the order of ``${NAME}`` in every file. Gym reads the merged
    document's pass-through entries once, after the merge, against the task's variables
    (:func:`~nemo_gym.sandbox.compose_fields.inline_env_files`), which is the task's own file's order; an overlay
    file's entries follow the overlay's order (the deployment's values above the task's sources) only when they are
    resolved before the merge, here, with the variables :func:`interpolate_overlay` took for the same file. A name
    nobody sets stays bare and is read after the merge as the task's entries are: unset in the container.
    """
    services = document.get("services") if isinstance(document, dict) else None
    if not isinstance(services, dict):
        return document
    for service in services.values():
        if not isinstance(service, dict) or not isinstance(service.get("environment"), (list, dict)):
            continue
        environment = service["environment"]
        if isinstance(environment, list):
            service["environment"] = type(environment)(_passthrough_entry(item, variables) for item in environment)
        else:
            service["environment"] = type(environment)(
                (key, variables[str(key)] if value is None and str(key) in variables else value)
                for key, value in environment.items()
            )
    return document


def _passthrough_entry(item: Any, variables: Mapping[str, str]) -> Any:
    """A list-form ``environment`` entry, a bare ``NAME`` that ``variables`` sets resolved to ``NAME=value``."""
    if not isinstance(item, str):
        return item
    name, separator, _ = item.partition("=")
    return f"{name}={variables[name]}" if not separator and name in variables else item


def overlay_env_file_resolver(project_dir: Path) -> EnvFileResolver:
    """The resolver of an overlay file's ``env_file`` entries: a relative path against ``project_dir``, the Compose
    project directory (the task's ``environment`` folder, Harbor's ``--project-directory``; Compose resolves every
    ``-f`` file's relative paths against it, not against the file that wrote them), an absolute path as written, any
    readable file accepted. The overlay files are the operator's own, named in the run config, so the containment rule
    of a task's entries (third-party content that prepare digested,
    :func:`~nemo_gym.data.tasks.harbor.compose_variables.under_task_folder`) does not apply, as it does not under Harbor,
    where Compose reads an extra file's env files from wherever they are."""

    def resolve(path: str, service: str) -> Path:
        file = Path(path).expanduser()
        return (file if file.is_absolute() else Path(project_dir) / file).resolve()

    return resolve


def absolute_env_files(document: Any, project_dir: Path) -> Any:
    """Each overlay service's ``env_file`` entries replaced by the absolute paths they resolve to against
    ``project_dir`` (:func:`overlay_env_file_resolver`,
    :func:`~nemo_gym.data.tasks.harbor.compose_variables.resolve_env_file_entries`), the files neither read nor checked to
    exist yet: Compose resolves each file's paths before the merge and reads the env files of the merged document once
    after it, the merged ``environment`` winning, so a file a later overlay file drops (``env_file: !reset []``) is
    never required to exist and a missing one that survives the merge is reported at that reading. The markers are
    kept for the merge: ``env_file: !reset []`` and a service tagged ``!reset`` pass through, an ``!override`` list
    keeps its tag and an ``!override`` scalar becomes an ``!override`` list of its one path (so it replaces the earlier
    files' lists, as Compose applies the tag), and ``environment`` is not touched (its own ``!reset`` and ``!override``
    markers, and its normalization, are the merge's and the final reading's). Bind sources are not rewritten: the
    volume planner resolves a relative source of any file against the same project directory, which is where Compose
    puts it."""
    from nemo_gym.data.tasks.harbor.compose_variables import resolve_env_file_entries

    services = document.get("services") if isinstance(document, dict) else None
    if not isinstance(services, dict):
        return document
    resolve = overlay_env_file_resolver(project_dir)
    for name, service in services.items():
        if not isinstance(service, dict) or "env_file" not in service or service["env_file"] is RESET:
            continue
        entries = service["env_file"]
        resolved = resolve_env_file_entries(name, entries, resolve)
        if isinstance(entries, OverrideScalar):
            service["env_file"] = OverrideSequence(resolved)
        else:
            service["env_file"] = type(entries)(resolved) if isinstance(entries, list) else resolved
    return document


# Paths (``*`` matches any service) whose value the later file replaces whole, as compose-go's ``override`` merger.
_REPLACED = (
    ("services", "*", "command"),
    ("services", "*", "entrypoint"),
    ("services", "*", "healthcheck", "test"),
)
# Paths merged as mappings whatever their written form (compose-go's ``mergeToSequence`` families): a later key
# wins, the rest is kept.
_KEYED_MAPPINGS = (
    ("services", "*", "environment"),
    ("services", "*", "labels"),
    ("services", "*", "extra_hosts"),
    ("services", "*", "sysctls"),
    ("services", "*", "build", "args"),
    ("services", "*", "build", "labels"),
    ("services", "*", "deploy", "labels"),
    ("networks", "*", "labels"),
    ("volumes", "*", "labels"),
)
_DEPENDS_ON = ("services", "*", "depends_on")
_SERVICE_NETWORKS = ("services", "*", "networks")
# Sequences unique on a key once appended (compose-go's ``EnforceUnicity``); the later entry wins.
_UNIQUE_SEQUENCES: dict[tuple[str, ...], Callable[[Any], Any]] = {}


def _match(path: tuple[str, ...], pattern: tuple[str, ...]) -> bool:
    return len(path) == len(pattern) and all(p == "*" or p == part for part, p in zip(path, pattern))


def _port_key(entry: Any) -> Any:
    if isinstance(entry, Mapping):
        return (
            str(entry.get("host_ip") or ""),
            str(entry.get("published") or ""),
            str(entry.get("target") or ""),
            str(entry.get("protocol") or "tcp"),
        )
    text, _, protocol = str(entry).partition("/")
    parts = text.rsplit(":", 2)
    target = parts[-1]
    published = parts[-2] if len(parts) > 1 else ""
    host_ip = parts[0] if len(parts) == 3 else ""
    return (host_ip, published, target, protocol or "tcp")


def _volume_key(entry: Any) -> Any:
    if isinstance(entry, Mapping):
        return str(entry.get("target") or "")
    parts = str(entry).split(":")
    return parts[1] if len(parts) > 1 else parts[0]


def _mapping_key(field: str) -> Callable[[Any], Any]:
    def key(entry: Any) -> Any:
        return str(entry.get(field) or "") if isinstance(entry, Mapping) else str(entry)

    return key


def _text_key(entry: Any) -> Any:
    return yaml.safe_dump(entry, sort_keys=True) if isinstance(entry, (Mapping, list)) else str(entry)


for _field, _key in (
    ("ports", _port_key),
    ("volumes", _volume_key),
    ("expose", _text_key),
    ("cap_add", _text_key),
    ("cap_drop", _text_key),
    ("security_opt", _text_key),
    ("tmpfs", _text_key),
    ("links", _text_key),
    ("profiles", _text_key),
    ("dns", _text_key),
    ("dns_search", _text_key),
    ("dns_opt", _text_key),
    ("env_file", _mapping_key("path")),
    ("devices", _mapping_key("target")),
    ("secrets", _mapping_key("source")),
    ("configs", _mapping_key("source")),
):
    _UNIQUE_SEQUENCES[("services", "*", _field)] = _key


def _plain(value: Any) -> Any:
    """``value`` without markers: ``!override`` wrappers become plain mappings, sequences and strings, and a
    ``!reset`` under a key no earlier file set leaves the key absent, as Compose leaves it."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items() if item is not RESET}
    if isinstance(value, list):
        items = [_plain(item) for item in value if item is not RESET]
        return items
    if isinstance(value, OverrideScalar):
        return str(value)
    return value


_ENV_KEY = re.compile(r"^[^=]+")


def _pairs(value: Any, path: tuple[str, ...]) -> dict[str, Any]:
    """A keyed field (``environment``, ``labels``, ``extra_hosts``, ...) as a mapping, whichever form it was
    written in: a list of ``KEY=value`` (``KEY`` alone is ``None``; ``host:ip`` for ``extra_hosts``)."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, list):
        pairs: dict[str, Any] = {}
        for item in value:
            if isinstance(item, Mapping):
                pairs.update(item)
                continue
            text = str(item)
            if path[-1] == "extra_hosts":
                # The field check's precedence (``normalize_extra_hosts``): ``=`` when it comes before any ``:``, so
                # ``db=::1`` (Compose's spelling for an IPv6 address) keys on ``db``; otherwise ``:``.
                first_equals = "=" in text and (":" not in text or text.index("=") < text.index(":"))
                key, separator, rest = text.partition("=" if first_equals else ":")
            else:
                key, separator, rest = text.partition("=")
            pairs[key] = rest if separator else None
        return pairs
    raise _shape_error(path, value)


def _shape_error(path: tuple[str, ...], value: Any) -> ComposeOverlayError:
    """A field of the wrong shape, named by its path and the value's type, never the value: a mistyped
    ``environment: API_KEY=...`` would otherwise print the key in prepare's ``skipped.json`` and the seed's reason."""
    return ComposeOverlayError(f"{'.'.join(path)} must be a mapping or a list, got a {type(value).__name__}")


def _dependencies(value: Any, path: tuple[str, ...]) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, list):
        return {str(name): {"condition": "service_started"} for name in value}
    if isinstance(value, Mapping):
        return {
            str(name): (dict(options) if isinstance(options, Mapping) else options) for name, options in value.items()
        }
    raise _shape_error(path, value)


def _networks(value: Any, path: tuple[str, ...]) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, list):
        return {str(name): None for name in value}
    if isinstance(value, Mapping):
        return dict(value)
    raise _shape_error(path, value)


def _unique(items: list[Any], key: Callable[[Any], Any]) -> list[Any]:
    """``items`` with one entry per key, the last one written, in the order its key first appeared."""
    position: dict[Any, int] = {}
    unique: list[Any] = []
    for item in items:
        k = key(item)
        if k in position:
            unique[position[k]] = item
        else:
            position[k] = len(unique)
            unique.append(item)
    return unique


def _merge(base: Any, over: Any, path: tuple[str, ...]) -> Any:
    if over is RESET:
        return RESET
    if isinstance(over, (OverrideMapping, OverrideSequence, OverrideScalar)) or any(
        _match(path, p) for p in _REPLACED
    ):
        return _plain(over)
    if over is None and base is not None:
        # A key written with no value (a bare ``redis_data:`` under ``volumes`` after a file that configured it) keeps the earlier
        # value: compose-go's ``mergeYaml`` returns the existing value for a nil override, once the replaced paths above
        # (its ``override`` merger) have taken the later one.
        return base
    if any(_match(path, p) for p in _KEYED_MAPPINGS):
        merged = _pairs(base, path)
        for key, value in _pairs(over, path).items():
            if value is RESET:
                merged.pop(key, None)
            else:
                merged[key] = _plain(value)
        return merged
    if _match(path, _DEPENDS_ON):
        base, over = _dependencies(base, path), _dependencies(over, path)
    elif _match(path, _SERVICE_NETWORKS):
        base, over = _networks(base, path), _networks(over, path)
    if isinstance(over, Mapping) and (isinstance(base, Mapping) or base is None):
        merged = dict(base or {})
        for key, value in over.items():
            key = str(key)
            if value is RESET:
                merged.pop(key, None)
            elif key in merged and merged[key] is not None:
                merged[key] = _merge(merged[key], value, path + (key,))
            else:
                merged[key] = _plain(value)
        return merged
    if base is not None and any(_match(path, pattern) for pattern in _UNIQUE_SEQUENCES):
        # Compose reads the scalar form of these fields (``tmpfs: /run``, ``dns: 10.0.0.2``, ``env_file: ./x.env``) as a
        # one-entry list before it merges (compose-go's canonical transforms run on each file), so ``tmpfs: /run``
        # followed by ``tmpfs: [/tmp]`` keeps both mounts, in either order; a scalar no earlier file set stays as
        # written, which the adapter reads too.
        base = [base] if isinstance(base, str) else base
        over = [over] if isinstance(over, str) else over
    if isinstance(over, list) and isinstance(base, list):
        merged = [*base, *_plain(over)]
        for pattern, key in _UNIQUE_SEQUENCES.items():
            if _match(path, pattern):
                return _unique(merged, key)
        return merged
    return _plain(over)


def merge_compose(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """``overlay`` merged onto ``base`` as Compose merges a later ``-f`` file onto the earlier ones.

    Mappings merge key by key; sequences are appended, the keyed ones (``ports`` by host, published port,
    target and protocol; ``volumes`` by target; ``expose``, ``cap_add``, ``env_file``, ... by value) unique
    on their key with the later entry winning; ``environment``, ``labels``, ``extra_hosts``, ``sysctls`` and
    ``build.args`` merge as mappings in either written form; ``depends_on`` and a service's ``networks`` merge
    as mappings (a list becomes ``{name: {condition: service_started}}`` and ``{name: null}``); ``command``,
    ``entrypoint`` and ``healthcheck.test`` are replaced whole; any other scalar, or a value of another shape,
    is the later file's. A key written with no value in a later file (a bare ``memory_net:`` under
    ``networks`` re-declaring one an earlier file configured) keeps the earlier value, as compose-go keeps it for a nil override, except
    on the replaced paths. A ``!reset`` removes the key the earlier files set; an ``!override`` mapping or
    sequence replaces instead of merging. Neither input is modified; the result carries no markers.
    """
    merged = _merge(dict(base), dict(overlay), ())
    return _plain(merged) if isinstance(merged, dict) else {}


# -- prompt template -----------------------------------------------------------------------------------------


def _jinja() -> Any:
    """The ``jinja2`` module, imported when a prompt template is read: the Harbor environment server's own
    requirements install it (Harbor renders its templates with Jinja2 too); the loader and the harnesses that import
    this module for the merge and the MCP rule do not need it."""
    try:
        import jinja2
        import jinja2.meta  # noqa: F401 (a submodule, imported for the attribute)
    except ImportError as exc:  # pragma: no cover - depends on the installed extras
        raise ComposeOverlayError(
            "A compose_overlay prompt_template needs jinja2 (pip install 'jinja2>=3.1'; the harbor environment "
            "server's requirements include it)"
        ) from exc
    return jinja2


def validate_prompt_template(path: Path) -> str:
    """The template's text, once it parses and names ``instruction``; otherwise :class:`ComposeOverlayError`.

    Harbor's ``render_prompt_template`` refuses a template that does not use the ``instruction`` variable
    (the task's instruction would be lost) and renders with ``StrictUndefined``, so any other variable fails.
    """
    jinja2 = _jinja()
    try:
        text = Path(path).read_text()
    except OSError as exc:
        raise ComposeOverlayError(f"Prompt template {path} cannot be read: {exc}") from exc
    try:
        ast = jinja2.Environment().parse(text)
    except jinja2.TemplateSyntaxError as exc:
        raise ComposeOverlayError(f"Prompt template {path} does not parse: {exc}") from exc
    if PROMPT_TEMPLATE_VARIABLE not in jinja2.meta.find_undeclared_variables(ast):
        raise ComposeOverlayError(
            f"Prompt template {path} must include an '{PROMPT_TEMPLATE_VARIABLE}' variable: use "
            f"{{{{ {PROMPT_TEMPLATE_VARIABLE} }}}} in the template"
        )
    return text


def compile_prompt_template(path: Path) -> Any:
    """The template at ``path`` compiled once it passes :func:`validate_prompt_template`: a ``jinja2.Template`` from
    an environment with ``StrictUndefined``, as Harbor's ``render_prompt_template`` builds one per render. The block
    compiles its template when it is read and renders the compiled one per episode, so the file is read and parsed
    once, not twice per episode on the event loop."""
    jinja2 = _jinja()
    text = validate_prompt_template(path)
    return jinja2.Environment(undefined=jinja2.StrictUndefined).from_string(text)


def render_compiled_template(template: Any, path: Path, instruction: str) -> str:
    """``instruction`` rendered into ``template`` (:func:`compile_prompt_template`), ``path`` named in the errors:
    ``instruction`` is the one variable, so a variable the template names and nothing defines raises
    :class:`ComposeOverlayError` (``StrictUndefined``), as under Harbor."""
    jinja2 = _jinja()
    try:
        return template.render(**{PROMPT_TEMPLATE_VARIABLE: instruction})
    except jinja2.UndefinedError as exc:
        raise ComposeOverlayError(f"Prompt template {path} has undefined variables: {exc}") from exc
    except Exception as exc:
        raise ComposeOverlayError(f"Error rendering prompt template {path}: {exc}") from exc


def render_prompt_template(path: Path, instruction: str) -> str:
    """``instruction`` rendered into the template at ``path``, as Harbor renders it (``render_prompt_template`` in
    ``harbor/utils/templating.py`` at v0.22.0): Jinja2 with ``StrictUndefined``, ``instruction`` the one
    variable. Reads and compiles the file on every call; the server renders a block's compiled template instead
    (:meth:`HarborComposeOverlay.render_instruction`)."""
    return render_compiled_template(compile_prompt_template(path), path, instruction)


def wrap_first_instruction(responses_create_params: Mapping[str, Any], render: Callable[[str], str]) -> dict[str, Any]:
    """A copy of ``responses_create_params`` with ``render`` applied to the instruction: the first user message's
    text (a string content, or the text of its first ``input_text`` part). The row's instruction is that
    message, as the loader materializes it; a request without one raises :class:`ComposeOverlayError`."""
    params = deepcopy(dict(responses_create_params))
    items = params.get("input")
    if isinstance(items, str):
        params["input"] = render(items)
        return params
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, Mapping) or item.get("role") != "user":
            continue
        content = item.get("content")
        if isinstance(content, str):
            item["content"] = render(content)
            return params
        for part in content if isinstance(content, list) else []:
            if (
                isinstance(part, Mapping)
                and part.get("type") in ("input_text", "text")
                and isinstance(part.get("text"), str)
            ):
                part["text"] = render(part["text"])
                return params
    raise ComposeOverlayError("the request carries no user instruction to render the prompt template around")
