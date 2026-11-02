# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic mirror of Harbor's ``task.toml`` (schema 1.4).

Field names are Harbor's. Deprecated spellings (``version``, ``memory``, ``storage``,
``allow_internet``) are read and normalized but never written back; the pre-1.3 ``multi_step_reward_mode``
is accepted and dropped, as Harbor drops it. Keys this mirror does not know are ignored, as Harbor ignores
them; :meth:`HarborSettings.unknown_keys` lists them so the loader can say so.
"""

import re
from pathlib import PurePosixPath
from typing import Any, ClassVar, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator


class HarborSettings(BaseModel):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)

    # Keys a `before` validator reads and renames; they are accepted, not unknown.
    deprecated_keys: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def unknown_keys(cls, data: Any, prefix: str = "") -> list[str]:
        """Dotted paths of the keys in ``data`` that no field of this model tree declares."""
        if not isinstance(data, dict):
            return []
        unknown: list[str] = []
        for key, value in data.items():
            field = cls.model_fields.get(key)
            if field is None:
                if key not in cls.deprecated_keys:
                    unknown.append(f"{prefix}{key}")
                continue
            nested = _nested_settings(field.annotation)
            if nested is None:
                continue
            if isinstance(value, list):
                for index, item in enumerate(value):
                    unknown += nested.unknown_keys(item, f"{prefix}{key}[{index}].")
            else:
                unknown += nested.unknown_keys(value, f"{prefix}{key}.")
        return unknown


def _nested_settings(annotation: Any) -> type[HarborSettings] | None:
    if isinstance(annotation, type) and issubclass(annotation, HarborSettings):
        return annotation
    for arg in get_args(annotation):
        nested = _nested_settings(arg)
        if nested is not None:
            return nested
    return None


class HarborHealthcheck(HarborSettings):
    """Mirrors Harbor's ``HealthcheckConfig``: plain numbers with Harbor's defaults and no bounds.

    Published tasks pin values at the lower bound (``retries = 0``: one attempt once the start
    period ends), and Harbor loads them; bounding the fields here rejected those tasks.
    """

    command: str
    interval_sec: float = 5.0
    timeout_sec: float = 30.0
    start_period_sec: float = 0.0
    start_interval_sec: float = 5.0
    retries: int = 3


class HarborMCPServer(HarborSettings):
    name: str
    transport: Literal["stdio", "sse", "streamable-http"] = "sse"
    url: str | None = None
    command: str | None = None
    args: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_transport(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("transport") == "http":
            data = {**data, "transport": "streamable-http"}
        return data

    @model_validator(mode="after")
    def require_endpoint(self) -> "HarborMCPServer":
        if not (self.command if self.transport == "stdio" else self.url):
            raise ValueError(f"MCP server {self.name!r}: stdio needs `command`, other transports need `url`")
        return self


class HarborPhaseSettings(HarborSettings):
    network_mode: Literal["public", "no-network", "allowlist"] | None = None
    allowed_hosts: list[str] | None = None


def _size_to_mb(raw: str) -> int:
    text = raw.strip().upper()
    units = {"G": 1024, "M": 1, "K": 1 / 1024}
    if not text or text[-1] not in units:
        raise ValueError(f"Cannot parse size {raw!r}; expected a number followed by K, M or G")
    return int(float(text[:-1]) * units[text[-1]])


class HarborEnvironment(HarborPhaseSettings):
    """``[environment]``: the sandbox the agent works in.

    ``docker_image`` is optional. A task without one builds from
    ``environment/Dockerfile`` or, when that file only names a base image, runs the
    base image directly.
    """

    deprecated_keys: ClassVar[frozenset[str]] = frozenset({"allow_internet", "memory", "storage"})

    docker_image: str | None = None
    build_timeout_sec: float = Field(default=600, gt=0)
    os: Literal["linux", "windows"] = "linux"
    cpus: int | None = None
    memory_mb: int | None = None
    storage_mb: int | None = None
    gpus: int | None = None
    gpu_types: list[str] | None = None
    env: dict[str, str] = Field(default_factory=dict)
    mcp_servers: list[HarborMCPServer] = Field(default_factory=list)
    skills_dir: str | None = None
    healthcheck: HarborHealthcheck | None = None
    workdir: str | None = None
    network_mode: Literal["public", "no-network", "allowlist"] = "public"

    @model_validator(mode="before")
    @classmethod
    def read_deprecated_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if isinstance(data.get("os"), str):
            data["os"] = data["os"].lower()
        if data.get("tpu") is not None:
            raise ValueError("TPU environments are not supported yet")
        allow = data.pop("allow_internet", None)
        if allow is not None and data.get("allowed_hosts") is None:
            data.setdefault("network_mode", "public" if allow else "no-network")
        for old, new in (("memory", "memory_mb"), ("storage", "storage_mb")):
            if old in data:
                raw = data.pop(old)
                # Harbor drops non-string legacy sizes rather than treating them as MB.
                if not isinstance(raw, str):
                    continue
                size = _size_to_mb(raw)
                if new in data and data[new] != size:
                    raise ValueError(f"Conflicting `{old}` and `{new}`")
                data[new] = size
        return data


class HarborAgent(HarborPhaseSettings):
    timeout_sec: float = Field(default=28800, gt=0)
    user: str | int | None = None


class HarborCollectHook(HarborSettings):
    """``[[verifier.collect]]``: a command run in a service before its artifacts are collected."""

    command: str
    service: str = "main"
    timeout_sec: float = Field(default=60, gt=0)
    user: str | int | None = None


# The files the shared verifier's own run leaves in /logs/verifier (the reward, the exit record, the wrapper's
# stdout): never a sidecar input, so `[verifier].inputs` cannot name them.
VERIFIER_OUTPUT_NAMES = ("reward.json", "reward.txt", "test-stdout.txt", ".ng-exit")


class HarborVerifier(HarborPhaseSettings):
    """``[verifier]``. ``inputs`` is a Gym extension Harbor has no field for, set from the dataset's ``dataset.toml``
    (``[gym.tasks."<task_id>".verifier] inputs = [...]``): the names of files under ``/logs/verifier`` a Compose
    sidecar writes during the rollout for ``test.sh`` to read (kumo's ``verifier`` service appends ``actions.jsonl``
    for ``main``'s ``tests/score.py``). The server empties ``/logs/verifier`` before a shared verification and
    before the next turn of a multi-step task; these names alone survive, as regular files, so an agent-planted
    report or reward is never left for the grader. Plain file names: no path separators, never the grader's own
    outputs (:data:`VERIFIER_OUTPUT_NAMES`)."""

    timeout_sec: float = Field(default=600, gt=0)
    user: str | int | None = None
    env: dict[str, str] = Field(default_factory=dict)
    environment_mode: Literal["separate", "shared"] | None = None
    environment: HarborEnvironment | None = None
    collect: list[HarborCollectHook] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def shared_has_no_environment(self) -> "HarborVerifier":
        if self.environment_mode == "shared" and self.environment is not None:
            raise ValueError("A shared verifier cannot declare its own [verifier.environment]")
        return self

    @model_validator(mode="after")
    def inputs_are_file_names(self) -> "HarborVerifier":
        for name in self.inputs:
            if not name or name in (".", "..") or "/" in name or "\\" in name or "\0" in name:
                raise ValueError(f"[verifier].inputs entry {name!r} must be a plain file name under /logs/verifier")
            # The server keeps the name with `find -path`, which matches it as a glob: a literal name only, or a
            # declared `log[1].jsonl` would be deleted and an undeclared `log1.jsonl` kept.
            if any(char in name for char in "*?["):
                raise ValueError(
                    f"[verifier].inputs entry {name!r} must be a literal file name, not a glob pattern (no *, ? or [)"
                )
            if name in VERIFIER_OUTPUT_NAMES:
                raise ValueError(f"[verifier].inputs entry {name!r} is the grader's own output, not a sidecar input")
        if len(set(self.inputs)) != len(self.inputs):
            raise ValueError(f"[verifier].inputs names a file twice: {self.inputs}")
        return self


class HarborSolution(HarborSettings):
    env: dict[str, str] = Field(default_factory=dict)


# The verifier's own directories in the sandbox: the uploaded tests and where test.sh writes its reward. An
# artifact there would hand the agent the grader's files when a separate verifier restores it, so the loader
# refuses it (and the environment server refuses the same paths again at the restore). /logs is not listed:
# it holds the agent's own folders too, and the server empties /logs/verifier again after restoring it.
PROTECTED_ARTIFACT_SOURCES = ("/tests", "/logs/verifier")


class HarborArtifact(HarborSettings):
    """One ``artifacts`` entry: a path copied out of the agent's sandbox after it finishes.

    A bare string is the ``source``. ``destination`` is where it lands relative to the run's
    artifacts folder (default: the source path without its leading slash). ``service`` names a
    Compose sidecar; ``None`` or ``"main"`` is the agent's own container.
    """

    source: str
    destination: str | None = None
    service: str | None = None
    exclude: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def from_string(cls, data: Any) -> Any:
        return {"source": data} if isinstance(data, str) else data

    @model_validator(mode="after")
    def contained_paths(self) -> "HarborArtifact":
        for path in (self.source, self.destination):
            if path and (".." in PurePosixPath(path).parts or "\\" in path):
                raise ValueError(f"Artifact paths must stay inside the sandbox and the artifacts folder: {path!r}")
        if self.destination and PurePosixPath(self.destination).is_absolute():
            raise ValueError(f"Artifact destination must be relative: {self.destination!r}")
        if self.source.startswith("/"):
            # One leading slash: POSIX lets "//x" name something other than "/x", and the server compares the
            # sandbox paths it guards by their text, so the spelling is settled here (a trailing slash stays).
            self.source = "/" + self.source.lstrip("/")
        source = PurePosixPath(self.source)
        tests, verifier_logs = (PurePosixPath(path) for path in PROTECTED_ARTIFACT_SOURCES)
        if source.is_relative_to(tests) or tests.is_relative_to(source) or source.is_relative_to(verifier_logs):
            replaced = tests if source.is_relative_to(tests) or tests.is_relative_to(source) else verifier_logs
            raise ValueError(
                f"Artifact source {self.source!r} would replace the verifier's {replaced} in a separate verifier "
                "and cannot be restored there"
            )
        return self

    @property
    def host_path(self) -> PurePosixPath:
        return PurePosixPath(self.destination or self.source.lstrip("/"))


_STEP_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class HarborTaskStepAgent(HarborPhaseSettings):
    """``[steps.agent]``: unset values fall back to the task's ``[agent]``, as in Harbor."""

    timeout_sec: float | None = Field(default=None, gt=0)
    user: str | int | None = None


# Harbor's `VerifierConfig.timeout_sec` default, which a step's verifier gets when `[steps.verifier]` names none:
# `StepConfig.verifier` is a full `VerifierConfig` (0.22.0 and main alike), so the task's `[verifier].timeout_sec`
# never reaches a step (`_step_verifier_timeout_sec`'s fallback to it is dead code there).
HARBOR_STEP_VERIFIER_TIMEOUT_SEC = 600.0


class HarborTaskStepVerifier(HarborPhaseSettings):
    """``[steps.verifier]``: ``user``, ``env``, ``environment_mode`` and ``environment`` fall back to the task's
    ``[verifier]`` as Harbor resolves them (``_step_verifier_user``, ``resolve_step_verifier_mode``,
    ``resolve_effective_verifier_env_config``). ``timeout_sec`` does not: Harbor's step verifier is a
    ``VerifierConfig`` of its own with the 600 s default, so a step that names none is graded within
    :data:`HARBOR_STEP_VERIFIER_TIMEOUT_SEC`, whatever the task's ``[verifier].timeout_sec`` says
    (:meth:`HarborTaskConfig.step_verifier_timeout_sec`; the loader warns when the two differ). ``collect`` is the
    step's own ``[[steps.verifier.collect]]`` hooks, run after the task's in the step's collection pass (Harbor's
    ``_collect_hooks_for``: the task's hooks, then the step's). A shared step cannot declare its own environment, as
    the task's verifier cannot (Harbor's ``_validate_mode_env_consistency`` is on the class both share)."""

    timeout_sec: float | None = Field(default=None, gt=0)
    user: str | int | None = None
    env: dict[str, str] = Field(default_factory=dict)
    environment_mode: Literal["separate", "shared"] | None = None
    environment: HarborEnvironment | None = None
    collect: list[HarborCollectHook] = Field(default_factory=list)

    @model_validator(mode="after")
    def shared_has_no_environment(self) -> "HarborTaskStepVerifier":
        if self.environment_mode == "shared" and self.environment is not None:
            raise ValueError(
                "[steps.verifier].environment_mode='shared' is incompatible with [steps.verifier.environment]; "
                "omit the environment or set environment_mode='separate'"
            )
        return self


class HarborTaskStep(HarborSettings):
    """One ``[[steps]]`` entry of ``task.toml`` (Harbor's ``StepConfig``): a step of the task, which the task
    author defines and grades, not a step of a trajectory (``nemo_gym.trajectory.atif_v1_7.AtifStep``).

    The loader pairs it with its ``steps/<name>/`` folder into a ``HarborTurn``, as ``HarborTaskConfig`` pairs
    with its folder into a ``HarborTask``. The step has its own instruction (``steps/<name>/instruction.md``)
    and verifier (the task's ``tests/`` with ``steps/<name>/tests/`` on top). ``min_reward`` stops the episode
    after this step when its rewards fall short: a number gates the ``reward`` key, a mapping gates each named
    key, and a step without a verifier result fails every gate.
    """

    name: str
    agent: HarborTaskStepAgent = Field(default_factory=HarborTaskStepAgent)
    verifier: HarborTaskStepVerifier = Field(default_factory=HarborTaskStepVerifier)
    min_reward: float | dict[str, float] | None = None
    healthcheck: HarborHealthcheck | None = None
    artifacts: list[HarborArtifact] = Field(default_factory=list)

    @model_validator(mode="after")
    def portable_name(self) -> "HarborTaskStep":
        if not _STEP_NAME.fullmatch(self.name) or self.name in (".", ".."):
            raise ValueError(f"Step name {self.name!r} must be a portable directory name under steps/")
        return self


class HarborTaskConfig(HarborSettings):
    """One ``task.toml``."""

    # `multi_step_reward_mode` is the pre-1.3 spelling older multi-step generators (the MemoryArena adapters
    # among them) write as "aggregate". Harbor 0.22.0 has no such field: `TaskConfig` drops the key and the
    # task rolls up with the default `mean`, or its own `multi_step_reward_strategy`. The mirror does the same
    # through `extra="ignore"`; naming the key here only keeps the loader from reporting it as unknown (the
    # loader warns about a value other than "aggregate", which Harbor drops silently).
    deprecated_keys: ClassVar[frozenset[str]] = frozenset({"version", "multi_step_reward_mode"})

    schema_version: str = "1.4"
    task: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str | None = None
    solution: HarborSolution = Field(default_factory=HarborSolution)
    agent: HarborAgent = Field(default_factory=HarborAgent)
    environment: HarborEnvironment = Field(default_factory=HarborEnvironment)
    verifier: HarborVerifier = Field(default_factory=HarborVerifier)
    artifacts: list[HarborArtifact] = Field(default_factory=list)
    # A task with steps runs them in order as turns in one sandbox; ``mean`` is Harbor's default roll-up.
    steps: list[HarborTaskStep] = Field(default_factory=list)
    multi_step_reward_strategy: Literal["mean", "final"] = "mean"
    # The dataset's `[gym.defaults].timeout_multiplier` once `apply_dataset_config` has scaled this task's budgets by
    # it (not a task.toml key): the default a step inherits for a budget it does not name reads scaled by it too,
    # like the budgets the task names (`step_verifier_timeout_sec`).
    _timeout_multiplier: float = PrivateAttr(default=1.0)

    @model_validator(mode="before")
    @classmethod
    def read_deprecated_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if "version" in data:
            data.setdefault("schema_version", str(data.pop("version")))
        return data

    @model_validator(mode="after")
    def unique_step_names(self) -> "HarborTaskConfig":
        names = [step.name for step in self.steps]
        if len(names) != len(set(names)):
            raise ValueError(f"Step names must be unique: {names}")
        return self

    @property
    def is_multi_turn(self) -> bool:
        return bool(self.steps)

    def step_agent_timeout_sec(self, step: HarborTaskStep) -> float:
        return step.agent.timeout_sec if step.agent.timeout_sec is not None else self.agent.timeout_sec

    def step_verifier_timeout_sec(self, step: HarborTaskStep) -> float:
        """The step's own ``[steps.verifier].timeout_sec``, else Harbor's step default of 600 s, never the task's
        ``[verifier].timeout_sec`` (:data:`HARBOR_STEP_VERIFIER_TIMEOUT_SEC`)."""
        if step.verifier.timeout_sec is not None:
            return step.verifier.timeout_sec
        # Harbor's `_step_verifier_timeout_sec` feeds that default through `_resolve_timeout_sec`, so it scales by
        # the dataset's `timeout_multiplier` like every budget the task names (`apply_dataset_config` carries it).
        return HARBOR_STEP_VERIFIER_TIMEOUT_SEC * self._timeout_multiplier

    @property
    def is_shared_verifier(self) -> bool:
        """Harbor runs the verifier in the agent's container unless the task says otherwise.

        As in Harbor, ``[verifier].environment_mode = "separate"`` or a ``[verifier.environment]``
        block selects the separate mode, where a fresh container receives ``/logs/artifacts`` and
        the declared ``artifacts``. A verifier is shared unless the mode is ``"separate"`` or a
        ``[verifier.environment]`` block is declared.
        """
        return self.verifier.environment_mode != "separate" and self.verifier.environment is None
