# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""`gym sandbox` — boot, inspect, and poke at a server's task sandbox.

Reproducing "what does this task's container actually look like?" normally means
starting the whole stack: model server, Ray, agent, rollout. These commands cut
straight to the sandbox, using the same provider, image, and spec a rollout would.
"""

import asyncio
import atexit
import json
import logging
import random
import signal
import time
import uuid
import weakref
from collections.abc import Mapping
from contextlib import suppress
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import rich
from omegaconf import DictConfig, OmegaConf
from pydantic import Field, model_validator
from rich.markup import escape

from nemo_gym import _resolve_under_cwd_or_install
from nemo_gym.cli.utils import did_you_mean, exit_cleanly_on_config_error
from nemo_gym.config_types import BaseNeMoGymCLIConfig, ConfigError, ConfigPathNotFoundError
from nemo_gym.global_config import (
    JSON_OUTPUT_KEY_NAME,
    NEMO_GYM_RESERVED_TOP_LEVEL_KEYS,
    VERBOSE_KEY_NAME,
    GlobalConfigDictParser,
    GlobalConfigDictParserConfig,
    get_global_config_dict,
)
from nemo_gym.sandbox import (
    AsyncSandbox,
    ConnectableProvider,
    RenewableProvider,
    SandboxSpec,
    create_provider,
    resolve_provider_config,
    resolve_provider_metadata,
)
from nemo_gym.sandbox.hooks import replace_image, resolve_spec, task_id_for_row, wrap_command


# Outcome taxonomy for a debug action. `not_run` covers a sandbox that booted but
# whose action never started (an earlier stage failed).
REASON_PASS = "pass"
REASON_NONZERO_EXIT = "nonzero_exit"
REASON_TIMEOUT = "timeout"
REASON_CANCELLED = "cancelled"
REASON_ERROR = "error"
REASON_NOT_RUN = "not_run"

# Uploaded scripts land here rather than in the task workdir: the point of debug
# is to observe the task's state, not to add to it.
REMOTE_SCRIPT_DIR = "/tmp"

# A provider-enforced timeout returns slightly under the requested budget once
# round-trip time is subtracted, so allow a little slack before calling it one.
_TIMEOUT_TOLERANCE = 0.9

# Default ceiling on how long a debugging sandbox lives. Server configs size
# `ttl_s` for a full rollout — mini-swe-agent asks for five hours — which is the
# wrong order of magnitude for poking at a container, and several servers set no
# ttl at all, which providers read as "never auto-terminate". Neither is safe for
# a tool that can be Ctrl-C'd, killed, or simply forgotten after `--keep`, so both
# are capped here. `--ttl` opts out for the cases that genuinely need longer.
DEFAULT_TTL_S = 900.0

# Above this, an explicit `--ttl` is worth questioning: sandboxes held this long
# are usually forgotten rather than needed.
TTL_WARN_THRESHOLD_S = 3600.0

_DEFAULT_OUTPUT_ROOT = "outputs"

# Sandboxes that must be torn down even if the loop dies under us. Weak so a
# stopped sandbox can still be collected normally.
_LIVE_SANDBOXES: "weakref.WeakSet[AsyncSandbox]" = weakref.WeakSet()
_cleaning_up = False


def _cleanup_at_exit() -> None:  # pragma: no cover - interpreter shutdown path
    """Last-resort teardown for sandboxes still alive at interpreter exit.

    By this point the event loop is usually gone, so a fresh one is spun up per
    sandbox. Failures are swallowed: this runs during shutdown, where raising
    would only mask whatever actually went wrong.
    """
    for sandbox in list(_LIVE_SANDBOXES):
        with suppress(Exception):
            asyncio.run(sandbox.stop())


atexit.register(_cleanup_at_exit)


def _install_interrupt_handler() -> None:  # pragma: no cover - signal path
    """Make the first Ctrl-C unwind cleanly and ignore the ones after it.

    A second Ctrl-C during teardown would orphan a running sandbox — which on a
    cluster keeps burning capacity until its TTL expires. Swallowing repeats is
    what makes "cleaning up, please wait" an honest message.
    """

    def handler(signum, frame):
        global _cleaning_up
        if _cleaning_up:
            rich.print("[yellow]cleanup in progress — please wait (ctrl-c ignored)[/yellow]")
            return
        _cleaning_up = True
        rich.print("[yellow]interrupted — cleaning up, please wait...[/yellow]")
        raise KeyboardInterrupt

    with suppress(ValueError):  # not on the main thread
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)


########################################
# Config models
########################################


class SandboxDebugConfig(BaseNeMoGymCLIConfig):
    """
    Boot the sandbox a server would create for a task and run a command inside it.

    Resolves the provider and spec exactly as the server does, so what you poke at is
    the environment the rollout sees. With no `-i/--input`, it boots the server's
    configured sandbox; with one, it binds a dataset row first.

    Examples:

    ```bash
    # Inspect without provisioning anything
    gym sandbox debug --config <config> --task <id> --dry-run

    # Run a command in one task's sandbox
    gym sandbox debug --config <config> --task <id> --command "pytest -x"

    # Upload and run a script, leaving the sandbox up afterwards
    gym sandbox debug --config <config> --script-path ./poke.sh --keep
    ```
    """

    server_name: Optional[str] = Field(
        default=None, description="Server whose sandbox to boot. Inferred when only one is configured."
    )
    sandbox_name: Optional[str] = Field(default=None, description="Override the server's sandbox_provider reference.")
    input_jsonl_fpath: Optional[str] = Field(
        default=None, description="Task rows. Defaults to the server's example dataset when a task is selected."
    )
    task_ids: List[str] = Field(default_factory=list, description="Select rows by task id; repeatable.")
    limit: Optional[int] = Field(default=None, description="Maximum number of tasks to run.")
    shuffle: bool = Field(default=False, description="Shuffle rows (fixed seed) before applying --limit.")

    command: Optional[str] = Field(default=None, description="Command to run inside the sandbox.")
    script_path: Optional[str] = Field(default=None, description="Local script to upload and run.")
    dry_run: bool = Field(default=False, description="Print the resolved sandbox and exit without provisioning.")
    list_tasks: bool = Field(default=False, description="List the task ids in the dataset and exit.")

    image: Optional[str] = Field(default=None, description="Override the resolved container image.")
    bare: bool = Field(default=False, description="Skip the server's exec_wrapper and run the command as typed.")
    cwd: Optional[str] = Field(default=None, description="Working directory for the command.")
    user: Optional[str] = Field(default=None, description="User to run the command as.")
    env: List[str] = Field(default_factory=list, description="Extra environment variables as KEY=VALUE.")

    concurrency: int = Field(default=4, description="Maximum sandboxes in flight.")
    timeout_setup: Optional[float] = Field(default=None, description="Seconds allowed for boot and upload.")
    timeout_total: Optional[float] = Field(default=600.0, description="Seconds allowed for the command itself.")
    ttl_s: Optional[float] = Field(
        default=None,
        description=f"Sandbox lifetime in seconds. Defaults to the server's ttl_s, or {DEFAULT_TTL_S:g}s if it sets none.",
    )

    keep: bool = Field(default=False, description="Leave sandboxes running and print their ids.")
    output_dirpath: Optional[str] = Field(default=None, description="Where to write config.yaml and traces.jsonl.")
    quiet: bool = Field(default=False, description="Suppress command stdout/stderr; keep the summary.")
    exit_zero: bool = Field(default=False, description="Always exit 0, even when the command failed.")

    @model_validator(mode="after")
    def check_action(self) -> "SandboxDebugConfig":
        if self.command and self.script_path:
            raise ValueError("pass at most one of --command or --script-path")
        # --dry-run is a modifier, not an action: passing a command with it shows
        # how that command would be wrapped, which is half the point of inspecting.
        if not (self.dry_run or self.list_tasks or self.command or self.script_path):
            raise ValueError("pass --command, --script-path, --dry-run, or --list-tasks")
        if self.script_path and not Path(self.script_path).is_file():
            raise ValueError(f"--script-path is not a file: {self.script_path}")
        if self.concurrency < 1:
            raise ValueError("--concurrency must be >= 1")
        if self.ttl_s is not None and self.ttl_s <= 0:
            # Providers impose their own floor (OpenSandbox requires 60s); only the
            # universally-wrong case is rejected here, so their message still shows.
            raise ValueError("--ttl must be > 0")
        return self


class SandboxExecConfig(BaseNeMoGymCLIConfig):
    """
    Run a command in an already-running sandbox, or delete one.

    Pairs with `gym sandbox debug --keep`, which prints the sandbox id. Requires a
    provider that can reattach by id.

    Examples:

    ```bash
    gym sandbox exec --config <config> --sandbox-id <id> --command "git log -1"
    gym sandbox rm --config <config> --sandbox-id <id>
    ```
    """

    sandbox_id: Optional[str] = Field(default=None, description="Id of the running sandbox.")
    server_name: Optional[str] = Field(default=None, description="Server whose provider config to connect with.")
    sandbox_name: Optional[str] = Field(default=None, description="Override the server's sandbox_provider reference.")
    command: Optional[str] = Field(default=None, description="Command to run (not used by `rm`).")
    bare: bool = Field(default=False, description="Skip the server's exec_wrapper.")
    cwd: Optional[str] = Field(default=None, description="Working directory for the command.")
    user: Optional[str] = Field(default=None, description="User to run the command as.")
    timeout_total: Optional[float] = Field(default=600.0, description="Seconds allowed for the command.")
    ttl_s: Optional[float] = Field(
        default=None,
        description=f"Seconds to extend the sandbox's expiry by before running (default {DEFAULT_TTL_S:g}s).",
    )
    quiet: bool = Field(default=False, description="Suppress stdout/stderr; keep the summary.")
    exit_zero: bool = Field(default=False, description="Always exit 0, even when the command failed.")


########################################
# Server / provider / spec resolution
########################################


def _redacted_config_yaml(global_config_dict: Any) -> str:
    """Serialize the resolved config with secrets masked.

    This file is written to a run directory that gets shared and attached to bug
    reports, so a resolved API key must not land in it. Values come from the
    environment anyway, so re-running from this config needs the same env vars
    set — which is the trade worth making.
    """
    redacted = deepcopy(global_config_dict)
    GlobalConfigDictParser()._recursively_hide_secrets(redacted)
    return OmegaConf.to_yaml(redacted, resolve=True)


def _warn_on_long_ttl(ttl_s: Optional[float]) -> None:
    """Flag a lifetime long enough that a forgotten sandbox becomes expensive."""
    if ttl_s is not None and ttl_s > TTL_WARN_THRESHOLD_S:
        rich.print(
            f"[yellow]warning[/yellow] --ttl {_format_duration(ttl_s)} keeps a sandbox alive well past a "
            f"normal debugging session; a forgotten one holds cluster capacity for that long. Delete it with "
            f"`gym sandbox rm --sandbox-id <id>` when you are done."
        )


def _quiet_transport_logs(global_config_dict: Any) -> None:
    """Silence per-request HTTP chatter unless `-v` was passed.

    The sandbox SDK logs every proxied call at INFO, which buries the command
    output this tool exists to show. Under `-v` the traffic is exactly what you
    want, so it is only suppressed at the default level.
    """
    if global_config_dict.get(VERBOSE_KEY_NAME):
        return
    for name in ("httpx", "httpcore", "opensandbox"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _plain(value: Any) -> Any:
    """Return a plain Python container for an OmegaConf node."""
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value


def discover_sandbox_servers(global_config_dict: Any) -> Dict[str, Dict[str, Any]]:
    """Find every server block that declares a sandbox.

    `sandbox_provider` is the one thing every sandbox-backed server agrees on, so
    it is what identifies them — regardless of whether they are agents or
    resources servers. The descent mirrors how `gym env start` locates servers.

    Returns:
        Mapping of server instance name to
        ``{"top_level": str, "server_type": str, "config": dict}``.
    """
    servers: Dict[str, Dict[str, Any]] = {}
    for top_level in global_config_dict:
        if top_level in NEMO_GYM_RESERVED_TOP_LEVEL_KEYS:
            continue
        by_type = global_config_dict[top_level]
        if not isinstance(by_type, DictConfig):
            continue
        for server_type in by_type:
            by_name = by_type[server_type]
            if not isinstance(by_name, DictConfig):
                continue
            for name in by_name:
                block = by_name[name]
                if not isinstance(block, DictConfig) or "sandbox_provider" not in block:
                    continue
                servers[str(name)] = {
                    "name": str(name),
                    "top_level": str(top_level),
                    "server_type": str(server_type),
                    "config": _plain(block),
                }
    return servers


def select_server(servers: Dict[str, Dict[str, Any]], server_name: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    """Pick the server to debug, inferring it when the choice is unambiguous."""
    if not servers:
        raise ConfigError(
            "No sandbox-backed server found in the merged config. A server opts in by declaring "
            "`sandbox_provider`; pass its config with --config."
        )
    if server_name:
        if server_name not in servers:
            raise ConfigError(
                f"Unknown server {server_name!r}. Configured sandbox servers: "
                f"{', '.join(sorted(servers))}.{did_you_mean(server_name, servers)}"
            )
        return server_name, servers[server_name]
    if len(servers) > 1:
        raise ConfigError(
            f"Several sandbox servers are configured ({', '.join(sorted(servers))}). Pick one with --server."
        )
    name = next(iter(servers))
    return name, servers[name]


def build_provider(server_config: Dict[str, Any], global_config_dict: Any, sandbox_name: Optional[str]) -> Any:
    """Instantiate the provider a server would use.

    `sandbox_provider` is either a name referencing a provider block elsewhere in
    the config or an inline single-key mapping; `resolve_provider_config` accepts
    both. `--sandbox` swaps the reference so the same task can run on a local
    provider and a cluster one without editing committed config.
    """
    reference = sandbox_name or server_config.get("sandbox_provider")
    if reference is None:
        raise ConfigError("Server does not declare `sandbox_provider`.")
    provider_config = resolve_provider_config(reference, global_config_dict)
    return create_provider(provider_config), provider_config


def provider_default_metadata(server_config: Dict[str, Any], global_config_dict: Any, sandbox_name: Optional[str]):
    reference = sandbox_name or server_config.get("sandbox_provider")
    return resolve_provider_metadata(reference, global_config_dict)


def _exec_kwargs(server_config: Dict[str, Any], config: Any) -> Dict[str, Any]:
    """Merge the server's exec defaults with CLI overrides.

    `sandbox_environment_kwargs` is where servers keep the shell context their
    commands assume (workdir, user, conda env), so it doubles as the argument set
    for an `exec_wrapper`.
    """
    kwargs = dict(server_config.get("sandbox_environment_kwargs") or {})
    if getattr(config, "cwd", None):
        kwargs["cwd"] = config.cwd
    if getattr(config, "user", None):
        kwargs["user"] = config.user
    return kwargs


def _parse_env(pairs: List[str]) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for pair in pairs:
        key, separator, value = pair.partition("=")
        if not separator or not key:
            raise ConfigError(f"--env expects KEY=VALUE, got {pair!r}")
        env[key] = value
    return env


########################################
# Task rows
########################################


def _default_input_fpath(server_config: Dict[str, Any], entry: Dict[str, Any]) -> Optional[str]:
    """Fall back to the server's example dataset when no input file is given.

    A declared `example` dataset wins; otherwise fall back to the committed
    `data/example.jsonl` every server is required to ship, which is what makes
    `--task <id>` work with no `-i`.
    """
    for dataset in server_config.get("datasets") or []:
        if isinstance(dataset, dict) and dataset.get("type") == "example" and dataset.get("jsonl_fpath"):
            return str(dataset["jsonl_fpath"])

    conventional = Path(entry["server_type"]) / entry["name"] / "data" / "example.jsonl"
    with suppress(Exception):
        if _resolve_under_cwd_or_install(conventional).exists():
            return str(conventional)
    return None


def load_rows(
    input_jsonl_fpath: Optional[str],
    *,
    entry: Dict[str, Any],
    task_ids: List[str],
    limit: Optional[int],
    shuffle: bool,
    id_from_row: Optional[str],
    require_selection: bool = False,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Load and select the dataset rows to debug.

    Returns ``([], None)`` when nothing selects a row, which is the signal to boot
    the server's configured sandbox unbound to any task. ``require_selection``
    forces the dataset to be read anyway, for callers that want the whole thing.
    """
    if not require_selection and input_jsonl_fpath is None and not task_ids and limit is None:
        return [], None

    fpath = input_jsonl_fpath or _default_input_fpath(entry["config"], entry)
    if fpath is None:
        raise ConfigError(
            "Selecting tasks needs a dataset. Pass -i/--input, or configure an `example` dataset on the server."
        )

    resolved = _resolve_under_cwd_or_install(Path(fpath))
    if not resolved.exists():
        raise ConfigPathNotFoundError(f"Input file not found: '{fpath}' (-i/--input).")

    rows: List[Dict[str, Any]] = []
    with open(resolved) as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ConfigError(f"Malformed JSON in '{resolved}' at line {line_no}: {e}") from e

    if task_ids:
        wanted = set(task_ids)
        selected = [r for r in rows if task_id_for_row(r, id_from_row=id_from_row) in wanted]
        found = {task_id_for_row(r, id_from_row=id_from_row) for r in selected}
        missing = wanted - found
        if missing:
            available = [task_id_for_row(r, id_from_row=id_from_row) or "<unnamed>" for r in rows]
            raise ConfigError(
                f"No row matched --task {', '.join(sorted(missing))} in '{resolved}'. "
                f"Available ids: {', '.join(available[:20])}{' ...' if len(available) > 20 else ''}"
            )
        rows = selected

    if shuffle:
        random.Random(0).shuffle(rows)
    if limit is not None:
        rows = rows[:limit]
    return rows, str(resolved)


########################################
# Planning
########################################


class TaskPlan:
    """Everything needed to boot and use one sandbox, resolved ahead of time.

    Built before any provisioning so `--dry-run` and a real run agree by
    construction — a dry run that diverged from the real thing would be worse
    than no dry run at all.
    """

    def __init__(
        self,
        *,
        index: int,
        task_id: Optional[str],
        row: Optional[Dict[str, Any]],
        spec: SandboxSpec,
        spec_source: str,
        image_source: str,
        exec_kwargs: Dict[str, Any],
        command: Optional[str],
        wrapped: bool,
        raw_command: Optional[str],
        wrapper_label: str,
    ) -> None:
        self.wrapper_label = wrapper_label
        self.index = index
        self.task_id = task_id
        self.row = row
        self.spec = spec
        self.spec_source = spec_source
        self.image_source = image_source
        self.exec_kwargs = exec_kwargs
        self.command = command
        self.wrapped = wrapped
        self.raw_command = raw_command

    @property
    def name(self) -> str:
        return self.task_id or f"task-{self.index}"


def build_plan(
    *,
    index: int,
    row: Optional[Dict[str, Any]],
    server_config: Dict[str, Any],
    sandbox_task: Dict[str, Any],
    default_metadata: Dict[str, Any],
    config: SandboxDebugConfig,
) -> TaskPlan:
    """Resolve one task into a concrete spec and command."""
    spec, spec_source = resolve_spec(
        sandbox_spec=server_config.get("sandbox_spec"),
        sandbox_task=sandbox_task,
        row=row,
        server_config=server_config,
    )

    image_source = spec_source
    if config.image:
        spec = replace_image(spec, config.image)
        image_source = "--image"
    if spec.image is None:
        raise ConfigError(
            "Could not determine a container image. Pass --image, set `sandbox_spec.image`, or declare "
            "`sandbox_task.image_from_row` / `sandbox_task.spec_resolver` on the server."
        )

    # Provider defaults sit underneath the server's own metadata, which sits
    # underneath what identifies this run — most specific wins.
    metadata = {**default_metadata, **spec.metadata}
    metadata["nemo_gym_tool"] = "gym-sandbox-debug"
    if row is not None:
        task_id = task_id_for_row(row, id_from_row=sandbox_task.get("id_from_row"))
        if task_id:
            metadata["task_id"] = task_id[:63]
    else:
        task_id = None

    exec_kwargs = _exec_kwargs(server_config, config)

    env = {**spec.env, **_parse_env(config.env)}
    # An explicit --ttl is taken at face value; otherwise cap at the debugging
    # default, whether the server asked for a rollout-sized lifetime or none at all
    # (which providers read as "never auto-terminate").
    if config.ttl_s is not None:
        ttl_s = config.ttl_s
    elif spec.ttl_s is None:
        ttl_s = DEFAULT_TTL_S
    else:
        ttl_s = min(float(spec.ttl_s), DEFAULT_TTL_S)
    spec = SandboxSpec(
        image=spec.image,
        ttl_s=ttl_s,
        ready_timeout_s=spec.ready_timeout_s,
        # Servers that keep their workdir in `sandbox_environment_kwargs` rather
        # than in the spec still expect the sandbox to land there.
        workdir=config.cwd or spec.workdir or exec_kwargs.get("cwd"),
        env=env,
        files=dict(spec.files),
        metadata=metadata,
        resources=spec.resources,
        entrypoint=list(spec.entrypoint) if spec.entrypoint else None,
        provider_options=dict(spec.provider_options),
    )

    raw_command = config.command
    if config.script_path:
        remote = f"{REMOTE_SCRIPT_DIR}/ng-debug-{index}-{Path(config.script_path).name}"
        raw_command = f"chmod +x {remote} && {remote}"

    command, wrapped = (None, False)
    if raw_command is not None:
        command, wrapped = wrap_command(
            raw_command, sandbox_task=sandbox_task, exec_kwargs=exec_kwargs, bare=config.bare
        )

    wrapper_ref = sandbox_task.get("exec_wrapper")
    if config.bare:
        wrapper_label = f"(skipped by --bare; server declares {wrapper_ref})" if wrapper_ref else "(none)"
    else:
        wrapper_label = str(wrapper_ref) if wrapper_ref else "(none)"

    return TaskPlan(
        index=index,
        task_id=task_id,
        row=row,
        spec=spec,
        spec_source=spec_source,
        image_source=image_source,
        exec_kwargs=exec_kwargs,
        command=command,
        wrapped=wrapped,
        raw_command=raw_command,
        wrapper_label=wrapper_label,
    )


def _spec_summary(spec: SandboxSpec) -> str:
    resources = spec.resources
    parts = [
        f"{key}={value}" for key, value in (("ttl_s", spec.ttl_s), ("ready_timeout_s", spec.ready_timeout_s)) if value
    ]
    for key, value in (
        ("cpu", resources.cpu),
        ("memory_mib", resources.memory_mib),
        ("disk_gib", resources.disk_gib),
        ("gpu", resources.gpu),
    ):
        if value:
            parts.append(f"{key}={value}")
    if spec.workdir:
        parts.append(f"workdir={spec.workdir}")
    return "  ".join(parts) or "(defaults)"


def render_dry_run(
    *,
    server_name: str,
    server_type: str,
    provider_config: Dict[str, Any],
    plans: List[TaskPlan],
    input_fpath: Optional[str],
    total_rows: int,
) -> None:
    """Show what would be provisioned, without provisioning it."""
    provider_name = next(iter(provider_config), "?")
    for plan in plans:
        rows: List[Tuple[str, str]] = [
            ("server", f"{server_name}  ({server_type})"),
            ("sandbox", provider_name),
        ]
        if plan.task_id:
            location = f"index {plan.index} of {total_rows}"
            if input_fpath:
                location += f", from {input_fpath}"
            rows.append(("task", f"{plan.task_id}  ({location})"))
        else:
            rows.append(("task", "(none — booting the server's configured sandbox)"))
        rows.append(("image", f"{plan.spec.image}\nvia {plan.image_source}"))
        rows.append(("spec", _spec_summary(plan.spec)))
        if plan.spec.metadata:
            rows.append(("metadata", " ".join(f"{k}={v}" for k, v in sorted(plan.spec.metadata.items()))))
        if plan.spec.files:
            rows.append(("files", " ".join(sorted(plan.spec.files))))
        rows.append(("exec", " ".join(f"{k}={v}" for k, v in sorted(plan.exec_kwargs.items())) or "(defaults)"))
        # Name the wrapper even with no command, since it is the main reason a
        # command behaves differently here than it does typed into a shell.
        rows.append(("wrapper", plan.wrapper_label))
        if plan.command:
            rows.append(("command", plan.command))

        # A key/value list rather than a table: one very long cell (the wrapped
        # command) would otherwise pad every other row out to its width.
        width = max(len(label) for label, _ in rows)
        for label, value in rows:
            first, *rest = str(value).split("\n")
            rich.print(f"[bold cyan]{label.ljust(width)}[/bold cyan]  {escape(first)}")
            for line in rest:
                rich.print(f"{' ' * width}  [dim]{escape(line)}[/dim]")
        rich.print("")


########################################
# Execution
########################################


async def _sandbox_id(sandbox: AsyncSandbox) -> Optional[str]:
    """Best-effort id for a running sandbox, for display and for `--keep`.

    Connectable providers expose it through `serialize()`, which is also the
    descriptor `gym sandbox exec` reattaches with. Others have no externally
    meaningful id, so the trace simply records none.
    """
    with suppress(Exception):
        descriptor = await sandbox.serialize()
        if isinstance(descriptor, Mapping) and descriptor.get("sandbox_id"):
            return str(descriptor["sandbox_id"])
    handle = getattr(sandbox, "_handle", None)
    return getattr(handle, "sandbox_id", None)


async def run_plan(
    plan: TaskPlan,
    *,
    provider_factory,
    provider_name: str,
    config: SandboxDebugConfig,
    semaphore: asyncio.Semaphore,
    write_trace,
) -> Dict[str, Any]:
    """Boot one sandbox, run the action in it, and return the trace record."""
    trace: Dict[str, Any] = {
        "task": {"index": plan.index, "id": plan.task_id},
        "provider": provider_name,
        "image": plan.spec.image,
        "image_source": plan.image_source,
        "ttl_s": plan.spec.ttl_s,
        "spec_source": plan.spec_source,
        "action": "script" if config.script_path else "command",
        "command": plan.command,
        "wrapped": plan.wrapped,
        "sandbox_id": None,
        "ok": False,
        "reason": REASON_NOT_RUN,
        "exit_code": None,
        "stdout": None,
        "stderr": None,
        "kept": False,
        "timing": {},
    }

    async with semaphore:
        sandbox = AsyncSandbox(provider_factory(), plan.spec)
        started = time.monotonic()
        boot_started = started
        # Which budget is in force right now, so a timeout can name the stage that
        # blew it rather than reporting whichever number happens to be handy.
        stage, budget = "boot", config.timeout_setup
        try:
            await asyncio.wait_for(sandbox.start(), timeout=config.timeout_setup)
            _LIVE_SANDBOXES.add(sandbox)
            trace["timing"]["boot"] = round(time.monotonic() - boot_started, 3)
            trace["sandbox_id"] = await _sandbox_id(sandbox)

            stage, budget = "setup", config.timeout_setup
            setup_started = time.monotonic()
            if config.script_path:
                remote = f"{REMOTE_SCRIPT_DIR}/ng-debug-{plan.index}-{Path(config.script_path).name}"
                await asyncio.wait_for(sandbox.upload(config.script_path, remote), timeout=config.timeout_setup)
            trace["timing"]["setup"] = round(time.monotonic() - setup_started, 3)

            stage, budget = "exec", config.timeout_total
            exec_started = time.monotonic()
            result = await sandbox.exec(
                plan.command,
                cwd=plan.exec_kwargs.get("cwd"),
                user=plan.exec_kwargs.get("user"),
                timeout_s=config.timeout_total,
            )
            elapsed = time.monotonic() - exec_started
            trace["timing"]["exec"] = round(elapsed, 3)
            # Split what the command spent from what the round-trip cost, so a slow
            # run points at the right culprit. Providers that can't measure the
            # command itself simply leave the breakdown out.
            if result.duration_ms is not None:
                command_s = result.duration_ms / 1000.0
                trace["timing"]["command"] = round(command_s, 3)
                trace["timing"]["overhead"] = round(max(elapsed - command_s, 0.0), 3)
            trace["stdout"] = result.stdout
            trace["stderr"] = result.stderr
            trace["exit_code"] = result.return_code
            trace["ok"] = result.return_code == 0
            trace["reason"] = REASON_PASS if result.return_code == 0 else REASON_NONZERO_EXIT
            if result.error_type:
                trace["reason"] = REASON_ERROR
                trace["ok"] = False
                trace["error_type"] = result.error_type
            elif result.return_code is not None and result.return_code < 0:
                # Providers enforce the timeout themselves and report it as an
                # abnormal (negative) code rather than raising, so a timeout would
                # otherwise read as an ordinary non-zero exit. Real process exits
                # are 0-255, so a negative code always means killed, not returned.
                timed_out = budget is not None and elapsed >= budget * _TIMEOUT_TOLERANCE
                trace["reason"] = REASON_TIMEOUT if timed_out else REASON_ERROR
                trace["error"] = (
                    f"command timed out after {budget}s"
                    if timed_out
                    else f"command terminated abnormally (code {result.return_code})"
                )
        except asyncio.TimeoutError:
            trace["reason"] = REASON_TIMEOUT
            trace["error"] = f"{stage} timed out after {budget}s"
            if stage == "boot":
                # A create that times out may already have produced a sandbox on
                # the provider's side that we never got a handle for, so there is
                # nothing to tear down here. Say so rather than leak quietly.
                trace["may_have_orphaned_sandbox"] = True
                rich.print(
                    f"[yellow]warning[/yellow] {plan.name}: boot timed out; a sandbox may have been created "
                    f"without returning a handle and cannot be cleaned up automatically. Check for orphans "
                    f"with metadata nemo_gym_tool=gym-sandbox-debug."
                )
        except (KeyboardInterrupt, asyncio.CancelledError):
            trace["reason"] = REASON_CANCELLED
            trace["error"] = "cancelled"
            raise
        except Exception as e:
            trace["reason"] = REASON_ERROR
            trace["error"] = str(e)
            trace["error_type"] = type(e).__name__
        finally:
            trace["timing"]["total"] = round(time.monotonic() - started, 3)
            if config.keep and trace["sandbox_id"]:
                trace["kept"] = True
                _LIVE_SANDBOXES.discard(sandbox)
            else:
                with suppress(Exception):
                    await asyncio.shield(sandbox.stop())
                _LIVE_SANDBOXES.discard(sandbox)
            await write_trace(trace)

    return trace


def _print_result(trace: Dict[str, Any], *, quiet: bool) -> None:
    name = trace["task"]["id"] or f"task-{trace['task']['index']}"
    if not quiet:
        for stream in ("stdout", "stderr"):
            text = trace.get(stream)
            if text:
                rich.print(f"[dim]--- {stream} ({name}) ---[/dim]")
                print(text)

    mark = "[green]✓[/green]" if trace["ok"] else "[red]✗[/red]"
    detail = f"reason={trace['reason']}"
    if trace.get("exit_code") is not None:
        detail += f" exit={trace['exit_code']}"
    if trace.get("error"):
        detail += f" error={trace['error']}"
    timing = trace.get("timing", {})
    duration = f"{timing.get('total', 0)}s"
    if "command" in timing:
        duration += (
            f" (command {timing['command']}s + overhead {timing.get('overhead', 0)}s, boot {timing.get('boot', 0)}s)"
        )
    rich.print(f"{mark} {name}  {detail}  {duration}")
    if trace.get("kept"):
        # State the expiry, not just that it was kept: a sandbox you forget about
        # holds cluster capacity until its TTL, so the deadline is the useful part.
        ttl = trace.get("ttl_s")
        expiry = f"expires in {_format_duration(ttl)}" if ttl else "no expiry set"
        rich.print(
            f"  [yellow]sandbox {trace['sandbox_id']} left running[/yellow] ({expiry})\n"
            f"  reattach: gym sandbox exec --sandbox-id {trace['sandbox_id']} --command ...\n"
            f"  delete:   gym sandbox rm --sandbox-id {trace['sandbox_id']}"
        )


def _format_duration(seconds: float) -> str:
    """Render a TTL the way a person reads a deadline, not as a raw second count."""
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes = remainder // 60
    if hours and minutes:
        return f"{hours}h{minutes:02d}m"
    if hours:
        return f"{hours}h"
    if minutes:
        return f"{minutes}m"
    return f"{seconds}s"


########################################
# Commands
########################################


def _resolve_server(config: Any, global_config_dict: Any):
    """Pick the server and read its hooks, without touching the provider.

    Kept separate so read-only work (listing a dataset) needs neither provider
    credentials nor a provider config on the command line.
    """
    servers = discover_sandbox_servers(global_config_dict)
    server_name, entry = select_server(servers, config.server_name)
    sandbox_task = dict(entry["config"].get("sandbox_task") or {})
    return server_name, entry, sandbox_task


def _resolve_context(config: Any, global_config_dict: Any):
    """Shared resolution: pick the server, build its provider, read its hooks."""
    server_name, entry, sandbox_task = _resolve_server(config, global_config_dict)
    server_config = entry["config"]
    provider, provider_config = build_provider(server_config, global_config_dict, config.sandbox_name)
    return server_name, entry, server_config, provider, provider_config, sandbox_task


# Row fields worth showing next to an id, in preference order. Enough to tell two
# tasks apart without dumping a problem statement into the terminal.
_TASK_DETAIL_FIELDS = ("repo", "subset", "split", "difficulty", "image_name")


def _list_tasks(config: Any, *, entry: Dict[str, Any], sandbox_task: Dict[str, Any], as_json: bool) -> None:
    """Print the task ids available to `--task`.

    Without this the only way to learn an id is to guess one and read it out of
    the error, which is a poor way to start.
    """
    id_from_row = sandbox_task.get("id_from_row")
    rows, source = load_rows(
        config.input_jsonl_fpath,
        entry=entry,
        task_ids=[],
        limit=config.limit,
        shuffle=config.shuffle,
        id_from_row=id_from_row,
        # Listing is about the dataset, so an unfiltered read is the point.
        require_selection=True,
    )

    listed = [
        {
            "index": index,
            "id": task_id_for_row(row, id_from_row=id_from_row),
            **{field: row[field] for field in _TASK_DETAIL_FIELDS if field in row},
        }
        for index, row in enumerate(rows)
    ]

    if as_json:
        print(json.dumps(listed, indent=2))
        return

    if not listed:
        rich.print(f"[yellow]No tasks found in {source}[/yellow]")
        return

    rich.print(f"[dim]{len(listed)} task(s) in {source}[/dim]")
    width = max((len(str(item["id"])) for item in listed), default=0)
    for item in listed:
        detail = "  ".join(f"{k}={v}" for k, v in item.items() if k not in ("index", "id"))
        rich.print(f"  [cyan]{str(item['id']).ljust(width)}[/cyan]  [dim]{escape(detail)}[/dim]")
    rich.print("\n[dim]Run one with --task <id>[/dim]")


@exit_cleanly_on_config_error
def debug() -> None:
    """Boot a task's sandbox and run a command or script inside it."""
    global_config_dict = get_global_config_dict(
        global_config_dict_parser_config=GlobalConfigDictParserConfig(
            initial_global_config_dict=GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        ),
    )
    config = SandboxDebugConfig.model_validate(global_config_dict)
    as_json = bool(global_config_dict.get(JSON_OUTPUT_KEY_NAME))
    _quiet_transport_logs(global_config_dict)
    _warn_on_long_ttl(config.ttl_s)

    # Listing only reads the dataset, so resolve the server but not the provider:
    # no provider config, no credentials, no cluster.
    if config.list_tasks:
        _, list_entry, list_hooks = _resolve_server(config, global_config_dict)
        _list_tasks(config, entry=list_entry, sandbox_task=list_hooks, as_json=as_json)
        return

    server_name, entry, server_config, provider, provider_config, sandbox_task = _resolve_context(
        config, global_config_dict
    )

    rows, input_fpath = load_rows(
        config.input_jsonl_fpath,
        entry=entry,
        task_ids=config.task_ids,
        limit=config.limit,
        shuffle=config.shuffle,
        id_from_row=sandbox_task.get("id_from_row"),
    )
    default_metadata = provider_default_metadata(server_config, global_config_dict, config.sandbox_name)

    # No row selected means "boot what this server is configured to boot".
    row_list: List[Optional[Dict[str, Any]]] = list(rows) if rows else [None]
    plans = [
        build_plan(
            index=index,
            row=row,
            server_config=server_config,
            sandbox_task=sandbox_task,
            default_metadata=default_metadata,
            config=config,
        )
        for index, row in enumerate(row_list)
    ]

    if config.dry_run:
        if as_json:
            print(
                json.dumps(
                    [
                        {
                            "server": server_name,
                            "server_type": entry["server_type"],
                            "provider": next(iter(provider_config), None),
                            "task_id": plan.task_id,
                            "image": plan.spec.image,
                            "image_source": plan.image_source,
                            "spec_source": plan.spec_source,
                            "exec": plan.exec_kwargs,
                            "command": plan.command,
                        }
                        for plan in plans
                    ],
                    indent=2,
                )
            )
        else:
            render_dry_run(
                server_name=server_name,
                server_type=entry["server_type"],
                provider_config=provider_config,
                plans=plans,
                input_fpath=input_fpath,
                total_rows=len(row_list),
            )
        return

    # Refuse --keep up front on a provider we could not reattach to: the sandbox
    # would sit there costing capacity with no way to reach or delete it.
    if config.keep and not isinstance(provider, ConnectableProvider):
        raise ConfigError(
            f"--keep needs a provider that can reattach by id, and "
            f"{getattr(provider, 'name', type(provider).__name__)!r} cannot. Without it the sandbox would "
            f"be left running with no way to reach or delete it."
        )

    _install_interrupt_handler()
    output_dir = Path(config.output_dirpath or Path(_DEFAULT_OUTPUT_ROOT) / f"{server_name}--debug" / uuid.uuid4().hex)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(_redacted_config_yaml(global_config_dict))
    traces_path = output_dir / "traces.jsonl"
    traces_path.write_text("")
    rich.print(f"[dim]results: {output_dir}/[/dim]")

    write_lock = asyncio.Lock()

    async def write_trace(trace: Dict[str, Any]) -> None:
        # Serialize whole lines and write off-loop so a cancellation cannot tear
        # a record in half; a trace line is either complete or absent.
        line = json.dumps(trace) + "\n"
        async with write_lock:
            await asyncio.shield(asyncio.to_thread(_append, traces_path, line))

    provider_name = next(iter(provider_config), "?")

    async def main() -> List[Dict[str, Any]]:
        semaphore = asyncio.Semaphore(config.concurrency)
        try:
            return await asyncio.gather(
                *(
                    run_plan(
                        plan,
                        # One provider per sandbox: providers own SDK clients, and
                        # AsyncSandbox.stop() closes the provider it was given.
                        provider_factory=lambda: create_provider(provider_config),
                        provider_name=provider_name,
                        config=config,
                        semaphore=semaphore,
                        write_trace=write_trace,
                    )
                    for plan in plans
                ),
                return_exceptions=True,
            )
        finally:
            # `provider` was built only to test the reattach capability; the run
            # itself uses per-sandbox providers. Release its client either way.
            with suppress(Exception):
                await provider.aclose()

    try:
        results = asyncio.run(main())
    except KeyboardInterrupt:  # pragma: no cover - signal path
        rich.print("[yellow]interrupted[/yellow]")
        raise SystemExit(130)

    traces = [r for r in results if isinstance(r, dict)]
    for trace in traces:
        _print_result(trace, quiet=config.quiet)

    failures = [t for t in traces if not t["ok"]] + [r for r in results if not isinstance(r, dict)]
    if failures and not config.exit_zero:
        raise SystemExit(1)


def _append(path: Path, line: str) -> None:
    with open(path, "a") as f:
        f.write(line)


async def _renew(provider, sandbox: AsyncSandbox, ttl_s: float) -> bool:
    """Push the sandbox's expiry out, if the provider can. Best effort.

    A provider without the capability simply keeps whatever lifetime it was
    created with; failing the command over an expiry bump would be worse than
    the shorter lifetime.
    """
    if not isinstance(provider, RenewableProvider):
        return False
    handle = getattr(sandbox, "_handle", None)
    if handle is None:
        return False
    try:
        await provider.renew(handle, ttl_s)
        return True
    except Exception as e:
        rich.print(f"[yellow]warning[/yellow] could not extend the sandbox expiry: {e}")
        return False


async def _connect(provider, sandbox_id: str):
    """Reattach to a running sandbox, or explain why the provider cannot.

    Only providers backed by an external control plane can be reached by id from
    a fresh process; a provider that keeps its sandboxes in-process has nothing
    to reconnect to.
    """
    if not isinstance(provider, ConnectableProvider):
        raise ConfigError(
            f"Provider {getattr(provider, 'name', type(provider).__name__)!r} cannot reattach to an existing "
            f"sandbox, so `gym sandbox exec` and `gym sandbox rm` are unavailable for it. Use "
            f"`gym sandbox debug --command ...` instead, which creates its own sandbox."
        )
    return await AsyncSandbox.connect({"sandbox_id": sandbox_id}, provider=provider)


@exit_cleanly_on_config_error
def exec_command() -> None:
    """Run a command in an already-running sandbox."""
    global_config_dict = get_global_config_dict(
        global_config_dict_parser_config=GlobalConfigDictParserConfig(
            initial_global_config_dict=GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        ),
    )
    config = SandboxExecConfig.model_validate(global_config_dict)
    _quiet_transport_logs(global_config_dict)
    if not config.sandbox_id:
        raise ConfigError("--sandbox-id is required.")
    if not config.command:
        raise ConfigError("--command is required.")

    _, _, server_config, provider, _, sandbox_task = _resolve_context(config, global_config_dict)
    exec_kwargs = _exec_kwargs(server_config, config)
    command, _ = wrap_command(config.command, sandbox_task=sandbox_task, exec_kwargs=exec_kwargs, bare=config.bare)

    # Bump the expiry before running rather than after, so the command has a full
    # window and a sandbox stays alive for as long as it is being used — a short
    # lifetime plus renewal beats a long lifetime nobody remembers to clean up.
    ttl_s = config.ttl_s if config.ttl_s is not None else DEFAULT_TTL_S
    renew_s = max(ttl_s, (config.timeout_total or 0) + 60)

    async def main():
        sandbox = await _connect(provider, config.sandbox_id)
        renewed = await _renew(provider, sandbox, renew_s)
        try:
            return await sandbox.exec(
                command,
                cwd=exec_kwargs.get("cwd"),
                user=exec_kwargs.get("user"),
                timeout_s=config.timeout_total,
            ), renewed
        finally:
            # Detach without killing: the sandbox outlives this command by design.
            with suppress(Exception):
                await provider.aclose()

    result, renewed = asyncio.run(main())
    if not config.quiet:
        for text in (result.stdout, result.stderr):
            if text:
                print(text)

    detail = f"exit={result.return_code}"
    if result.duration_ms is not None:
        detail += f"  command {round(result.duration_ms / 1000.0, 3)}s"
    if renewed:
        detail += f"  expires in {_format_duration(renew_s)}"
    mark = "[green]✓[/green]" if result.return_code == 0 else "[red]✗[/red]"
    rich.print(f"{mark} {config.sandbox_id}  {detail}")

    if result.return_code != 0 and not config.exit_zero:
        raise SystemExit(result.return_code or 1)


@exit_cleanly_on_config_error
def rm() -> None:
    """Delete a running sandbox."""
    global_config_dict = get_global_config_dict(
        global_config_dict_parser_config=GlobalConfigDictParserConfig(
            initial_global_config_dict=GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        ),
    )
    config = SandboxExecConfig.model_validate(global_config_dict)
    _quiet_transport_logs(global_config_dict)
    if not config.sandbox_id:
        raise ConfigError("--sandbox-id is required.")

    _, _, _, provider, _, _ = _resolve_context(config, global_config_dict)

    async def main():
        sandbox = await _connect(provider, config.sandbox_id)
        await sandbox.stop()

    asyncio.run(main())
    rich.print(f"[green]✓[/green] deleted sandbox {config.sandbox_id}")
