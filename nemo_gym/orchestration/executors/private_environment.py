# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep task environments in private files rather than the scheduler's environment or script."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from nemo_gym.orchestration.executors.connection import is_local_hostname


if TYPE_CHECKING:
    from nemo_gym.orchestration.api import SubmitConfig


PRIVATE_ENVIRONMENT_MARKER = "# Gym task environments are loaded after srun (v3)."
DRIVER_SELECTION_MARKER = "# Gym private driver container selections: "

# Executed by the task's Python, including when its installed Gym predates this launcher.
_LOADER = """import json, os, stat, sys
from pathlib import Path
directory, role = Path(sys.argv[1]), sys.argv[2]
def checked(path, directory=False):
    flags = os.O_RDONLY | os.O_NOFOLLOW | (os.O_DIRECTORY if directory else 0)
    fd = os.open(path, flags)
    info = os.fstat(fd)
    expected = 0o700 if directory else 0o600
    mode = stat.S_IMODE(info.st_mode)
    if directory: mode &= ~stat.S_ISGID
    if info.st_uid != os.geteuid() or mode != expected:
        os.close(fd)
        raise PermissionError("Gym private environment must be owner-only: " + str(path))
    return fd
parent = checked(directory, directory=True)
try:
    with os.fdopen(checked(directory / "environment.json")) as stream:
        payload = json.load(stream)
finally:
    os.close(parent)
if payload["version"] != 3:
    raise ValueError("Unsupported Gym private environment contract; submit a new run")
runtime = os.environ.copy()
runtime.update(payload["forwarded"])
if role == "driver": runtime.update(payload["driver_forwarded"])
environment = payload["inherited"].copy()
environment.update(runtime)
environment.update({
    name: runtime[value[8:]] if value.startswith("runtime:") else value
    for name, value in payload["roles"][role].items()
})
os.execvpe(sys.argv[3], sys.argv[3:], environment)
"""

# The scheduler receives only executable/user locale settings. Task values are restored after srun.
_SCHEDULER_ENV = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "LD_LIBRARY_PATH",
    "SLURM_CONF",
    "SLURM_CONF_SERVER",
    "SLURM_CLUSTERS",
    "SLURM_ACCOUNT",
    "SLURM_PARTITION",
    "SLURM_QOS",
    "SLURM_RESERVATION",
)


def scheduler_command(command: str, *, reader: bool = False) -> str:
    """Run a scheduler client without inheriting credentials from its caller."""
    names = (*_SCHEDULER_ENV, "GYM_RESUBMIT", "GYM_RESUBMIT_KEEP_BUDGET")
    if reader:
        names += ("UV_LOCK_TIMEOUT", "GYM_READER_TOKEN", "SLURM_JOB_ID", "SLURM_CLUSTER_NAME")
    assignments = " ".join(f'${{{name}+{name}="${name}"}}' for name in names)
    wrapper = scheduler_environment_check() + '\nexec "$@"'
    return f"bash -c {shlex.quote(wrapper)} gym-scheduler env -i {assignments} {command}"


def scheduler_environment_check() -> str:
    """Reject ambient Slurm option defaults at the client boundary without printing values."""
    return """for _gym_default in $(compgen -e); do
    case "$_gym_default" in
        SBATCH_*|SALLOC_*|SRUN_*)
            printf '%s\\n' "Gym cannot safely inherit scheduler option $_gym_default; configure compute/node-pool extra_args or reader arguments and unset it" >&2
            exit 2
            ;;
    esac
done"""


def validate_scheduler_environment_options(extra_args: Mapping[str, str]) -> None:
    """Refuse Slurm options that can replace the restricted scheduler environment."""
    forbidden = ("export", "export-file", "get-user-env")
    for key in extra_args:
        option = key.lstrip("-").split("=", 1)[0].replace("_", "-")
        # getopt accepts unambiguous long-option abbreviations, including --get-u and --export-f.
        if option and any(name.startswith(option) for name in forbidden):
            raise ValueError(
                f"Slurm option --{option} can import credentials into scheduler accounting; "
                "use compute.forward_env and driver/service env instead"
            )
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", key):
            raise ValueError("Slurm extra_args keys must be single option names without leading dashes or values")


def forwarded_names(config: SubmitConfig) -> set[str]:
    """Explicit caller overrides restored privately, excluding protected task context."""
    compute = next(iter(config.compute.values()))
    names = set(compute.forward_env)
    if config.otel.enabled and config.otel.remote_export and config.otel.token_env in os.environ:
        names.add(config.otel.token_env)
    return _without_task_context(names)


def _without_task_context(names: set[str]) -> set[str]:
    return {
        name
        for name in names
        if not (
            name.startswith(
                (
                    "SLURM_",
                    "PMI_",
                    "PMIX_",
                    "OMPI_COMM_WORLD_",
                    "MV2_COMM_WORLD_",
                    "MPI_LOCAL",
                    "GYM_POOL_",
                    "GYM_RAY_ADDRESS_",
                )
            )
            or (name.startswith("GYM_RAY_") and name.endswith("_WORKERS"))
            or name
            in (
                "PWD",
                "OLDPWD",
                "HOSTNAME",
                "HEAD_NODE_IP",
                "RAY_HEAD_NODE_IP",
                "CUDA_VISIBLE_DEVICES",
                "NVIDIA_VISIBLE_DEVICES",
                "RANK",
                "LOCAL_RANK",
                "WORLD_SIZE",
                "LOCAL_WORLD_SIZE",
                "GROUP_RANK",
                "ROLE_RANK",
                "ROLE_WORLD_SIZE",
                "NODE_RANK",
                "GYM_RESUBMIT",
                "GYM_RESUBMIT_KEEP_BUDGET",
                "GYM_EVAL_SUCCEEDED_MARKER",
                "GYM_DRIVER_FAILURE_FILE",
                "GYM_PREVIOUS_STATE",
                "GYM_PREVIOUS_JOB_ID",
                "GYM_PREVIOUS_EXIT_CODE",
                "GYM_FOLLOWUP_SENTINEL",
                "GYM_CHAIN_ACCOUNTING",
            )
        )
    }


def driver_forwarded_names(config: SubmitConfig) -> set[str]:
    """Caller-present container selections restored only inside the driver, before runtime aliases."""
    return _without_task_context(set(config.driver.container_env)) & set(os.environ)


def driver_selection_check(config: SubmitConfig, script: Path) -> list[str]:
    """Refuse loss of a private selection whose native flag was omitted from the saved script."""
    current = driver_forwarded_names(config)
    if not (_without_task_context(set(config.driver.container_env)) - current):
        return []
    check = """import json, sys
from pathlib import Path
prefix = sys.argv[2]
lines = [line[len(prefix):] for line in Path(sys.argv[1]).read_text().splitlines() if line.startswith(prefix)]
if len(lines) != 1:
    raise ValueError("Saved job.sh lacks driver selection metadata; submit a new run")
previous = json.loads(lines[0])
if not isinstance(previous, list) or any(not isinstance(name, str) or not name.isidentifier() for name in previous):
    raise ValueError("Saved job.sh has invalid driver selection metadata; submit a new run")
missing = sorted(set(previous) - set(json.loads(sys.argv[3])))
if missing:
    raise ValueError("Driver container selections are no longer set in the caller: " + ", ".join(missing) +
                     "; restore those variables or submit a new run")
"""
    return [
        f"python3 -c {shlex.quote(check)} {shlex.quote(str(script))} "
        f"{shlex.quote(DRIVER_SELECTION_MARKER)} {shlex.quote(json.dumps(sorted(current)))}\n"
        "[ $? = 0 ] || exit 1"
    ]


def forwarded_values(config: SubmitConfig) -> dict[str, str]:
    """Capture explicit caller overrides for task-side restoration."""
    compute = next(iter(config.compute.values()))
    missing = [name for name in compute.forward_env if name not in os.environ]
    if missing:
        raise ValueError(f"compute.forward_env variables are unset: {', '.join(missing)}")
    return {name: os.environ[name] for name in sorted(forwarded_names(config))}


def _inherited_values(config: SubmitConfig) -> dict[str, str]:
    compute = next(iter(config.compute.values()))
    names = _without_task_context(set(os.environ)) if is_local_hostname(compute.hostname) else set()
    return {name: os.environ[name] for name in sorted(names - forwarded_names(config))}


def private_directory(benchmark_dir: Path) -> Path:
    """A store inside the writable output root, outside run trees that can be shared or exported."""
    output_root = benchmark_dir.parent.parent
    store = output_root / (".gym-private-" + hashlib.sha256(str(benchmark_dir.parent).encode()).hexdigest()[:16])
    return store / hashlib.sha256(str(benchmark_dir).encode()).hexdigest()


_PRIVATE_DIRECTORY_SETUP = """import os, stat, sys, tempfile
from pathlib import Path
directory = Path(sys.argv[1])
directory.parent.parent.mkdir(parents=True, exist_ok=True)
for path in [directory.parent, directory]:
    path.mkdir(mode=0o700, exist_ok=True)
    info = path.lstat()
    mode = stat.S_IMODE(info.st_mode) & ~stat.S_ISGID
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or mode != 0o700:
        raise PermissionError("Gym private environment directory must be owner-only: " + str(path))
"""


def prepare_commands(benchmark_dir: Path) -> list[str]:
    """Probe private-store access before copying a fresh run, without writing credentials."""
    probe = (
        _PRIVATE_DIRECTORY_SETUP
        + """
fd, temporary = tempfile.mkstemp(dir=directory)
os.close(fd)
os.unlink(temporary)
"""
    )
    return [
        scheduler_environment_check(),
        f"python3 -c {shlex.quote(probe)} {shlex.quote(str(private_directory(benchmark_dir)))}\n[ $? = 0 ] || exit 1",
    ]


def stage_commands(config: SubmitConfig, roles: Mapping[str, Mapping[str, str]], benchmark_dir: Path) -> list[str]:
    """Atomically refresh private task values; the caller must exclude concurrent jobs first."""
    directory = private_directory(benchmark_dir)
    payload = json.dumps(
        {
            "version": 3,
            "forwarded": forwarded_values(config),
            "driver_forwarded": {name: os.environ[name] for name in sorted(driver_forwarded_names(config))},
            "inherited": _inherited_values(config),
            "roles": roles,
        },
        sort_keys=True,
    )
    writer = (
        _PRIVATE_DIRECTORY_SETUP
        + """
fd, temporary = tempfile.mkstemp(dir=directory)
try:
    with os.fdopen(fd, "w") as stream:
        stream.write(sys.stdin.read())
    os.replace(temporary, directory / "environment.json")
finally:
    if os.path.exists(temporary): os.unlink(temporary)
"""
    )
    return [
        f"python3 -c {shlex.quote(writer)} {shlex.quote(str(directory))} <<'GYM_PRIVATE_ENVIRONMENT'\n"
        f"{payload}\nGYM_PRIVATE_ENVIRONMENT\n"
        "[ $? = 0 ] || exit 1"
    ]


def task_command(
    command: str,
    *,
    directory: Path,
    role: str,
    environment: Mapping[str, str],
    arguments: str = "",
    shell: bool = True,
) -> tuple[str, str]:
    """A task-side loader and the nonsecret runtime shell variables it needs from the batch shell."""
    runtime_names = sorted(
        {
            value[8:]
            for value in environment.values()
            if value in ("runtime:GYM_EVAL_SUCCEEDED_MARKER", "runtime:HOSTNAME", "runtime:OLDPWD")
        }
    )
    prefix = " ".join(f'{name}="${{{name}:-}}"' for name in runtime_names)
    loader = shlex.join(["python3", "-c", _LOADER, str(directory), role])
    loader += " " + (shlex.join(["bash", "-c", command]) if shell else command)
    return loader + arguments, f"env {prefix} " if prefix else ""
