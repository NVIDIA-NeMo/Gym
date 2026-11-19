# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import json
import os
import shlex
import signal
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from nemo_gym.orchestration.api import RayServiceConfig, SubmitConfig
from nemo_gym.orchestration.executors.connection import LocalConnection
from nemo_gym.orchestration.executors.private_environment import (
    prepare_commands,
    private_directory,
    scheduler_command,
    stage_commands,
    task_command,
)
from nemo_gym.orchestration.executors.slurm import executor as slurm_module
from nemo_gym.orchestration.executors.slurm.script import build_sbatch_script, task_environments


@pytest.fixture
def caller_environment(tmp_path, monkeypatch):
    values = {
        "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "CANARY_TOKEN": "complex-secret-canary 'quoted' \"double\" $HOME `command` $(command)\\tail\nsecond line ☃",
        "UNRELATED_PRIVATE_KEY": "another-private-canary",
    }
    for name in tuple(os.environ):
        monkeypatch.delenv(name)
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    return values


def _config(tmp_path: Path, *, hostname: str | None = "login.example") -> SubmitConfig:
    return SubmitConfig.model_validate(
        {
            "services": {},
            "compute": {
                "test": {
                    "type": "slurm",
                    "account": "test-account",
                    "hostname": hostname,
                    "forward_env": ["CANARY_TOKEN"],
                }
            },
            "driver": {"container": "python:3", "benchmarks": {"benchmark": {}}},
            "job": {"output_path": str(tmp_path / "output")},
            "otel": {"enabled": False},
        }
    )


def _shell(command: str, *, environment: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", command],
        env=environment,
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )


def _read_environment_command(names: list[str]) -> str:
    source = "import json,os; print(json.dumps({name: os.environ.get(name) for name in " + repr(names) + "}))"
    return shlex.join([sys.executable, "-c", source])


@pytest.mark.parametrize("hostname", [None, "login.example"])
@pytest.mark.parametrize("directory_mode", [0o700, 0o2700])
def test_private_values_reach_each_task_without_becoming_script_content(
    tmp_path, caller_environment, hostname, directory_mode
):
    config = _config(tmp_path, hostname=hostname)
    benchmark_dir = Path(config.job.output_path) / "run" / "benchmark"
    literal = "literal 'quote' $value\nsecond line"
    roles = {
        role: {"ROLE": role, "TOKEN_ALIAS": "runtime:CANARY_TOKEN", "LITERAL_SECRET": literal}
        for role in ("driver", "policy", "collector")
    }
    directory = private_directory(benchmark_dir)
    if directory_mode == 0o2700:
        directory.parent.mkdir(mode=0o700, parents=True)
        directory.mkdir(mode=0o700)
        directory.parent.chmod(directory_mode)
        directory.chmod(directory_mode)
    staged = _shell("\n".join(stage_commands(config, roles, benchmark_dir)))
    assert staged.returncode == 0, staged.stderr

    assert directory.is_relative_to(Path(config.job.output_path))
    assert not directory.is_relative_to(benchmark_dir.parent)
    assert stat.S_IMODE(directory.parent.stat().st_mode) == directory_mode
    assert stat.S_IMODE(directory.stat().st_mode) == directory_mode
    assert stat.S_IMODE((directory / "environment.json").stat().st_mode) == 0o600

    task_environment = {"PATH": caller_environment["PATH"]}
    names = ["CANARY_TOKEN", "TOKEN_ALIAS", "LITERAL_SECRET", "ROLE", "UNRELATED_PRIVATE_KEY"]
    for role, environment in roles.items():
        command, prefix = task_command(
            _read_environment_command(names),
            directory=directory,
            role=role,
            environment=environment,
        )
        assert caller_environment["CANARY_TOKEN"] not in command + prefix
        assert literal not in command + prefix
        for _ in range(2):
            result = _shell(prefix + command, environment=task_environment)
            assert result.returncode == 0, result.stderr
            assert json.loads(result.stdout) == {
                "CANARY_TOKEN": caller_environment["CANARY_TOKEN"],
                "TOKEN_ALIAS": caller_environment["CANARY_TOKEN"],
                "LITERAL_SECRET": literal,
                "ROLE": role,
                "UNRELATED_PRIVATE_KEY": caller_environment["UNRELATED_PRIVATE_KEY"] if hostname is None else None,
            }


def test_scheduler_parent_is_sanitized_and_task_restores_its_private_environment(tmp_path, caller_environment):
    config = _config(tmp_path)
    benchmark_dir = Path(config.job.output_path) / "run" / "benchmark"
    roles = {"driver": {"TOKEN_ALIAS": "runtime:CANARY_TOKEN"}}
    staged = _shell("\n".join(stage_commands(config, roles, benchmark_dir)))
    assert staged.returncode == 0, staged.stderr
    command, prefix = task_command(
        _read_environment_command(["CANARY_TOKEN", "TOKEN_ALIAS"]),
        directory=private_directory(benchmark_dir),
        role="driver",
        environment=roles["driver"],
    )
    task = tmp_path / "task.sh"
    task.write_text(prefix + command)
    parent_receipt = tmp_path / "parent.json"
    boundary = tmp_path / "scheduler.py"
    boundary.write_text(
        "import json,os,subprocess,sys\n"
        "from pathlib import Path\n"
        "Path(sys.argv[1]).write_text(json.dumps({'env': dict(os.environ), 'argv': sys.argv}))\n"
        "subprocess.run(['bash', sys.argv[2]], check=True)\n"
    )
    result = _shell(scheduler_command(shlex.join([sys.executable, str(boundary), str(parent_receipt), str(task)])))
    assert result.returncode == 0, result.stderr
    parent = json.loads(parent_receipt.read_text())
    assert "CANARY_TOKEN" not in parent["env"]
    assert "UNRELATED_PRIVATE_KEY" not in parent["env"]
    assert "SLURM_CONF" not in parent["env"]
    assert "SLURM_CONF_SERVER" not in parent["env"]
    assert "SLURM_CLUSTERS" not in parent["env"]
    assert all(caller_environment["CANARY_TOKEN"] not in value for value in parent["env"].values())
    assert all(caller_environment["CANARY_TOKEN"] not in value for value in parent["argv"])
    assert caller_environment["CANARY_TOKEN"] not in task.read_text()
    assert parent["env"]["HOME"] == caller_environment["HOME"]
    assert json.loads(result.stdout) == {
        "CANARY_TOKEN": caller_environment["CANARY_TOKEN"],
        "TOKEN_ALIAS": caller_environment["CANARY_TOKEN"],
    }


def test_private_payload_refresh_preserves_modes_and_replaces_the_value(tmp_path, caller_environment, monkeypatch):
    config = _config(tmp_path)
    benchmark_dir = Path(config.job.output_path) / "run" / "benchmark"
    roles = {"driver": {}}
    first = _shell("\n".join(stage_commands(config, roles, benchmark_dir)))
    assert first.returncode == 0, first.stderr
    monkeypatch.setenv("CANARY_TOKEN", "refreshed-private-canary")
    second = _shell("\n".join(stage_commands(config, roles, benchmark_dir)))
    assert second.returncode == 0, second.stderr
    directory = private_directory(benchmark_dir)
    payload = directory / "environment.json"
    assert payload.stat().st_mode & 0o777 == 0o600
    command, prefix = task_command(
        _read_environment_command(["CANARY_TOKEN"]),
        directory=directory,
        role="driver",
        environment={},
    )
    result = _shell(prefix + command, environment={"PATH": caller_environment["PATH"]})
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"CANARY_TOKEN": "refreshed-private-canary"}
    assert caller_environment["CANARY_TOKEN"] not in payload.read_text()


@pytest.mark.parametrize("hostname", [None, "login.example"])
def test_task_uses_current_scheduler_context_and_preserves_argument_boundaries(
    tmp_path, caller_environment, monkeypatch, hostname
):
    config = _config(tmp_path, hostname=hostname)
    config.compute["test"].forward_env += ["SLURM_JOB_ID", "GYM_POOL_GPU_NODES"]
    monkeypatch.setenv("SLURM_JOB_ID", "stale-submit-job")
    monkeypatch.setenv("GYM_POOL_GPU_NODES", "stale-submit-host")
    runtime_context = {
        "OLDPWD": "/previous/task/location",
        "HOSTNAME": "task-host",
        "HEAD_NODE_IP": "192.0.2.1",
        "RAY_HEAD_NODE_IP": "192.0.2.1:6379",
        "GYM_RAY_ADDRESS_POLICY": "192.0.2.2:6380",
        "GYM_RAY_POLICY_GPU_WORKERS": "worker-3,worker-4",
        "GYM_DRIVER_FAILURE_FILE": "/task/failure-marker",
        "GYM_PREVIOUS_JOB_ID": "41",
    }
    for name in ["PWD", *runtime_context]:
        monkeypatch.setenv(name, "stale-submit-context")
        config.compute["test"].forward_env.append(name)
    benchmark_dir = Path(config.job.output_path) / "run" / "benchmark"
    environment = {
        "CURRENT_JOB": "runtime:SLURM_JOB_ID",
        "POOL_HOST": "runtime:GYM_POOL_GPU_NODES",
        **{f"TASK_{name}": f"runtime:{name}" for name in ["PWD", *runtime_context]},
    }
    staged = _shell("\n".join(stage_commands(config, {"driver": environment}, benchmark_dir)))
    assert staged.returncode == 0, staged.stderr
    source = (
        "import json,os,sys; "
        "print(json.dumps({'job': os.environ['CURRENT_JOB'], 'host': os.environ['POOL_HOST'], 'args': sys.argv[1:], "
        "'context': {name: os.environ['TASK_' + name] for name in " + repr(["PWD", *runtime_context]) + "}}))"
    )
    command, prefix = task_command(
        shlex.join([sys.executable, "-c", source]) + ' "$@"',
        directory=private_directory(benchmark_dir),
        role="driver",
        environment=environment,
        arguments=" -- 'argument with spaces' '$literal'",
    )
    result = _shell(
        "\n".join(f"export {name}={shlex.quote(value)}" for name, value in runtime_context.items())
        + "\n"
        + prefix
        + command,
        environment={
            "PATH": caller_environment["PATH"],
            "SLURM_JOB_ID": "42",
            "GYM_POOL_GPU_NODES": "worker-3",
        },
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "job": "42",
        "host": "worker-3",
        "args": ["argument with spaces", "$literal"],
        "context": {"PWD": str(Path.cwd()), **runtime_context},
    }


def test_runtime_alias_reads_the_caller_value_before_role_assignments(tmp_path, caller_environment):
    config = _config(tmp_path)
    benchmark_dir = Path(config.job.output_path) / "run" / "benchmark"
    environment = {"CANARY_TOKEN": "role-local-value", "Z_ALIAS": "runtime:CANARY_TOKEN"}
    staged = _shell("\n".join(stage_commands(config, {"driver": environment}, benchmark_dir)))
    assert staged.returncode == 0, staged.stderr
    command, prefix = task_command(
        _read_environment_command(["CANARY_TOKEN", "Z_ALIAS"]),
        directory=private_directory(benchmark_dir),
        role="driver",
        environment=environment,
    )
    result = _shell(prefix + command, environment={"PATH": caller_environment["PATH"]})
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "CANARY_TOKEN": "role-local-value",
        "Z_ALIAS": caller_environment["CANARY_TOKEN"],
    }


@pytest.mark.parametrize("hostname", [None, "this-host", "login.example"])
def test_implicit_local_defaults_preserve_task_values_and_runtime_aliases(
    tmp_path, caller_environment, monkeypatch, hostname
):
    monkeypatch.setattr("nemo_gym.orchestration.executors.connection.socket.gethostname", lambda: "this-host")
    config = _config(tmp_path, hostname=hostname)
    config.compute["test"].forward_env = []
    monkeypatch.setenv("LOCAL_DEFAULT", "caller-default")
    context = {
        "PATH": f"/task/image/bin:{Path(sys.executable).parent}:/usr/bin:/bin",
        "SLURMD_NODENAME": "task-node",
        "GPU_DEVICE_ORDINAL": "2,3",
        "ROCR_VISIBLE_DEVICES": "2,3",
        "ZE_AFFINITY_MASK": "2,3",
        "HIP_VISIBLE_DEVICES": "2,3",
        "OMPI_UNIVERSE_SIZE": "8",
        "OMPI_MCA_test_setting": "task-setting",
    }
    for name in context:
        if name != "PATH":
            monkeypatch.setenv(name, "caller-context")
    environment = {"TASK_" + name: "runtime:" + name for name in context}
    benchmark_dir = Path(config.job.output_path) / "run" / "benchmark"
    staged = _shell("\n".join(stage_commands(config, {"driver": environment}, benchmark_dir)))
    assert staged.returncode == 0, staged.stderr
    command, prefix = task_command(
        _read_environment_command([*context, *environment, "LOCAL_DEFAULT"]),
        directory=private_directory(benchmark_dir),
        role="driver",
        environment=environment,
    )
    result = _shell(prefix + command, environment=context)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        **context,
        **{"TASK_" + name: value for name, value in context.items()},
        "LOCAL_DEFAULT": "caller-default" if hostname != "login.example" else None,
    }


@pytest.mark.parametrize("hostname", [None, "this-host", "login.example"])
def test_runtime_alias_never_uses_an_implicit_caller_default(tmp_path, caller_environment, monkeypatch, hostname):
    monkeypatch.setattr("nemo_gym.orchestration.executors.connection.socket.gethostname", lambda: "this-host")
    monkeypatch.setenv("CALLER_ONLY", "implicit-caller-value")
    config = _config(tmp_path, hostname=hostname)
    config.compute["test"].forward_env = []
    benchmark_dir = Path(config.job.output_path) / "run" / "benchmark"
    environment = {"ALIAS": "runtime:CALLER_ONLY"}
    staged = _shell("\n".join(stage_commands(config, {"driver": environment}, benchmark_dir)))
    assert staged.returncode == 0, staged.stderr
    marker = tmp_path / "workload-started"
    command, prefix = task_command(
        shlex.join([sys.executable, "-c", "from pathlib import Path; Path(" + repr(str(marker)) + ").touch()"]),
        directory=private_directory(benchmark_dir),
        role="driver",
        environment=environment,
    )
    result = _shell(prefix + command, environment={"PATH": caller_environment["PATH"]})
    assert result.returncode != 0
    assert "CALLER_ONLY" in result.stderr
    assert "implicit-caller-value" not in result.stderr
    assert not marker.exists()


@pytest.mark.parametrize(
    "name",
    [
        "SBATCH_QOS",
        "SBATCH_RESERVATION",
        "SBATCH_EXPORT",
        "SBATCH_GET_USER_ENV",
        "SALLOC_ACCOUNT",
        "SRUN_CPUS_PER_TASK",
    ],
)
def test_scheduler_rejects_target_shell_option_defaults_without_printing_values(tmp_path, caller_environment, name):
    marker = tmp_path / "scheduler-ran"
    command = scheduler_command(
        shlex.join([sys.executable, "-c", "from pathlib import Path; Path(" + repr(str(marker)) + ").touch()"])
    )
    result = _shell(command, environment={"PATH": caller_environment["PATH"], name: "private-value-not-to-print"})
    assert result.returncode != 0
    assert name in result.stderr
    assert "extra_args" in result.stderr and "unset" in result.stderr
    assert "private-value-not-to-print" not in result.stderr + result.stdout
    assert not marker.exists()


@pytest.mark.parametrize("name", ["SLURM_ACCOUNT", "SLURM_PARTITION", "SLURM_QOS", "SLURM_RESERVATION"])
def test_safe_placement_defaults_reach_the_actual_scheduler_boundary(tmp_path, caller_environment, name):
    command = scheduler_command(_read_environment_command([name, "CANARY_TOKEN"]))
    result = _shell(
        command,
        environment={"PATH": caller_environment["PATH"], name: "test-placement", "CANARY_TOKEN": "private-canary"},
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {name: "test-placement", "CANARY_TOKEN": None}


def test_private_store_uses_writable_output_root_without_parent_write_access(tmp_path, caller_environment):
    parent = tmp_path / "shared-parent"
    output = parent / "jobs"
    output.mkdir(parents=True)
    parent.chmod(0o555)
    config = _config(tmp_path)
    config.job.output_path = str(output)
    benchmark_dir = output / "run" / "benchmark"
    try:
        prepared = _shell("\n".join(prepare_commands(benchmark_dir)))
        assert prepared.returncode == 0, prepared.stderr
        directory = private_directory(benchmark_dir)
        assert directory.is_relative_to(output)
        assert not directory.is_relative_to(benchmark_dir.parent)
        assert not (directory / "environment.json").exists()
        staged = _shell("\n".join(stage_commands(config, {"driver": {}}, benchmark_dir)))
        assert staged.returncode == 0, staged.stderr
        assert stat.S_IMODE(directory.stat().st_mode) in (0o700, 0o2700)
        assert stat.S_IMODE((directory / "environment.json").stat().st_mode) == 0o600
        assert list(parent.iterdir()) == [output]
    finally:
        parent.chmod(0o755)


@pytest.mark.parametrize("refusal", ["private_permissions", "remote_scheduler_default"])
def test_private_preflight_refuses_before_a_fresh_run_is_copied(tmp_path, caller_environment, monkeypatch, refusal):
    config = _config(tmp_path, hostname="login.example")
    benchmark_dir = Path(config.job.output_path) / "fixed-run" / "benchmark"
    directory = private_directory(benchmark_dir)
    directory.mkdir(mode=0o700, parents=True)
    directory.parent.chmod(0o700)
    directory.chmod(0o755 if refusal == "private_permissions" else 0o700)
    previous = directory / "environment.json"
    previous.write_bytes(b"existing-private-payload")
    previous.chmod(0o600)
    copied = []

    class Connection(LocalConnection):
        def run(self, commands):
            with monkeypatch.context() as remote:
                if refusal == "remote_scheduler_default":
                    remote.setenv("SBATCH_QOS", "private-default-not-to-print")
                return super().run(commands)

        def copy(self, local, remote):
            copied.append(remote)
            raise AssertionError("private preflight refusal must precede copy")

    monkeypatch.setattr(slurm_module, "get_connection", lambda hostname: Connection())
    monkeypatch.setattr(slurm_module, "new_gym_job_id", lambda now: "fixed-run")
    with pytest.raises(
        RuntimeError, match="owner-only" if refusal == "private_permissions" else "SBATCH_QOS"
    ) as error:
        slurm_module.SlurmExecutor().run(config)
    assert "private-default-not-to-print" not in str(error.value)
    assert copied == []
    assert not benchmark_dir.parent.exists()
    assert previous.read_bytes() == b"existing-private-payload"


@pytest.mark.parametrize("hostname", [None, "login.example"])
@pytest.mark.parametrize("explicit_forward", [False, True])
def test_private_restoration_preserves_task_allocation_context(
    tmp_path, caller_environment, monkeypatch, hostname, explicit_forward
):
    config = _config(tmp_path, hostname=hostname)
    task_context = {
        "CUDA_VISIBLE_DEVICES": "2,3",
        "NVIDIA_VISIBLE_DEVICES": "GPU-task-2,GPU-task-3",
        "PMI_RANK": "3",
        "PMI_SIZE": "8",
        "PMIX_RANK": "3",
        "OMPI_COMM_WORLD_LOCAL_RANK": "1",
        "MV2_COMM_WORLD_RANK": "3",
        "MPI_LOCALRANKID": "1",
        "RANK": "3",
        "LOCAL_RANK": "1",
        "WORLD_SIZE": "8",
        "LOCAL_WORLD_SIZE": "2",
        "GROUP_RANK": "1",
        "ROLE_RANK": "3",
        "ROLE_WORLD_SIZE": "8",
        "NODE_RANK": "1",
    }
    for name in task_context:
        monkeypatch.setenv(name, "0,1" if name == "CUDA_VISIBLE_DEVICES" else "stale-caller-context")
    config.compute["test"].forward_env = list(task_context) if explicit_forward else []
    benchmark_dir = Path(config.job.output_path) / "run" / "benchmark"
    environment = {f"TASK_{name}": f"runtime:{name}" for name in task_context}
    environment["CUDA_VISIBLE_DEVICES"] = "runtime:CUDA_VISIBLE_DEVICES"
    staged = _shell("\n".join(stage_commands(config, {"driver": environment}, benchmark_dir)))
    assert staged.returncode == 0, staged.stderr
    command, prefix = task_command(
        _read_environment_command([*task_context, *environment]),
        directory=private_directory(benchmark_dir),
        role="driver",
        environment=environment,
    )
    result = _shell(prefix + command, environment={"PATH": caller_environment["PATH"], **task_context})
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        **task_context,
        **{f"TASK_{name}": value for name, value in task_context.items()},
    }


@pytest.mark.parametrize("hostname", [None, "login.example"])
def test_missing_task_gpu_context_does_not_fall_back_to_the_caller(
    tmp_path, caller_environment, monkeypatch, hostname
):
    config = _config(tmp_path, hostname=hostname)
    config.compute["test"].forward_env = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    benchmark_dir = Path(config.job.output_path) / "run" / "benchmark"
    environment = {"GPU_ALIAS": "runtime:CUDA_VISIBLE_DEVICES"}
    staged = _shell("\n".join(stage_commands(config, {"driver": environment}, benchmark_dir)))
    assert staged.returncode == 0, staged.stderr
    marker = tmp_path / "workload-started"
    command, prefix = task_command(
        shlex.join([sys.executable, "-c", "from pathlib import Path; Path(" + repr(str(marker)) + ").touch()"]),
        directory=private_directory(benchmark_dir),
        role="driver",
        environment=environment,
    )
    result = _shell(prefix + command, environment={"PATH": caller_environment["PATH"]})
    assert result.returncode != 0
    assert "CUDA_VISIBLE_DEVICES" in result.stderr
    assert not marker.exists()


@pytest.mark.parametrize("hostname", [None, "login.example"])
def test_forwarded_credentials_and_configured_literals_keep_their_task_precedence(
    tmp_path, caller_environment, monkeypatch, hostname
):
    config = _config(tmp_path, hostname=hostname)
    config.compute["test"].forward_env = ["HF_TOKEN"]
    monkeypatch.setenv("HF_TOKEN", "fake-caller-token")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    benchmark_dir = Path(config.job.output_path) / "run" / "benchmark"
    environment = {
        "TOKEN_ALIAS": "runtime:HF_TOKEN",
        "CUDA_VISIBLE_DEVICES": "configured-mask",
        "TASK_MASK": "runtime:CUDA_VISIBLE_DEVICES",
    }
    staged = _shell("\n".join(stage_commands(config, {"driver": environment}, benchmark_dir)))
    assert staged.returncode == 0, staged.stderr
    command, prefix = task_command(
        _read_environment_command(["HF_TOKEN", *environment]),
        directory=private_directory(benchmark_dir),
        role="driver",
        environment=environment,
    )
    result = _shell(
        prefix + command,
        environment={
            "PATH": caller_environment["PATH"],
            "HF_TOKEN": "baked-placeholder",
            "CUDA_VISIBLE_DEVICES": "2,3",
        },
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "HF_TOKEN": "fake-caller-token",
        "TOKEN_ALIAS": "fake-caller-token",
        "CUDA_VISIBLE_DEVICES": "configured-mask",
        "TASK_MASK": "2,3",
    }


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "symlink_file",
        "symlink_directory",
        "file_permissions",
        "directory_permissions",
        "setgid_file",
        "setgid_group_read",
        "setgid_group_write",
        "setgid_group_execute",
    ],
)
def test_task_does_not_execute_with_an_unavailable_or_unsafe_private_payload(tmp_path, caller_environment, damage):
    config = _config(tmp_path)
    benchmark_dir = Path(config.job.output_path) / "run" / "benchmark"
    staged = _shell("\n".join(stage_commands(config, {"driver": {}}, benchmark_dir)))
    assert staged.returncode == 0, staged.stderr
    directory = private_directory(benchmark_dir)
    payload = directory / "environment.json"
    if damage == "missing":
        payload.unlink()
    elif damage == "symlink_file":
        target = tmp_path / "moved.json"
        payload.rename(target)
        payload.symlink_to(target)
    elif damage == "symlink_directory":
        target = tmp_path / "moved"
        directory.rename(target)
        directory.symlink_to(target, target_is_directory=True)
    elif damage == "file_permissions":
        payload.chmod(0o644)
    elif damage == "directory_permissions":
        directory.chmod(0o755)
    elif damage == "setgid_file":
        payload.chmod(0o2600)
        assert stat.S_IMODE(payload.stat().st_mode) == 0o2600
    else:
        directory.chmod(
            {"setgid_group_read": 0o2740, "setgid_group_write": 0o2720, "setgid_group_execute": 0o2710}[damage]
        )
    executed = tmp_path / "task-executed"
    command, prefix = task_command(
        shlex.join(
            [sys.executable, "-c", "from pathlib import Path; Path(__import__('sys').argv[1]).touch()", str(executed)]
        ),
        directory=directory,
        role="driver",
        environment={},
    )
    result = _shell(prefix + command, environment={"PATH": caller_environment["PATH"]})
    assert result.returncode != 0
    assert not executed.exists()
    assert str(directory) in result.stderr
    assert caller_environment["CANARY_TOKEN"] not in result.stderr


@pytest.mark.parametrize("damage", ["permissions", "symlink"])
def test_staging_refuses_an_unsafe_private_root(tmp_path, caller_environment, damage):
    config = _config(tmp_path)
    benchmark_dir = Path(config.job.output_path) / "run" / "benchmark"
    directory = private_directory(benchmark_dir)
    directory.parent.parent.mkdir(parents=True)
    if damage == "permissions":
        directory.parent.mkdir(mode=0o755)
    else:
        target = tmp_path / "other-root"
        target.mkdir(mode=0o700)
        directory.parent.symlink_to(target, target_is_directory=True)
    result = _shell("\n".join(stage_commands(config, {"driver": {}}, benchmark_dir)))
    assert result.returncode != 0
    assert "owner-only" in result.stderr
    assert not (directory / "environment.json").exists()


@pytest.mark.parametrize("location", ["store", "benchmark"])
@pytest.mark.parametrize("access", [0o040, 0o020, 0o010])
def test_staging_refuses_group_access_on_setgid_directories(tmp_path, caller_environment, location, access):
    config = _config(tmp_path)
    benchmark_dir = Path(config.job.output_path) / "run" / "benchmark"
    directory = private_directory(benchmark_dir)
    directory.parent.mkdir(mode=0o700, parents=True)
    directory.mkdir(mode=0o700)
    unsafe = directory.parent if location == "store" else directory
    unsafe.chmod(0o2700 | access)

    result = _shell("\n".join(stage_commands(config, {"driver": {}}, benchmark_dir)))

    assert result.returncode != 0
    assert "owner-only" in result.stderr
    assert not (directory / "environment.json").exists()
    assert stat.S_IMODE(unsafe.stat().st_mode) == 0o2700 | access


def test_missing_forwarded_value_fails_before_staging(tmp_path, caller_environment, monkeypatch):
    config = _config(tmp_path)
    benchmark_dir = Path(config.job.output_path) / "run" / "benchmark"
    monkeypatch.delenv("CANARY_TOKEN")
    with pytest.raises(ValueError, match="compute.forward_env variables are unset: CANARY_TOKEN"):
        stage_commands(config, {"driver": {}}, benchmark_dir)
    assert not private_directory(benchmark_dir).exists()


@pytest.mark.parametrize("container_env", [[], ["UV_CONSTRAINT"], ["UV_CONSTRAINT", "PATH"]])
def test_private_driver_value_overrides_baked_image_env_without_passing_it_through_pyxis(
    tmp_path, caller_environment, container_env
):
    config = _config(tmp_path)
    config.driver.env = {"UV_CONSTRAINT": "/host/constraints.txt"}
    config.driver.container_env = container_env
    benchmark_dir = Path(config.job.output_path) / "run" / "benchmark"
    environments = task_environments(config, "benchmark", config.compute["test"], benchmark_dir)
    staged = _shell("\n".join(stage_commands(config, environments, benchmark_dir)))
    assert staged.returncode == 0, staged.stderr
    command, prefix = task_command(
        _read_environment_command(["UV_CONSTRAINT"]),
        directory=private_directory(benchmark_dir),
        role="driver",
        environment=environments["driver"],
    )
    result = _shell(
        prefix + command,
        environment={"PATH": caller_environment["PATH"], "UV_CONSTRAINT": "/image/constraints.txt"},
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"UV_CONSTRAINT": "/host/constraints.txt"}
    script = build_sbatch_script(
        config, "benchmark", config.driver.benchmarks["benchmark"], config.compute["test"], benchmark_dir
    )
    assert "/host/constraints.txt" not in script
    assert ("--container-env=UV_CONSTRAINT" in script) is ("UV_CONSTRAINT" in container_env)
    assert "--container-env=PATH" not in script
    assert config.model_dump()["driver"]["container_env"] == container_env


@pytest.mark.parametrize("hostname", [None, "login.example"])
@pytest.mark.parametrize("role_literal", [False, True])
def test_container_selection_is_private_driver_only_and_precedes_runtime_aliases(
    tmp_path, caller_environment, monkeypatch, hostname, role_literal
):
    selected = "SELECTED_CANARY"
    caller = "runtime:literal-selected-canary"
    monkeypatch.setenv(selected, caller)
    config = _config(tmp_path, hostname=hostname)
    config.compute["test"].forward_env = []
    config.services["policy"] = RayServiceConfig(type="ray", container="python:3", placement="test")
    config.driver.container_env = [selected]
    config.driver.env = {"ALIAS": f"runtime:{selected}"}
    if role_literal:
        config.driver.env[selected] = "driver-role-literal"
    benchmark_dir = Path(config.job.output_path) / "run" / "benchmark"
    roles = task_environments(config, "benchmark", config.compute["test"], benchmark_dir)
    staged = _shell("\n".join(stage_commands(config, roles, benchmark_dir)))
    assert staged.returncode == 0, staged.stderr
    directory = private_directory(benchmark_dir)
    for role in ("driver", "service:policy", "collector"):
        command, prefix = task_command(
            _read_environment_command([selected, "ALIAS"]), directory=directory, role=role, environment=roles[role]
        )
        result = _shell(
            prefix + command,
            environment={"PATH": caller_environment["PATH"], "SLURM_JOB_ID": "42", selected: "image-placeholder"},
        )
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == (
            {selected: "driver-role-literal" if role_literal else caller, "ALIAS": caller}
            if role == "driver"
            else {selected: "image-placeholder", "ALIAS": None}
        )
    script = build_sbatch_script(
        config, "benchmark", config.driver.benchmarks["benchmark"], config.compute["test"], benchmark_dir
    )
    assert f"--container-env={selected}" not in script
    assert caller not in script
    assert "driver-role-literal" not in script


@pytest.mark.parametrize("selected", ["TASK_RUNTIME_SELECTION", "CUDA_VISIBLE_DEVICES"])
def test_runtime_container_selections_keep_native_host_priority(tmp_path, caller_environment, monkeypatch, selected):
    config = _config(tmp_path)
    config.compute["test"].forward_env = []
    config.driver.container_env = [selected]
    config.driver.env = {selected: f"runtime:{selected}", "ALIAS": f"runtime:{selected}"}
    if selected == "CUDA_VISIBLE_DEVICES":
        monkeypatch.setenv(selected, "stale-caller-mask")
    else:
        monkeypatch.delenv(selected, raising=False)
    benchmark_dir = Path(config.job.output_path) / "run" / "benchmark"
    roles = task_environments(config, "benchmark", config.compute["test"], benchmark_dir)
    staged = _shell("\n".join(stage_commands(config, roles, benchmark_dir)))
    assert staged.returncode == 0, staged.stderr
    script = build_sbatch_script(
        config, "benchmark", config.driver.benchmarks["benchmark"], config.compute["test"], benchmark_dir
    )
    flags = next(word for word in shlex.split(script) if word.startswith("--container-env="))
    selected_names = flags.split("=", 1)[1].split(",")
    image = {"PATH": caller_environment["PATH"], selected: "image-placeholder"}
    if selected in selected_names:
        image[selected] = "host-runtime-value"
    command, prefix = task_command(
        _read_environment_command([selected, "ALIAS"]),
        directory=private_directory(benchmark_dir),
        role="driver",
        environment=roles["driver"],
    )
    result = _shell(prefix + command, environment=image)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {selected: "host-runtime-value", "ALIAS": "host-runtime-value"}


def test_rendered_job_keeps_scheduler_steps_clean_and_restores_driver_service_and_collector(
    tmp_path, caller_environment, monkeypatch
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    monkeypatch.setenv("PATH", f"{bin_dir}:{caller_environment['PATH']}")
    monkeypatch.setenv("OTEL_CANARY", "telemetry-private-canary")
    roles = ("driver", "policy", "collector")
    for role in roles:
        binary = bin_dir / {"driver": "gym", "policy": "vllm", "collector": "otelcol"}[role]
        binary.write_text(
            f"#!{sys.executable}\n"
            "import json,os,sys,time\n"
            "from pathlib import Path\n"
            f"receipts = Path({str(receipts)!r})\n"
            f"role = {role!r}\n"
            "names = ['CANARY_TOKEN', 'TOKEN_ALIAS', 'ROLE_SECRET', 'OTEL_CANARY', 'SLURM_JOB_ID']\n"
            "receipt = {'env': {name: os.environ.get(name) for name in names}, 'argv': sys.argv[1:], 'pid': os.getpid()}\n"
            "(receipts / (role + '.json')).write_text(json.dumps(receipt))\n"
            "if role == 'driver':\n"
            "    deadline = time.monotonic() + 5\n"
            "    while not all((receipts / (name + '.json')).exists() for name in ['policy', 'collector']):\n"
            "        if time.monotonic() > deadline: raise RuntimeError('task children did not start')\n"
            "        time.sleep(0.01)\n"
            "else:\n"
            "    time.sleep(30)\n"
        )
        binary.chmod(0o755)
    srun = bin_dir / "srun"
    srun.write_text(
        f"#!{sys.executable}\n"
        "import json,os,sys\n"
        "from pathlib import Path\n"
        "arguments = sys.argv[1:]\n"
        "output = next(value.split('=', 1)[1] for value in arguments if value.startswith('--output='))\n"
        "role = Path(output).name.rsplit('-', 1)[0]\n"
        "role = 'collector' if role == 'otel_collector' else role\n"
        f"receipt = Path({str(receipts)!r}) / ('scheduler-' + role + '.json')\n"
        "receipt.write_text(json.dumps({'env': dict(os.environ), 'argv': arguments}))\n"
        "while arguments and arguments[0].startswith('--'):\n"
        "    arguments.pop(0)\n"
        "os.execvpe(arguments[0], arguments, os.environ)\n"
    )
    srun.chmod(0o755)
    pkill = bin_dir / "pkill"
    pkill.write_text(
        f"#!{sys.executable}\n"
        "import json,os,signal\n"
        f"with open({str(receipts / 'collector.json')!r}) as stream: pid = json.load(stream)['pid']\n"
        "os.kill(pid, signal.SIGTERM)\n"
    )
    pkill.chmod(0o755)
    for name, source in {"curl": "exit 0", "sleep": "exec /bin/sleep 0.01"}.items():
        binary = bin_dir / name
        binary.write_text("#!/bin/bash\n" + source + "\n")
        binary.chmod(0o755)

    config = SubmitConfig.model_validate(
        {
            "services": {
                "policy": {
                    "type": "vllm",
                    "container": "policy:test",
                    "model": "org/model",
                    "env": {"TOKEN_ALIAS": "runtime:CANARY_TOKEN", "ROLE_SECRET": "lit:policy-private-canary"},
                }
            },
            "compute": {
                "test": {
                    "type": "slurm",
                    "account": "test-account",
                    "hostname": "login.example",
                    "forward_env": ["CANARY_TOKEN"],
                }
            },
            "driver": {
                "container": "driver:test",
                "env": {"TOKEN_ALIAS": "runtime:CANARY_TOKEN", "ROLE_SECRET": "lit:driver-private-canary"},
                "benchmarks": {"benchmark": {"run": {"message": "argument with spaces", "literal": "$literal"}}},
            },
            "job": {"output_path": str(tmp_path / "output")},
            "otel": {
                "enabled": True,
                "remote_export": True,
                "token_env": "OTEL_CANARY",
                "endpoint": "https://telemetry.example",
                "service_name": "test",
                "binary": str(bin_dir / "otelcol"),
                "container": "collector:test",
                "gym_telemetry": False,
            },
        }
    )
    benchmark_dir = Path(config.job.output_path) / "run" / "benchmark"
    benchmark_dir.mkdir(parents=True)
    (benchmark_dir / "logs").mkdir()
    compute = config.compute["test"]
    environments = task_environments(config, "benchmark", compute, benchmark_dir)
    staged = _shell("\n".join(stage_commands(config, environments, benchmark_dir)))
    assert staged.returncode == 0, staged.stderr
    script = build_sbatch_script(config, "benchmark", config.driver.benchmarks["benchmark"], compute, benchmark_dir)
    canaries = [
        caller_environment["CANARY_TOKEN"],
        "complex-secret-canary",
        "driver-private-canary",
        "policy-private-canary",
        "telemetry-private-canary",
    ]
    for canary in canaries:
        assert canary not in script
    job = benchmark_dir / "job.sh"
    job.write_text(script)
    with (benchmark_dir / "job.log").open("w") as output:
        process = subprocess.Popen(
            ["bash", "-c", scheduler_command('env SLURM_JOB_ID=42 bash "$1"'), "--", str(job)],
            cwd=benchmark_dir,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=20)
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
    assert returncode == 0, (benchmark_dir / "job.log").read_text()
    children = {role: json.loads((receipts / f"{role}.json").read_text()) for role in roles}
    assert children["driver"]["argv"][:2] == ["eval", "run"]
    assert "+message=argument with spaces" in children["driver"]["argv"]
    assert "+literal=$literal" in children["driver"]["argv"]
    for role in ("driver", "policy"):
        assert children[role]["env"]["CANARY_TOKEN"] == caller_environment["CANARY_TOKEN"]
        assert children[role]["env"]["TOKEN_ALIAS"] == caller_environment["CANARY_TOKEN"]
        assert children[role]["env"]["ROLE_SECRET"] == f"{role}-private-canary"
    assert children["collector"]["env"]["OTEL_CANARY"] == "telemetry-private-canary"
    assert children["collector"]["env"]["SLURM_JOB_ID"] == "42"
    for role in roles:
        scheduler = json.loads((receipts / f"scheduler-{role}.json").read_text())
        assert "CANARY_TOKEN" not in scheduler["env"]
        assert "TOKEN_ALIAS" not in scheduler["env"]
        assert "ROLE_SECRET" not in scheduler["env"]
        assert "OTEL_CANARY" not in scheduler["env"]
        for canary in canaries:
            assert all(canary not in value for value in scheduler["env"].values())
            assert all(canary not in value for value in scheduler["argv"])
        assert any(
            f"{private_directory(benchmark_dir)}:{private_directory(benchmark_dir)}:ro" in arg
            for arg in scheduler["argv"]
        )


@pytest.mark.parametrize("name", ["BAD-NAME", "X=Y", "", "1NAME", "TOKEN\nOTHER"])
def test_driver_container_selection_requires_an_environment_name(tmp_path, name):
    raw = _config(tmp_path).model_dump()
    raw["driver"]["container_env"] = [name]
    with pytest.raises(ValueError, match="container_env entries must be environment variable names"):
        SubmitConfig.model_validate(raw)


def test_driver_container_selection_survives_saved_config_roundtrip(tmp_path):
    config = _config(tmp_path)
    config.driver.container_env = ["HF_TOKEN", "CUDA_VISIBLE_DEVICES"]
    restored = SubmitConfig.model_validate(config.model_dump())
    assert restored.driver.container_env == ["HF_TOKEN", "CUDA_VISIBLE_DEVICES"]
