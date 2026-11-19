# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Submit one allocation with shared serving and separately pinned, supervised drivers."""

import json
import shlex
import tempfile
from collections.abc import Iterator
from pathlib import Path

import yaml

from nemo_gym.orchestration.api import HOST_ENV_REFS
from nemo_gym.orchestration.executors.base import BaseExecutor
from nemo_gym.orchestration.executors.capabilities import Inspectable, LogReadable, Terminable
from nemo_gym.orchestration.executors.connection import get_connection
from nemo_gym.orchestration.executors.git_ref import validate_gym_install_ref
from nemo_gym.orchestration.executors.otel import (
    COLLECTOR_CONFIG_NAME,
    COLLECTOR_DIR,
    COLLECTOR_SERVICE_NAME,
    otel_active,
    render_collector_config,
    resolve_token,
    validate_destination,
    validate_gym_telemetry,
)
from nemo_gym.orchestration.executors.script_templates import bash_var, render_driver_entrypoint, render_health_check
from nemo_gym.orchestration.executors.slurm.executor import SlurmExecutor, _validate_mounts
from nemo_gym.orchestration.executors.slurm.scheduler import _sbatch_command, _submission_env_exports
from nemo_gym.orchestration.executors.slurm.script import (
    _driver_node,
    _node_totals,
    _places_services,
    _probe_host,
    _render_collector_shutdown,
    _render_directives,
    _render_service_command,
    build_sbatch_parts,
    planned_log_sources,
)
from nemo_gym.orchestration.jobs import (
    ARTIFACTS_DIRNAME,
    LOGS_DIRNAME,
    BenchmarkJob,
    JobKey,
    JobRef,
    LogSource,
    ObservedStatus,
    SubmissionRecord,
    new_gym_job_id,
    utc_now,
)
from nemo_gym.orchestration.shared_allocation import SharedAllocationConfig


ALLOCATION_NAME = "shared-allocation"
ADMISSION_NAME = "gym_admission"


def render_shared_scripts(config: SharedAllocationConfig, directory: Path) -> dict[str, str]:
    """Render the controller and members through the same Slurm phases as ordinary submission."""
    common = config.common
    compute = next(iter(common.compute.values()))
    first_name = next(iter(config.members))
    parent = build_sbatch_parts(
        common,
        first_name,
        common.driver.benchmarks[first_name],
        compute,
        directory,
        shared_services_dir=directory,
    )
    services = [*common.services, ADMISSION_NAME]
    service_vars = [bash_var(name) + "_PID" for name in services]
    if otel_active(common):
        service_vars.append(bash_var(COLLECTOR_SERVICE_NAME) + "_PID")
    scripts = {}
    for name, member in config.members.items():
        child = member.model_copy(deep=True)
        child.driver.supervise = True
        benchmark = child.driver.benchmarks[name]
        benchmark.run.update(config.member_ports(name))
        benchmark.run["policy_base_url"] = f"http://localhost:{config.admission.port}/v1"
        parts = build_sbatch_parts(
            child,
            name,
            benchmark,
            compute,
            directory / name,
            shared_services_dir=directory,
            extra_watched_services=(ADMISSION_NAME,),
        )
        scripts[f"{name}/driver.sh"] = "#!/bin/bash\n" + parts.ray_prelude + "\n" + parts.driver_command + "\n"

    admission = config.admission
    policy = common.services[common.driver.policy_model]
    command = (
        f"python -m nemo_gym.models.policy_admission --upstream http://127.0.0.1:{policy.port} "
        f"--host 0.0.0.0 --port {admission.port} --max-inflight {admission.max_inflight} --max-queued {admission.max_queued}"
    )
    entrypoint = render_driver_entrypoint(
        repo=admission.gym_install.repo, ref=admission.gym_install.ref, prepare_cmd=None, command=command
    )
    node = _driver_node(common, compute)
    admission_step = _render_service_command(
        ADMISSION_NAME,
        admission.container,
        entrypoint,
        mounts=[f"{directory}:{directory}"],
        workdir=str(directory),
        nodes=1,
        ntasks=1,
        single_node=True,
        nodelist=f"gym_nodes[{node}]" if _places_services(common, _node_totals(compute)[0] > 1) else None,
        account=compute.account,
    )
    health = render_health_check(
        ADMISSION_NAME, admission.port, "/health", admission.health_check_timeout_seconds, _probe_host(node)
    )
    pid_list = " ".join(f"${{{name}:-}}" for name in service_vars)
    collector_shutdown = _render_collector_shutdown(common, directory, node) if otel_active(common) else ""
    # Each member shell waits for its supervised worker. Shared services stay alive until every such wait ends.
    cleanup = f"""set -uo pipefail
exec >> "logs/controller-$SLURM_JOB_ID.log" 2>&1
_gym_members=()
_gym_shared_cleanup() {{
    _gym_result=$?
    trap '' TERM INT
    trap - EXIT
    for pid in "${{_gym_members[@]}}"; do kill -TERM "$pid" 2>/dev/null || true; done
    for pid in "${{_gym_members[@]}}"; do wait "$pid" 2>/dev/null || true; done
    # The collector's own shutdown waits for its flush before other service steps end.
    ( (exit "$_gym_result"); {collector_shutdown or ":"} )
    _gym_services=({pid_list})
    for pid in "${{_gym_services[@]}}"; do kill -TERM "$pid" 2>/dev/null || true; done
    for _gym_pass in {{1..50}}; do
        _gym_live=0
        for pid in "${{_gym_services[@]}}"; do kill -0 "$pid" 2>/dev/null && _gym_live=1; done
        (( _gym_live )) || break
        sleep 0.1
    done
    for pid in "${{_gym_services[@]}}"; do kill -KILL "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; done
    printf '%s\\n' "$_gym_result" > eval.exit
    exit "$_gym_result"
}}
trap _gym_shared_cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
"""
    start = "\n".join(f"export {name}" for name in service_vars)
    for name in config.members:
        start += f'\n( cd {shlex.quote(str(directory / name))} && exec bash driver.sh ) &\n_gym_members+=("$!")\n'
    start += """
_gym_result=0
for pid in "${_gym_members[@]}"; do
    if wait "$pid"; then :; else _gym_result=1; fi
done
exit "$_gym_result"
"""
    scripts["job.sh"] = (
        "\n\n".join(
            [
                "#!/bin/bash\n" + _render_directives(compute, directory, ALLOCATION_NAME),
                cleanup,
                parent.ray_prelude,
                parent.service_commands,
                parent.health_checks,
                admission_step,
                health,
                start,
            ]
        )
        + "\n"
    )
    return scripts


class SharedSlurmExecutor(BaseExecutor, Terminable, LogReadable, Inspectable):
    """One scheduler job and one standard record; recovery is deliberately not a supported capability yet."""

    name = "slurm-shared"

    def run(self, config: SharedAllocationConfig, *, dry_run: bool = False) -> SubmissionRecord | None:
        common = config.common
        compute = next(iter(common.compute.values()))
        now = utc_now()
        gym_job_id = new_gym_job_id(now)
        remote = Path(config.job.output_path) / gym_job_id
        directory = remote / ALLOCATION_NAME
        for member in config.members.values():
            validate_gym_install_ref(member)
            if otel_active(member):
                validate_gym_telemetry(member)
        admission_config = common.model_copy(deep=True)
        admission_config.driver.gym_install = config.admission.gym_install
        validate_gym_install_ref(admission_config)
        token = None
        if otel_active(common):
            token = resolve_token(common)
            validate_destination(common)
        scripts = render_shared_scripts(config, directory)
        if dry_run:
            for name, text in scripts.items():
                print(f"[dry-run] {directory / name}\n{text}")
            return None
        exports = _submission_env_exports(compute)
        if token is not None:
            exports.append(f"export {common.otel.token_env}={shlex.quote(token)}")
        with tempfile.TemporaryDirectory(prefix="gym-submit-shared-") as staging_name:
            staging = Path(staging_name)
            staging.chmod(0o755)
            root = staging / ALLOCATION_NAME
            for name, script in scripts.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                (path.parent / LOGS_DIRNAME).mkdir(exist_ok=True)
                (path.parent / ARTIFACTS_DIRNAME).mkdir(exist_ok=True)
                path.write_text(script)
                path.chmod(0o600)
            for name, member in config.members.items():
                (root / name / "resolved-config.yaml").write_text(
                    yaml.safe_dump(member.model_dump(mode="json", context={HOST_ENV_REFS: True}))
                )
            (root / "members.json").write_text(
                json.dumps(
                    {
                        name: {
                            "directory": str(directory / name),
                            "ports": config.member_ports(name),
                            "policy_base_url": f"http://localhost:{config.admission.port}/v1",
                        }
                        for name in config.members
                    },
                    indent=2,
                )
                + "\n"
            )
            if otel_active(common):
                (root / COLLECTOR_DIR).mkdir()
                (root / COLLECTOR_DIR / COLLECTOR_CONFIG_NAME).write_text(
                    render_collector_config(common, ALLOCATION_NAME, directory)
                )
            with get_connection(compute.hostname) as conn:
                for member in config.members.values():
                    _validate_mounts(member, conn)
                conn.copy(staging, remote)
                output = conn.run(exports + [_sbatch_command(ALLOCATION_NAME, directory / "job.sh")])
                logs = [source for source in planned_log_sources(common, compute) if source.kind != "driver"]
                logs.append(LogSource(name="driver", kind="driver", location="logs/controller-{job_id}.log"))
                logs.append(LogSource(name=ADMISSION_NAME, kind="service", location="logs/gym_admission-{job_id}.log"))
                logs.extend(
                    LogSource(name=name, kind="driver", location=f"{name}/logs/driver-{{job_id}}.log")
                    for name in config.members
                )
                record = SlurmExecutor()._build_record(
                    next(iter(common.compute)), compute, gym_job_id, now, remote, [ALLOCATION_NAME], output, logs
                )
                record.executor = self.name
                record.environment = self.submission_environment(
                    config,
                    required_env=compute.forward_env,
                    forwarded_env=[*compute.forward_env, *([common.otel.token_env] if token is not None else [])],
                )
                self.persist(record, config, conn.write_text)
        return record

    def connection_key(self, record: SubmissionRecord) -> str:
        return SlurmExecutor().connection_key(record)

    def query_states(self, refs: list[JobRef]) -> dict[JobKey, ObservedStatus]:
        return SlurmExecutor().query_states(refs)

    def read_run_files(self, record: SubmissionRecord, paths: list[Path]) -> dict[Path, str | None]:
        return SlurmExecutor().read_run_files(record, paths)

    def terminate(self, record: SubmissionRecord, job: BenchmarkJob) -> None:
        SlurmExecutor().terminate(record, job)

    def stream_logs(
        self, record: SubmissionRecord, job: BenchmarkJob, *, source: LogSource, tail: int | None, follow: bool
    ) -> Iterator[str]:
        return SlurmExecutor().stream_logs(record, job, source=source, tail=tail, follow=follow)
