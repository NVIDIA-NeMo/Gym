# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Kubernetes executor: one `kubectl apply` per benchmark, no SSH/rsync.

Assumes the kubeconfig is already set up (e.g. `tsh kube login <cluster>`) the same way the Slurm
executor assumes SSH keys/agent are already set up -- this executor does not drive cluster auth.
"""

import getpass
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import yaml

from nemo_gym import __version__
from nemo_gym.orchestration.api import KubernetesComputeConfig, SubmitConfig
from nemo_gym.orchestration.executors.base import BaseExecutor
from nemo_gym.orchestration.executors.kubernetes_script import build_job_manifest, job_name
from nemo_gym.orchestration.executors.otel import otel_active
from nemo_gym.orchestration.jobs import (
    BenchmarkJob,
    SubmissionRecord,
    installed_gym_commit,
    new_gym_job_id,
    utc_now,
    utc_timestamp,
)


# Kept in sync with the Slurm executor's benchmark-name check (`slurm.py:_VALID_BENCHMARK_NAME`):
# the name becomes part of a k8s Job name (DNS-1123), so it's restricted further than Slurm needs.
_VALID_BENCHMARK_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def _validate_benchmark_names(benchmarks: list[str]) -> None:
    bad = [name for name in benchmarks if not _VALID_BENCHMARK_NAME.match(name)]
    if bad:
        raise ValueError(
            f"Invalid benchmark name(s) {', '.join(map(repr, bad))}: names must match "
            f"{_VALID_BENCHMARK_NAME.pattern} to become part of a Kubernetes Job name."
        )


# kubectl can hang indefinitely (e.g. an expired tsh session retrying login), blocking the whole
# submit -- bound it and let the caller treat it as a per-benchmark failure instead.
_KUBECTL_TIMEOUT_SECONDS = 60


def _kubectl(compute: KubernetesComputeConfig, *args: str, input: str | None = None) -> subprocess.CompletedProcess:
    cmd = ["kubectl", *args, "-n", compute.namespace]
    if compute.context:
        cmd += ["--context", compute.context]
    return subprocess.run(cmd, input=input, text=True, capture_output=True, timeout=_KUBECTL_TIMEOUT_SECONDS)


class KubernetesExecutor(BaseExecutor):
    """Kubernetes executor. One Job per benchmark; see `kubernetes_script.py` for the pod shape."""

    def run(self, config: SubmitConfig, *, dry_run: bool = False) -> SubmissionRecord | None:
        compute = next(iter(config.compute.values()))
        assert isinstance(compute, KubernetesComputeConfig)
        cluster = next(iter(config.compute))
        benchmark_names = list(config.driver.benchmarks)
        _validate_benchmark_names(benchmark_names)
        if otel_active(config):
            raise ValueError(
                "otel is enabled (the default) but the kubernetes executor does not wire up a collector "
                "yet. Set `otel.enabled: false` explicitly to submit without it."
            )

        now = utc_now()
        gym_job_id = new_gym_job_id(now)
        base_run_dir = Path(config.job.output_path) / gym_job_id

        manifests = self._build_manifests(config, compute, gym_job_id, now, base_run_dir, benchmark_names)

        if dry_run:
            self._dry_run(manifests, gym_job_id)
            return None

        if shutil.which("kubectl") is None:
            raise RuntimeError("kubectl is not on PATH; install it and run `tsh kube login <cluster>` first.")

        benchmarks = []
        for name, job in manifests:
            rendered = yaml.safe_dump(job, sort_keys=False)
            try:
                result = _kubectl(compute, "apply", "-f", "-", input=rendered)
            except subprocess.TimeoutExpired:
                error = (
                    f"kubectl apply timed out after {_KUBECTL_TIMEOUT_SECONDS}s -- check that the kubeconfig "
                    "context is reachable and `tsh kube login <cluster>` (or equivalent) hasn't expired."
                )
                benchmarks.append(
                    BenchmarkJob(benchmark=name, job_dir=str(base_run_dir / name), job_id=None, error=error)
                )
                continue
            if result.returncode == 0:
                benchmarks.append(
                    BenchmarkJob(benchmark=name, job_dir=str(base_run_dir / name), job_id=job["metadata"]["name"])
                )
            else:
                error = (result.stderr or result.stdout).strip() or f"kubectl apply exited {result.returncode}"
                benchmarks.append(
                    BenchmarkJob(benchmark=name, job_dir=str(base_run_dir / name), job_id=None, error=error)
                )

        record = SubmissionRecord(
            gym_job_id=gym_job_id,
            gym_version=__version__,
            gym_commit=installed_gym_commit(),
            submitted_at=utc_timestamp(now),
            run_dir=str(base_run_dir),
            cluster=cluster,
            executor="kubernetes",
            hostname=None,
            submitted_by=getpass.getuser(),
            benchmarks=benchmarks,
            executor_metadata={"namespace": compute.namespace, "context": compute.context or ""},
        )
        # The manifest/config write already happened in each Job's own apply (kubernetes_script.py),
        # so write_manifest is a no-op here; only persist()'s local index write matters.
        self.persist(record, config, lambda _path, _content, *, private=False: None)
        return record

    def _build_manifests(
        self,
        config: SubmitConfig,
        compute: KubernetesComputeConfig,
        gym_job_id: str,
        now: datetime,
        base_run_dir: Path,
        benchmark_names: list[str],
    ) -> list[tuple[str, dict]]:
        resolved_config_yaml = yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)
        manifests = []
        for name in benchmark_names:
            benchmark = config.driver.benchmarks[name]
            run_dir = base_run_dir / name
            name_in_cluster = job_name(gym_job_id, name)
            record_stub = SubmissionRecord(
                gym_job_id=gym_job_id,
                gym_version=__version__,
                gym_commit=installed_gym_commit(),
                submitted_at=utc_timestamp(now),
                run_dir=str(run_dir),
                cluster=next(iter(config.compute)),
                executor="kubernetes",
                hostname=None,
                submitted_by=getpass.getuser(),
                benchmarks=[BenchmarkJob(benchmark=name, job_dir=str(run_dir), job_id=name_in_cluster)],
                executor_metadata={"namespace": compute.namespace, "context": compute.context or ""},
            )
            job = build_job_manifest(
                config,
                name,
                benchmark,
                compute,
                run_dir,
                name=name_in_cluster,
                gym_job_id=gym_job_id,
                resolved_config=resolved_config_yaml,
                manifest=record_stub.dumps(),
            )
            manifests.append((name, job))
        return manifests

    def _dry_run(self, manifests: list[tuple[str, dict]], gym_job_id: str) -> None:
        staging = Path(tempfile.mkdtemp(prefix=f"gym-dry-run-{gym_job_id}-"))
        for name, job in manifests:
            rendered = yaml.safe_dump(job, sort_keys=False)
            (staging / f"{name}.job.yaml").write_text(rendered)
            print(f"\n{'=' * 60}")
            print(f"[dry-run] kubernetes Job for benchmark: {name}")
            print(f"{'=' * 60}")
            print(rendered)
        print(f"\n[dry-run] rendered manifests written to {staging}")
