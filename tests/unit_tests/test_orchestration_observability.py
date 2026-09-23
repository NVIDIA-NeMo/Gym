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

import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from nemo_gym.orchestration.api import SubmitConfig
from nemo_gym.orchestration.executors import otel as obs_module
from nemo_gym.orchestration.executors import slurm as slurm_module
from nemo_gym.orchestration.executors.connection import LocalConnection
from nemo_gym.orchestration.executors.otel import (
    COLLECTOR_HEALTH_PORT,
    OTLP_GRPC_PORT,
    OTLP_HTTP_PORT,
    collector_config_path,
    observability_active,
    render_collector_config,
    resolve_token,
    scrape_targets,
)
from nemo_gym.orchestration.executors.slurm import SlurmExecutor
from nemo_gym.orchestration.executors.slurm_script import build_sbatch_script
from nemo_gym.orchestration.jobs import RESOLVED_CONFIG_NAME


BENCH_DIR = Path("/remote/jobs/gym-job-20260921T100000Z-abc123/scicode")


def _config(**overrides):
    base = {
        "services": {"policy": {"type": "vllm", "container": "vllm:latest", "model": "/checkpoint", "port": 8000}},
        "compute": {"cluster-a": {"type": "slurm", "account": "acct", "hostname": None}},
        "driver": {"container": "gym:latest", "policy_model": "policy", "benchmarks": {"scicode": {}}},
        "job": {"output_path": "/remote/jobs"},
        "observability": {"endpoint": "https://otlp.example.com", "service_name": "my-registered-service"},
    }
    if "observability" in overrides:
        overrides = {**overrides, "observability": {**base["observability"], **overrides["observability"]}}
    return SubmitConfig.model_validate({**base, **overrides})


def _script(config, bench_dir=BENCH_DIR):
    compute = next(iter(config.compute.values()))
    return build_sbatch_script(config, "scicode", config.driver.benchmarks["scicode"], compute, bench_dir)


@pytest.fixture(autouse=True)
def _fixed_user(monkeypatch):
    monkeypatch.setattr(obs_module.getpass, "getuser", lambda: "someone")


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


def test_observability_is_on_by_default():
    assert _config().observability.enabled is True


def test_observability_can_be_disabled():
    assert _config(observability={"enabled": False}).observability.enabled is False


def test_observability_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        _config(observability={"enabld": False})


def test_observability_rejects_invalid_token_env_name():
    with pytest.raises(ValidationError, match="token_env"):
        _config(observability={"token_env": "not a var"})


def test_observability_is_part_of_the_resolved_config():
    dumped = _config().model_dump(mode="json")
    assert dumped["observability"]["service_name"] == "my-registered-service"
    assert dumped["observability"]["token_env"] == "OBSERVABILITY_TOKEN"


def test_observability_has_no_destination_by_default():
    bare = SubmitConfig.model_validate(
        {
            "services": {},
            "compute": {"c": {"type": "slurm", "account": "a"}},
            "driver": {"container": "gym:latest", "benchmarks": {}},
            "job": {"output_path": "/remote"},
        }
    )
    assert bare.observability.enabled is True
    assert bare.observability.endpoint is None
    assert bare.observability.service_name is None


# ---------------------------------------------------------------------------
# activation + token
# ---------------------------------------------------------------------------


def test_scrape_targets_are_the_model_services():
    config = _config(
        services={
            "policy": {"type": "vllm", "container": "vllm:latest", "model": "m", "port": 8000},
            "judge": {"type": "vllm", "container": "vllm:latest", "model": "j", "port": 8100},
            "head": {"type": "ray", "container": "ray:latest"},
        }
    )
    assert scrape_targets(config) == {"policy": 8000, "judge": 8100}


def test_inactive_without_a_model_service():
    driver = {"container": "gym:latest", "benchmarks": {"scicode": {}}}
    assert not observability_active(_config(services={}, driver=driver))
    assert not observability_active(
        _config(services={"head": {"type": "ray", "container": "ray:latest"}}, driver=driver)
    )


def test_inactive_when_disabled():
    assert not observability_active(_config(observability={"enabled": False}))


def test_resolve_token_reads_the_submitting_env(monkeypatch):
    monkeypatch.setenv("OBSERVABILITY_TOKEN", "secret-token")
    assert resolve_token(_config()) == "secret-token"


def test_resolve_token_honours_token_env(monkeypatch):
    monkeypatch.delenv("OBSERVABILITY_TOKEN", raising=False)
    monkeypatch.setenv("MY_TOKEN", "other")
    assert resolve_token(_config(observability={"token_env": "MY_TOKEN"})) == "other"


def test_resolve_token_missing_names_the_variable(monkeypatch):
    monkeypatch.delenv("OBSERVABILITY_TOKEN", raising=False)
    with pytest.raises(ValueError, match="OBSERVABILITY_TOKEN"):
        resolve_token(_config())


# ---------------------------------------------------------------------------
# collector config
# ---------------------------------------------------------------------------


def _rendered(config=None):
    return yaml.safe_load(render_collector_config(config or _config(), "scicode", BENCH_DIR))


def _attrs(doc):
    return {a["key"]: (a["value"], a["action"]) for a in doc["processors"]["resource"]["attributes"] if "value" in a}


def test_collector_converts_the_slurm_job_id_to_a_string():
    actions = _rendered()["processors"]["resource"]["attributes"]
    convert = [a for a in actions if a["action"] == "convert"]
    assert convert == [{"key": "slurm_job_id", "action": "convert", "converted_type": "string"}]
    assert actions.index(convert[0]) > max(
        i for i, a in enumerate(actions) if a["key"] == "slurm_job_id" and "value" in a
    )


def test_collector_scrapes_the_node_exporters_by_default_and_can_skip_them():
    by_job = {s["job_name"]: s for s in _rendered()["receivers"]["prometheus"]["config"]["scrape_configs"]}
    assert by_job["dcgm"]["static_configs"][0]["targets"] == ["localhost:9400"]
    assert by_job["node"]["static_configs"][0]["targets"] == ["localhost:9100"]
    assert by_job["dcgm"]["scrape_interval"] == "15s"

    off = _config(observability={"gpu_metrics_port": None, "node_metrics_port": 9200})
    jobs = {s["job_name"]: s for s in _rendered(off)["receivers"]["prometheus"]["config"]["scrape_configs"]}
    assert "dcgm" not in jobs
    assert jobs["node"]["static_configs"][0]["targets"] == ["localhost:9200"]


def test_collector_scrapes_every_model_service_on_localhost():
    config = _config(
        services={
            "policy": {"type": "vllm", "container": "vllm:latest", "model": "m", "port": 8000},
            "judge": {"type": "vllm", "container": "vllm:latest", "model": "j", "port": 8100},
        }
    )
    scrapes = _rendered(config)["receivers"]["prometheus"]["config"]["scrape_configs"]
    by_job = {s["job_name"]: s for s in scrapes}
    assert by_job["vllm-policy"]["static_configs"][0]["targets"] == ["localhost:8000"]
    assert by_job["vllm-judge"]["static_configs"][0]["targets"] == ["localhost:8100"]
    assert by_job["vllm-policy"]["static_configs"][0]["labels"] == {"gym_service": "policy"}
    assert by_job["vllm-policy"]["scrape_interval"] == "15s"


def test_collector_renames_colon_metrics_to_underscores_before_export():
    doc = _rendered()
    statements = doc["processors"]["transform/metric_names"]["metric_statements"]
    assert statements == [
        {"context": "metric", "statements": ['replace_pattern(metric.name, "^([^:]+):(.+)$", "$${1}_$${2}")']}
    ]
    assert doc["service"]["pipelines"]["metrics"]["processors"][0] == "transform/metric_names"


def test_collector_stamps_the_dashboard_labels():
    attrs = _attrs(_rendered())
    assert attrs["user"] == ("someone", "upsert")
    assert attrs["run_id"] == ("gym-job-20260921T100000Z-abc123", "upsert")
    assert attrs["slurm_job_id"] == ("${env:SLURM_JOB_ID}", "upsert")
    assert attrs["benchmark"] == ("scicode", "upsert")
    assert attrs["cluster"] == ("cluster-a", "upsert")
    assert attrs["model"] == ("/checkpoint", "upsert")


def test_collector_model_label_prefers_served_model_name():
    config = _config(
        services={
            "policy": {
                "type": "vllm",
                "container": "vllm:latest",
                "model": "/checkpoint",
                "served_model_name": "super-bf16",
            }
        }
    )
    assert _attrs(_rendered(config))["model"][0] == "super-bf16"


def test_collector_omits_model_label_without_a_policy_model():
    config = _config(driver={"container": "gym:latest", "benchmarks": {"scicode": {}}})
    assert "model" not in _attrs(_rendered(config))


def test_collector_routes_via_service_name_and_token_attribute():
    doc = _rendered()
    attrs = _attrs(doc)
    assert attrs["service.name"] == ("my-registered-service", "upsert")
    assert attrs["service.name.override"] == ("gym-vllm", "insert")
    assert attrs["Authorization"] == ("${env:OBSERVABILITY_TOKEN}", "upsert")
    exporter = doc["exporters"]["otlp_http/managed"]
    assert exporter["endpoint"] == "https://otlp.example.com"
    assert exporter["headers"]["Authorization"] == "Bearer ${env:OBSERVABILITY_TOKEN}"


def test_collector_token_reference_follows_token_env():
    doc = _rendered(_config(observability={"token_env": "MY_TOKEN"}))
    assert _attrs(doc)["Authorization"][0] == "${env:MY_TOKEN}"
    assert doc["exporters"]["otlp_http/managed"]["headers"]["Authorization"] == "Bearer ${env:MY_TOKEN}"


def test_collector_writes_a_local_copy_next_to_the_managed_export():
    doc = _rendered()
    assert doc["exporters"]["file/metrics"]["path"] == str(BENCH_DIR / "otel" / "metrics.jsonl")
    for signal in ("metrics", "traces", "logs"):
        exporters = doc["service"]["pipelines"][signal]["exporters"]
        assert exporters == ["otlp_http/managed", f"file/{signal}"]


def test_collector_receives_otlp_for_the_job_processes():
    protocols = _rendered()["receivers"]["otlp"]["protocols"]
    assert protocols["grpc"]["endpoint"] == f"0.0.0.0:{OTLP_GRPC_PORT}"
    assert protocols["http"]["endpoint"] == f"0.0.0.0:{OTLP_HTTP_PORT}"


def test_collector_exposes_a_health_check():
    doc = _rendered()
    assert doc["extensions"]["health_check"]["endpoint"] == f"0.0.0.0:{COLLECTOR_HEALTH_PORT}"
    assert doc["service"]["extensions"] == ["health_check"]


def test_collector_config_never_contains_the_token_value(monkeypatch):
    monkeypatch.setenv("OBSERVABILITY_TOKEN", "secret-token")
    assert "secret-token" not in render_collector_config(_config(), "scicode", BENCH_DIR)


# ---------------------------------------------------------------------------
# sbatch script
# ---------------------------------------------------------------------------


def test_script_starts_the_collector_before_the_model_service():
    script = _script(_config())
    assert script.index("# service: otel_collector") < script.index("# service: policy")


def _collector_line(script):
    return next(line for line in script.splitlines() if "--output=logs/otel_collector.log" in line)


def test_script_pins_the_collector_to_one_node_of_a_multi_node_job():
    """srun without node flags fans a step out to every node of the allocation; the collector must
    run once, on the batch host, where the vLLM head and the driver are."""
    multi = _config(
        compute={
            "cluster-a": {
                "type": "slurm",
                "account": "acct",
                "hostname": None,
                "node_pools": {"gpu": {"partition": "p", "nodes": 2, "ntasks_per_node": 1, "gpus_per_node": 8}},
            }
        }
    )
    line = _collector_line(_script(multi))
    assert " --nodes=1 --ntasks=1" in line
    assert " --nodes=1 --ntasks=1" not in _collector_line(_script(_config()))


def test_script_runs_the_collector_on_the_node_by_default():
    line = next(line for line in _script(_config()).splitlines() if "--output=logs/otel_collector.log" in line)
    assert "--container" not in line
    assert line.startswith("env ")
    assert (
        f" srun --overlap --output=logs/otel_collector.log otelcol-contrib --config {collector_config_path(BENCH_DIR)} &"
        in line
    )


def test_script_runs_the_collector_in_a_container_with_the_job_dir_mounted_when_one_is_set():
    config = _config(observability={"container": "/shared/images/otelcol.sqsh", "binary": "/otelcol-contrib"})
    line = next(line for line in _script(config).splitlines() if "--output=logs/otel_collector.log" in line)
    assert "--container-image=/shared/images/otelcol.sqsh" in line
    assert "--no-container-mount-home" in line
    assert f"--container-mounts={BENCH_DIR}:{BENCH_DIR}" in line
    assert f"--container-workdir={BENCH_DIR}" in line
    assert f"/otelcol-contrib --config {collector_config_path(BENCH_DIR)}" in line
    assert line.rstrip().endswith("&")


def test_script_forwards_the_token_from_the_job_environment_not_a_literal(monkeypatch):
    monkeypatch.setenv("OBSERVABILITY_TOKEN", "secret-token")
    line = next(line for line in _script(_config()).splitlines() if "--output=logs/otel_collector.log" in line)
    assert "OBSERVABILITY_TOKEN=${OBSERVABILITY_TOKEN}" in line
    assert "SLURM_JOB_ID=${SLURM_JOB_ID}" in line
    assert "secret-token" not in line


def test_script_honours_a_binary_path_on_shared_storage():
    config = _config(observability={"binary": "/shared/tools/otelcol-contrib"})
    line = next(line for line in _script(config).splitlines() if "--output=logs/otel_collector.log" in line)
    assert " /shared/tools/otelcol-contrib --config " in line
    assert "--container" not in line


def test_script_health_checks_the_collector_before_the_model_service():
    script = _script(_config())
    assert f"Waiting for otel_collector at http://localhost:{COLLECTOR_HEALTH_PORT}" in script
    assert script.index("Waiting for otel_collector") < script.index("Waiting for policy")
    assert "seq 1 60" in script  # 300s / 5s


def test_script_flushes_the_collector_after_the_driver_and_keeps_the_driver_exit_code():
    script = _script(_config())
    tail = script[script.index("--output=logs/driver.log") :]
    assert "DRIVER_RC=$?" in tail
    # Anchored to the binary so the launching srun, whose command line also carries the path, is
    # not signalled: TERM to srun kills the step before the collector can flush.
    expected = re.escape("otelcol-contrib") + " --config " + re.escape(str(collector_config_path(BENCH_DIR)))
    assert f"pkill -TERM -u \"$USER\" -f -- '^{expected}'" in tail
    assert tail.rstrip().endswith("exit $DRIVER_RC")
    assert tail.index("sleep 20") < tail.index("pkill -TERM") < tail.index("kill -0 $OTEL_COLLECTOR_PID")
    # srun only gets TERM as a last resort, after the collector had its chance to flush.
    assert tail.index("kill -0 $OTEL_COLLECTOR_PID") < tail.index("kill -TERM $OTEL_COLLECTOR_PID")


def test_script_has_no_collector_when_disabled():
    script = _script(_config(observability={"enabled": False}))
    assert "otel_collector" not in script
    assert "DRIVER_RC" not in script


def test_script_has_no_collector_without_a_model_service():
    config = _config(services={}, driver={"container": "gym:latest", "benchmarks": {"scicode": {}}})
    assert "otel_collector" not in _script(config)


# ---------------------------------------------------------------------------
# executor
# ---------------------------------------------------------------------------


class _FakeConnection(LocalConnection):
    def __init__(self, replies):
        self._replies = replies
        self.commands = []

    def run(self, commands):
        self.commands.append(commands)
        return self._replies.pop(0)


def _install(monkeypatch, conn, tmp_path):
    monkeypatch.setattr(slurm_module, "get_connection", lambda hostname: conn)
    monkeypatch.setattr(slurm_module, "_validate_mounts", lambda config, connection: None)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


def _executor_config(tmp_path, **overrides):
    return _config(job={"output_path": str(tmp_path / "jobs")}, **overrides)


def test_submit_fails_before_staging_when_the_token_is_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("OBSERVABILITY_TOKEN", raising=False)
    conn = _FakeConnection([])
    _install(monkeypatch, conn, tmp_path)

    with pytest.raises(ValueError, match="OBSERVABILITY_TOKEN"):
        SlurmExecutor().run(_executor_config(tmp_path))

    assert conn.commands == []
    assert not (tmp_path / "jobs").exists()


def test_submit_fails_when_the_destination_is_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("OBSERVABILITY_TOKEN", "secret-token")
    config = _executor_config(tmp_path)
    config.observability.endpoint = None
    with pytest.raises(ValueError, match="observability.endpoint"):
        SlurmExecutor().run(config, dry_run=True)


def test_dry_run_fails_when_the_token_is_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("OBSERVABILITY_TOKEN", raising=False)
    with pytest.raises(ValueError, match="OBSERVABILITY_TOKEN"):
        SlurmExecutor().run(_executor_config(tmp_path), dry_run=True)


def test_dry_run_prints_the_collector_config(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OBSERVABILITY_TOKEN", "secret-token")
    SlurmExecutor().run(_executor_config(tmp_path), dry_run=True)
    out = capsys.readouterr().out
    assert "otel/collector.yaml for benchmark: scicode" in out
    assert "otlp_http/managed" in out
    assert "secret-token" not in out


def test_submit_stages_the_collector_config_and_exports_the_token_only_to_the_shell(tmp_path, monkeypatch):
    monkeypatch.setenv("OBSERVABILITY_TOKEN", "secret-token")
    conn = _FakeConnection(["__GYM_JOB:scicode:0:111 "])
    _install(monkeypatch, conn, tmp_path)

    record = SlurmExecutor().run(_executor_config(tmp_path))

    bench_dir = Path(record.run_dir) / "scicode"
    collector = yaml.safe_load((bench_dir / "otel" / "collector.yaml").read_text())
    assert _attrs(collector)["run_id"] == (record.gym_job_id, "upsert")
    assert "secret-token" not in (bench_dir / "job.sh").read_text()
    assert "secret-token" not in (bench_dir / "otel" / "collector.yaml").read_text()
    assert "secret-token" not in (Path(record.run_dir) / RESOLVED_CONFIG_NAME).read_text()

    commands = conn.commands[0]
    assert commands[0] == "export OBSERVABILITY_TOKEN=secret-token"
    assert commands[1].startswith("out=$(sbatch --parsable ")


def test_submit_without_observability_neither_stages_nor_exports(tmp_path, monkeypatch):
    monkeypatch.delenv("OBSERVABILITY_TOKEN", raising=False)
    conn = _FakeConnection(["__GYM_JOB:scicode:0:111 "])
    _install(monkeypatch, conn, tmp_path)

    record = SlurmExecutor().run(_executor_config(tmp_path, observability={"enabled": False}))

    assert not (Path(record.run_dir) / "scicode" / "otel").exists()
    assert not conn.commands[0][0].startswith("export ")
