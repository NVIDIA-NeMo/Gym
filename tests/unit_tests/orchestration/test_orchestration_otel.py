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

import glob
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import time
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from nemo_gym.orchestration.api import SubmitConfig
from nemo_gym.orchestration.executors import otel as obs_module
from nemo_gym.orchestration.executors.connection import LocalConnection
from nemo_gym.orchestration.executors.otel import (
    COLLECTOR_HEALTH_PORT,
    OTLP_GRPC_PORT,
    OTLP_HTTP_PORT,
    collector_config_path,
    otel_active,
    render_collector_config,
    resolve_token,
    scrape_targets,
)
from nemo_gym.orchestration.executors.private_environment import private_directory
from nemo_gym.orchestration.executors.slurm import SlurmExecutor
from nemo_gym.orchestration.executors.slurm import chain as slurm_chain
from nemo_gym.orchestration.executors.slurm import executor as slurm_executor
from nemo_gym.orchestration.executors.slurm import judge_job as slurm_judge_job
from nemo_gym.orchestration.executors.slurm.resume_script import render_resume_epilogue
from nemo_gym.orchestration.executors.slurm.script import _collector_command, build_sbatch_script, task_environments
from nemo_gym.orchestration.jobs import RESOLVED_CONFIG_NAME


BENCH_DIR = Path("/remote/jobs/gym-job-20260921T100000Z-abc123/scicode")


def _config(**overrides):
    base = {
        "services": {"policy": {"type": "vllm", "container": "vllm:latest", "model": "/checkpoint", "port": 8000}},
        "compute": {"cluster-a": {"type": "slurm", "account": "acct", "hostname": None}},
        "driver": {"container": "gym:latest", "policy_model": "policy", "benchmarks": {"scicode": {}}},
        "job": {"output_path": "/remote/jobs"},
        "otel": {"endpoint": "https://otlp.example.com", "service_name": "my-registered-service"},
    }
    if "otel" in overrides:
        overrides = {**overrides, "otel": {**base["otel"], **overrides["otel"]}}
    return SubmitConfig.model_validate({**base, **overrides})


def _sync_extras(script: str) -> str:
    """The extras the driver install requests of uv; the checkout's pyproject.toml decides which of them it keeps."""
    match = re.search(r'_gym_sync_extras="\$\(_gym_declared_extras((?: --extra [\w-]+)+)\)"', script)
    assert match, "the driver install requests no extras"
    return match.group(1)


def _script(config, bench_dir=BENCH_DIR):
    compute = next(iter(config.compute.values()))
    return build_sbatch_script(config, "scicode", config.driver.benchmarks["scicode"], compute, bench_dir)


@pytest.fixture(autouse=True)
def _fixed_user(monkeypatch):
    monkeypatch.setattr(obs_module.getpass, "getuser", lambda: "someone")


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


def test_otel_is_on_by_default():
    assert _config().otel.enabled is True
    assert _config().otel.remote_export is True


def test_otel_can_be_disabled():
    assert _config(otel={"enabled": False}).otel.enabled is False


def test_otel_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        _config(otel={"enabld": False})


def test_otel_rejects_invalid_token_env_name():
    with pytest.raises(ValidationError, match="token_env"):
        _config(otel={"token_env": "not a var"})


def test_otel_is_part_of_the_resolved_config():
    dumped = _config().model_dump(mode="json")
    assert dumped["otel"]["service_name"] == "my-registered-service"
    assert dumped["otel"]["token_env"] == "OTEL_TOKEN"


def test_otel_has_no_destination_by_default():
    bare = SubmitConfig.model_validate(
        {
            "services": {},
            "compute": {"c": {"type": "slurm", "account": "a"}},
            "driver": {"container": "gym:latest", "benchmarks": {"b": {}}},
            "job": {"output_path": "/remote"},
        }
    )
    assert bare.otel.enabled is True
    assert bare.otel.endpoint is None
    assert bare.otel.service_name is None


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


def test_scrape_targets_include_both_tiers_of_a_pd_service():
    pd = {
        "type": "vllm_pd",
        "container": "vllm:latest",
        "model": "m",
        "prefill": {"node_pool": "prefill"},
        "decode": {"node_pool": "decode"},
    }
    pools = {"prefill": {"partition": "batch"}, "decode": {"partition": "batch"}}
    config = _config(
        services={"policy": pd},
        compute={"cluster-a": {"type": "slurm", "account": "acct", "node_pools": pools}},
    )
    assert scrape_targets(config) == {"policy-prefill": 8001, "policy-decode": 8002, "policy": 8000}


def test_active_without_a_model_service():
    """Gym's own servers produce telemetry with or without a local model, so the collector runs."""
    driver = {"container": "gym:latest", "benchmarks": {"scicode": {}}}
    assert otel_active(_config(services={}, driver=driver))
    assert otel_active(_config(services={"head": {"type": "ray", "container": "ray:latest"}}, driver=driver))


def test_collector_without_scrape_targets_has_no_prometheus_receiver():
    driver = {"container": "gym:latest", "benchmarks": {"scicode": {}}}
    doc = _rendered(_config(services={}, driver=driver, otel={"gpu_metrics_port": None, "node_metrics_port": None}))
    assert "prometheus" not in doc["receivers"]
    assert doc["service"]["pipelines"]["metrics"]["receivers"] == ["otlp", "span_metrics"]
    # With the node exporters on, the scrape jobs alone justify the receiver.
    doc = _rendered(_config(services={}, driver=driver))
    assert [s["job_name"] for s in doc["receivers"]["prometheus"]["config"]["scrape_configs"]] == ["dcgm", "node"]
    assert doc["service"]["pipelines"]["metrics"]["receivers"] == ["prometheus", "otlp", "span_metrics"]


def test_inactive_when_disabled():
    assert not otel_active(_config(otel={"enabled": False}))


def test_resolve_token_reads_the_submitting_env(monkeypatch):
    monkeypatch.setenv("OTEL_TOKEN", "secret-token")
    assert resolve_token(_config()) == "secret-token"


def test_resolve_token_honours_token_env(monkeypatch):
    monkeypatch.delenv("OTEL_TOKEN", raising=False)
    monkeypatch.setenv("MY_TOKEN", "other")
    assert resolve_token(_config(otel={"token_env": "MY_TOKEN"})) == "other"


@pytest.mark.parametrize("token", [None, ""])
def test_missing_token_disables_remote_export_with_a_warning(monkeypatch, caplog, token):
    monkeypatch.delenv("NVAUTH_TOKEN", raising=False)
    if token is not None:
        monkeypatch.setenv("NVAUTH_TOKEN", token)
    config = _config(otel={"token_env": "NVAUTH_TOKEN"})

    assert resolve_token(config) is None
    assert config.otel.enabled is True
    assert config.otel.remote_export is False
    assert "NVAUTH_TOKEN" in caplog.text
    assert "otel.remote_export=false" in caplog.text
    assert "otel/" in caplog.text
    resolve_token(config)
    assert len(caplog.records) == 1


@pytest.mark.parametrize("otel", [{"enabled": False}, {"remote_export": False}])
def test_disabled_remote_export_does_not_resolve_a_token(monkeypatch, caplog, otel):
    monkeypatch.setenv("OTEL_TOKEN", "unused-secret")
    assert resolve_token(_config(otel=otel)) is None
    assert not caplog.records


# ---------------------------------------------------------------------------
# collector config
# ---------------------------------------------------------------------------


def _rendered(config=None):
    return yaml.safe_load(render_collector_config(config or _config(), "scicode", BENCH_DIR))


def _attrs(doc, processor="resource/managed"):
    return {a["key"]: (a["value"], a["action"]) for a in doc["processors"][processor]["attributes"] if "value" in a}


def test_collector_converts_the_slurm_job_id_to_a_string():
    actions = _rendered()["processors"]["resource/local"]["attributes"]
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

    off = _config(otel={"gpu_metrics_port": None, "node_metrics_port": 9200})
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
    assert by_job["gym-vllm/policy"]["static_configs"][0]["targets"] == ["localhost:8000"]
    assert by_job["gym-vllm/judge"]["static_configs"][0]["targets"] == ["localhost:8100"]
    assert by_job["gym-vllm/policy"]["static_configs"][0]["labels"] == {"gym_service": "policy"}
    assert by_job["gym-vllm/policy"]["scrape_interval"] == "15s"


def test_scraped_engine_display_name_is_component_slash_service():
    """The scrape job name becomes the engine's `service.name.override`; the docs promise this shape."""
    jobs = {s["job_name"] for s in _rendered()["receivers"]["prometheus"]["config"]["scrape_configs"]}
    assert "gym-vllm/policy" in jobs
    custom = _rendered(_config(otel={"component": "engine"}))
    assert "engine/policy" in {s["job_name"] for s in custom["receivers"]["prometheus"]["config"]["scrape_configs"]}


def test_collector_keeps_each_producers_own_name_as_display_identity():
    doc = _rendered()
    identity = doc["processors"]["transform/identity"]
    rule = (
        'set(resource.attributes["service.name.override"], resource.attributes["service.name"]) '
        'where resource.attributes["service.name.override"] == nil and resource.attributes["service.name"] != nil'
    )
    for signal in ("metric", "trace", "log"):
        assert identity[f"{signal}_statements"] == [{"context": "resource", "statements": [rule]}]
    for name, pipeline in doc["service"]["pipelines"].items():
        processors = pipeline["processors"]
        resource = "resource/local" if name.endswith("/local") else "resource/managed"
        assert processors.index("transform/identity") < processors.index(resource), name
    assert "service.name.override" not in _attrs(doc)


def test_collector_metrics_pipeline_also_accepts_otlp_metrics():
    assert _rendered()["service"]["pipelines"]["metrics"]["receivers"] == ["prometheus", "otlp", "span_metrics"]


def test_collector_derives_metrics_from_spans_with_display_identity_and_sandbox_provider():
    doc = _rendered()
    connector = doc["connectors"]["span_metrics"]
    assert connector["dimensions"] == [{"name": "service.name.override"}, {"name": "nemo.gym.sandbox.provider"}]
    assert connector["metrics_flush_interval"] == "15s"
    assert "span_metrics" in doc["service"]["pipelines"]["traces/local"]["exporters"]
    assert "span_metrics" not in doc["service"]["pipelines"]["traces"]["exporters"]
    traces = doc["service"]["pipelines"]["traces/local"]["processors"]
    assert traces.index("transform/identity") < traces.index("resource/local")
    for name in ("metrics", "metrics/local"):
        assert "span_metrics" in doc["service"]["pipelines"][name]["receivers"], name


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
    assert attrs["Authorization"] == ("${env:OTEL_TOKEN}", "upsert")
    exporter = doc["exporters"]["otlp_http/managed"]
    assert exporter["endpoint"] == "https://otlp.example.com"
    assert exporter["headers"]["Authorization"] == "Bearer ${env:OTEL_TOKEN}"


def test_collector_token_reference_follows_token_env():
    doc = _rendered(_config(otel={"token_env": "MY_TOKEN"}))
    assert _attrs(doc)["Authorization"][0] == "${env:MY_TOKEN}"
    assert doc["exporters"]["otlp_http/managed"]["headers"]["Authorization"] == "Bearer ${env:MY_TOKEN}"


def test_collector_writes_a_local_copy_next_to_the_managed_export():
    doc = _rendered()
    # Node-local while the job runs; the collector's step appends it into the job directory when it stops.
    assert doc["exporters"]["file/metrics"]["path"] == "${env:GYM_OTEL_LOCAL_DIR}/metrics.jsonl"
    for signal in ("metrics", "traces", "logs"):
        managed, local = doc["service"]["pipelines"][signal], doc["service"]["pipelines"][f"{signal}/local"]
        assert managed["exporters"] == ["otlp_http/managed"]
        assert local["exporters"][0] == f"file/{signal}"
        assert managed["receivers"] == local["receivers"]


def test_local_only_collector_needs_no_destination_or_token():
    config = _config(otel={"remote_export": False, "endpoint": None, "service_name": None})
    doc = _rendered(config)

    assert set(doc["exporters"]) == {"file/metrics", "file/traces", "file/logs"}
    assert set(doc["service"]["pipelines"]) == {"metrics/local", "traces/local", "logs/local"}
    assert "span_metrics" in doc["service"]["pipelines"]["traces/local"]["exporters"]
    assert "service.name" not in _attrs(doc, "resource/local")
    for signal in ("metrics", "traces", "logs"):
        local = doc["service"]["pipelines"][f"{signal}/local"]
        assert f"file/{signal}" in local["exporters"]
        assert doc["exporters"][f"file/{signal}"]["path"] == f"${{env:GYM_OTEL_LOCAL_DIR}}/{signal}.jsonl"
    rendered = yaml.safe_dump(doc)
    assert "OTEL_TOKEN" not in rendered
    assert "Authorization" not in rendered
    assert "managed" not in rendered


def test_collector_for_a_job_script_without_the_copy_writes_into_the_job_directory():
    doc = yaml.safe_load(render_collector_config(_config(), "scicode", BENCH_DIR, node_local=False))
    for signal in ("metrics", "traces", "logs"):
        assert doc["exporters"][f"file/{signal}"]["path"] == str(BENCH_DIR / "otel" / f"{signal}.jsonl")


def test_the_token_never_reaches_the_job_directory_copies():
    doc = _rendered()
    assert "Authorization" in _attrs(doc, "resource/managed")
    assert "Authorization" not in _attrs(doc, "resource/local")
    local_keys = {a["key"] for a in doc["processors"]["resource/local"]["attributes"]}
    assert local_keys == {a["key"] for a in doc["processors"]["resource/managed"]["attributes"]} - {"Authorization"}
    for name, pipeline in doc["service"]["pipelines"].items():
        writes_files = any(e.startswith("file/") for e in pipeline["exporters"])
        assert writes_files == ("resource/local" in pipeline["processors"]), name
        assert writes_files == name.endswith("/local"), name
        assert ("otlp_http/managed" in pipeline["exporters"]) == ("resource/managed" in pipeline["processors"]), name


def test_collector_log_level_does_not_print_batches():
    assert _rendered()["service"]["telemetry"]["logs"]["level"] == "info"


def test_collector_receives_otlp_for_the_job_processes():
    protocols = _rendered()["receivers"]["otlp"]["protocols"]
    assert protocols["grpc"]["endpoint"] == f"0.0.0.0:{OTLP_GRPC_PORT}"
    assert protocols["http"]["endpoint"] == f"0.0.0.0:{OTLP_HTTP_PORT}"


def test_collector_exposes_a_health_check():
    doc = _rendered()
    assert doc["extensions"]["health_check"]["endpoint"] == f"0.0.0.0:{COLLECTOR_HEALTH_PORT}"
    assert doc["service"]["extensions"] == ["health_check"]


def test_collector_config_never_contains_the_token_value(monkeypatch):
    monkeypatch.setenv("OTEL_TOKEN", "secret-token")
    assert "secret-token" not in render_collector_config(_config(), "scicode", BENCH_DIR)


# ---------------------------------------------------------------------------
# sbatch script
# ---------------------------------------------------------------------------


def test_script_starts_the_collector_before_the_model_service():
    script = _script(_config())
    assert script.index("# service: otel_collector") < script.index("# service: policy")


def _step_command(script, name):
    """Read one shell command, including the private loader's quoted multiline Python."""
    lines = script.splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if f"--output=logs/{name}-$SLURM_JOB_ID.log" in line)
    command = ""
    for line in lines[start:]:
        command += line
        try:
            argv = shlex.split(command)
        except ValueError:
            continue
        assert "srun" in argv, command
        return command.rstrip()
    raise AssertionError(f"Unterminated {name} step")


def _task_argv(script, name, role):
    argv = shlex.split(_step_command(script, name))
    loader = argv.index("python3")
    assert argv[loader : loader + 2] == ["python3", "-c"]
    assert argv[loader + 3 : loader + 5] == [str(private_directory(BENCH_DIR)), role]
    return argv[loader + 5 : -1] if argv[-1] == "&" else argv[loader + 5 :]


def _role_environment(config, role):
    name = "otel_collector" if role == "collector" else role.removeprefix("service:")
    _task_argv(_script(config), name, role)
    return task_environments(config, "scicode", next(iter(config.compute.values())), BENCH_DIR)[role]


def _collector_line(script):
    return _step_command(script, "otel_collector")


def _collector_shell(script):
    argv = _task_argv(script, "otel_collector", "collector")
    assert argv[:2] == ["sh", "-c"]
    assert len(argv) == 3
    return argv[2]


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
    # Named, not left to Slurm: the same node the driver is placed on.
    assert '--nodelist="${gym_nodes[0]}"' in line
    assert " --nodes=1 --ntasks=1" not in _collector_line(_script(_config()))


def test_script_runs_the_collector_on_the_node_by_default():
    line = _collector_line(_script(_config()))
    assert "--container" not in line
    assert line.startswith("srun --overlap --account=acct --output=logs/otel_collector-$SLURM_JOB_ID.log python3 -c ")
    assert line.rstrip().endswith("&")
    line = _collector_shell(_script(_config()))
    # A node-local copy of the binary runs, with its jsonl written beside it.
    assert "src=$(command -v otelcol-contrib)" in line
    assert 'dir=$(mktemp -d "/tmp/gym-otel-$SLURM_JOB_ID.XXXXXX") && cp "$src" "$dir"/otelcol-contrib' in line
    assert 'export GYM_OTEL_LOCAL_DIR="$dir"' in line
    assert f'"$dir"/otelcol-contrib --config {collector_config_path(BENCH_DIR)} & pid=$!' in line


def test_script_appends_the_node_local_jsonl_into_the_job_directory_once_the_collector_stops():
    line = _collector_shell(_script(_config()))
    copy = f'cat "$dir/$s.jsonl" >> {BENCH_DIR / "otel"}/"$s.jsonl" && rm -f "$dir/$s.jsonl"'
    assert f'for s in metrics traces logs; do if [ -e "$dir/$s.jsonl" ]; then {copy}; fi; done' in line
    assert line.index("wait $pid") < line.index('trap "" TERM INT') < line.index(copy)
    assert line.rstrip(" &'").endswith("exit $rc")


def _run_collector_step(config, bench_dir, job_id, *, stop):
    """Run the collector step's own command as srun would, then `stop` it; returns its exit code and leftovers."""
    local_dirs = f"/tmp/gym-otel-{job_id}.*"
    step = subprocess.Popen(
        shlex.split(_collector_command(config, bench_dir)), env={**os.environ, "SLURM_JOB_ID": job_id}
    )
    try:
        deadline = time.monotonic() + 10
        while not glob.glob(f"{local_dirs}/traces.jsonl") and time.monotonic() < deadline:
            time.sleep(0.05)
        stop(step)
        return step.wait(timeout=10), glob.glob(local_dirs)
    finally:
        step.kill()
        for leftover in glob.glob(local_dirs):
            shutil.rmtree(leftover, ignore_errors=True)


def _fake_collector(tmp_path):
    """Writes one span node-local at start and one more when TERM asks it to flush, like the collector's exporter."""
    fake = tmp_path / "bin" / "otelcol-contrib"
    fake.parent.mkdir()
    fake.write_text(
        "#!/bin/sh\n"
        'echo start >> "$GYM_OTEL_LOCAL_DIR/traces.jsonl"\n'
        "trap 'echo flushed >> \"$GYM_OTEL_LOCAL_DIR/traces.jsonl\"; exit 0' TERM\n"
        "while :; do sleep 0.05; done\n"
    )
    fake.chmod(0o755)
    return fake


@pytest.mark.skipif(shutil.which("sh") is None or shutil.which("mktemp") is None, reason="sh and mktemp needed")
def test_collector_step_appends_after_an_earlier_segment_when_slurm_terms_the_step(tmp_path):
    # At the time limit Slurm sends TERM to every process of the step: the shell passes it on, waits, then copies.
    bench_dir = tmp_path / "gym-job-x" / "scicode"
    (bench_dir / "otel").mkdir(parents=True)
    (bench_dir / "otel" / "traces.jsonl").write_text("earlier segment\n")
    config = _config(otel={"binary": str(_fake_collector(tmp_path))})

    rc, leftovers = _run_collector_step(config, bench_dir, f"t{os.getpid()}", stop=lambda p: p.send_signal(15))

    assert rc == 0
    assert (bench_dir / "otel" / "traces.jsonl").read_text() == "earlier segment\nstart\nflushed\n"
    assert not (bench_dir / "otel" / "metrics.jsonl").exists()
    assert leftovers == []


@pytest.mark.skipif(shutil.which("sh") is None, reason="sh needed")
def test_collector_step_fails_clearly_without_its_binary(tmp_path):
    config = _config(otel={"binary": str(tmp_path / "missing" / "otelcol-contrib")})
    step = subprocess.run(
        shlex.split(_collector_command(config, BENCH_DIR)),
        env={**os.environ, "SLURM_JOB_ID": "1"},
        capture_output=True,
        text=True,
    )
    assert step.returncode == 127
    assert "OTel collector binary not found" in step.stderr


def test_script_runs_the_collector_in_a_container_with_the_job_dir_mounted_when_one_is_set():
    config = _config(otel={"container": "/shared/images/otelcol.sqsh", "binary": "/otelcol-contrib"})
    line = _collector_line(_script(config))
    assert "--container-image=/shared/images/otelcol.sqsh" in line
    assert "--no-container-mount-home" in line
    # The job dir holds the config and receives the copied jsonl; the node-local dir is the container's own /tmp.
    assert f"--container-mounts={BENCH_DIR}:{BENCH_DIR}" in line
    assert f"--container-workdir={BENCH_DIR}" in line
    assert line.rstrip().endswith("&")
    shell = _collector_shell(_script(config))
    assert "src=$(command -v /otelcol-contrib)" in shell
    assert f'"$dir"/otelcol-contrib --config {collector_config_path(BENCH_DIR)}' in shell


def test_script_forwards_the_token_from_the_job_environment_not_a_literal(monkeypatch):
    monkeypatch.setenv("OTEL_TOKEN", "secret-token")
    config = _config()
    assert _role_environment(config, "collector") == {
        "OTEL_TOKEN": "runtime:OTEL_TOKEN",
        "SLURM_JOB_ID": "runtime:SLURM_JOB_ID",
    }
    assert "secret-token" not in _script(config)
    assert "OTEL_TOKEN=" not in _collector_line(_script(config))


def test_local_only_script_keeps_collection_without_forwarding_the_token():
    config = _config(otel={"remote_export": False}, driver=_DRIVER_WITH_INSTALL)
    script = _script(config)
    assert "--output=logs/otel_collector-$SLURM_JOB_ID.log" in script
    environment = _role_environment(config, "driver")
    assert environment["NEMO_GYM_OTEL_ENABLED"] == "1"
    assert environment["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://localhost:4318"
    assert "OTEL_TOKEN" not in script
    assert "DRIVER_RC=$?" in script


def test_script_copies_a_binary_on_shared_storage_and_never_runs_it_there():
    config = _config(otel={"binary": "/shared/tools/otelcol-contrib"})
    script = _script(config)
    line = _collector_shell(script)
    assert "src=$(command -v /shared/tools/otelcol-contrib)" in line
    assert "/shared/tools/otelcol-contrib --config" not in line
    assert f'"$dir"/otelcol-contrib --config {collector_config_path(BENCH_DIR)}' in line
    assert "--container" not in _collector_line(script)


def test_script_health_checks_the_collector_before_the_model_service():
    script = _script(_config())
    assert f"Waiting for otel_collector at http://localhost:{COLLECTOR_HEALTH_PORT}" in script
    assert script.index("Waiting for otel_collector") < script.index("Waiting for policy")
    assert "seq 1 60" in script  # 300s / 5s


def test_script_flushes_the_collector_after_the_driver_and_keeps_the_driver_exit_code():
    script = _script(_config())
    tail = script[script.index("--output=logs/driver-$SLURM_JOB_ID.log") :]
    assert "DRIVER_RC=$?" in tail
    assert 'pkill -TERM -u "$USER" -f -- ' in tail
    assert tail.rstrip().endswith("exit $DRIVER_RC")
    assert tail.index("sleep 20") < tail.index("pkill -TERM") < tail.index("kill -0 $OTEL_COLLECTOR_PID")
    # srun only gets TERM as a last resort, after the collector had its chance to flush.
    assert tail.index("kill -0 $OTEL_COLLECTOR_PID") < tail.index("kill -TERM $OTEL_COLLECTOR_PID")


def _pkill_regex(script, job_id):
    """The regular expression pkill receives once the batch shell has expanded the pattern's words."""
    line = next(line for line in script.splitlines() if "pkill -TERM" in line)
    words = line[line.index(" -- ") + len(" -- ") : line.index(" || true")]
    printed = subprocess.run(
        ["bash", "-c", f"printf %s {words}"],
        env={**os.environ, "SLURM_JOB_ID": job_id},
        capture_output=True,
        text=True,
    )
    return printed.stdout


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash needed")
def test_the_shutdown_signals_only_this_jobs_node_local_collector():
    # The srun and the shell that launched the collector carry the same `--config <path>`; a TERM to srun makes
    # Slurm kill the step before the flush and the copy complete.
    script = _script(_config())
    regex = _pkill_regex(script, "4242")
    config_arg = f"--config {collector_config_path(BENCH_DIR)}"
    assert re.search(regex, f"/tmp/gym-otel-4242.Xy12Ab/otelcol-contrib {config_arg}")
    assert not re.search(regex, f"/tmp/gym-otel-4243.Xy12Ab/otelcol-contrib {config_arg}")
    assert not re.search(regex, f"otelcol-contrib {config_arg}")
    collector_step = _collector_line(script)
    assert not re.search(regex, collector_step[collector_step.index("srun") :])
    assert not re.search(regex, " ".join(shlex.split(_collector_command(_config(), BENCH_DIR))))


def test_a_resumable_script_ends_its_own_chain_before_the_flush_reads_the_driver_status():
    driver = {"container": "gym:latest", "policy_model": "policy", "benchmarks": {"scicode": {"resumable": {}}}}
    script = _script(_config(driver=driver))
    tail = script[script.index("--output=logs/driver-$SLURM_JOB_ID.log") :]
    # The epilogue returns the driver's status, which the flush keeps as the job's.
    assert render_resume_epilogue() + "\nDRIVER_RC=$?" in tail
    assert tail.rstrip().endswith("exit $DRIVER_RC")


_DRIVER_WITH_INSTALL = {
    "container": "gym:latest",
    "policy_model": "policy",
    "benchmarks": {"scicode": {}},
    "gym_install": {"ref": "main"},
}


def test_script_switches_on_gym_lens_telemetry_toward_the_collector():
    config = _config(driver=_DRIVER_WITH_INSTALL)
    script = _script(config)
    environment = _role_environment(config, "driver")
    assert environment["NEMO_GYM_OTEL_ENABLED"] == "1"
    assert environment["NEMO_GYM_OTEL_RUN_ID"] == "gym-job-20260921T100000Z-abc123"  # pragma: allowlist secret
    assert environment["NEMO_GYM_OTEL_SPAN_GROUPS"] == "default,verify"
    assert environment["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://localhost:4318"
    assert environment["OTEL_EXPORTER_OTLP_PROTOCOL"] == "http/protobuf"
    assert _sync_extras(script) == " --extra telemetry --extra ray"


def test_script_installs_the_sdk_of_a_configured_exporter():
    run = {
        "mlflow_tracking_uri": "https://tracking.example",
        "mlflow_experiment_name": "gym",
        "mlflow_run_name": "run",
    }
    driver = {**_DRIVER_WITH_INSTALL, "benchmarks": {"scicode": {"run": run}}}
    assert _sync_extras(_script(_config(driver=driver))) == " --extra telemetry --extra mlflow --extra ray"


def test_script_skips_the_sdk_of_a_partly_configured_exporter():
    # A tracking URI alone (a registry lookup) opens no exporter, so needs no SDK.
    driver = {**_DRIVER_WITH_INSTALL, "benchmarks": {"scicode": {"run": {"mlflow_tracking_uri": "https://x"}}}}
    assert _sync_extras(_script(_config(driver=driver))) == " --extra telemetry --extra ray"


@pytest.mark.parametrize("section", ["prepare", "run"])
def test_script_installs_mlflow_for_gitlab_dataset_storage(section):
    # A tracking URI and token without an experiment open no exporter, but a GitLab download needs mlflow.
    creds = {"mlflow_tracking_uri": "https://gitlab.example", "mlflow_tracking_token": "token"}
    driver = {**_DRIVER_WITH_INSTALL, "benchmarks": {"scicode": {section: creds}}}
    script = _script(_config(driver=driver))
    assert _sync_extras(script) == " --extra telemetry --extra mlflow --extra ray"
    assert "uv pip install" not in script


def test_script_installs_mlflow_once_for_an_exporter_and_gitlab_storage():
    run = {
        "mlflow_tracking_uri": "https://tracking.example",
        "mlflow_tracking_token": "token",
        "mlflow_experiment_name": "gym",
        "mlflow_run_name": "run",
    }
    driver = {**_DRIVER_WITH_INSTALL, "benchmarks": {"scicode": {"prepare": run, "run": run}}}
    script = _script(_config(driver=driver))
    assert _sync_extras(script) == " --extra telemetry --extra mlflow --extra ray"
    assert script.count("--extra mlflow") == 1
    assert "uv pip install" not in script


def test_script_lets_an_explicit_driver_env_win_over_telemetry_defaults():
    driver = {**_DRIVER_WITH_INSTALL, "env": {"OTEL_EXPORTER_OTLP_ENDPOINT": "lit:http://elsewhere:4318"}}
    environment = _role_environment(_config(driver=driver), "driver")
    assert environment["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://elsewhere:4318"
    assert environment["NEMO_GYM_OTEL_ENABLED"] == "1"


def test_script_ships_gym_logs_by_default_and_can_switch_them_off():
    environment = _role_environment(_config(driver=_DRIVER_WITH_INSTALL), "driver")
    assert environment["NEMO_GYM_OTEL_LOGS_ENABLED"] == "1"
    # No per-signal logs endpoint: the pinned nemo-lens builds the log exporter from
    # OTEL_EXPORTER_OTLP_PROTOCOL like the span and metric exporters, so logs follow the shared
    # endpoint and protocol in every environment.
    assert "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT" not in environment
    assert environment["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://localhost:4318"
    assert environment["OTEL_EXPORTER_OTLP_PROTOCOL"] == "http/protobuf"
    off = _config(driver=_DRIVER_WITH_INSTALL, otel={"gym_logs": False})
    assert _role_environment(off, "driver")["NEMO_GYM_OTEL_LOGS_ENABLED"] == "0"


def test_script_honours_configured_gym_span_groups():
    config = _config(driver=_DRIVER_WITH_INSTALL, otel={"gym_span_groups": "per_rollout,sandbox"})
    assert _role_environment(config, "driver")["NEMO_GYM_OTEL_SPAN_GROUPS"] == "per_rollout,sandbox"


def _policy_line(script):
    return _step_command(script, "policy")


def test_vllm_traces_are_off_by_default():
    assert "--otlp-traces-endpoint" not in _script(_config())


def test_vllm_traces_point_the_engine_at_the_collector_over_http():
    config = _config(otel={"vllm_traces": True})
    line = _policy_line(_script(config))
    assert f"--otlp-traces-endpoint http://localhost:{OTLP_HTTP_PORT}/v1/traces" in line
    environment = _role_environment(config, "service:policy")
    assert environment["OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"] == "http/protobuf"
    assert environment["OTEL_SERVICE_NAME"] == "gym-vllm/policy"


def test_vllm_traces_keep_the_services_own_flags_and_let_its_env_win():
    config = _config(
        services={
            "policy": {
                "type": "vllm",
                "container": "vllm:latest",
                "model": "/checkpoint",
                "port": 8000,
                "extra_args": "--max-model-len 8192",
                "env": {"OTEL_SERVICE_NAME": "lit:mine"},
            }
        },
        otel={"vllm_traces": True, "component": "engine"},
    )
    line = _policy_line(_script(config))
    assert "--max-model-len 8192 --otlp-traces-endpoint" in line
    environment = _role_environment(config, "service:policy")
    assert environment["OTEL_SERVICE_NAME"] == "mine"
    assert environment["OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"] == "http/protobuf"


def test_vllm_traces_need_the_collector():
    assert "--otlp-traces-endpoint" not in _script(_config(otel={"enabled": False, "vllm_traces": True}))


def test_vllm_traces_skip_a_tier_whose_api_server_is_on_another_node():
    config = _config(
        services={
            "policy": {
                "type": "vllm_pd",
                "container": "vllm:latest",
                "model": "/checkpoint",
                "port": 8000,
                "prefill": {"node_pool": "prefill"},
                "decode": {"node_pool": "decode"},
            }
        },
        compute={
            "cluster-a": {
                "type": "slurm",
                "account": "acct",
                "hostname": None,
                "node_pools": {"prefill": {"partition": "batch"}, "decode": {"partition": "batch"}},
            }
        },
        otel={"vllm_traces": True},
    )
    script = _script(config)
    traced = [line for line in script.splitlines() if "--otlp-traces-endpoint" in line]
    assert len(traced) == 1
    assert "--port 8001" in traced[0]  # the prefill tier, on the collector's node


def test_script_leaves_the_driver_alone_when_disabled():
    script = _script(_config(otel={"enabled": False}, driver=_DRIVER_WITH_INSTALL))
    assert "NEMO_GYM_OTEL" not in script
    assert "uv sync --locked" in script
    assert "[telemetry]" not in script


def test_script_has_no_collector_when_disabled():
    script = _script(_config(otel={"enabled": False}))
    assert "otel_collector" not in script
    assert "DRIVER_RC" not in script


def test_script_has_a_collector_without_a_model_service():
    config = _config(services={}, driver={"container": "gym:latest", "benchmarks": {"scicode": {}}})
    script = _script(config)
    assert "otel_collector" in script
    assert _role_environment(config, "driver")["NEMO_GYM_OTEL_ENABLED"] == "1"


# ---------------------------------------------------------------------------
# executor
# ---------------------------------------------------------------------------


class _FakeConnection(LocalConnection):
    def __init__(self, replies):
        self._replies = replies
        self.commands = []

    def run(self, commands):
        self.commands.append(commands)
        scheduler = [command for command in commands if "sbatch --parsable " in command]
        # Exercise the actual permission probe and private writer; only scheduler submission is simulated.
        local = [command for command in commands if command not in scheduler]
        output = super().run(local) if local else ""
        return self._replies.pop(0) if scheduler else output


def _scheduler_commands(conn):
    commands = [command for batch in conn.commands for command in batch if "sbatch --parsable " in command]
    assert len(commands) == 1
    assert "env -i " in commands[0]
    return commands


def _private_payload(record):
    directory = private_directory(Path(record.run_dir) / "scicode")
    for path in [directory.parent, directory]:
        assert stat.S_IMODE(path.stat().st_mode) & ~stat.S_ISGID == 0o700
        assert path.stat().st_uid == os.geteuid()
    path = directory / "environment.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert path.stat().st_uid == os.geteuid()
    return json.loads(path.read_text())


def _install(monkeypatch, conn, tmp_path):
    monkeypatch.setattr(slurm_chain, "get_connection", lambda hostname: conn)
    monkeypatch.setattr(slurm_judge_job, "get_connection", lambda hostname: conn)
    monkeypatch.setattr(slurm_executor, "get_connection", lambda hostname: conn)
    monkeypatch.setattr(slurm_executor, "_validate_mounts", lambda config, connection: None)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


_DRIVER_WITHOUT_INSTALL = {"container": "gym:latest", "policy_model": "policy", "benchmarks": {"scicode": {}}}


def _executor_config(tmp_path, **overrides):
    """A submittable config: Gym telemetry is on by default, so the driver needs a checkout to install it from."""
    return _config(**{"job": {"output_path": str(tmp_path / "jobs")}, "driver": _DRIVER_WITH_INSTALL, **overrides})


def test_submit_fails_when_gym_telemetry_has_no_checkout_to_install_it_from(tmp_path, monkeypatch):
    monkeypatch.setenv("OTEL_TOKEN", "secret-token")
    conn = _FakeConnection([])
    _install(monkeypatch, conn, tmp_path)

    with pytest.raises(ValueError, match=r"driver\.gym_install.*otel\.gym_telemetry: false"):
        SlurmExecutor().run(_executor_config(tmp_path, driver=_DRIVER_WITHOUT_INSTALL))

    assert conn.commands == []


def test_gym_telemetry_off_keeps_the_collector_and_skips_lens():
    script = _script(_config(otel={"gym_telemetry": False}, driver=_DRIVER_WITH_INSTALL))
    assert "--output=logs/otel_collector-$SLURM_JOB_ID.log" in script
    assert "NEMO_GYM_OTEL_ENABLED" not in script
    assert _sync_extras(script) == " --extra ray"


def test_submit_without_a_token_stages_and_persists_local_only_collection(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv("OTEL_TOKEN", raising=False)
    conn = _FakeConnection(["__GYM_JOB:scicode:0:111 "])
    _install(monkeypatch, conn, tmp_path)

    record = SlurmExecutor().run(_executor_config(tmp_path, otel={"endpoint": None, "service_name": None}))

    bench_dir = Path(record.run_dir) / "scicode"
    collector = yaml.safe_load((bench_dir / "otel" / "collector.yaml").read_text())
    assert set(collector["exporters"]) == {"file/metrics", "file/traces", "file/logs"}
    assert "OTEL_TOKEN" not in (bench_dir / "job.sh").read_text()
    payload = _private_payload(record)
    assert "OTEL_TOKEN" not in payload["forwarded"]
    assert "OTEL_TOKEN" not in payload["roles"]["collector"]
    assert "OTEL_TOKEN" not in "\n".join(_scheduler_commands(conn))
    resolved = yaml.safe_load((Path(record.run_dir) / RESOLVED_CONFIG_NAME).read_text())
    assert resolved["otel"]["enabled"] is True
    assert resolved["otel"]["remote_export"] is False
    assert "otel.remote_export=false" in caplog.text


def test_submit_fails_when_the_destination_is_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("OTEL_TOKEN", "secret-token")
    config = _executor_config(tmp_path)
    config.otel.endpoint = None
    with pytest.raises(ValueError, match="otel.endpoint"):
        SlurmExecutor().run(config, dry_run=True)


def test_dry_run_without_a_token_keeps_local_collection(tmp_path, monkeypatch, capsys, caplog):
    monkeypatch.delenv("OTEL_TOKEN", raising=False)
    SlurmExecutor().run(_executor_config(tmp_path), dry_run=True)
    out = capsys.readouterr().out
    assert "file/metrics" in out
    assert "otlp_http/managed" not in out
    assert "OTEL_TOKEN" not in out
    assert "otel.remote_export=false" in caplog.text


def test_explicit_local_only_submit_ignores_available_credentials(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("NVAUTH_TOKEN", "unused-secret")
    conn = _FakeConnection(["__GYM_JOB:scicode:0:111 "])
    _install(monkeypatch, conn, tmp_path)
    config = _executor_config(
        tmp_path, otel={"remote_export": False, "token_env": "NVAUTH_TOKEN", "endpoint": None, "service_name": None}
    )

    record = SlurmExecutor().run(config)

    collector = (Path(record.run_dir) / "scicode" / "otel" / "collector.yaml").read_text()
    assert "file/metrics" in collector
    assert "managed" not in collector
    payload = _private_payload(record)
    assert "NVAUTH_TOKEN" not in payload["forwarded"]
    assert "NVAUTH_TOKEN" not in payload["roles"]["collector"]
    assert "unused-secret" not in "\n".join(_scheduler_commands(conn))
    assert "unused-secret" not in (Path(record.run_dir) / "scicode" / "job.sh").read_text()
    assert not [r for r in caplog.records if r.name == obs_module.__name__]


def test_dry_run_prints_the_collector_config(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OTEL_TOKEN", "secret-token")
    SlurmExecutor().run(_executor_config(tmp_path), dry_run=True)
    out = capsys.readouterr().out
    assert "otel/collector.yaml for benchmark: scicode" in out
    assert "otlp_http/managed" in out
    assert "secret-token" not in out


def test_submit_stages_the_collector_config_and_keeps_the_token_in_the_private_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("OTEL_TOKEN", "secret-token")
    conn = _FakeConnection(["__GYM_JOB:scicode:0:111 "])
    _install(monkeypatch, conn, tmp_path)

    record = SlurmExecutor().run(_executor_config(tmp_path))

    bench_dir = Path(record.run_dir) / "scicode"
    collector = yaml.safe_load((bench_dir / "otel" / "collector.yaml").read_text())
    assert _attrs(collector)["run_id"] == (record.gym_job_id, "upsert")
    assert "secret-token" not in (bench_dir / "job.sh").read_text()
    assert "secret-token" not in (bench_dir / "otel" / "collector.yaml").read_text()
    assert "secret-token" not in (Path(record.run_dir) / RESOLVED_CONFIG_NAME).read_text()

    payload = _private_payload(record)
    assert payload["forwarded"]["OTEL_TOKEN"] == "secret-token"
    assert payload["roles"]["collector"] == {
        "OTEL_TOKEN": "runtime:OTEL_TOKEN",
        "SLURM_JOB_ID": "runtime:SLURM_JOB_ID",
    }
    assert "secret-token" not in "\n".join(_scheduler_commands(conn))
    assert all(not command.startswith("export ") for batch in conn.commands for command in batch)


def test_submit_without_otel_neither_stages_nor_exports(tmp_path, monkeypatch):
    monkeypatch.delenv("OTEL_TOKEN", raising=False)
    conn = _FakeConnection(["__GYM_JOB:scicode:0:111 "])
    _install(monkeypatch, conn, tmp_path)

    record = SlurmExecutor().run(_executor_config(tmp_path, otel={"enabled": False}))

    assert not (Path(record.run_dir) / "scicode" / "otel").exists()
    payload = _private_payload(record)
    assert "OTEL_TOKEN" not in payload["forwarded"]
    assert "OTEL_TOKEN" not in payload["roles"]["collector"]
    assert "OTEL_TOKEN" not in "\n".join(_scheduler_commands(conn))


def _engine_rollout_id_statements(doc):
    copy, trim, write = doc["processors"]["transform/engine_rollout_id"]["trace_statements"][0]["statements"]
    pattern = re.search(r'replace_pattern\(cache\["engine_rollout_id"\], "([^"]+)", "([^"]+)"\)', trim)
    # The collector expands `${...}` itself, so OTTL regex syntax is written with `$$`; Go's `${1}` is
    # Python's `\g<1>`.
    regex = pattern.group(1).replace("$$", "$")
    replacement = pattern.group(2).replace("$$", "$").replace("${1}", "\\g<1>")
    return copy, trim, write, regex, replacement


def test_collector_copies_the_rollout_id_out_of_the_engine_request_id_when_vllm_traces_are_on():
    doc = _rendered(_config(otel={"vllm_traces": True}))
    copy, trim, write, regex, replacement = _engine_rollout_id_statements(doc)
    # Derived in the span-local cache; the attribute is written only while absent, so an id the span
    # already carries is never touched, whatever its value (see the collector round trip in the PR).
    assert copy.startswith('set(cache["engine_rollout_id"], attributes["gen_ai.request.id"])')
    assert 'where name == "llm_request"' in copy and 'attributes["nemo.gym.rollout.id"] == nil' in copy
    assert "IsMatch(" in copy, "only Gym's own request ids are derived"
    assert trim.endswith('where cache["engine_rollout_id"] != nil')
    assert write.startswith('set(attributes["nemo.gym.rollout.id"], cache["engine_rollout_id"])')
    assert 'attributes["nemo.gym.rollout.id"] == nil' in write
    assert "attributes[" not in trim, "the trim edits the cache, never an attribute"
    for pipeline in ("traces", "traces/local"):
        processors = doc["service"]["pipelines"][pipeline]["processors"]
        assert processors.index("transform/engine_rollout_id") < processors.index(processors[-2]), pipeline
    # Same semantics in Python's engine as in Go's for these inputs.
    assert re.sub(regex, replacement, "chatcmpl-0-0-5bf494fb") == "0-0"
    assert re.sub(regex, replacement, "chatcmpl-3-0-a2-deadbeef") == "3-0-a2"
    assert re.sub(regex, replacement, "custom.id_7-00c0ffee") == "custom.id_7"
    plain_uuid = "chatcmpl-0123456789abcdef0123456789abcdef"
    assert re.search(regex, plain_uuid) is None, "a request id that is not Gym's is not derived"


def test_collector_leaves_engine_spans_alone_when_vllm_traces_are_off():
    doc = _rendered()
    assert "transform/engine_rollout_id" not in doc["processors"]
    for pipeline in ("traces", "traces/local"):
        assert "transform/engine_rollout_id" not in doc["service"]["pipelines"][pipeline]["processors"]
