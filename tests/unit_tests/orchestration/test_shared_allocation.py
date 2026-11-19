# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from nemo_gym.cli.main import _eval_submit
from nemo_gym.orchestration.executors.capabilities import Resubmittable
from nemo_gym.orchestration.executors.slurm.shared import SharedSlurmExecutor, render_shared_scripts
from nemo_gym.orchestration.jobs import SubmissionRecord
from nemo_gym.orchestration.shared_allocation import SharedAllocationConfig


def group_config():
    members = {}
    for name in ("fast", "slow"):
        members[name] = {
            "services": {"policy": {"type": "vllm", "container": "policy-image", "model": "example/model"}},
            "compute": {"cluster": {"type": "slurm", "account": "account"}},
            "otel": {"enabled": False},
            "driver": {
                "container": name + "-image",
                "policy_model": "policy",
                "benchmarks": {
                    name: {
                        "run": {
                            "num_repeats": 3,
                            "label": "two words",
                            "responses_create_params": {"temperature": 0.7},
                        }
                    },
                },
            },
            "job": {"output_path": "/member-output"},
        }
    return {
        "kind": "shared_allocation",
        "members": members,
        "job": {"output_path": "/shared-output"},
        "admission": {
            "container": "admission-image",
            "gym_install": {"ref": "admission-commit"},
            "max_inflight": 2,
            "max_queued": 4,
        },
    }


def test_preserves_independent_runtime_pins_and_protocols(tmp_path):
    raw = group_config()
    for name, member in raw["members"].items():
        member["driver"]["gym_install"] = {"ref": name + "-commit"}
        member["driver"]["env"] = {"MEMBER_SETTING": "lit:" + name}
    config = SharedAllocationConfig.model_validate(raw)
    before = config.model_dump()
    scripts = render_shared_scripts(config, tmp_path)
    assert config.model_dump() == before
    for name in config.members:
        driver = scripts[name + "/driver.sh"]
        assert name + "-commit" in driver and name + "-image" in driver
        assert "+num_repeats=3" in driver and "+responses_create_params.temperature=0.7" in driver
        assert f"MEMBER_SETTING={name}" in driver
        assert f"+head_server.port={config.member_ports(name)['head_server.port']}" in driver
        assert "+policy_base_url=http://localhost:18001/v1" in driver
        assert "driver-worker" in driver
    assert scripts["job.sh"].count("# service: policy\n") == 1
    assert "admission-commit" in scripts["job.sh"]


@pytest.mark.parametrize(
    "change,match",
    [
        (
            lambda c: c["members"]["slow"]["services"]["policy"].update(model="another/model"),
            "different shared services",
        ),
        (lambda c: c["members"]["slow"]["compute"]["cluster"].update(account="other"), "different compute"),
        (lambda c: c["members"]["slow"]["driver"]["benchmarks"]["slow"].update(resumable=True), "grouped recovery"),
        (
            lambda c: c["members"]["slow"]["driver"]["benchmarks"]["slow"]["run"].update({"head_server.port": 5000}),
            "sets ports",
        ),
        (
            lambda c: c["members"]["slow"]["driver"]["benchmarks"]["slow"]["run"].update(
                {"policy_model.responses_api_models.vllm_model.base_url": ["http://localhost:8000/v1"]}
            ),
            "endpoint directly",
        ),
        (lambda c: c["admission"].update(port=8000), "conflicts"),
        (lambda c: c.update(first_driver_port=65500), "exceed"),
        (lambda c: c.update(first_driver_port=8000), "overlap"),
    ],
)
def test_rejects_incompatible_or_silently_changed_members(change, match):
    raw = group_config()
    change(raw)
    with pytest.raises(ValidationError, match=match):
        SharedAllocationConfig.model_validate(raw)


def test_pool_order_is_part_of_the_shared_placement_contract():
    raw = group_config()
    pools = {"policy": {"partition": "gpu"}, "judge": {"partition": "gpu"}}
    for name, member in raw["members"].items():
        member["compute"]["cluster"]["node_pools"] = pools if name == "fast" else dict(reversed(list(pools.items())))
    with pytest.raises(ValidationError, match="node-pool order"):
        SharedAllocationConfig.model_validate(raw)


def test_members_share_policy_placement_collector_and_tracking_identity(tmp_path):
    raw = group_config()
    for member in raw["members"].values():
        member["compute"]["cluster"]["node_pools"] = {"other": {"partition": "gpu"}, "policy": {"partition": "gpu"}}
        member["services"]["policy"]["node_pool"] = "policy"
        member["otel"] = {"remote_export": False}
        member["driver"]["gym_install"] = {"ref": "member-pin"}
        member["job"]["export"] = {"evalrdl": {}}
    scripts = render_shared_scripts(
        SharedAllocationConfig.model_validate(raw), tmp_path / "job-id" / "shared-allocation"
    )
    assert scripts["job.sh"].count("# service: otel_collector\n") == 1
    admission = next(
        line for line in scripts["job.sh"].splitlines() if "srun " in line and "logs/gym_admission-" in line
    )
    assert '--nodelist="${gym_nodes[1]}" --nodes=1 --ntasks=1' in admission
    for name in raw["members"]:
        driver = scripts[name + "/driver.sh"]
        assert "NEMO_GYM_OTEL_RUN_ID=job-id" in driver
        assert f"+label={name}-job-id" in driver
        assert '--nodelist="${gym_nodes[1]}" --nodes=1 --ntasks=1' in driver
        assert "# service: otel_collector" not in driver


def test_submit_cli_resolves_shared_schema_without_contacting_scheduler(tmp_path, capsys):
    path = tmp_path / "shared.yaml"
    path.write_text(yaml.safe_dump(group_config()))
    args = argparse.Namespace(config=str(path), select=None, resolve_only=True, json=True, dry_run=False)
    _eval_submit(args, [])
    resolved = json.loads(capsys.readouterr().out)
    assert resolved["kind"] == "shared_allocation" and set(resolved["members"]) == {"fast", "slow"}


def _wait(path, process):
    deadline = time.monotonic() + 15
    while not path.exists():
        assert process.poll() is None and time.monotonic() < deadline
        time.sleep(0.02)


@pytest.fixture
def rendered_controller(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "41")
    monkeypatch.setenv("GROUP_TEST_ROOT", str(tmp_path))
    commands = tmp_path / "bin"
    commands.mkdir()
    monkeypatch.setenv("PATH", str(commands) + os.pathsep + os.environ["PATH"])
    programs = {
        "curl": "",
        "squeue": "",
        "scancel": "",
        "srun": """import os,sys,time,subprocess
from pathlib import Path
root=Path(os.environ['GROUP_TEST_ROOT'])
args=sys.argv[1:]
output=next(a for a in args if a.startswith('--output='))
if 'driver-' not in output:
    name='admission' if 'gym_admission' in output else 'policy'
    (root/(name+'.pid')).write_text(str(os.getpid()))
    while not (root/'break-services').exists(): time.sleep(.03)
    raise SystemExit(2)
while args[0].startswith('--'): args.pop(0)
raise SystemExit(subprocess.call(args))
""",
        "python3": """import os,sys,time
from pathlib import Path
if sys.argv[1:3] == ['-m', 'nemo_gym.orchestration.export_step']:
    root=Path(os.environ['GROUP_TEST_ROOT'])
    Path('export-started').touch()
    if Path.cwd().name=='slow':
        while not (root/'release-slow').exists(): time.sleep(.03)
    Path('export-finished').touch()
    raise SystemExit(0)
os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
""",
        "gym": """import json,os,sys,time
from pathlib import Path
root=Path(os.environ['GROUP_TEST_ROOT']);name=Path.cwd().name
Path('argv.json').write_text(json.dumps(sys.argv[1:]));Path('started').touch()
if name=='slow' and not os.environ.get('EXPORT_SLOW'):
    while not (root/'release-slow').exists(): time.sleep(.03)
Path('finished').touch()
raise SystemExit(7 if name=='fast' and os.environ.get('FAIL_FAST') else 0)
""",
    }
    for name, source in programs.items():
        path = commands / name
        path.write_text(f"#!{sys.executable}\n" + source)
        path.chmod(0o755)
    config = group_config()
    if os.environ.get("EXPORT_SLOW"):
        for member in config["members"].values():
            member["job"]["export"] = {"evalrdl": {}}
    for name, script in render_shared_scripts(SharedAllocationConfig.model_validate(config), tmp_path).items():
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        (path.parent / "logs").mkdir(exist_ok=True)
        path.write_text(script)
        subprocess.run(["bash", "-n", str(path)], check=True)
    with (tmp_path / "controller.log").open("w") as output:
        process = subprocess.Popen(
            ["bash", str(tmp_path / "job.sh")],
            cwd=tmp_path,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        yield process
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux worker supervision")
@pytest.mark.parametrize("failure", [False, True])
def test_shared_services_live_until_every_worker_finishes(tmp_path, monkeypatch, request, failure):
    if failure:
        monkeypatch.setenv("FAIL_FAST", "1")
    process = request.getfixturevalue("rendered_controller")
    _wait(tmp_path / "fast/finished", process)
    _wait(tmp_path / "slow/started", process)
    assert process.poll() is None
    for name in ("policy", "admission"):
        os.kill(int((tmp_path / (name + ".pid")).read_text()), 0)
    (tmp_path / "release-slow").touch()
    assert process.wait(timeout=15) == int(failure), (tmp_path / "controller.log").read_text()
    for name in ("fast", "slow"):
        receipts = list((tmp_path / name / "logs").glob("driver-worker.*.json"))
        assert len(receipts) == 1 and json.loads(receipts[0].read_text())["cleanup_confirmed"]
    for name in ("policy", "admission"):
        assert not Path(f"/proc/{(tmp_path / (name + '.pid')).read_text()}").exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux worker supervision")
@pytest.mark.parametrize("stop", ["cancel", "service"])
def test_controller_stops_members_and_waits_for_cleanup(tmp_path, rendered_controller, stop):
    process = rendered_controller
    _wait(tmp_path / "slow/started", process)
    if stop == "cancel":
        process.terminate()
    else:
        (tmp_path / "break-services").touch()
    assert process.wait(timeout=15) == (143 if stop == "cancel" else 1), (
        tmp_path / "logs/controller-41.log"
    ).read_text()
    (receipt,) = (tmp_path / "slow/logs").glob("driver-worker.*.json")
    assert json.loads(receipt.read_text())["cleanup_confirmed"]
    assert not (tmp_path / "slow/finished").exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux worker supervision")
def test_service_loss_does_not_interrupt_exports_after_evaluation_succeeded(tmp_path, monkeypatch, request):
    monkeypatch.setenv("EXPORT_SLOW", "1")
    process = request.getfixturevalue("rendered_controller")
    _wait(tmp_path / "fast/export-finished", process)
    _wait(tmp_path / "slow/export-started", process)
    (tmp_path / "break-services").touch()
    time.sleep(6)
    assert process.poll() is None
    (tmp_path / "release-slow").touch()
    assert process.wait(timeout=15) == 0, (tmp_path / "logs/controller-41.log").read_text()
    assert (tmp_path / "slow/export-finished").exists()


def test_submits_one_job_and_preserves_group_and_member_records(tmp_path, monkeypatch):
    import nemo_gym.orchestration.executors.slurm.shared as module

    config = SharedAllocationConfig.model_validate(group_config())
    writes = {}
    staged = {}
    calls = []

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def copy(self, source, target):
            staged.update(
                {str(path.relative_to(source)): path.read_text() for path in source.rglob("*") if path.is_file()}
            )

        def write_text(self, path, text):
            writes[str(path)] = text

        def run(self, commands):
            calls.extend(commands)
            return "__GYM_JOB:shared-allocation:0:123"

    monkeypatch.setattr(module, "get_connection", lambda host: Connection())
    monkeypatch.setattr(module, "validate_gym_install_ref", lambda config: None)
    monkeypatch.setattr(SubmissionRecord, "write_local_index", lambda record: None)
    result = SharedSlurmExecutor().run(config)
    assert result.executor == "slurm-shared"
    assert len(result.benchmarks) == 1 and result.benchmarks[0].job_id == "123"
    assert sum("sbatch --parsable" in command for command in calls) == 1
    assert json.loads(staged["shared-allocation/members.json"])["fast"]["ports"]["head_server.port"] == 20000
    assert yaml.safe_load(staged["shared-allocation/slow/resolved-config.yaml"])["driver"]["container"] == "slow-image"
    assert any(
        yaml.safe_load(value).get("kind") == "shared_allocation"
        for key, value in writes.items()
        if key.endswith("yaml")
    )
    assert not isinstance(SharedSlurmExecutor(), Resubmittable)
