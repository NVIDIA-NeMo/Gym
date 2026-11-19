# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from nemo_gym.orchestration.api import SubmitConfig
from nemo_gym.orchestration.executors.slurm.script import build_sbatch_script


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux child subreaper")


def _executable(directory: Path, name: str, code: str) -> None:
    path = directory / name
    path.write_text(f"#!{sys.executable}\n" + code)
    path.chmod(0o755)


def _render(tmp_path: Path) -> str:
    config = SubmitConfig.model_validate(
        {
            "services": {},
            "otel": {"enabled": False},
            "compute": {"cluster": {"type": "slurm", "account": "test-account", "hostname": "host"}},
            "driver": {
                "supervise": True,
                "container": "test-image",
                "benchmarks": {"test": {"run": {"label": "two words"}}},
            },
            "job": {"output_path": str(tmp_path)},
        }
    )
    return build_sbatch_script(config, "test", config.driver.benchmarks["test"], config.compute["cluster"], tmp_path)


@pytest.fixture
def scheduler(tmp_path, monkeypatch):
    commands = tmp_path / "bin"
    commands.mkdir()
    monkeypatch.setenv("PATH", str(commands) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("SLURM_JOB_ID", "41")
    monkeypatch.setenv("GYM_RESUBMIT", "0")
    _executable(commands, "sleep", "")
    _executable(
        commands,
        "squeue",
        """import os,sys
from pathlib import Path
if os.environ.get('BROKEN_QUEUE'): raise SystemExit(3)
print('41.3|shared-policy')
if Path('step').exists(): print('41.2|' + Path('step').read_text())
""",
    )
    _executable(
        commands,
        "scancel",
        """import json,sys
from pathlib import Path
with Path('cancelled').open('a') as f: f.write(json.dumps(sys.argv[1:]) + '\\n')
assert sys.argv[-1] == '41.2'
Path('step').unlink(missing_ok=True)
""",
    )
    _executable(
        commands,
        "srun",
        """import os,subprocess,sys
from pathlib import Path
name = next(a.split('=',1)[1] for a in sys.argv if a.startswith('--job-name='))
Path('step').write_text(name)
args=sys.argv[1:]
while args[0].startswith('--'): args.pop(0)
raise SystemExit(subprocess.call(args))
""",
    )
    _executable(
        commands,
        "gym",
        """import json,os,sys,time
from pathlib import Path
Path('argv.json').write_text(json.dumps(sys.argv[1:]))
Path('resume').write_text(os.environ['GYM_RESUBMIT'])
Path('started').touch()
if os.environ.get('BLOCK_DRIVER'): time.sleep(60)
raise SystemExit(int(os.environ.get('DRIVER_EXIT', '0')))
""",
    )
    script = tmp_path / "job.sh"
    script.write_text(_render(tmp_path))
    subprocess.run(["bash", "-n", str(script)], check=True)
    return script


@pytest.mark.parametrize("exit_code", [0, 7])
def test_rendered_worker_preserves_native_argv_and_drains_only_its_step(tmp_path, monkeypatch, scheduler, exit_code):
    monkeypatch.setenv("DRIVER_EXIT", str(exit_code))
    result = subprocess.run(["bash", str(scheduler)], cwd=tmp_path, timeout=15)
    assert result.returncode == exit_code
    args = json.loads((tmp_path / "argv.json").read_text())
    assert args[:2] == ["eval", "run"]
    assert "+label=two words" in args
    assert (tmp_path / "resume").read_text() == "0"
    assert not (tmp_path / "step").exists()
    assert json.loads((tmp_path / "cancelled").read_text()) == ["--signal=TERM", "41.2"]
    (receipt,) = (tmp_path / "logs").glob("driver-worker.*.json")
    assert json.loads(receipt.read_text())["cleanup_confirmed"]


def test_cancelling_the_rendered_shell_waits_for_worker_cleanup(tmp_path, monkeypatch, scheduler):
    monkeypatch.setenv("BLOCK_DRIVER", "1")
    process = subprocess.Popen(["bash", str(scheduler)], cwd=tmp_path)
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "started").exists():
            assert process.poll() is None and time.monotonic() < deadline
            time.sleep(0.02)
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=15) == 143
        assert not (tmp_path / "step").exists()
        (receipt,) = (tmp_path / "logs").glob("driver-worker.*.json")
        value = json.loads(receipt.read_text())
        assert value["state"] == "cancelled" and value["cleanup_confirmed"]
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_unreadable_scheduler_does_not_confirm_cleanup(tmp_path, monkeypatch, scheduler):
    monkeypatch.setenv("BROKEN_QUEUE", "1")
    result = subprocess.run(["bash", str(scheduler)], cwd=tmp_path, timeout=15)
    assert result.returncode == 70
    (receipt,) = (tmp_path / "logs").glob("driver-worker.*.json")
    value = json.loads(receipt.read_text())
    assert value["state"] == "cleanup_failed" and not value["cleanup_confirmed"]
