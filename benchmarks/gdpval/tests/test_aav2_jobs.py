# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the manual launcher and spooled jobs with local process fixtures."""

import fcntl
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from benchmarks.gdpval.hsg.aav2 import snapshot


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / snapshot.PACKAGE
PYTHON = sys.executable


def write(path, content, executable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    if executable:
        path.chmod(0o755)
    return path


# Platform adapters and expensive external process boundaries. Bash control flow,
# source extraction, environment setup, CLI argument parsing, and completion run normally.
TOOL = r"""import fcntl, json, os, shutil, sys
from pathlib import Path
name = Path(sys.argv[0]).name
if name == 'readlink':
    print(Path(sys.argv[-1]).resolve(strict=True))
elif name == 'stat':
    print(os.environ.get('FIXTURE_FILESYSTEM', 'ext4'))
elif name == 'flock':
    fcntl.flock(int(sys.argv[-1]), fcntl.LOCK_EX | fcntl.LOCK_NB)
elif name == 'setsid':
    os.setsid()
    os.execvp(sys.argv[1], sys.argv[1:])
elif name == 'scontrol':
    print('fixture-node')
elif name == 'squeue':
    print('01:00:00')
elif name == 'curl':
    print('{"data": [{"id": "fixture-model"}]}')
elif name == 'sbatch':
    with Path(os.environ['FIXTURE_RECORD']).open('a') as stream:
        stream.write(json.dumps({'sbatch': sys.argv[1:]}) + '\n')
    print('12345')
elif name == 'uv':
    root = Path.cwd()
    assert (root / 'cache').is_dir() and (root / 'uv.lock').is_file()
    assert os.environ['HF_DATASETS_CACHE'].startswith(str(root.parent))
    with Path(os.environ['FIXTURE_RECORD']).open('a') as stream:
        stream.write(json.dumps({'uv': sys.argv[1:], 'project': os.environ['UV_PROJECT_ENVIRONMENT'],
                                'cache': os.environ['UV_CACHE_DIR']}) + '\n')
    venv = root / '.venv/bin'
    venv.mkdir(parents=True)
    if os.environ.get('FIXTURE_UV_FAIL'):
        (venv / 'partial').touch()
        sys.exit(42)
    managed = Path(os.environ['UV_PYTHON_INSTALL_DIR']) / 'bin/python'
    managed.parent.mkdir()
    managed.write_text('#!/bin/bash\nexec ' + os.environ['FIXTURE_PYTHON'] + ' ' + os.environ['FIXTURE_DRIVER'] + ' "$@"\n')
    managed.chmod(0o755)
    (venv / 'python').symlink_to(os.environ['FIXTURE_PYTHON'] if os.environ.get('FIXTURE_SHARED_PYTHON') else managed)
    for command in ('setsid', 'scontrol', 'squeue', 'curl'):
        destination = Path(os.environ['UV_PYTHON_BIN_DIR']) / command
        shutil.copyfile(Path(os.environ['FIXTURE_TOOLS']) / command, destination)
        destination.chmod(0o755)
"""

DRIVER = r"""import json, os, sys, time
from pathlib import Path
args = sys.argv[1:]
def record(value):
    with Path(os.environ['FIXTURE_RECORD']).open('a') as stream:
        stream.write(json.dumps(value) + '\n')
if args and args[0].endswith(('/rollout_runtime.py', '/preconvert.py')):
    # The Linux-only interpreter guard and native conversion are covered in
    # their own tests; this fixture records that jobs invoke those boundaries.
    record({'check': args})
    sys.exit(0)
if args[:2] == ['-c', 'from nemo_gym.cli.main import main; main()']:
    sys.path.insert(0, os.environ['FIXTURE_SOURCE'])
    import nemo_gym.cli.main as cli
    def dispatch(target, overrides):
        values = dict(token.lstrip('+').split('=', 1) for token in overrides)
        output = Path(values['output_jsonl_fpath'])
        judge = output.parent.name.startswith('judge_')
        record({'target': target, 'values': values, 'cwd': os.getcwd(), 'pid': os.getpid(),
                'file_limit': os.environ['GDPVAL_MAX_FILE_BYTES_FOR_JUDGE']})
        if os.environ.get('FIXTURE_HOLD'):
            time.sleep(120)
        if os.environ.get('FIXTURE_GYM_RC'):
            raise SystemExit(int(os.environ['FIXTURE_GYM_RC']))
        if os.environ.get('FIXTURE_INCOMPLETE'):
            return
        if not judge:
            finish = Path(os.environ['PERSIST_DELIVERABLES_DIR']) / 'task_a/repeat_0/finish_params.json'
            finish.parent.mkdir(parents=True, exist_ok=True)
            finish.write_text('{}')
        else:
            trials = int(values['gdpval_resources_server.resources_servers.gdpval.num_comparison_trials'])
            stage = 0 if output.parent.name == 'judge_smoke' else 1
            output.write_text(json.dumps({'task_id': 'a', 'expected_final_stage_index': stage,
                'stage_index': stage, 'judge_response': {'total_judged': trials, 'total_invalid': 0}}) + '\n')
            metrics = {'comparison/final_stage_' + key: value for key, value in
                {'present': 1, 'complete': 1, 'fit': 1, 'degraded': 0}.items()}
            output.with_stem(output.stem + '_aggregate_metrics').with_suffix('.json').write_text(
                json.dumps([{'agent_metrics': metrics}]))
    cli.dispatch = dispatch
    sys.argv = ['gym', *args[2:]]
    cli.main()
else:
    os.execv(os.environ['FIXTURE_PYTHON'], [os.environ['FIXTURE_PYTHON'], *args])
"""


@pytest.fixture
def job(tmp_path, request):
    with tempfile.TemporaryDirectory(prefix="gj", dir="/private/tmp") as short:
        local, source, tools = Path(short), tmp_path / "source", tmp_path / "tools"
        package = source / snapshot.PACKAGE
        shutil.copytree(PACKAGE, package, ignore=shutil.ignore_patterns("__pycache__"))
        for name in ("readlink", "stat", "flock", "setsid", "scontrol", "squeue", "curl", "uv", "sbatch"):
            write(tools / name, f"#!{PYTHON}\n" + TOOL, True)
        write(tools / "python3", f'#!/bin/sh\nexec {shlex.quote(PYTHON)} "$@"\n', True)
        for path in package.iterdir():
            if path.suffix not in (".sh", ".sbatch", ".py"):
                continue
            content = path.read_text().replace("/raid/scratch", str(local))
            if path.suffix in (".sh", ".sbatch"):
                for utility in ("readlink", "stat", "flock"):
                    content = content.replace(f"{utility} ", f"{shlex.quote(str(tools / utility))} ")
            path.write_text(content)
        write(source / ".python-version", "3.13.14\n")
        write(source / "pyproject.toml", '[tool.distutils.egg_info]\negg_base="cache"\n')
        write(source / "uv.lock", "# pinned fixture dependencies\n")
        subprocess.run(["git", "init", "-q", str(source)], check=True)
        subprocess.run(["git", "add", "."], cwd=source, check=True)
        subprocess.run(
            ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.com", "commit", "-qm", "fixture"],
            cwd=source,
            check=True,
        )
        dataset = write(
            tmp_path / "dataset.jsonl",
            "".join(
                json.dumps(
                    {"task_id": f"a{index}" if index else "a", "reference_file_urls": ["file:///lustre/ref.txt"]}
                )
                + "\n"
                for index in range(getattr(request, "param", 1))
            ),
        )
        config = write(tmp_path / "judge.yaml", "multistage:\n  stages: [{num_tasks: 1}]\n")
        credentials = write(tmp_path / "credentials.env", "export HF_DATASETS_CACHE=/lustre/stale-cache\n")
        driver = write(tmp_path / "driver.py", DRIVER)
        image = write(tmp_path / "vllm:fixture.sqsh", "image")
        model = tmp_path / "model"
        for name, value in {
            "config.json": "{}",
            "modeling_fixture.py": "VALUE=1\n",
            "part.safetensors": "weights",
        }.items():
            write(model / name, value)
        parser = write(tmp_path / "parsers/parser.py", 'NAME="parser"\n')
        sif = write(tmp_path / "agent.sif", "agent image")
        apptainer = write(tmp_path / "apptainer/bin/apptainer", "#!/bin/sh\nexit 0\n", True)
        serving = write(
            tmp_path / "serve.sh",
            r"""#!/bin/bash
set -eu
[[ -z ${FIXTURE_SERVE_FAIL:-} ]] || exit 37
mkdir -p "$OUTPUT_DIR/server_info"
printf 'SERVER_URL=http://fixture/v1\n' > "$OUTPUT_DIR/server_info/ready.env"
printf '%s\n' "$VLLM_EXTRA_ARGS" > "$OUTPUT_DIR/extra-args.txt"
printf '%s\n' "$$" > "$OUTPUT_DIR/pid"
exec sleep 120
""",
            True,
        )
        extra = '--skip-mm-profiling --speculative-config \'{"method":"mtp","num_speculative_tokens":1}\''
        profile = write(
            tmp_path / "serving.env",
            "".join(
                f"export {key}={shlex.quote(value)}\n"
                for key, value in {
                    "POLICY_SERVE_SCRIPT": str(serving),
                    "MODEL_NAME": "fixture-model",
                    "MODEL_PATH": str(model),
                    "CONTAINER_IMAGE": str(image),
                    "EXTRA_MOUNTS": f"{parser.parent}:/parsers:ro",
                    "VLLM_EXTRA_ARGS": extra,
                    "SERVE_WAIT_SECONDS": "5",
                }.items()
            ),
        )
        args = SimpleNamespace(
            run_dir=tmp_path / "run",
            source=source,
            revision="HEAD",
            dataset=dataset,
            smoke_dataset=dataset,
            profile=profile,
            existing_rollout=None,
            judge_config=config,
            env_file=credentials,
            uv_source=tools / "uv",
            agent_sif=sif,
            judge_sif=None,
            apptainer_bin=apptainer.parent,
            concurrency=2,
            agent_max_turns=32,
        )
        snapshot.prepare(args)
        write(args.run_dir / "prepared/candidate/task_a/repeat_0/finish_params.json", "{}")
        env = {
            **os.environ,
            "AAV2_RUN_DIR": str(args.run_dir),
            "SLURM_JOB_USER": "u",
            "SLURM_JOB_ID": "41",
            "SLURM_JOB_NODELIST": "fixture-node",
            "FIXTURE_RECORD": str(tmp_path / "record.jsonl"),
            "FIXTURE_PYTHON": PYTHON,
            "FIXTURE_DRIVER": str(driver),
            "FIXTURE_TOOLS": str(tools),
            "FIXTURE_SOURCE": str(ROOT),
            "PATH": f"{tools}:/usr/bin:/bin:/usr/sbin:/sbin",
        }
        yield SimpleNamespace(
            run=args.run_dir,
            package=args.run_dir / "package",
            local=local,
            env=env,
            extra=extra,
            args=args,
            tools=tools,
        )


def records(job):
    path = Path(job.env["FIXTURE_RECORD"])
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def run_job(job, phase, **environment):
    spooled = write(job.run.parent / "slurm_script", (job.package / f"aav2_{phase}.sbatch").read_text())
    return subprocess.run(
        ["bash", str(spooled)],
        cwd=job.run.parent,
        env={**job.env, **environment},
        text=True,
        capture_output=True,
        timeout=30,
    )


@pytest.mark.parametrize("mode,trials,concurrency", [("smoke", 1, 4), ("pilot", 2, 8), ("full", 4, 16)])
def test_judge_modes_use_prepared_candidate_and_frozen_references(job, mode, trials, concurrency):
    result = run_job(job, "judge", AAV2_MODE=mode, GDPVAL_MAX_FILE_BYTES_FOR_JUDGE="1")
    assert result.returncode == 0, (result.stdout, result.stderr)
    entries = records(job)
    assert not any(item.get("check", [""])[0].endswith("/preconvert.py") for item in entries)
    assert any(item.get("check", [""])[0].endswith("/rollout_runtime.py") for item in entries)
    call = next(item for item in entries if "target" in item)
    assert call["target"] == "nemo_gym.cli.eval:e2e_rollout_collection"
    assert call["file_limit"] == json.loads((job.run / "run.json").read_text())["GDPVAL_MAX_FILE_BYTES_FOR_JUDGE"]
    values = call["values"]
    assert values["resume_from_cache"] == "true"
    assert values["num_samples_in_parallel"] == str(concurrency)
    assert values["model_endpoint_readiness_timeout_seconds"] == "0"
    assert values["head_server.host"] == "127.0.0.1"
    assert 0 < int(values["head_server.port"]) < 65536
    prefix = "gdpval_resources_server.resources_servers.gdpval."
    assert values[prefix + "num_comparison_trials"] == str(trials)
    assert values[prefix + "preconvert_office_to_pdf"] == "false"
    assert prefix + "strict_comparison_trials" not in values
    assert values[prefix + "judge_reference_files_recursive"] == "true"
    assert "benchmarks/gdpval/config.yaml" in values["config_paths"]
    staged_judge = Path(call["cwd"]).parent / "judge.yaml"
    assert str(staged_judge) in values["config_paths"]
    assert str(job.run / "prepared/judge.yaml") not in values["config_paths"]
    frozen_judge = Path(json.loads((job.run / "run.json").read_text())["JUDGE_CONFIG"])
    assert staged_judge.read_bytes() == frozen_judge.read_bytes()
    assert not (job.run / "prepared/manifest.json").exists()
    assert "true3_transport" not in values["config_paths"]
    assert not any(key.startswith(prefix + "judge_panel") or key == prefix + "judge_media_mode" for key in values)
    assert values["output_jsonl_fpath"] == str(job.run / f"judge_{mode}/rollouts.jsonl")
    assert values["gdpval_stirrup_agent.responses_api_agents.stirrup_agent.persist_deliverables_dir"] == str(
        job.run / "prepared/candidate"
    )
    overlay = json.loads((Path(call["cwd"]).parent / "dataset.yaml").read_text())
    expected = job.run / ("smoke_dataset.jsonl" if mode == "smoke" else "dataset.jsonl")
    assert overlay["gdpval_stirrup_agent"]["responses_api_agents"]["stirrup_agent"]["datasets"][0][
        "jsonl_fpath"
    ] == str(expected)
    assert not list(job.local.rglob("agent.sif"))


@pytest.mark.parametrize("job", [220], indirect=True)
def test_full_judge_accepts_partial_calibration_and_retains_all_final_tasks(job):
    result = run_job(job, "judge", AAV2_MODE="full", FIXTURE_GYM_RC="37")
    assert result.returncode == 37, (result.stdout, result.stderr)
    call = next(item for item in records(job) if "target" in item)
    stages = OmegaConf.to_container(OmegaConf.create(call["values"]["multistage.stages"]))
    assert stages == [
        {
            "num_tasks": 45,
            "partial_completion": {
                "min_success_fraction": 0.97,
                "min_per_reference_success_fraction": 0.8,
                "min_successful_rows_per_reference": 1,
                "tolerate_unresolved": True,
            },
        },
        {"num_tasks": 220, "num_models": 4},
    ]


def test_judge_requires_a_prepared_candidate(job):
    shutil.rmtree(job.run / "prepared/candidate")
    result = run_job(job, "judge")
    assert result.returncode == 64
    assert "prepared candidate directory is missing" in result.stderr
    assert not any("target" in item for item in records(job))


def test_rollout_preserves_quoted_serving_args_and_uses_fresh_resume_environments(job):
    for job_id in ("41", "42"):
        result = run_job(job, "rollout", SLURM_JOB_ID=job_id)
        assert result.returncode == 0, (result.stdout, result.stderr)
        serving = job.run / f"serving-{job_id}"
        assert (serving / "extra-args.txt").read_text().strip() == job.extra
        with pytest.raises(ProcessLookupError):
            os.kill(int((serving / "pid").read_text()), 0)
    installs = [item for item in records(job) if "uv" in item]
    assert installs[0]["project"] != installs[1]["project"]
    assert installs[0]["cache"] == installs[1]["cache"] == str(job.local / "u/uv-cache")
    assert (job.local / "u/uv-cache").stat().st_mode & 0o777 == 0o700
    assert all(
        item["uv"] == ["sync", "--frozen", "--no-dev", "--managed-python", "--python", "3.13.14"] for item in installs
    )
    calls = [item for item in records(job) if "target" in item]
    assert all(item["values"]["resume_from_cache"] == "true" for item in calls)
    assert all(item["values"]["dispatch_budget_s"] == "3300" for item in calls)
    assert calls[0]["values"]["output_jsonl_fpath"] == calls[1]["values"]["output_jsonl_fpath"]
    assert all(call["values"]["uv_cache_dir"] == installs[0]["cache"] for call in calls)


@pytest.mark.parametrize(
    "environment,expected",
    [
        ({"FIXTURE_UV_FAIL": "1"}, 42),
        ({"FIXTURE_SHARED_PYTHON": "1"}, 64),
        ({"FIXTURE_SERVE_FAIL": "1"}, 64),
        ({"FIXTURE_GYM_RC": "37"}, 37),
        ({"FIXTURE_INCOMPLETE": "1"}, 1),
    ],
)
def test_failures_preserve_process_status_and_never_report_completion(job, environment, expected):
    result = run_job(job, "rollout", **environment)
    assert result.returncode == expected, (result.stdout, result.stderr)
    assert "COMPLETE:" not in result.stdout
    if "FIXTURE_UV_FAIL" in environment:
        partial = next(job.local.rglob("partial"))
        retry = run_job(job, "rollout", SLURM_JOB_ID="42")
        assert retry.returncode == 0, (retry.stdout, retry.stderr)
        assert partial.exists()
        installs = [item for item in records(job) if "uv" in item]
        assert installs[0]["project"] != installs[1]["project"]


def test_run_lock_rejects_concurrent_phases_before_setup(job):
    with (job.run / ".phase.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = run_job(job, "judge")
    assert result.returncode == 64
    assert "another phase is already running" in result.stderr
    assert not records(job)


def test_checksum_failure_precedes_sourcing_run_settings(job):
    marker = job.run.parent / "must-not-exist"
    (job.run / "run.env").chmod(0o600)
    with (job.run / "run.env").open("a") as stream:
        stream.write(f"touch {shlex.quote(str(marker))}\n")
    result = run_job(job, "judge")
    assert result.returncode != 0
    assert not marker.exists() and not records(job)


@pytest.mark.parametrize("office_image", [False, True])
def test_sandbox_stages_selected_office_image_or_default_agent(job, office_image):
    image = write(job.run.parent / "python-3.12.gdpval.sqsh", "Office image with Java")
    stage = job.local / "sandbox-stage"
    stage.mkdir()
    command = 'source "$1"; shift; gdpval_prepare_sandbox "$@"; cat "$GDPVAL_CONTAINER_PATH"'
    arguments = [str(image)] if office_image else []
    result = subprocess.run(
        ["bash", "-euc", command, "fixture", str(job.package / "node_local.sh"), *arguments],
        env={
            **job.env,
            "JOB_ROOT": str(stage),
            "AGENT_SIF": str(job.args.agent_sif),
            "APPTAINER_BIN": str(job.args.apptainer_bin),
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == (image if office_image else job.args.agent_sif).read_text()
    assert not (stage / "agent.sif").is_symlink()


def test_term_reaps_gym_and_serving_groups(job):
    with subprocess.Popen(
        ["bash", str(job.package / "aav2_rollout.sbatch")],
        cwd=job.run.parent,
        env={**job.env, "FIXTURE_HOLD": "1"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as process:
        deadline = time.monotonic() + 20
        calls = []
        while time.monotonic() < deadline and process.poll() is None:
            calls = [item for item in records(job) if "target" in item]
            if calls:
                break
            time.sleep(0.1)
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=15)
    assert calls, (stdout, stderr)
    assert process.returncode == 143, (stdout, stderr)
    for pid in (calls[0]["pid"], int((job.run / "serving-41/pid").read_text())):
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_original_manual_cli_submits_one_phase_and_keeps_stages_out_of_slurm_exports(job):
    config = write(job.run.parent / "aav2.env", "ACCOUNT=fixture\nGPU_QOS=gpu-fixture\nCPU_QOS=cpu-fixture\n")
    environment = {**job.env, "AAV2_CONFIG": str(config)}
    for arguments, expected in [
        (["rollout", "resume", str(job.run)], "rollout"),
        (["preconvert", str(job.run)], "preconvert"),
        (["judge", str(job.run), "pilot"], "judge"),
    ]:
        result = subprocess.run(
            ["bash", str(job.package / "run_aav2.sh"), *arguments], env=environment, text=True, capture_output=True
        )
        assert result.returncode == 0, (result.stdout, result.stderr)
        command = records(job)[-1]["sbatch"]
        assert command[-1] == str(job.package / f"aav2_{expected}.sbatch")
        assert not any("dependency" in argument or "STAGES=" in argument for argument in command)
        assert f"--qos={'gpu' if expected == 'rollout' else 'cpu'}-fixture" in command
        receipt = "judge_pilot" if expected == "judge" else expected
        assert (job.run / f"{receipt}.jobid").read_text() == "12345\n"
    assert len(records(job)) == 3
    assert f"--export=ALL,AAV2_RUN_DIR={job.run},AAV2_MODE=pilot" in records(job)[-1]["sbatch"]


@pytest.mark.parametrize("phase", ["import", "smoke", "full"])
def test_manual_cli_freezes_new_run_and_import_submits_no_job(job, phase):
    evidence = job.run / "deliverables/task_a/repeat_0"
    write(evidence / "finish_params.json", "{}")
    write(evidence / "history.json", '[{"role":"assistant","content":"done"}]')
    write(evidence / "answer.txt", "candidate evidence\n")
    write(job.run / "deliverables/task_a/repeat_0_verify_response.json", '{"old_judgment":true}')
    settings = {
        "ACCOUNT": "fixture",
        "GYM_SOURCE": job.args.source,
        "DATASET": job.args.dataset,
        "SMOKE_DATASET": job.args.smoke_dataset,
        "JUDGE_CONFIG": job.args.judge_config,
        "ENV_FILE": job.args.env_file,
        "UV_SOURCE": job.args.uv_source,
        "AGENT_SIF": job.args.agent_sif,
        "JUDGE_SIF": write(job.run.parent / "office.sqsh", "Office image"),
        "APPTAINER_BIN": job.args.apptainer_bin,
        "RUNS_DIR": job.run.parent / "new-runs",
        "GDPVAL_MAX_FILE_BYTES_FOR_JUDGE": "1048576",
    }
    if phase != "import":
        settings["PROFILE"] = job.args.profile
    config = write(
        job.run.parent / "aav2.env", "".join(f"{key}={shlex.quote(str(value))}\n" for key, value in settings.items())
    )
    arguments = ["import", str(job.run)] if phase == "import" else ["rollout", phase]
    result = subprocess.run(
        ["bash", str(job.package / "run_aav2.sh"), *arguments],
        env={**job.env, "AAV2_CONFIG": str(config)},
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    created = next(settings["RUNS_DIR"].iterdir())
    snapshot.verify(created)
    assert json.loads((created / "run.json").read_text())["GDPVAL_MAX_FILE_BYTES_FOR_JUDGE"] == "1048576"
    assert json.loads((created / "run.json").read_text())["JUDGE_SIF"] == str(settings["JUDGE_SIF"])
    if phase == "import":
        assert not records(job)
        assert not (created / "serving.env").exists()
        assert (created / "deliverables/task_a/repeat_0/answer.txt").read_bytes() == (
            evidence / "answer.txt"
        ).read_bytes()
        assert not list(created.rglob("*verify_response*"))
        assert (job.run / "deliverables/task_a/repeat_0_verify_response.json").exists()
        resume = subprocess.run(
            ["bash", str(created / "package/run_aav2.sh"), "rollout", "resume", str(created)],
            env={**job.env, "AAV2_CONFIG": str(config)},
            text=True,
            capture_output=True,
        )
        assert resume.returncode == 64 and "Imported evidence cannot resume" in resume.stderr
        assert not records(job)
    else:
        assert len(records(job)) == 1
        assert records(job)[0]["sbatch"][-1] == str(created / "package/aav2_rollout.sbatch")
        assert (created / "serving.env").read_bytes() == job.args.profile.read_bytes()
