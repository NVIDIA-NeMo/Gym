# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
import os
import shutil
import subprocess
import sys
import tomllib
import types
from pathlib import Path

import pytest

from benchmarks.posttrainbench import prepare


# The lines of upstream's src/harbor_adapter/template/task.toml (at prepare.PTB_REVISION) that prepare changes.
UPSTREAM_TOML = """version = "1.0"

[environment]
gpus = 1
gpu_types = ["H100!"]
allow_internet = true

[environment.healthcheck]
command = "pgrep -f 'tail -F' > /dev/null && (test -f /timer_start || date +%s > /timer_start)"

[verifier]
timeout_sec = 18000.0
environment_mode = "separate"

[verifier.env]
OPENAI_API_KEY = "${OPENAI_API_KEY}"
CODEX_API_KEY = "${OPENAI_API_KEY}"
HF_TOKEN = "${HF_TOKEN}"
PTB_MODEL_DIR = "/mnt/ptb_final_model"

[verifier.environment]
gpus = 1
gpu_types = ["H100!"]

[verifier.environment.healthcheck]
command = "pgrep -f 'tail -F' > /dev/null && (test -f /timer_start || date +%s > /timer_start)"
retries = 3

# Post-agent collection.
[[verifier.collect]]
service = "main"
timeout_sec = 3600.0
command = "bash /usr/local/bin/ptb_collect.sh"

[agent]
timeout_sec = 36000.0

[environment.env]
HF_TOKEN = "${HF_TOKEN}"
"""


def test_task_toml_hands_off_the_model_and_routes_the_judges() -> None:
    adapted = tomllib.loads(
        prepare.adapt_task_toml(
            UPSTREAM_TOML, judge_base_url="https://judge.example/v1", judge_api_key_env="JUDGE_KEY"
        )
    )

    verifier_env = adapted["verifier"]["env"]
    assert verifier_env["OPENAI_API_KEY"] == verifier_env["CODEX_API_KEY"] == "${JUDGE_KEY}"
    assert verifier_env["OPENAI_BASE_URL"] == "https://judge.example/v1"
    assert verifier_env["PTB_MODEL_DIR"] == prepare.MODEL_HANDOFF_DIR == "/logs/artifacts/final_model"
    assert adapted["environment"]["env"] == {"HF_TOKEN": "${HF_TOKEN}", "BASH_MAX_TIMEOUT_MS": "36000000"}
    command = adapted["verifier"]["collect"][0]["command"]
    assert command == (
        "bash -c 'rm -rf /logs/artifacts/final_model; "
        "PTB_VOLUME_DIR=/logs/artifacts/final_model bash /usr/local/bin/ptb_collect.sh'"
    )
    # The verifier is built from tests/Dockerfile with the task's own GPU, not upstream's H100-only table.
    assert "environment" not in adapted["verifier"]
    assert adapted["verifier"]["collect"][0]["timeout_sec"] == 3600.0
    # The Modal-only parts that Gym does not read are left as upstream wrote them.
    assert adapted["environment"]["gpu_types"] == ["H100!"]
    assert adapted["agent"]["timeout_sec"] == 36000.0


def test_the_adapted_task_builds_its_verifier_from_tests_on_the_task_gpu(tmp_path: Path) -> None:
    import tomllib

    from nemo_gym.tasks.harbor.models import HarborTaskConfig
    from nemo_gym.tasks.harbor.verifier_env import effective_verifier_environment, verifier_build_context

    adapted = prepare.adapt_task_toml(UPSTREAM_TOML, judge_base_url="https://judge.example/v1", judge_api_key_env="K")
    config = HarborTaskConfig.model_validate(tomllib.loads(adapted))
    task_dir = tmp_path / "posttrainbench-gsm8k-qwen3-1.7b"
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "tests" / "Dockerfile").write_text("FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04\n")

    # Upstream's tests/Dockerfile is the verifier image, built with the task environment's one GPU.
    assert verifier_build_context(task_dir, config) == task_dir / "tests"
    assert effective_verifier_environment(config).gpus == 1


@pytest.mark.parametrize("agent_key, expected", [(None, None), ("PTB_AGENT_KEY", "${PTB_AGENT_KEY}")])
def test_the_agent_never_gets_the_judges_key(agent_key: str | None, expected: str | None) -> None:
    # arenahardwriting and healthbench list OPENAI_API_KEY in required_api_keys, so upstream gives it to the agent.
    upstream = UPSTREAM_TOML + 'OPENAI_API_KEY = "${OPENAI_API_KEY}"\n'
    adapted = tomllib.loads(
        prepare.adapt_task_toml(
            upstream,
            judge_base_url="https://judge.example/v1",
            judge_api_key_env="JUDGE_KEY",
            agent_api_key_env=agent_key,
        )
    )

    agent_env = adapted["environment"]["env"]
    assert agent_env.get("OPENAI_API_KEY") == expected
    assert agent_env.get("OPENAI_BASE_URL") == (expected and "https://judge.example/v1")
    assert adapted["verifier"]["env"]["OPENAI_API_KEY"] == "${JUDGE_KEY}"


def test_task_toml_refuses_an_upstream_change() -> None:
    changed = UPSTREAM_TOML.replace('PTB_MODEL_DIR = "/mnt/ptb_final_model"', 'PTB_MODEL_DIR = "/vol"')
    with pytest.raises(ValueError, match="adapter output changed"):
        prepare.adapt_task_toml(changed, judge_base_url="https://judge.example/v1", judge_api_key_env="JUDGE_KEY")


def test_judge_models_are_renamed_in_code_but_not_in_comments() -> None:
    text = '# defaults to gpt-5.6-terra\nJUDGE_DEFAULT_MODEL="gpt-5.6-terra"\nJUDGE_MODEL = "gpt-5-mini"\nX = "gpt-5-mini-2"\n'
    renamed = prepare.rename_judge_models(text, prepare.DEFAULT_JUDGE_MODELS)

    assert renamed == (
        "# defaults to gpt-5.6-terra\n"
        'JUDGE_DEFAULT_MODEL="azure/openai/gpt-5.6-terra"\n'
        'JUDGE_MODEL = "azure/openai/gpt-5-mini"\n'
        'X = "gpt-5-mini-2"\n'
    )


def test_adapt_task_rewrites_the_verifier_side_only(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    (source / "scripts/utils.py").write_text("# upstream verdict readers\n")
    task = tmp_path / "posttrainbench-gsm8k-qwen3-1.7b"
    (task / "environment").mkdir(parents=True)
    (task / "environment/evaluate.py").write_text('JUDGE_MODEL = "gpt-5-mini"\n')
    judge_lib = task / "tests/ptb/src/judges/judge_lib.sh"
    judge_lib.parent.mkdir(parents=True)
    judge_lib.write_text('JUDGE_DEFAULT_MODEL="gpt-5.6-terra"\n')
    (task / "tests/test.sh").write_text('#!/bin/bash\necho "=== Verification complete ==="\n')
    (task / "task.toml").write_text(UPSTREAM_TOML)

    prepare.adapt_task(
        task,
        source,
        judge_base_url="https://judge.example/v1",
        judge_api_key_env="INFERENCE_API_KEY",
        judge_models=prepare.DEFAULT_JUDGE_MODELS,
    )

    assert judge_lib.read_text() == 'JUDGE_DEFAULT_MODEL="azure/openai/gpt-5.6-terra"\n'
    assert (task / "environment/evaluate.py").read_text() == 'JUDGE_MODEL = "azure/openai/gpt-5-mini"\n'
    assert (task / "tests/ptb/scripts/utils.py").read_text() == "# upstream verdict readers\n"
    assert (task / "tests/gym_score.py").read_text() == prepare.GYM_SCORE_SCRIPT.read_text()
    test_sh = (task / "tests/test.sh").read_text()
    assert prepare.append_gym_score(test_sh) == test_sh
    assert test_sh.startswith("#!/bin/bash\n# Gym (benchmarks/posttrainbench/prepare.py)")
    assert test_sh.endswith('echo "=== Verification complete ==="\n')


def test_both_image_records_land_where_the_loader_reads_them(tmp_path: Path) -> None:
    from nemo_gym.tasks.harbor.records import (
        BUILT_IMAGES_FILE,
        IMAGE_CONFIGS_FILE,
        read_built_images,
        read_image_configs,
    )

    built = {"hb__abc": {"image": "registry.example/hb__abc", "pushed": True}}
    configs = {"registry.example/hb__abc": {"os": "linux", "architecture": "amd64", "inputs_digest": "sha256:1"}}
    records = (tmp_path / BUILT_IMAGES_FILE, tmp_path / IMAGE_CONFIGS_FILE)
    records[0].write_text(json.dumps(built))
    records[1].write_text(json.dumps(configs))
    tasks_dir = tmp_path / "data" / "tasks" / "posttrainbench"
    (tasks_dir / "posttrainbench-gsm8k-qwen3-1.7b").mkdir(parents=True)

    prepare.restore_image_records(tasks_dir, records)

    # The loader and the builder read both from a task folder's parent.
    assert read_built_images(tasks_dir) == built
    assert read_image_configs(tasks_dir) == configs


def test_one_image_record_without_the_other_is_refused(tmp_path: Path) -> None:
    records = (tmp_path / "built-images.json", tmp_path / "compose-images.json")
    records[0].write_text("{}")

    with pytest.raises(FileNotFoundError, match="together"):
        prepare.restore_image_records(tmp_path, records)


def run_adapted_test_sh(tmp_path: Path, model_dir: Path) -> str:
    """Run the adapted test.sh around a stand-in for upstream's, which reports what final_model it sees."""
    upstream = '#!/bin/bash\n[ -d "$PTB_MODEL_DIR" ] && echo "model present" || echo "model absent"\nexit 0\n'
    script = tmp_path / "test.sh"
    script.write_text(prepare.append_gym_score(upstream))
    env = {"PATH": os.environ["PATH"], "PTB_MODEL_DIR": str(model_dir), "TESTS": str(tmp_path), "LOGS_DIR": "/x"}
    return subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, check=False).stdout


@pytest.mark.parametrize("link", ["config.json", None])
def test_a_model_tree_with_links_is_not_submitted(tmp_path: Path, link: str | None) -> None:
    secrets = tmp_path / "environ"
    secrets.write_text("OPENAI_API_KEY=secret")
    model = tmp_path / "final_model"
    if link is None:  # final_model itself is the link
        model.symlink_to(tmp_path)
    else:
        model.mkdir()
        (model / "model.safetensors").write_bytes(b"\0")
        (model / link).symlink_to(secrets)

    out = run_adapted_test_sh(tmp_path, model)

    assert "model absent" in out and "links or special files" in out
    assert not model.exists() and not model.is_symlink()
    assert secrets.read_text() == "OPENAI_API_KEY=secret"


def test_a_plain_model_tree_reaches_upstream_unchanged(tmp_path: Path) -> None:
    model = tmp_path / "final_model"
    (model / "sub").mkdir(parents=True)
    (model / "config.json").write_text("{}")

    assert "model present" in run_adapted_test_sh(tmp_path, model)
    assert (model / "config.json").read_text() == "{}"


STAND_IN_UTILS = """
import json, os

def _read(logs, name):
    path = os.path.join(logs, name)
    return json.load(open(path)) if os.path.exists(path) else None

def load_judgement(logs):
    verdict = _read(logs, "judgement_gpt5_4.json")
    if verdict is None:
        raise FileNotFoundError(logs)
    return verdict

def load_api_judgement(logs):
    verdict = _read(logs, "judgement_api.json")
    return None if verdict is None else verdict["disallowed_api_usage"]

def load_ptb_lookup_judgement(logs):
    verdict = _read(logs, "judgement_ptb_lookup.json")
    return None if verdict is None else verdict["disallowed_ptb_lookup"]

def judgement_to_cell(judgement, api_usage):
    return "M" * judgement["disallowed_model"] + "C" * judgement["contamination"] + "A" * bool(api_usage)
"""

# What upstream test.sh's write_no_model_results leaves before its early exit 0 (verdict files and metrics.json).
NO_MODEL_STAND_IN = """#!/bin/bash
echo '{"contamination": false, "disallowed_model": false}' > "$LOGS_DIR/judgement_gpt5_4.json"
echo '{"disallowed_api_usage": false}' > "$LOGS_DIR/judgement_api.json"
echo '{"disallowed_ptb_lookup": false}' > "$LOGS_DIR/judgement_ptb_lookup.json"
echo '{"accuracy": 0.25, "error": "No final model submitted"}' > "$LOGS_DIR/metrics.json"
exit 0
echo "never reached"
"""


def run_verifier(tmp_path: Path, upstream: str, planted: dict[str, str] | None = None) -> tuple[Path, Path]:
    """Lay out /tests and /logs/verifier as Gym does (tests/ world-writable) and run the adapted test.sh."""
    tests, logs = tmp_path / "tests", tmp_path / "logs"
    (tests / "ptb" / "scripts").mkdir(parents=True)
    logs.mkdir()
    (tests / "ptb" / "scripts" / "utils.py").write_text(STAND_IN_UTILS)
    shutil.copy2(prepare.GYM_SCORE_SCRIPT, tests / "gym_score.py")
    (tests / "metadata.json").write_text(json.dumps({"baseline_accuracy": 0.25}))
    for name, text in (planted or {}).items():
        (tests / name).write_text(text)
    (tests / "test.sh").write_text(prepare.append_gym_score(upstream))
    for path in [tests, *tests.rglob("*")]:
        path.chmod(0o777 if path.is_dir() else 0o666)
    env = {
        "PATH": os.environ["PATH"],
        "TESTS": str(tests),
        "LOGS_DIR": str(logs),
        "PTB_MODEL_DIR": str(tmp_path / "m"),
    }
    subprocess.run(["bash", str(tests / "test.sh")], env=env, check=True, capture_output=True)
    return tests, logs


def test_an_early_no_model_exit_is_scored_as_the_baseline(tmp_path: Path) -> None:
    _, logs = run_verifier(tmp_path, NO_MODEL_STAND_IN)

    fields = json.loads((logs / "reward.json").read_text())
    assert fields["reward"] == 0.25 and fields["no_model"] == 1.0
    assert fields["trace_missing"] == fields["judgement_missing"] == 0.0


def test_the_judges_cannot_change_the_scorer(tmp_path: Path) -> None:
    # A module planted beside gym_score.py, as a judge running as nobody could have written it before the fix.
    tests, logs = run_verifier(tmp_path, NO_MODEL_STAND_IN, planted={"json.py": "raise SystemExit('planted')\n"})

    assert all(not path.stat().st_mode & 0o022 for path in [tests, *tests.rglob("*")])
    assert json.loads((logs / "reward.json").read_text())["reward"] == 0.25


FACTORS = {"gsm8k": 0.5, "aime2025": 0.5}


def write_ptb_task(tasks_dir: Path, name: str, baseline: float, dockerfile: str = "FROM ubuntu:24.04\nWORKDIR /app"):
    from environment_servers.harbor.tests.conftest import write_task

    task = write_task(tasks_dir / name, dockerfile=dockerfile)
    (task / "tests" / "metadata.json").write_text(json.dumps({"baseline_accuracy": baseline}))


def test_materialize_writes_the_taskset_the_server_serves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from environment_servers.harbor.tests.conftest import make_client
    from environment_servers.posttrainbench.app import (
        PostTrainBenchEnvironmentServer,
        PostTrainBenchEnvironmentServerConfig,
    )
    from nemo_gym.tasks.harbor import DIGEST_KEY

    tasks_dir = tmp_path / "tasks" / "posttrainbench"
    baselines = {"posttrainbench-gsm8k-qwen3-1.7b": 0.12, "posttrainbench-aime2025-qwen3-4b": 0.03}
    for name, baseline in baselines.items():
        write_ptb_task(tasks_dir, name, baseline)
    monkeypatch.setattr(prepare, "ROWS_ROOT", tmp_path / "rows")
    monkeypatch.setattr(prepare, "TASKSET_PATH", tmp_path / "taskset.json")

    rows_path = prepare.materialize(tasks_dir, builds={"enabled": False}, factors=FACTORS)

    taskset = json.loads((tmp_path / "taskset.json").read_text())
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    assert taskset["taskset"] == "posttrainbench" and taskset["folder"] == str(tasks_dir.resolve())
    assert taskset["baselines"] == baselines and taskset["factors"] == FACTORS
    digests = {row["task_id"]["task_id"]: row["task_input"]["task_data"][DIGEST_KEY] for row in rows}
    assert digests == taskset["tasks"]
    server = PostTrainBenchEnvironmentServer(
        config=PostTrainBenchEnvironmentServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="posttrainbench_environment_server",
            agent_server={"type": "responses_api_agents", "name": "agent"},
            taskset_file=tmp_path / "taskset.json",
            artifacts_dir=tmp_path / "artifacts",
        ),
        server_client=make_client(),
    )
    assert server.config.tasksets["posttrainbench"].tasks == taskset["tasks"]
    assert server._baselines == baselines


def test_materialize_refuses_a_task_that_cannot_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tasks_dir = tmp_path / "tasks" / "posttrainbench"
    write_ptb_task(tasks_dir, "posttrainbench-gsm8k-qwen3-1.7b", 0.12)
    # A Dockerfile that needs a build, with builds off and no recorded image: Gym skips the task.
    write_ptb_task(tasks_dir, "posttrainbench-aime2025-qwen3-4b", 0.03, dockerfile="FROM ubuntu:24.04\nCOPY x /x")
    monkeypatch.setattr(prepare, "ROWS_ROOT", tmp_path / "rows")
    monkeypatch.setattr(prepare, "TASKSET_PATH", tmp_path / "taskset.json")

    with pytest.raises(RuntimeError, match="posttrainbench-aime2025-qwen3-4b"):
        prepare.materialize(tasks_dir, builds={"enabled": False}, factors=FACTORS)
    assert not (tmp_path / "taskset.json").exists()


def test_unknown_selection_is_refused() -> None:
    with pytest.raises(ValueError, match="bfcl"):
        prepare._selection(["gsm8k", "bfcl"], prepare.BENCHMARKS, "benchmark")
    assert prepare._selection(["humaneval", "gsm8k"], prepare.BENCHMARKS, "benchmark") == ["gsm8k", "humaneval"]


@pytest.fixture
def gym_score(monkeypatch: pytest.MonkeyPatch):
    """gym_score.py with stand-ins for upstream's verdict readers, driven by files in the logs dir."""

    def read(logs: str, name: str):
        path = Path(logs) / name
        if not path.exists():
            return None
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError(f"{path}: top-level JSON is not an object")
        return value

    def load_judgement(logs: str) -> dict:
        verdict = read(logs, "judgement_gpt5_4.json")
        if verdict is None:
            raise FileNotFoundError(logs)
        return verdict

    def flag(name: str, field: str):
        return lambda logs: None if (verdict := read(logs, name)) is None else verdict[field]

    def judgement_to_cell(judgement: dict, api_usage: bool | None) -> str:
        return "M" * judgement["disallowed_model"] + "C" * judgement["contamination"] + "A" * bool(api_usage)

    fake = types.ModuleType("utils")
    fake.load_judgement = load_judgement
    fake.load_api_judgement = flag("judgement_api.json", "disallowed_api_usage")
    fake.load_ptb_lookup_judgement = flag("judgement_ptb_lookup.json", "disallowed_ptb_lookup")
    fake.judgement_to_cell = judgement_to_cell
    monkeypatch.setitem(sys.modules, "utils", fake)
    monkeypatch.delitem(sys.modules, "benchmarks.posttrainbench.gym_score", raising=False)
    return importlib.import_module("benchmarks.posttrainbench.gym_score")


def write_verdicts(logs: Path, *, contamination=False, api=False, lookup=False, accuracy: object = 0.6) -> None:
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "judgement_gpt5_4.json").write_text(
        json.dumps({"contamination": contamination, "disallowed_model": False})
    )
    (logs / "judgement_api.json").write_text(json.dumps({"disallowed_api_usage": api}))
    (logs / "judgement_ptb_lookup.json").write_text(json.dumps({"disallowed_ptb_lookup": lookup}))
    if accuracy is not None:
        (logs / "metrics.json").write_text(json.dumps({"accuracy": accuracy}))
    (logs / "solve_out.txt").write_text('{"type":"result"}\n')


@pytest.mark.parametrize(
    "verdicts, reward",
    [
        ({}, 0.6),
        ({"contamination": True}, 0.1),
        ({"api": True}, 0.1),
        ({"lookup": True}, 0.1),
        ({"accuracy": None}, 0.1),
        ({"accuracy": "0.6"}, 0.1),
    ],
)
def test_score_takes_the_baseline_where_collect_py_does(gym_score, tmp_path: Path, verdicts: dict, reward: float):
    write_verdicts(tmp_path, **verdicts)

    fields = gym_score.score(str(tmp_path), 0.1)

    assert fields["reward"] == reward
    assert fields["baseline_accuracy"] == 0.1
    assert fields["judgement_missing"] == fields["judgement_unreadable"] == 0.0


def test_an_unjudged_run_takes_the_baseline_and_says_so(gym_score, tmp_path: Path) -> None:
    write_verdicts(tmp_path)
    (tmp_path / "judgement_gpt5_4.json").unlink()

    fields = gym_score.score(str(tmp_path), 0.1)

    assert fields["reward"] == 0.1
    assert fields["accuracy"] == 0.6
    assert fields["judgement_missing"] == 1.0


def test_judges_without_the_agent_transcript_did_not_check_the_run(gym_score, tmp_path: Path) -> None:
    write_verdicts(tmp_path)
    (tmp_path / "solve_out.txt").unlink()

    fields = gym_score.score(str(tmp_path), 0.1)

    assert fields["trace_missing"] == fields["judgement_missing"] == 1.0
    assert fields["reward"] == 0.1


def test_an_unreadable_verdict_takes_the_baseline_and_says_so(gym_score, tmp_path: Path) -> None:
    write_verdicts(tmp_path)
    (tmp_path / "judgement_api.json").write_text("[]")

    fields = gym_score.score(str(tmp_path), 0.1)

    assert fields["reward"] == 0.1
    assert fields["judgement_unreadable"] == 1.0


def test_main_writes_reward_json(gym_score, tmp_path: Path) -> None:
    write_verdicts(tmp_path / "logs")
    (tmp_path / "metadata.json").write_text(json.dumps({"baseline_accuracy": 0.25}))

    gym_score.main(str(tmp_path / "logs"), str(tmp_path / "metadata.json"))

    assert json.loads((tmp_path / "logs/reward.json").read_text())["reward"] == 0.6
