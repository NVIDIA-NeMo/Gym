# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate PostTrainBench v1.2's Harbor tasks with the upstream adapter and adapt them to Gym's Harbor server.

The task folders come from ``src/harbor_adapter`` of the pinned PostTrainBench revision, unchanged except
for what Modal provided and Gym does not: the trained weights reach the separate verifier through
``/logs/artifacts`` instead of a Modal volume, the agent's Claude Code transcript is staged where the
judges read it, the judges and graders (and the agent's own grader, for the two benchmarks that give it an
OpenAI key, with a key of its own) call an OpenAI-compatible endpoint by its model names, and the verifier
writes the post-fallback score of ``scripts/collect.py`` as the reward. ``environment/`` changes only in the
grader model name of those two benchmarks' ``evaluate.py``.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any


PTB_REPO = "https://github.com/aisa-group/PostTrainBench.git"
PTB_REVISION = "ea8d65bc375e61c8ac77a7aea86effcc995378fb"
BENCHMARKS = ("aime2025", "arenahardwriting", "gpqamain", "gsm8k", "healthbench", "humaneval")
MODELS = ("qwen3-1.7b", "qwen3-4b", "smollm3-3b", "gemma3-4b")
TASKSET = "posttrainbench"

BENCHMARK_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCHMARK_DIR / "data"
SOURCE_DIR = DATA_DIR / "source"
TASKS_DIR = DATA_DIR / "tasks" / TASKSET
ROWS_ROOT = DATA_DIR / "rows"
ROWS_PATH = ROWS_ROOT / TASKSET / "tasks.jsonl"
TASKSET_PATH = DATA_DIR / "taskset.json"
# Images built once for these task folders (see README); copied next to the tasks so prepare reuses them.
# The builder's two records (see README): the built images, and the configurations with their inputs digests.
IMAGE_RECORDS = (BENCHMARK_DIR / "built-images.json", BENCHMARK_DIR / "compose-images.json")
GYM_SCORE_SCRIPT = BENCHMARK_DIR / "gym_score.py"

# Where ptb_collect.sh puts final_model; Gym carries /logs/artifacts into the separate verifier.
MODEL_HANDOFF_DIR = "/logs/artifacts/final_model"
# gym_score.py imports upstream's verdict readers from here in the verifier.
UPSTREAM_UTILS_DEST = "ptb/scripts/utils.py"
# The agent writes its transcript to /logs/agent/claude-code.txt (sandbox_transcript_path in config.yaml), where
# ptb_collect.sh picks it up for the judges.
COLLECT_COMMAND = (
    f"bash -c 'rm -rf {MODEL_HANDOFF_DIR}; PTB_VOLUME_DIR={MODEL_HANDOFF_DIR} bash /usr/local/bin/ptb_collect.sh'"
)
# A table header and its key lines, up to the next table or comment.
VERIFIER_ENVIRONMENT_TABLES = re.compile(r"^\[verifier\.environment(?:\.healthcheck)?\]\n(?:[^\[#\n].*\n|\n)*", re.M)
DEFAULT_JUDGE_MODELS = {"gpt-5.6-terra": "azure/openai/gpt-5.6-terra", "gpt-5-mini": "azure/openai/gpt-5-mini"}
# Files that name a judge or grader model, relative to the task folder; the first is in every task.
JUDGE_MODEL_FILES = (
    "tests/ptb/src/judges/judge_lib.sh",
    "tests/ptb/src/judges/general_judge/judge.conf",
    "tests/ptb/src/eval/tasks/arenahardwriting/evaluate_final_eval.py",
    "tests/ptb/src/eval/tasks/healthbench/evaluate_final_eval.py",
    # The agent's grader, run with the OpenAI key upstream gives the agent for these two benchmarks.
    "environment/evaluate.py",
)


def prepare(
    benchmarks: list[str] | None = None,
    models: list[str] | None = None,
    num_hours: int = 10,
    judge_base_url: str = "https://inference-api.nvidia.com/v1",
    judge_api_key_env: str = "INFERENCE_API_KEY",
    judge_models: dict[str, str] | None = None,
    agent_api_key_env: str | None = None,
    builds: dict[str, Any] | None = None,
) -> Path:
    """Write the adapted task folders, the rows file and the taskset file the environment server loads."""
    benchmarks = _selection(benchmarks, BENCHMARKS, "benchmark")
    models = _selection(models, MODELS, "model")
    source = checkout_source(SOURCE_DIR)
    download_test_data(source, benchmarks)
    generate_tasks(source, TASKS_DIR, benchmarks=benchmarks, models=models, num_hours=num_hours)
    for task_dir in sorted(TASKS_DIR.iterdir()):
        adapt_task(
            task_dir,
            source,
            judge_base_url=judge_base_url,
            judge_api_key_env=judge_api_key_env,
            judge_models={**DEFAULT_JUDGE_MODELS, **(judge_models or {})},
            agent_api_key_env=agent_api_key_env,
        )
    restore_image_records(TASKS_DIR)
    factors = json.loads((source / "scripts/factors.json").read_text())
    return materialize(TASKS_DIR, builds=builds, factors=factors)


def restore_image_records(tasks_dir: Path, records: tuple[Path, ...] = IMAGE_RECORDS) -> None:
    """Put the committed image records where the loader and the builder read them: beside the task folders.

    A built image is reused only with both records: built-images.json names it, and compose-images.json holds its
    configuration and the inputs digest it was built from. One without the other is refused.
    """
    present = [record for record in records if record.is_file()]
    if present and len(present) != len(records):
        raise FileNotFoundError(f"Commit {[r.name for r in records]} together; found only {[r.name for r in present]}")
    for record in present:
        shutil.copy2(record, tasks_dir / record.name)


def _selection(values: list[str] | None, known: tuple[str, ...], kind: str) -> list[str]:
    if values is None:
        return list(known)
    unknown = sorted(set(values) - set(known))
    if unknown or not values:
        raise ValueError(f"Unknown or empty PostTrainBench {kind} selection {values!r}; choose from {list(known)}")
    return [value for value in known if value in values]


def checkout_source(dest: Path, revision: str = PTB_REVISION) -> Path:
    """A checkout of PostTrainBench at ``revision``; an existing one at that revision is reused."""
    if (dest / ".git").is_dir() and _git(dest, "rev-parse", "HEAD") == revision:
        return dest
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    _git(dest, "init", "-q")
    _git(dest, "fetch", "-q", "--depth", "1", PTB_REPO, revision)
    _git(dest, "checkout", "-q", "FETCH_HEAD")
    return dest


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def download_test_data(source: Path, benchmarks: list[str]) -> None:
    """Upstream's download script; it reads HF_TOKEN (read-only) for the gated GPQA set."""
    missing = [b for b in benchmarks if not (source / "src/eval/tasks" / b / "test_data.json").is_file()]
    if not missing:
        return
    script = source / "src/judges/test_data_download/download_test_data.py"
    subprocess.run([sys.executable, str(script), "--tasks", *missing], cwd=source, check=True)


def generate_tasks(source: Path, dest: Path, *, benchmarks: list[str], models: list[str], num_hours: int) -> None:
    """Run upstream's adapter into a fresh ``dest``, one task per benchmark and model."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    adapter_dir = source / "src/harbor_adapter"
    for benchmark in benchmarks:
        for model in models:
            subprocess.run(
                [sys.executable, "run_adapter.py", "--benchmark", benchmark, "--model", model]
                + ["--output", str(dest), "--num-hours", str(num_hours)],
                cwd=adapter_dir,
                check=True,
                stdout=subprocess.DEVNULL,
            )


def adapt_task(
    task_dir: Path,
    source: Path,
    *,
    judge_base_url: str,
    judge_api_key_env: str,
    judge_models: dict[str, str],
    agent_api_key_env: str | None = None,
) -> None:
    """Apply the Gym deviations to one generated task folder (see the module docstring)."""
    toml_path = task_dir / "task.toml"
    toml_path.write_text(
        adapt_task_toml(
            toml_path.read_text(),
            judge_base_url=judge_base_url,
            judge_api_key_env=judge_api_key_env,
            agent_api_key_env=agent_api_key_env,
        )
    )
    if not (task_dir / JUDGE_MODEL_FILES[0]).is_file():
        raise FileNotFoundError(f"{task_dir.name} has no {JUDGE_MODEL_FILES[0]}; the adapter output changed")
    for relative in JUDGE_MODEL_FILES:
        path = task_dir / relative
        if path.is_file():
            path.write_text(rename_judge_models(path.read_text(), judge_models))
    tests = task_dir / "tests"
    utils_dest = tests / UPSTREAM_UTILS_DEST
    utils_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "scripts/utils.py", utils_dest)
    shutil.copy2(GYM_SCORE_SCRIPT, tests / GYM_SCORE_SCRIPT.name)
    test_sh = tests / "test.sh"
    test_sh.write_text(append_gym_score(test_sh.read_text()))


def _replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one {old!r} in the upstream task; the adapter output changed")
    return text.replace(old, new)


def adapt_task_toml(
    text: str, *, judge_base_url: str, judge_api_key_env: str, agent_api_key_env: str | None = None
) -> str:
    """The upstream task.toml with the model hand-off, the agent env and the judge endpoint for Gym.

    arenahardwriting and healthbench also give the agent an OpenAI key for its own grader. The agent can read and
    use it, so it never gets the judges' key: it gets agent_api_key_env with the same endpoint, or no key.
    """
    openai_key = 'OPENAI_API_KEY = "${OPENAI_API_KEY}"'
    endpoint = f"OPENAI_BASE_URL = {json.dumps(judge_base_url)}"
    head, header, agent_env = text.partition("\n[environment.env]\n")
    if not header or "\n[" in agent_env or agent_env.count(openai_key) > 1:
        raise ValueError(
            "Expected the agent's [environment.env] last in the upstream task; the adapter output changed"
        )
    if agent_api_key_env:
        agent_env = agent_env.replace(openai_key, f'OPENAI_API_KEY = "${{{agent_api_key_env}}}"\n{endpoint}')
    else:
        agent_env = agent_env.replace(openai_key, "")
    key = f'"${{{judge_api_key_env}}}"'
    head = _replace_once(head, 'PTB_MODEL_DIR = "/mnt/ptb_final_model"', f'PTB_MODEL_DIR = "{MODEL_HANDOFF_DIR}"')
    head = _replace_once(head, 'CODEX_API_KEY = "${OPENAI_API_KEY}"', f"CODEX_API_KEY = {key}")
    head = _replace_once(head, openai_key, f"OPENAI_API_KEY = {key}\n{endpoint}")
    text = head + header + agent_env
    text = _replace_once(
        text, 'command = "bash /usr/local/bin/ptb_collect.sh"', f"command = {json.dumps(COLLECT_COMMAND)}"
    )
    # Upstream's launcher passes this one to claude-code with `--ae`; Gym reads agent env from the task only.
    text = _replace_once(
        text,
        '[environment.env]\nHF_TOKEN = "${HF_TOKEN}"',
        '[environment.env]\nHF_TOKEN = "${HF_TOKEN}"\nBASH_MAX_TIMEOUT_MS = "36000000"',
    )
    # Without the table the verifier image is still built from tests/Dockerfile, with the task's resources, which
    # upstream's table only repeats (one H100, pinned to the H100 type).
    text, dropped = VERIFIER_ENVIRONMENT_TABLES.subn("", text)
    if dropped != 2:
        raise ValueError("Expected [verifier.environment] and its healthcheck in the upstream task")
    parsed = tomllib.loads(text)
    if "environment" in parsed["verifier"] or parsed["environment"].get("gpus") != 1:
        raise ValueError("The adapted task must verify in the task image on the task's one GPU")
    return text


def rename_judge_models(text: str, judge_models: dict[str, str]) -> str:
    """Replace each quoted upstream model name with the endpoint's name for it; comments stay as they are."""
    for upstream, endpoint in judge_models.items():
        text = re.sub(rf'(["\']){re.escape(upstream)}\1', rf"\g<1>{endpoint}\g<1>", text)
    return text


GYM_PREAMBLE = """# Gym (benchmarks/posttrainbench/prepare.py): the judges run as nobody, so the verifier's code is owned by this
# user and read-only for others before any judge starts (Gym uploads /tests world-writable).
GYM_TESTS_DIR="${TESTS:-/tests}"
if ! { chown -R "$(id -u):$(id -g)" "$GYM_TESTS_DIR" && chmod -R go-w "$GYM_TESTS_DIR"; }; then
    echo "could not make $GYM_TESTS_DIR read-only for the judges"
    exit 1
fi
# A model tree with links or special files could point the verifier's logs at its own secrets
# (config.json -> /proc/self/environ), so it counts as not submitted.
GYM_MODEL_DIR="${PTB_MODEL_DIR:-/logs/artifacts/final_model}"
if [ -e "$GYM_MODEL_DIR" ] || [ -L "$GYM_MODEL_DIR" ]; then
    if [ -n "$(find "$GYM_MODEL_DIR" ! -type f ! -type d -print -quit)" ]; then
        echo "final_model holds links or special files; treating it as not submitted"
        rm -rf "$GYM_MODEL_DIR"
    fi
fi
# gym_score.py writes reward.json (Gym reads it before reward.txt) whenever test.sh ends, also on upstream's
# early exits when there is no model. -I: nothing the judges left beside it is imported.
trap 'python3 -I "$GYM_TESTS_DIR/gym_score.py" "${LOGS_DIR:-/logs/verifier}" "$GYM_TESTS_DIR/metadata.json"' EXIT
"""


def append_gym_score(test_sh: str) -> str:
    """Upstream's verifier with Gym's checks and its scorer (an EXIT trap) placed before upstream's own steps."""
    if "gym_score.py" in test_sh:
        return test_sh
    shebang, _, body = test_sh.partition("\n")
    if not shebang.startswith("#!"):
        raise ValueError("Expected a shebang on the first line of the upstream test.sh")
    return f"{shebang}\n{GYM_PREAMBLE}{body}"


def materialize(tasks_dir: Path, *, builds: dict[str, Any] | None, factors: dict[str, float]) -> Path:
    """Load the tasks with Gym's Harbor loader, write the rows and the taskset file, and return the rows path."""
    from nemo_gym.tasks.harbor.build_options import BuildOptions
    from nemo_gym.tasks.harbor.cli import prepare_target
    from nemo_gym.tasks.harbor.materialize import taskset_mapping

    prepared = prepare_target(
        str(tasks_dir), output_root=ROWS_ROOT, build=BuildOptions.from_config({"builds": builds or {}})
    )
    if prepared.skipped:
        raise RuntimeError(
            f"{len(prepared.skipped)} PostTrainBench task(s) cannot run, and the score needs every task: "
            + "; ".join(f"{task}: {reason}" for task, reason in sorted(prepared.skipped.items()))
        )
    baselines = {
        task.task_id: json.loads((task.path / "tests/metadata.json").read_text())["baseline_accuracy"]
        for task in prepared.tasks
    }
    TASKSET_PATH.write_text(
        json.dumps(
            {
                "taskset": prepared.taskset,
                "revision": PTB_REVISION,
                **taskset_mapping(prepared.tasks, prepared.folder),
                "baselines": baselines,
                "factors": factors,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return prepared.rows_path


if __name__ == "__main__":
    prepare(**json.loads(os.environ.get("PTB_PREPARE_ARGS", "{}")))
