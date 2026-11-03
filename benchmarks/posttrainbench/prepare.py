# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate PostTrainBench v1.2's Harbor tasks with the upstream adapter and adapt them to Gym's Harbor server.

The task folders come from ``src/harbor_adapter`` of the pinned PostTrainBench revision, unchanged except
for what Modal provided and Gym does not: the trained weights reach the separate verifier through
``/logs/artifacts`` instead of a Modal volume, the agent's Claude Code transcript is staged where the
judges read it, the judges and graders (and the agent's own grader, for the two benchmarks that give it an
OpenAI key, with a key of its own) call an OpenAI-compatible endpoint by its model names (the judges' Codex
config names it as ``openai_base_url``), and the verifier
writes the post-fallback score of ``scripts/collect.py`` as the reward. ``environment/`` changes only in the
grader model name of those two benchmarks' ``evaluate.py``, and its Dockerfile starts from one prebuilt image
for all tasks (see README, Images).
"""

import base64
import hashlib
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
# The prebuilt task and verifier images (see README, Images): each one's reference and the digest of the task
# files it holds (per benchmark for the task image).
IMAGES_FILE = BENCHMARK_DIR / "images.json"
TASK_DOCKERFILE = BENCHMARK_DIR / "image" / "task.Dockerfile"
VERIFIER_DOCKERFILE = BENCHMARK_DIR / "image" / "verifier.Dockerfile"
# The images' recorded configurations, copied next to the tasks: the loader reads them, and the registry is private.
IMAGE_CONFIGS_FILE = BENCHMARK_DIR / "compose-images.json"
# Where the image holds each benchmark's files; a task's overlay copies its own into place and deletes the rest.
IMAGE_TASK_FILES = "/opt/ptb"
# The environment/ files that differ per task, which the overlay writes, and the files upstream's Dockerfile copies
# out of the workspace to fixed paths (the same for every task).
PER_TASK_FILES = ("metadata.json", "timer.sh")
SHARED_FILES = (
    "entrypoint.sh",
    "system_monitor.sh",
    "ptb_collect.sh",
    "requirements-direct.txt",
    "contamination_check.py",
)
# The tests/ files the verifier image is built from; Gym uploads the rest of tests/ before test.sh runs.
VERIFIER_FILES = ("Dockerfile", "entrypoint.sh", "system_monitor.sh", "requirements-direct.txt")
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
    images = json.loads(IMAGES_FILE.read_text())
    for task_dir in sorted(TASKS_DIR.iterdir()):
        use_images(task_dir, images)
    shutil.copy2(IMAGE_CONFIGS_FILE, TASKS_DIR / IMAGE_CONFIGS_FILE.name)
    factors = json.loads((source / "scripts/factors.json").read_text())
    return materialize(TASKS_DIR, builds=builds, factors=factors)


def image_files(env: Path) -> list[Path]:
    """The files of a task's ``environment/`` that the task image holds: all but the Dockerfile, the per-task files
    and hidden files (upstream's .dockerignore, which only keeps the Dockerfile out of the workspace)."""
    skipped = {"Dockerfile", *PER_TASK_FILES}
    return [
        path
        for path in sorted(env.rglob("*"))
        if path.is_file()
        and path.relative_to(env).as_posix() not in skipped
        and not any(part.startswith(".") for part in path.relative_to(env).parts)
    ]


def files_digest(root: Path, paths: list[Path]) -> str:
    """sha256 over the path below ``root`` and the content digest of each of ``paths``, in their order."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def image_files_digest(env: Path) -> str:
    return files_digest(env, image_files(env))


def verifier_files_digest(tests: Path) -> str:
    return files_digest(tests, [tests / name for name in VERIFIER_FILES])


def task_benchmark(task_dir: Path) -> str:
    return json.loads((task_dir / "environment/metadata.json").read_text())["benchmark_id"]


def _check_image(task_dir: Path, folder: str, image: str, expected: str | None, actual: str) -> None:
    if actual != expected:
        raise ValueError(
            f"{task_dir.name}: {folder} files have sha256 {actual}, but {image} holds {expected}; "
            "rebuild the images (README, Images)"
        )


def use_images(task_dir: Path, images: dict[str, Any]) -> None:
    """Point the task at the prebuilt task and verifier images, which must hold this task's files.

    Upstream builds one task image and one verifier image per task, from ``environment/`` and ``tests/``. The task
    images differ only in ``metadata.json`` and ``timer.sh``, and the verifier images only in the ``tests/`` they
    copy, which Gym uploads before ``test.sh`` runs. So the task's Dockerfile becomes an overlay on the task image
    that does what upstream's last steps do: it puts the benchmark's files into the workspace and ``test_data.json``
    beside it, writes the two per-task files, and removes the other benchmarks' files. The separate verifier gets
    a ``[verifier.environment]`` with the task's resources and the verifier image.
    """
    env = task_dir / "environment"
    benchmark = task_benchmark(task_dir)
    task, verifier = images["task"], images["verifier"]
    _check_image(task_dir, "environment/", task["image"], task["files_sha256"].get(benchmark), image_files_digest(env))
    tests_digest = verifier_files_digest(task_dir / "tests")
    _check_image(task_dir, "tests/", verifier["image"], verifier["files_sha256"], tests_digest)
    source = f"{IMAGE_TASK_FILES}/benchmarks/{benchmark}"
    steps = [
        f"cp -a {source}/workspace/. /home/agent/workspace/",
        f"cp {source}/test_data.json /home/agent/test_data.json",
        f"rm -rf {IMAGE_TASK_FILES}",
    ]
    for name in PER_TASK_FILES:
        encoded = base64.b64encode((env / name).read_bytes()).decode()
        steps.append(f"printf %s {encoded} | base64 -d > /home/agent/workspace/{name}")
    steps += ["chmod -R a+rw /home/agent/workspace/", "chmod +x /home/agent/workspace/timer.sh"]
    (env / "Dockerfile").write_text(
        "# benchmarks/posttrainbench/prepare.py: upstream's task image, prebuilt for all tasks (README, Images).\n"
        f"FROM {task['image']}\nRUN " + " \\\n    && ".join(steps) + "\n"
    )
    toml_path = task_dir / "task.toml"
    text = toml_path.read_text()
    resources = tomllib.loads(text)["environment"]
    settings = "".join(
        f"{key} = {json.dumps(value)}\n" for key, value in resources.items() if not isinstance(value, dict)
    )
    toml_path.write_text(
        f"{text.rstrip()}\n\n# benchmarks/posttrainbench/prepare.py: upstream's verifier image, prebuilt for all tasks.\n"
        f"[verifier.environment]\n{settings}docker_image = {json.dumps(verifier['image'])}\n"
    )


def write_image_context(tasks_dir: Path, dest: Path) -> dict[str, Any]:
    """Write the build contexts of both images from generated tasks, and return the ``files_sha256`` of each.

    ``task/`` is ``image/task.Dockerfile``, the shared files, and ``benchmarks/<benchmark>/`` with the workspace
    files and ``test_data.json`` of each benchmark. ``verifier/`` is ``image/verifier.Dockerfile`` and the files it
    copies from ``tests/``. Every task of a benchmark must have the same files, and every task the same verifier files.
    """
    if dest.exists():
        shutil.rmtree(dest)
    task_context, verifier_context = dest / "task", dest / "verifier"
    task_context.mkdir(parents=True)
    verifier_context.mkdir()
    shutil.copy2(TASK_DOCKERFILE, task_context / "Dockerfile")
    shutil.copy2(VERIFIER_DOCKERFILE, verifier_context / "Dockerfile")
    digests: dict[str, str] = {}
    verifier_digest = None
    for task_dir in sorted(tasks_dir.iterdir()):
        if not (task_dir / "environment").is_dir():
            continue
        env, tests = task_dir / "environment", task_dir / "tests"
        benchmark = task_benchmark(task_dir)
        digest = image_files_digest(env)
        if digests.setdefault(benchmark, digest) != digest:
            raise ValueError(f"{task_dir.name}: environment/ files differ from another {benchmark} task")
        tests_digest = verifier_files_digest(tests)
        verifier_digest = verifier_digest or tests_digest
        if tests_digest != verifier_digest:
            raise ValueError(f"{task_dir.name}: tests/ verifier files differ from another task's")
        for path in image_files(env):
            relative = path.relative_to(env).as_posix()
            if relative in SHARED_FILES:
                target = task_context / relative
                if target.exists() and target.read_bytes() != path.read_bytes():
                    raise ValueError(f"{task_dir.name}: {relative} differs from another benchmark's")
            elif relative == "test_data.json":
                target = task_context / "benchmarks" / benchmark / relative
            else:
                target = task_context / "benchmarks" / benchmark / "workspace" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        for name in VERIFIER_FILES:
            if name != "Dockerfile":
                shutil.copy2(tests / name, verifier_context / name)
    return {"task": digests, "verifier": verifier_digest}


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
    judge_lib = task_dir / JUDGE_MODEL_FILES[0]
    judge_lib.write_text(point_judges_at(judge_lib.read_text(), judge_base_url))
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
    # Upstream's table gives the verifier one H100. use_images declares the verifier's environment instead: the
    # prebuilt verifier image with all the task's resources (one H100 too).
    text, dropped = VERIFIER_ENVIRONMENT_TABLES.subn("", text)
    if dropped != 2:
        raise ValueError("Expected [verifier.environment] and its healthcheck in the upstream task")
    parsed = tomllib.loads(text)
    if "environment" in parsed["verifier"] or parsed["environment"].get("gpus") != 1:
        raise ValueError("The adapted task must leave the verifier's environment to use_images, on one GPU")
    return text


def rename_judge_models(text: str, judge_models: dict[str, str]) -> str:
    """Replace each quoted upstream model name with the endpoint's name for it; comments stay as they are."""
    for upstream, endpoint in judge_models.items():
        text = re.sub(rf'(["\']){re.escape(upstream)}\1', rf"\g<1>{endpoint}\g<1>", text)
    return text


# The Codex config judge_lib.sh writes for JUDGE_RUNTIME=local (the Harbor verifier), as a printf format.
CODEX_LOCAL_CONFIG = "printf '[projects."


def point_judges_at(judge_lib: str, judge_base_url: str) -> str:
    """judge_lib.sh with the judges' Codex config naming ``judge_base_url`` as its ``openai_base_url``.

    The pinned Codex CLI (0.144.5) ignores OPENAI_BASE_URL, so without it every judge calls api.openai.com with
    the judges' key, gets 401 and writes no verdict.
    """
    if re.search(r"""['"%\\\s]""", judge_base_url):
        raise ValueError(f"judge_base_url {judge_base_url!r} cannot go into judge_lib.sh's printf format")
    return _replace_once(judge_lib, CODEX_LOCAL_CONFIG, f'printf \'openai_base_url = "{judge_base_url}"\\n[projects.')


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
