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
"""Prepare the IOI'26 benchmark for CCC-backed evaluation.

Everything comes from the official task archive, `github.com/ioi/task-archive`,
which publishes each year's statements, graders, subtask definitions and the
full private test data. Running this script is the only step:

    python benchmarks/ioi26/prepare.py

It emits CCC-shaped artifacts:

- ``benchmarks/ioi26/data/ioi26_benchmark.jsonl``: one row per
  ``(problem, subtask)`` with the top-level fields CCC expects --
  ``competition_id``, ``problem_id`` (the task's short code, which is both the
  metadata key AND the grader filename ``graders/{problem_id}.cpp`` that IOI's
  ``compile`` script expects), ``subtask``, ``subtask_score``, ``name``,
  ``question``.

- ``benchmarks/ioi26/data/ioi26_metadata.json``: single-line JSONL with CCC's
  competition wrapper ``{"competition_id": "ioi26", "metadata": {"partition":
  {...}, ...}}``. Each problem value is a subtask-keyed dict carrying per-subtask
  ``tests`` / ``subtask_score`` / ``score_precision`` / ``run`` / ``compile`` /
  ``grader_files``. CCC's ``_normalize_problem_metadata`` converts this legacy
  shape into its normalized ``subtasks`` / ``all_tests`` structure at load time.
"""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import requests
from subtask_split import split as split_subtasks


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
BENCHMARK_FPATH = DATA_DIR / "ioi26_benchmark.jsonl"
METADATA_FPATH = DATA_DIR / "ioi26_metadata.json"
# NeMo Skills consumes a dataset directory, not Gym's two loose files.
NS_DIR = DATA_DIR / "nemo_skills" / "ioi26"
ARCHIVE_DIR = DATA_DIR / "_task_archive"

ARCHIVE_REPO = "https://github.com/ioi/task-archive.git"
ARCHIVE_YEAR = "2026"

# The archive ships no compile/run harness, so we reuse the IOI-generic pair the
# HuggingFace IOI project publishes. They handle every shape present in 2026: a
# checker, a manager, a stub, and a plain grader.
RUN_URL = "https://raw.githubusercontent.com/huggingface/ioi/refs/heads/main/run_tests/custom_setup/run"
COMPILE_URL = "https://raw.githubusercontent.com/huggingface/ioi/refs/heads/main/run_tests/custom_setup/compile"

# IOI'26 graded with C++20; those scripts still say gnu++17, which would reject
# valid submissions. Substituted rather than vendored so upstream fixes survive.
CXX_STANDARD_FROM = "gnu++17"
CXX_STANDARD_TO = "gnu++20"

COMPETITION_ID = "ioi26"

#: NeMo Skills discovers a benchmark through this module. `ccc` is the evaluator
#: that reads the normalized metadata written beside it.
NS_INIT = """GENERATION_ARGS = "++prompt_config=generic/default ++eval_type=ccc"
METRICS_TYPE = "ccc"
DATASET_GROUP = "code"
EVAL_ARGS = "++eval_type=ccc"

SANDBOX_ENV_VARS = [
    "UWSGI_PROCESSES=1024",
    "UWSGI_CPU_AFFINITY=8",
    "UWSGI_CHEAPER=1023",
    "NUM_WORKERS=1",
    "STATEFUL_SANDBOX=0",
]
"""

# Only these paths are checked out. `solutions/` and `translations/` are most of
# the year's bulk and neither is used for evaluation.
SPARSE_PATHS = [
    f"/{ARCHIVE_YEAR}/*/*/problem.json",
    f"/{ARCHIVE_YEAR}/*/*/en.pdf",
    f"/{ARCHIVE_YEAR}/*/*/subtasks/**",
    f"/{ARCHIVE_YEAR}/*/*/tests/**",
    f"/{ARCHIVE_YEAR}/*/*/graders/**",
    f"/{ARCHIVE_YEAR}/*/*/checker/**",
]


def _run(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True, stdout=subprocess.DEVNULL)


def fetch_archive(force: bool = False) -> Path:
    """Check out just the parts of the year we need, ~390 MB of it test data.

    A blobless clone means git fetches only the blobs the sparse checkout names,
    so the other years and this year's translations never cross the network.
    """
    year_dir = ARCHIVE_DIR / ARCHIVE_YEAR
    if year_dir.is_dir() and not force:
        print(f"Reusing archive checkout at {ARCHIVE_DIR}")
        return year_dir

    if shutil.which("git") is None:
        raise SystemExit("git is required to fetch the IOI task archive")

    if ARCHIVE_DIR.exists():
        shutil.rmtree(ARCHIVE_DIR)
    print(f"Cloning {ARCHIVE_REPO} ({ARCHIVE_YEAR} only; this pulls a few hundred MB) ...")
    _run("git", "clone", "--filter=blob:none", "--no-checkout", "--depth", "1", ARCHIVE_REPO, str(ARCHIVE_DIR))
    _run("git", "sparse-checkout", "init", "--no-cone", cwd=ARCHIVE_DIR)
    _run("git", "sparse-checkout", "set", *SPARSE_PATHS, cwd=ARCHIVE_DIR)
    _run("git", "checkout", cwd=ARCHIVE_DIR)

    if not year_dir.is_dir():
        raise SystemExit(f"{ARCHIVE_REPO} has no {ARCHIVE_YEAR}/ directory after checkout")
    return year_dir


def statement_markdown(pdf_path: Path) -> str:
    """Recover the markdown the setters attached inside the statement PDF.

    Not a conversion: the bytes are the setter's own file, and the PDF records
    its size and MD5 so both are checked.
    """
    try:
        import pikepdf
    except ModuleNotFoundError as exc:
        if exc.name != "pikepdf":
            raise
        raise SystemExit("pikepdf is required to extract IOI statements: pip install pikepdf") from exc

    # Bind the Pdf. It owns the stream, and reading an attachment off a
    # temporary has been observed to segfault or silently yield zero bytes.
    pdf = pikepdf.open(pdf_path)
    try:
        if "markdown" not in pdf.attachments:
            raise SystemExit(f"{pdf_path}: no 'markdown' attachment; found {list(pdf.attachments) or 'none'}")
        spec = pdf.attachments["markdown"]
        data = spec.get_file().read_bytes()
        params = spec.obj.get("/EF").get("/F").get("/Params")
        if params is not None:
            size = params.get("/Size")
            if size is not None and int(size) != len(data):
                raise SystemExit(f"{pdf_path}: got {len(data)} bytes, PDF declares {int(size)}")
            checksum = params.get("/CheckSum")
            if checksum is not None:
                want, got = bytes(checksum).hex(), hashlib.md5(data).hexdigest()
                if want != got:
                    raise SystemExit(f"{pdf_path}: checksum {got} != declared {want}")
        return data.decode("utf-8")
    finally:
        pdf.close()


#: The run harness only branches on Batch and Communication; anything else exits
#: "Unsupported task type". BatchAndOutput is graded exactly like Batch here
#: because none of 2026's subtasks are output-only -- every one is scored by
#: running the submitted program.
RUN_TASK_TYPE = {"BatchAndOutput": "Batch"}


def grader_config(problem: dict) -> str:
    """The `graders/grader_config.json` the run harness reads before every test.

    Without it the harness aborts with "grader_config.json not found" and every
    test scores zero, however good the submission -- compilation succeeds and the
    program is simply never executed. Field names are what the script greps for.
    """
    task_type = problem.get("task_type", "Batch")
    config = {
        "task_type": RUN_TASK_TYPE.get(task_type, task_type),
        "code": problem["code"],
        "time_limit": problem.get("time_limit", 1.0),
        "memory_limit": problem.get("memory_limit", 2147483648),
    }
    params = problem.get("task_type_params") or []
    if config["task_type"] == "Communication" and params:
        # [num_processes, "stub", user_io]
        config["task_type_parameters_Communication_num_processes"] = int(params[0])
        if len(params) > 2:
            config["task_type_parameters_Communication_user_io"] = str(params[2])
    return json.dumps(config, indent=2)


def read_support_files(task_dir: Path) -> list[list[str]]:
    """`graders/` and `checker/` as CCC's [relpath, content] pairs.

    Skips dotfiles and anything that is not UTF-8 text: the archive carries a
    stray `.DS_Store` and, for some tasks, a compiled `manager` binary beside
    its source.
    """
    files: list[list[str]] = []
    for folder in ("graders", "checker"):
        directory = task_dir / folder
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.name.startswith("."):
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue  # compiled artifact checked into the archive
            files.append([f"{folder}/{path.name}", content])
    return files


def prepare() -> Path:
    """Build the IOI'26 benchmark + metadata from the official task archive.

    Returns the benchmark JSONL path (the framework's expected primary
    artifact). The metadata JSONL is written alongside and referenced by
    config.yaml's ``test_file``.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    year_dir = fetch_archive()

    print("Downloading the IOI compile/run harness ...")
    run_code = requests.get(RUN_URL, timeout=60).text
    compile_code = requests.get(COMPILE_URL, timeout=60).text
    if CXX_STANDARD_FROM not in compile_code:
        print(
            f"  NOTE: upstream compile script no longer mentions {CXX_STANDARD_FROM}; "
            f"check it still targets {CXX_STANDARD_TO}"
        )
    compile_code = compile_code.replace(CXX_STANDARD_FROM, CXX_STANDARD_TO)
    run_code = run_code.replace(CXX_STANDARD_FROM, CXX_STANDARD_TO)

    task_dirs = sorted(
        p
        for day in sorted(year_dir.iterdir())
        if day.is_dir()
        for p in sorted(day.iterdir())
        if (p / "problem.json").is_file()
    )
    if not task_dirs:
        raise SystemExit(f"no tasks with a problem.json under {year_dir}")

    benchmark_rows: list[dict] = []
    metadata: dict[str, dict] = {}

    for task_dir in task_dirs:
        problem = json.loads((task_dir / "problem.json").read_text(encoding="utf-8"))
        problem_id = problem["code"]
        statement = statement_markdown(task_dir / "en.pdf")
        grader_files = read_support_files(task_dir)
        grader_files.append(["graders/grader_config.json", grader_config(problem)])
        tests_dir = task_dir / "tests"

        archive = [
            (p.stem, json.loads(p.read_text(encoding="utf-8"))) for p in sorted((task_dir / "subtasks").glob("*.json"))
        ]
        scored = [(name, d) for name, d in archive if d.get("score")]

        # Normalized CCC shape: one `all_tests` pool plus per-subtask test
        # name lists. Required by NeMo Skills' evaluator, accepted as-is by
        # Gym's, and it stops each shared test being embedded once per subtask.
        all_tests: dict[str, dict] = {}
        subtasks: dict[str, dict] = {}
        for name, entry in scored:
            for test_name in entry["testcases"]:
                if test_name in all_tests:
                    continue
                in_path = tests_dir / f"{test_name}.in"
                out_path = tests_dir / f"{test_name}.out"
                if not in_path.is_file() or not out_path.is_file():
                    raise SystemExit(f"{problem_id}/{name}: missing test {test_name!r}")
                all_tests[test_name] = {
                    "input": in_path.read_text(encoding="utf-8"),
                    "output": out_path.read_text(encoding="utf-8"),
                    "group": "sample" if test_name.startswith(("0-", "sample")) else "secret",
                }
            names = list(entry["testcases"])
            subtasks[name] = {
                "aggregation": "min",
                "score": float(entry["score"]),
                "score_precision": int(problem.get("score_precision", 2)),
                "test_names": names,
                "sample_test_names": [n for n in names if all_tests[n]["group"] == "sample"],
                "secret_test_names": [n for n in names if all_tests[n]["group"] == "secret"],
            }

        metadata[problem_id] = {
            "name": problem["name"],
            "problem_header_include": f"{problem_id}.h",
            "task_type": problem.get("task_type", "Batch"),
            "compile": compile_code,
            "run": run_code,
            "grader_files": grader_files,
            "subtasks": subtasks,
            "all_tests": all_tests,
        }

        # One prompt per scoring row. For five tasks that is one per subtask;
        # for magiccity the statement itself bands 50 subtasks into 8 rows and
        # we follow it, so the budget tracks the 100 points rather than the
        # subtask count. `subtask` is a prompt label only -- NeMo Skills' CCC
        # evaluator always runs every test and scores every subtask.
        variants = split_subtasks(statement)
        if not variants:
            raise SystemExit(f"{problem_id}: no scoring table found in the statement")
        for index, variant in enumerate(variants, start=1):
            benchmark_rows.append(
                {
                    "id": len(benchmark_rows) + 1,
                    "competition_id": COMPETITION_ID,
                    "problem_id": problem_id,
                    "ioi_id": problem_id,
                    "name": problem["name"],
                    "subtask": scored[index - 1][0] if not variant.band else f"band-{variant.label.replace(' ', '')}",
                    "subtask_score": variant.score,
                    "problem": variant.statement,
                    "question": variant.statement,
                }
            )

        print(
            f"  {problem_id:<12} {problem['task_type']:<15} "
            f"{len(scored):>2} subtasks -> {len(variants):>2} prompts, "
            f"{sum(s['score'] for s in subtasks.values()):g} points, "
            f"{len(all_tests):>4} tests"
        )

    with open(BENCHMARK_FPATH, "w") as f:
        f.write("\n".join(json.dumps(r) for r in benchmark_rows))
    print(f"Wrote {len(benchmark_rows)} rows to {BENCHMARK_FPATH}")

    wrapped = {"competition_id": COMPETITION_ID, "metadata": dict(metadata)}
    with open(METADATA_FPATH, "w") as f:
        json.dump(wrapped, f)
    grand_total = sum(st["score"] for problem in metadata.values() for st in problem["subtasks"].values())
    print(
        f"Wrote CCC-wrapped metadata for {len(metadata)} problems to {METADATA_FPATH} "
        f"(keys: {sorted(metadata)}; {grand_total:g} points total)"
    )

    NS_DIR.mkdir(parents=True, exist_ok=True)
    (NS_DIR / "test.jsonl").write_text("\n".join(json.dumps(r) for r in benchmark_rows) + "\n", encoding="utf-8")
    with open(NS_DIR / "test_metadata.json", "w") as f:
        json.dump(metadata, f)  # bare problem_id -> metadata, as CCC wants
    (NS_DIR / "__init__.py").write_text(NS_INIT, encoding="utf-8")
    print(f"Wrote the NeMo Skills dataset to {NS_DIR}")

    return BENCHMARK_FPATH


if __name__ == "__main__":
    prepare()
