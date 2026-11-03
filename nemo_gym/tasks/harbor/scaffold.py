# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``gym dataset init``: a dataset folder in exactly the shape the loader accepts.

One task, ready to validate and run. Everything written here is read back by the loader in a
test, so the scaffold cannot drift from what ``gym eval run`` and ``gym dataset validate`` expect.
"""

import re
from pathlib import Path


DEFAULT_IMAGE = "python:3.12-slim"
# The org `gym dataset init <name>` supplies when the name has none: Harbor 0.22.0 requires `org/name`.
DEFAULT_ORG = "local"
# One segment of Harbor's `org/name` rule (its ORG_NAME_PATTERN): alphanumerics, hyphens, underscores and
# dots, not starting with a dot. Harbor also refuses a name containing `..`.
_NAME_SEGMENT = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")

DATASET_TOML = """# Harbor's dataset file. The [dataset] table is Harbor's; the [gym] table is Gym's and holds
# what the dataset needs on top of its tasks, as data. Deployment settings (sandbox provider,
# registry credentials, image mirrors) never go here: a dataset runs unchanged everywhere.
#
# Round-trip caveat: Harbor's tooling (`harbor add`, `harbor remove`, `harbor sync`) rewrites this file
# from its manifest model, which keeps only Harbor's own tables. A Harbor round trip therefore drops
# the [gym] table and resets the multipliers and overrides below. Keep the dataset in git and
# restore the table from there after one.
[dataset]
name = "{org}/{name}"
version = "0.1.0"
description = "One-line description of the dataset."

[gym]
# Which split the rows enter (train, validation, example or benchmark) and the dataset's license,
# one of Gym's dataset license values. A license is required for train and validation; with
# neither key the dataset evaluates as a benchmark and never enters a training split.
# type = "validation"
# license = "Apache 2.0"
# The harnesses admitted to run the dataset (responses_api_agents/<name>); unset admits any. Catalog
# metadata: one of Gym's domain values, and whether the rewards were baselined and reviewed.
# allowed_agents = ["hermes_agent"]
# domain = "coding"
# verified = false
# A committed entry under environments/harbor/ names the hub dataset `gym dataset pull <entry>` fetches
# into its folder; an authored dataset, whose task folders are its own, leaves this unset.
# source = "harbor:<org>/<name>@<tag>"

[gym.defaults]
# Rollouts per task for an evaluation; `+num_repeats=` on the command line replaces it.
num_repeats = 1
# Scale every task's [agent] and [verifier] timeout_sec, and its cpus, memory_mb and storage_mb.
timeout_multiplier = 1.0
resource_multiplier = 1.0

# Per-task overrides use Harbor's own [environment] fields; `env` merges over the task's own.
# [gym.tasks."{task}".environment]
# env = {{ EXAMPLE = "1" }}
"""

TASK_TOML = """schema_version = "1.4"

[task]
name = "{org}/{task}"

[environment]
docker_image = "{image}"
cpus = 1
memory_mb = 2048
storage_mb = 10240

[agent]
timeout_sec = 600.0

[verifier]
timeout_sec = 120.0
"""

# The template's task.toml: prepare rewrites `[task].name` to `<org>/<task_id>` in every rendered task, so each has
# the name Harbor keys it on (`harbor add` keeps one folder per name).
TEMPLATE_TASK_TOML = TASK_TOML.replace(
    "[task]\n",
    "# Rendered per row as `<org>/<task_id>`: Harbor names a task by this field, so each rendered task gets its own.\n"
    "[task]\n",
)

INSTRUCTION_MD = """Create a file called `hello.txt` in the working directory containing the single line `hello`.
"""

TEST_SH = """#!/bin/bash
# Harbor's verifier contract: write the reward to /logs/verifier/reward.txt (or reward.json).
mkdir -p /logs/verifier
if [ "$(cat hello.txt 2>/dev/null)" = "hello" ]; then
    echo 1 > /logs/verifier/reward.txt
else
    echo 0 > /logs/verifier/reward.txt
fi
"""

SOLVE_SH = """#!/bin/bash
# The reference solution `gym dataset validate` runs; it must score 1.
echo hello > hello.txt
"""


TASK_DATA_PY = """# The TaskData model every task's fields (tasks.jsonl rows, rendered as task_data.json) are validated
# against at prepare, at seed and at collate. Unknown keys fail; keep this the complete description.
from pydantic import BaseModel, Field


class TaskData(BaseModel):
    question: str
    answer: str = Field(min_length=1)
"""

TASKS_JSONL = """{"task_id": "{task}-1", "task_data": {"question": "What is 2 + 2?", "answer": "4"}}
{"task_id": "{task}-2", "task_data": {"question": "What is 6 * 7?", "answer": "42"}}
"""

INSTRUCTION_TEMPLATE_MD = """Solve this problem:

{{ task_data.question }}

Write only the final answer to a file called `answer.md` in the working directory.
"""

VERIFIER_PY = """# The Python verifier: runs inside the task's sandbox through a test.sh shim NeMo Gym writes (never on the
# host). Declare the arguments you read: `workspace` (the working directory), `task_data` (the row's
# fields), `response` (the agent's final response; Gym-only) or `state`. Return the reward.
# It reads workspace/answer.md: the file the agent is asked for, and the one the reference answer
# (solution/answer.md) is written to by `gym dataset validate` and by the published solve.sh alike.
from pathlib import Path

from nemo_gym.verifiers.files import read_text


def verify(workspace: Path, task_data: dict) -> float:
    answer = read_text(workspace, "answer.md")
    if answer is None:
        return 0.0
    return float(answer.strip() == task_data["answer"])
"""

# The reference answer, rendered per row into solution/answer.md. Placeholders are inserted verbatim, so a
# reference answer lives in a text file like this one, never inside a quoted shell literal of a solve.sh.
ANSWER_TEMPLATE_MD = """{{ task_data.answer }}
"""


def split_dataset_name(name: str) -> tuple[str, str]:
    """``org/name`` as Harbor 0.22.0 requires it; a bare name gets :data:`DEFAULT_ORG`.

    Raises ``ValueError`` for a name Harbor's manifest would reject, so nothing is written for it.
    """
    org, slash, short = name.rpartition("/")
    if not slash:
        org = DEFAULT_ORG
    valid = all(_NAME_SEGMENT.match(part) for part in (org, short)) and ".." not in name
    if not valid:
        raise ValueError(
            f"{name!r} is not a Harbor dataset name: use `org/name` (or a bare name, which becomes "
            f"`{DEFAULT_ORG}/name`) with alphanumeric characters, hyphens, underscores and dots, "
            "not starting with a dot and without `..`"
        )
    return org, short


def init_dataset(
    root: Path, name: str, *, image: str = DEFAULT_IMAGE, task: str = "hello", parameterized: bool = False
) -> Path:
    """Write ``<root>/<name without org>/`` with ``dataset.toml`` and one task folder; refuses to overwrite.

    ``name`` is ``org/name`` or a bare name, which is written as ``local/<name>`` so that Harbor accepts the
    manifest. The folder is the name without the org, as ``harbor task init`` does for a task. With
    ``parameterized`` the folder is a template instead: ``task.toml`` at its root (its ``[task].name`` is rewritten to
    ``<org>/<task_id>`` in every rendered task, the name Harbor keys a task on), ``instruction.template.md``,
    ``tasks.jsonl`` with two rows, ``task_data.py``, a Python verifier that reads ``answer.md`` and the reference
    answer as ``solution/answer.template.md``, which prepare renders into one task folder per row
    (:mod:`nemo_gym.tasks.harbor.parameterized`).
    """
    org, short = split_dataset_name(name)
    folder = Path(root) / short
    if folder.exists():
        raise FileExistsError(f"{folder} already exists")
    if parameterized:
        for sub in ("environment", "tests", "solution"):
            (folder / sub).mkdir(parents=True)
        (folder / "dataset.toml").write_text(DATASET_TOML.format(org=org, name=short, task=task))
        (folder / "task.toml").write_text(TEMPLATE_TASK_TOML.format(org=org, task=task, image=image))
        (folder / "instruction.template.md").write_text(INSTRUCTION_TEMPLATE_MD)
        (folder / "tasks.jsonl").write_text(TASKS_JSONL.replace("{task}", task))
        (folder / "task_data.py").write_text(TASK_DATA_PY)
        (folder / "tests" / "verifier.py").write_text(VERIFIER_PY)
        (folder / "solution" / "answer.template.md").write_text(ANSWER_TEMPLATE_MD)
        return folder
    task_dir = folder / task
    for sub in ("environment", "tests", "solution"):
        (task_dir / sub).mkdir(parents=True)
    (folder / "dataset.toml").write_text(DATASET_TOML.format(org=org, name=short, task=task))
    (task_dir / "task.toml").write_text(TASK_TOML.format(org=org, task=task, image=image))
    (task_dir / "instruction.md").write_text(INSTRUCTION_MD)
    (task_dir / "tests" / "test.sh").write_text(TEST_SH)
    (task_dir / "solution" / "solve.sh").write_text(SOLVE_SH)
    for script in (task_dir / "tests" / "test.sh", task_dir / "solution" / "solve.sh"):
        script.chmod(0o755)
    return folder
