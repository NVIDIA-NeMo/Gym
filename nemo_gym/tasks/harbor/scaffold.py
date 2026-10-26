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

[gym.defaults]
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


def init_dataset(root: Path, name: str, *, image: str = DEFAULT_IMAGE, task: str = "hello") -> Path:
    """Write ``<root>/<name without org>/`` with ``dataset.toml`` and one task folder; refuses to overwrite.

    ``name`` is ``org/name`` or a bare name, which is written as ``local/<name>`` so that Harbor accepts the
    manifest. The folder is the name without the org, as ``harbor task init`` does for a task.
    """
    org, short = split_dataset_name(name)
    folder = Path(root) / short
    if folder.exists():
        raise FileExistsError(f"{folder} already exists")
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
