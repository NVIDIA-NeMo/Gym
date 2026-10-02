# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``gym dataset init``: a dataset folder in exactly the shape the loader accepts.

One task, ready to validate and run. Everything written here is read back by the loader in a
test, so the scaffold cannot drift from what ``gym eval run`` and ``gym dataset validate`` expect.
"""

from pathlib import Path


DEFAULT_IMAGE = "python:3.12-slim"

DATASET_TOML = """# Harbor's dataset file. The [dataset] table is Harbor's; the [gym] table is Gym's and holds
# what the dataset needs on top of its tasks, as data. Deployment settings (sandbox provider,
# registry credentials, image mirrors) never go here: a dataset runs unchanged everywhere.
[dataset]
name = "{name}"
version = "0.1.0"
description = "One-line description of the dataset."

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
name = "{name}/{task}"

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


def init_dataset(root: Path, name: str, *, image: str = DEFAULT_IMAGE, task: str = "hello") -> Path:
    """Write ``<root>/<name>/`` with ``dataset.toml`` and one task folder; refuses to overwrite."""
    folder = Path(root) / name
    if folder.exists():
        raise FileExistsError(f"{folder} already exists")
    task_dir = folder / task
    for sub in ("environment", "tests", "solution"):
        (task_dir / sub).mkdir(parents=True)
    (folder / "dataset.toml").write_text(DATASET_TOML.format(name=name, task=task))
    (task_dir / "task.toml").write_text(TASK_TOML.format(name=name, task=task, image=image))
    (task_dir / "instruction.md").write_text(INSTRUCTION_MD)
    (task_dir / "tests" / "test.sh").write_text(TEST_SH)
    (task_dir / "solution" / "solve.sh").write_text(SOLVE_SH)
    for script in (task_dir / "tests" / "test.sh", task_dir / "solution" / "solve.sh"):
        script.chmod(0o755)
    return folder
