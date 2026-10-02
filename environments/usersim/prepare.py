# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize canonical NeMo UserSim episode inputs."""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


ENVIRONMENT_DIR = Path(__file__).parent
DATA_DIR = ENVIRONMENT_DIR / "data"
TASKS_FPATH = DATA_DIR / "example.jsonl"
PREPARE_REQUIREMENTS_FPATH = ENVIRONMENT_DIR / "requirements.txt"
USERSIM_REVISION = "a5f676bf6dc5a73914c8a0860f97c10dd2c214ee"  # pragma: allowlist secret
_MATERIALIZE_SCRIPT = """
import json
import sys
from pathlib import Path
from usersim.cli._pipeline import known_probes
from usersim.engine.external import materialize_episode_inputs

locale, seed, destination = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
rows = []
for offset, probe in enumerate(known_probes()):
    [row] = materialize_episode_inputs(
        locale=locale,
        num_rows=1,
        probe_mix={probe: 1.0},
        random_seed=seed + offset,
    )
    rows.append(row)
destination.write_text("".join(json.dumps(row, ensure_ascii=False, default=str) + "\\n" for row in rows))
"""


def prepare(
    locale: str = "en_US",
    random_seed: int = 1042,
    uv_executable: str = "uv",
    timeout_seconds: float = 3_600,
) -> Path:
    """Materialize one canonical resolved row for each registered UserSim probe."""
    executable = shutil.which(uv_executable)
    if executable is None:
        raise RuntimeError(f"{uv_executable!r} is not on PATH; it is required to prepare UserSim inputs.")
    TASKS_FPATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="usersim-materialize-") as working_dir:
        resolved_path = Path(working_dir) / "resolved.jsonl"
        command = [
            executable,
            "run",
            "--no-config",
            "--no-project",
            "--isolated",
            "--with-requirements",
            str(PREPARE_REQUIREMENTS_FPATH),
            "python",
            "-c",
            _MATERIALIZE_SCRIPT,
            locale,
            str(random_seed),
            str(resolved_path),
        ]
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                errors="replace",
                env={**os.environ, "USERSIM_CODE_SHA": USERSIM_REVISION},
                timeout=timeout_seconds,
                cwd=working_dir,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            stderr = getattr(exc, "stderr", "") or ""
            raise RuntimeError(f"Failed to materialize NeMo UserSim inputs: {stderr.strip() or exc}") from exc
        resolved_rows = [json.loads(line) for line in resolved_path.read_text().splitlines() if line.strip()]

    tasks = [
        {
            "task_id": {"taskset": "usersim:example", "task_id": str(index)},
            "task_input": {"resolved_row": row},
        }
        for index, row in enumerate(resolved_rows)
    ]
    temporary_tasks = TASKS_FPATH.with_suffix(".jsonl.tmp")
    temporary_tasks.write_text("".join(f"{json.dumps(row, separators=(',', ':'))}\n" for row in tasks))
    os.replace(temporary_tasks, TASKS_FPATH)
    print(f"Prepared {len(tasks)} NeMo UserSim tasks at {TASKS_FPATH}")
    return TASKS_FPATH.absolute()


if __name__ == "__main__":
    prepare()
