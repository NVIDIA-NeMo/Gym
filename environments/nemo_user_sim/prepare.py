# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize canonical NeMo UserSim episode inputs."""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


ENVIRONMENT_DIR = Path(__file__).parent
DATA_DIR = ENVIRONMENT_DIR / "data"
TASKS_FPATH = DATA_DIR / "usersim.jsonl"
PREPARE_REQUIREMENTS_FPATH = ENVIRONMENT_DIR / "requirements.txt"
USERSIM_REVISION = "a5f676bf6dc5a73914c8a0860f97c10dd2c214ee"  # pragma: allowlist secret
_MATERIALIZE_SCRIPT = """
import json
import sys
from pathlib import Path

repository_root = sys.argv[1]
sys.path.insert(0, repository_root)
from environments.nemo_user_sim.prepare import _probe_seed
from usersim.engine.core.probes import known_probes
from usersim.engine.external import materialize_episode_inputs

locale, seed, models_path, destination = sys.argv[2], int(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5])
rows = []
for probe in known_probes():
    [row] = materialize_episode_inputs(
        locale=locale,
        num_rows=1,
        probe_mix={probe: 1.0},
        random_seed=_probe_seed(seed, probe),
        models_path=models_path,
    )
    rows.append(row)
destination.write_text("".join(json.dumps(row, ensure_ascii=False, default=str) + "\\n" for row in rows))
"""


def _probe_seed(seed: int, probe: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}\0{probe}".encode()).digest()[:4], "big")


def _task_id(row: dict[str, object]) -> str:
    task_id = row.get("trajectory_id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("Each UserSim row must have a non-empty string trajectory_id")
    return task_id


def _models_config(*, policy_model_name: str, support_model_name: str) -> str:
    model_names = {
        "user_model": policy_model_name,
        "assistant_model": policy_model_name,
        "api_response_model": support_model_name,
        "judge_model": support_model_name,
        "summary_model": support_model_name,
        "evaluator_model": support_model_name,
    }
    specs = ",\n".join(
        f'  {{ alias = {json.dumps(alias)}, model = {json.dumps(model)}, provider = "openai" }}'
        for alias, model in model_names.items()
    )
    return f"models = [\n{specs},\n]\n"


def prepare(
    locale: str = "en_US",
    random_seed: int = 1042,
    policy_model_name: str = "policy_model",
    support_model_name: str = "support_model",
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
        models_path = Path(working_dir) / "models.toml"
        models_path.write_text(
            _models_config(policy_model_name=policy_model_name, support_model_name=support_model_name)
        )
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
            str(ENVIRONMENT_DIR.parents[1]),
            locale,
            str(random_seed),
            str(models_path),
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

    tasks = []
    task_ids: set[str] = set()
    for row in resolved_rows:
        usersim_config = row.get("usersim_config")
        if isinstance(usersim_config, dict):
            usersim_config.pop("assets_dir", None)
        task_id = _task_id(row)
        if task_id in task_ids:
            raise ValueError(f"Duplicate UserSim trajectory_id: {task_id}")
        task_ids.add(task_id)
        tasks.append({"task_id": task_id, "resolved_row": row})
    temporary_tasks = TASKS_FPATH.with_suffix(".jsonl.tmp")
    temporary_tasks.write_text("".join(f"{json.dumps(row, separators=(',', ':'))}\n" for row in tasks))
    os.replace(temporary_tasks, TASKS_FPATH)
    print(f"Prepared {len(tasks)} NeMo UserSim tasks at {TASKS_FPATH}")
    return TASKS_FPATH.absolute()


if __name__ == "__main__":
    prepare()
