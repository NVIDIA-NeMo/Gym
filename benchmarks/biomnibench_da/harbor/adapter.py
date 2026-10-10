# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import shutil
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import tomli_w
import tyro
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm


@dataclass(slots=True)
class BOBTask:
    """A BiomniBench task from the source."""

    id: str
    task_dir: str | Path

    def __post_init__(self):
        self.task_dir = Path(self.task_dir).expanduser()

    @property
    def toml_config(self) -> dict:
        toml_file = self.task_dir / "task.toml"

        with toml_file.open("rb") as file:
            task_config = tomllib.load(file)

        return task_config

    @classmethod
    def from_dir(cls, benchmark_dir: Path) -> Iterator["BOBTask"]:
        for da_dir in sorted(benchmark_dir.glob("da-*")):
            if not da_dir.is_dir():
                continue

            yield cls(da_dir.name, da_dir)


@dataclass(slots=True)
class HarborBOBTask:
    source: BOBTask
    docker_image: str
    template_dir: str | Path = Path(__file__).parent / "task-template"
    timeout: int = 3600

    @property
    def id(self) -> str:
        return f"bob-task__{self.source.id}"

    @property
    def toml(self) -> str:
        task_config = self.source.toml_config

        task_config["task"]["name"] = f"phylo/{self.id}"

        task_config["environment"] = {
            "docker_image": self.docker_image,
            "workdir": "/app",
        }

        task_config["verifier"]["env"] = {
            "JUDGE_MODEL": r"${JUDGE_MODEL}",
            "JUDGE_MODEL_API_BASE": r"${JUDGE_MODEL_API_BASE}",
            "JUDGE_MODEL_API_KEY": r"${JUDGE_MODEL_API_KEY:-}",
        }

        return tomli_w.dumps(task_config)

    @property
    def gym_input(self) -> dict:
        return {"task_name": self.id, "responses_create_params": {"input": []}}

    def write(self, output_dir: Path, force: bool = False) -> Path:
        task_dir = output_dir / self.id

        if task_dir.exists():
            if not force:
                raise FileExistsError(f"Harbor task output directory already exists: {task_dir}")
            shutil.rmtree(task_dir)

        shutil.copytree(self.source.task_dir, task_dir, dirs_exist_ok=True)
        shutil.copytree(self.template_dir, task_dir, dirs_exist_ok=True)

        ## Do not need custom Dockerfile per task.
        (task_dir / "environment/Dockerfile").unlink()
        (task_dir / "task.toml").write_text(self.toml + "\n")

        return task_dir


@dataclass(slots=True)
class HarborBOB:
    docker_image: str
    """Docker image for task execution."""

    output_dir: Annotated[str | Path, tyro.conf.arg(aliases=["-o"])]
    """Directory for generated Harbor tasks."""

    data_dir: Annotated[str | Path, tyro.conf.arg(aliases=["-d"])] = "phylobio/BiomniBench-DA"
    """HuggingFace repo id or directory containing the source benchmark data."""

    limit: Annotated[int | None, tyro.conf.arg(aliases=["-l"])] = None
    """Maximum number of tasks to convert."""

    overwrite: bool = False
    """Whether existing generated tasks may be replaced."""

    def __post_init__(self):
        self.output_dir = Path(self.output_dir).expanduser()

        self.data_dir = Path(self.data_dir).expanduser()

        if not self.data_dir.is_dir():
            self.data_dir = Path(snapshot_download(str(self.data_dir), repo_type="dataset"))

    @property
    def tasks(self) -> Iterator[HarborBOBTask]:
        for source_task in BOBTask.from_dir(self.data_dir):
            yield HarborBOBTask(source_task, docker_image=self.docker_image)

    def write(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        gym_rows: list[str] = []
        for task in tqdm(self.tasks, desc="bob"):
            if self.limit is not None and len(gym_rows) >= self.limit:
                break

            task.write(self.output_dir, force=self.overwrite)
            gym_rows.append(json.dumps(task.gym_input) + "\n")

        with (self.output_dir / "gym.jsonl").open("w") as gym_file:
            gym_file.writelines(gym_rows)

        print(f"Adapted {len(gym_rows)} tasks to {self.output_dir}")


def prepare(
    *,
    output_dir: str | Path,
    docker_image: str,
    data_dir: str | Path = "phylobio/BiomniBench-DA",
    limit: int | None = None,
    overwrite: bool = False,
) -> Path:
    """Prepare Harbor tasks and return their Gym JSONL index path."""
    benchmark = HarborBOB(
        docker_image=docker_image,
        output_dir=output_dir,
        data_dir=data_dir,
        limit=limit,
        overwrite=overwrite,
    )
    benchmark.write()
    return Path(benchmark.output_dir) / "gym.jsonl"


if __name__ == "__main__":
    tyro.cli(HarborBOB).write()
