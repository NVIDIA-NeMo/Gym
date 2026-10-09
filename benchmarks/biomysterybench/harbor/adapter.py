# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import csv
import json
import shutil
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import tyro
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm


@dataclass(slots=True)
class BMBTask:
    """A BioMysteryBench problem from the source CSV."""

    id: str
    question: str
    answer_rubric: str
    allowed_domains: str | list[str]
    human_solvable: str | bool
    data: str | Path

    def __post_init__(self):
        self.data = Path(self.data).expanduser()

        if isinstance(self.allowed_domains, str):
            self.allowed_domains = [domain.strip() for domain in self.allowed_domains.split(",")]

    @classmethod
    def from_dir(cls, benchmark_dir: Path) -> Iterator["BMBTask"]:
        problems_csv = benchmark_dir / "problems.csv"
        data_dir = benchmark_dir / "data"

        with problems_csv.open(newline="") as problems_file:
            reader = csv.DictReader(problems_file)

            for row in reader:
                if not (data_dir / f"{row['id']}.zip").is_file():
                    raise FileNotFoundError(f"Task data archive does not exist for ID: {row['id']}")

                yield cls(**row, data=data_dir / f"{row['id']}.zip")


@dataclass(slots=True)
class HarborBMBTask:
    source: BMBTask
    docker_image: str
    template_dir: str | Path = Path(__file__).parent / "task-template"

    @property
    def id(self) -> str:
        return f"bmb-task__{self.source.id}"

    @property
    def toml(self) -> str:
        toml_template = (self.template_dir / "task.toml").read_text().strip()
        allowed_domains = sorted(
            set(self.source.allowed_domains)
            | {f"*.{domain}" for domain in self.source.allowed_domains if not domain.startswith("*.")}
        )
        return toml_template.format(
            id=self.id,
            docker_image=self.docker_image,
            allowed_domains=json.dumps(allowed_domains),
        )

    @property
    def instruction(self) -> str:
        instruction_template = (self.template_dir / "instruction.md").read_text().strip()
        return instruction_template.format(question=self.source.question)

    @property
    def judge_prompt(self) -> str:
        prompt_template = (self.template_dir / "tests/prompt.txt").read_text().strip()
        return prompt_template.format(
            question=self.source.question,
            answer_rubric=self.source.answer_rubric,
        )

    @property
    def gym_input(self) -> dict:
        return {"task_name": self.id, "responses_create_params": {"input": []}}

    def write(self, output_dir: Path, force: bool = False) -> Path:
        task_dir = output_dir / self.id

        if task_dir.exists():
            if not force:
                raise FileExistsError(f"Harbor task output directory already exists: {task_dir}")
            shutil.rmtree(task_dir)

        shutil.copytree(self.template_dir, task_dir, dirs_exist_ok=True)

        (task_dir / "task.toml").write_text(self.toml + "\n")
        (task_dir / "instruction.md").write_text(self.instruction + "\n")
        (task_dir / "environment/data").mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(self.source.data) as data_zip:
            data_zip.extractall(task_dir / "environment/data")
        (task_dir / "tests/prompt.txt").write_text(self.judge_prompt + "\n")

        return task_dir


@dataclass(slots=True)
class HarborBMB:
    docker_image: str
    """Docker image for task execution."""

    output_dir: Annotated[str | Path, tyro.conf.arg(aliases=["-o"])]
    """Directory for generated Harbor tasks."""

    data_dir: Annotated[str | Path, tyro.conf.arg(aliases=["-d"])] = "Anthropic/BioMysteryBench-full"
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
    def tasks(self) -> Iterator[HarborBMBTask]:
        for source_task in BMBTask.from_dir(self.data_dir):
            yield HarborBMBTask(source_task, docker_image=self.docker_image)

    def write(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        gym_rows: list[str] = []

        for task in tqdm(self.tasks, desc="bmb"):
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
    data_dir: str | Path = "Anthropic/BioMysteryBench-full",
    limit: int | None = None,
    overwrite: bool = False,
) -> Path:
    """Prepare Harbor tasks and return their Gym JSONL index path."""
    benchmark = HarborBMB(
        docker_image=docker_image,
        output_dir=output_dir,
        data_dir=data_dir,
        limit=limit,
        overwrite=overwrite,
    )
    benchmark.write()
    return Path(benchmark.output_dir) / "gym.jsonl"


if __name__ == "__main__":
    tyro.cli(HarborBMB).write()
