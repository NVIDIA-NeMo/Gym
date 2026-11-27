# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import csv
import json
import shutil
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import tyro
from huggingface_hub import hf_hub_download
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
                yield cls(**row, data=data_dir / f"{row['id']}.zip")


@dataclass(slots=True)
class HarborBMBTask:
    source: BMBTask
    docker_image: str
    template_dir: str | Path = Path(__file__).parent / "task-template"
    skip_data: bool = False
    data_path_prefix: str | None = None

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
    def environment_data(self) -> str:
        """Return the task data path visible to the agent."""
        if self.skip_data and self.data_path_prefix:
            return str(Path(self.data_path_prefix) / self.id / "environment/data")
        return "/app/data"

    @property
    def instruction(self) -> str:
        """Render the task question and data location with a trailing newline."""
        instruction_template = (self.template_dir / "instruction.md").read_text().strip() + "\n"
        return instruction_template.format(question=self.source.question, data_path=self.environment_data)

    @property
    def task_data(self) -> dict[str, str]:
        """Return the question and private rubric used by the verifier."""
        return {"question": self.source.question, "answer_rubric": self.source.answer_rubric}

    def write(
        self,
        output_dir: Path,
        force: bool = False,
        *,
        data_repo_id: str | None = None,
    ) -> Path:
        """Reuse completed tasks unless forced, rebuilding incomplete task directories."""
        task_dir = output_dir / self.id
        done_marker = task_dir / ".done"

        if task_dir.exists():
            if not force and done_marker.is_file() and (self.skip_data or (task_dir / "environment/data").is_dir()):
                (task_dir / "instruction.md").write_text(self.instruction)
                return task_dir
            shutil.rmtree(task_dir)

        shutil.copytree(self.template_dir, task_dir, dirs_exist_ok=True)

        (task_dir / "task.toml").write_text(self.toml + "\n")
        (task_dir / "instruction.md").write_text(self.instruction)
        if not self.skip_data:
            archive_path = Path(self.source.data)
            if data_repo_id is not None:
                archive_path = Path(
                    hf_hub_download(data_repo_id, filename=f"data/{self.source.id}.zip", repo_type="dataset")
                )
            if not archive_path.is_file():
                raise FileNotFoundError(f"Task data archive does not exist for ID: {self.source.id}")
            (task_dir / "environment/data").mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive_path) as data_zip:
                data_zip.extractall(task_dir / "environment/data")
        (task_dir / "tests/task_data.json").write_text(json.dumps(self.task_data) + "\n")

        # Only mark success after every requested file has been written.
        done_marker.touch()

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
    """Rebuild completed tasks too; incomplete tasks are always rebuilt."""

    skip_data: bool = False
    """Skip downloading and extracting data archives; fetch only problems.csv and supply task data separately."""

    data_path_prefix: str | None = None
    """Root containing <task-id>/environment/data when skip_data is enabled; None uses /app/data."""

    _data_repo_id: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.output_dir = Path(self.output_dir).expanduser()

        self.data_dir = Path(self.data_dir).expanduser()

        if not self.data_dir.is_dir():
            self._data_repo_id = str(self.data_dir)
            self.data_dir = Path(
                hf_hub_download(self._data_repo_id, filename="problems.csv", repo_type="dataset")
            ).parent

    @property
    def tasks(self) -> Iterator[HarborBMBTask]:
        for source_task in BMBTask.from_dir(self.data_dir):
            yield HarborBMBTask(
                source_task,
                docker_image=self.docker_image,
                skip_data=self.skip_data,
                data_path_prefix=self.data_path_prefix,
            )

    def write(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        task_count = 0

        for task in tqdm(self.tasks, desc="bmb"):
            if self.limit is not None and task_count >= self.limit:
                break

            task.write(
                self.output_dir,
                data_repo_id=self._data_repo_id,
                force=self.overwrite,
            )
            task_count += 1

        print(f"Adapted {task_count} tasks to {self.output_dir}")


def prepare(
    *,
    output_dir: str | Path,
    docker_image: str,
    data_dir: str | Path = "Anthropic/BioMysteryBench-full",
    limit: int | None = None,
    overwrite: bool = False,
    skip_data: bool = False,
    data_path_prefix: str | None = None,
) -> Path:
    """Prepare Harbor tasks and return their output directory."""
    benchmark = HarborBMB(
        docker_image=docker_image,
        output_dir=output_dir,
        data_dir=data_dir,
        limit=limit,
        overwrite=overwrite,
        skip_data=skip_data,
        data_path_prefix=data_path_prefix,
    )
    benchmark.write()
    return Path(benchmark.output_dir)


if __name__ == "__main__":
    tyro.cli(HarborBMB).write()
