# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Annotated, TypedDict, cast

import tyro
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm


class GeneBenchConfig(TypedDict):
    """Source fields needed to prepare a public case study."""

    id: str
    task: str
    data_files: list[str]


@dataclass(slots=True)
class GeneBenchTask:
    """A GeneBench-Pro task from the source package."""

    id: str
    task_dir: Path

    @property
    def eval_config(self) -> GeneBenchConfig:
        return cast(GeneBenchConfig, json.loads((self.task_dir / "eval_config.json").read_text()))

    @classmethod
    def from_dir(cls, benchmark_dir: Path) -> Iterator["GeneBenchTask"]:
        for config_file in sorted((benchmark_dir / "problems").glob("*/eval_config.json")):
            config = json.loads(config_file.read_text())
            yield cls(id=config["id"], task_dir=config_file.parent)


@dataclass(slots=True)
class HarborGeneBenchTask:
    """Prepare a Harbor task with isolated inputs and reference grading assets."""

    source: GeneBenchTask
    docker_image: str
    template_dir: Path = Path(__file__).parent / "task-template"

    @property
    def id(self) -> str:
        return f"genebench-pro__{self.source.id}"

    @property
    def instruction(self) -> str:
        return (
            self.source.eval_config["task"]
            + "\n\nInput files are available under /app/data_files/.\n"
            + "Also save your final JSON object to /app/eval_answer.json.\n"
        )

    @property
    def toml(self) -> str:
        return Template((self.template_dir / "task.toml").read_text()).substitute(
            task_name=json.dumps(f"openai/{self.id}"),
            docker_image=json.dumps(self.docker_image),
        )

    @property
    def gym_input(self) -> dict[str, object]:
        return {"task_name": self.id, "responses_create_params": {"input": []}}

    def write(self, output_dir: Path, *, force: bool = False) -> Path:
        """Write agent-visible inputs separately from reference grading assets."""
        config = self.source.eval_config
        task_dir = output_dir / self.id

        if task_dir.exists():
            if not force:
                raise FileExistsError(f"Harbor task output directory already exists: {task_dir}")
            shutil.rmtree(task_dir)

        shutil.copytree(self.template_dir, task_dir)
        environment_dir = task_dir / "environment"
        tests_dir = task_dir / "tests"
        tests_dir.mkdir(exist_ok=True)

        for filename in config["data_files"]:
            relative_path = Path(filename)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"Expected a problem-relative input path, got: {filename}")
            destination = environment_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.source.task_dir / relative_path, destination)

        (task_dir / "instruction.md").write_text(self.instruction)
        (task_dir / "task.toml").write_text(self.toml)
        shutil.copy2(self.source.task_dir / "eval_config.json", tests_dir / "eval_config.json")
        shutil.copy2(self.source.task_dir.parent.parent / "reference_grader.py", tests_dir / "reference_grader.py")
        shutil.copy2(self.source.task_dir.parent.parent / "LICENSE", tests_dir / "LICENSE")
        return task_dir


@dataclass(slots=True)
class HarborGeneBench:
    """Convert the public package into task assets and a Gym index."""

    docker_image: str
    """Docker image for task execution."""

    output_dir: Annotated[str | Path, tyro.conf.arg(aliases=["-o"])]
    """Directory for generated Harbor tasks."""

    data_dir: Annotated[str | Path, tyro.conf.arg(aliases=["-d"])] = "openai/genebench-pro-public-package"
    """Hugging Face repo id or local source package directory."""

    limit: Annotated[int | None, tyro.conf.arg(aliases=["-l"])] = None
    """Maximum number of tasks to convert; does not limit the download."""

    overwrite: bool = False
    """Whether existing generated tasks may be replaced."""

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir).expanduser()
        data_dir = Path(self.data_dir).expanduser()
        if not data_dir.is_dir():
            data_dir = Path(snapshot_download(str(self.data_dir), repo_type="dataset"))
        self.data_dir = data_dir

    @property
    def tasks(self) -> Iterator[HarborGeneBenchTask]:
        for source_task in GeneBenchTask.from_dir(Path(self.data_dir)):
            yield HarborGeneBenchTask(source=source_task, docker_image=self.docker_image)

    def write(self) -> None:
        """Write converted tasks and their Gym JSONL index."""
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        gym_rows: list[str] = []
        for task in tqdm(self.tasks, desc="genebench-pro"):
            if self.limit is not None and len(gym_rows) >= self.limit:
                break
            task.write(output_dir, force=self.overwrite)
            gym_rows.append(json.dumps(task.gym_input) + "\n")

        (output_dir / "gym.jsonl").write_text("".join(gym_rows))
        print(f"Adapted {len(gym_rows)} tasks to {output_dir}")


def prepare(
    *,
    output_dir: str | Path,
    docker_image: str,
    data_dir: str | Path = "openai/genebench-pro-public-package",
    limit: int | None = None,
    overwrite: bool = False,
) -> Path:
    """Prepare task assets and return the Gym JSONL index path."""
    benchmark = HarborGeneBench(
        docker_image=docker_image,
        output_dir=output_dir,
        data_dir=data_dir,
        limit=limit,
        overwrite=overwrite,
    )
    benchmark.write()
    return Path(benchmark.output_dir) / "gym.jsonl"


if __name__ == "__main__":
    tyro.cli(HarborGeneBench).write()
