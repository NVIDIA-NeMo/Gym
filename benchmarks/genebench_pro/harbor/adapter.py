# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import shutil
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from string import Template
from typing import Annotated, TypedDict, cast

import tyro
from huggingface_hub import RepoFile, list_repo_tree, snapshot_download
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
    data_path_prefix: str | None = field(default=None, kw_only=True)

    @property
    def id(self) -> str:
        return f"genebench-pro__{self.source.id}"

    def instruction(self, *, skip_data: bool = False) -> str:
        """Return the prompt with the input location and submission contract."""
        data_path = (
            f"{Path(self.data_path_prefix) / self.id / 'environment/data_files'}/"
            if skip_data and self.data_path_prefix
            else "/app/data_files/"
        )
        return (
            self.source.eval_config["task"]
            + f"\n\nInput files are available under {data_path}.\n"
            + "Also save your final JSON object to /app/eval_answer.json.\n"
        )

    @property
    def toml(self) -> str:
        return Template((self.template_dir / "task.toml").read_text()).substitute(
            task_name=json.dumps(f"openai/{self.id}"),
            docker_image=json.dumps(self.docker_image),
        )

    def _is_complete(self, output_dir: Path, *, skip_data: bool) -> bool:
        task_dir = output_dir / self.id
        return (task_dir / ".done").is_file() and (
            not (task_dir / "environment").exists() if skip_data else (task_dir / "environment").is_dir()
        )

    def write(self, output_dir: Path, *, force: bool = False, skip_data: bool = False) -> Path:
        """Reuse completed tasks, rebuilding incomplete tasks or a changed data mode."""
        task_dir = output_dir / self.id
        if task_dir.exists():
            if not force and self._is_complete(output_dir, skip_data=skip_data):
                if skip_data:
                    (task_dir / "instruction.md").write_text(self.instruction(skip_data=True))
                return task_dir
            shutil.rmtree(task_dir)

        config = self.source.eval_config
        shutil.copytree(
            self.template_dir,
            task_dir,
            ignore=lambda directory, _names: (
                ("environment",) if skip_data and Path(directory) == self.template_dir else ()
            ),
        )
        environment_dir = task_dir / "environment"
        tests_dir = task_dir / "tests"
        tests_dir.mkdir(exist_ok=True)

        if not skip_data:
            environment_dir.mkdir(exist_ok=True)
        for filename in config["data_files"]:
            relative_path = Path(filename)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"Expected a problem-relative input path, got: {filename}")
            if skip_data:
                continue
            destination = environment_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.source.task_dir / relative_path, destination)

        (task_dir / "instruction.md").write_text(self.instruction(skip_data=skip_data))
        (task_dir / "task.toml").write_text(self.toml)
        shutil.copy2(self.source.task_dir / "eval_config.json", tests_dir / "eval_config.json")
        shutil.copy2(self.source.task_dir.parent.parent / "reference_grader.py", tests_dir / "reference_grader.py")
        shutil.copy2(self.source.task_dir.parent.parent / "LICENSE", tests_dir / "LICENSE")
        (task_dir / ".done").touch()
        return task_dir


@dataclass(slots=True)
class HarborGeneBench:
    """Convert the public package into Harbor task folders."""

    docker_image: str
    """Docker image for task execution."""

    output_dir: Annotated[str | Path, tyro.conf.arg(aliases=["-o"])]
    """Directory for generated Harbor tasks."""

    data_dir: Annotated[str | Path, tyro.conf.arg(aliases=["-d"])] = "openai/genebench-pro-public-package"
    """Hugging Face repo id or local source package directory."""

    limit: Annotated[int | None, tyro.conf.arg(aliases=["-l"])] = None
    """Maximum number of tasks to convert and download."""

    overwrite: bool = False
    """Rebuild completed tasks too; incomplete tasks are always rebuilt."""

    skip_data: bool = False
    """Omit input files and environment/; supply data separately at the path in the instruction."""

    data_path_prefix: str | None = None
    """Root containing <task-id>/environment/data_files when skip_data is enabled; None uses /app/data_files/."""

    _data_repo_id: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir).expanduser()
        self.data_dir = Path(self.data_dir).expanduser()
        if not self.data_dir.is_dir():
            self._data_repo_id = str(self.data_dir)

    @property
    def tasks(self) -> Iterator[HarborGeneBenchTask]:
        if self.limit is not None and self.limit <= 0:
            return
        if self._data_repo_id is None:
            source_tasks = GeneBenchTask.from_dir(Path(self.data_dir))
        else:
            config_paths = sorted(
                entry.path
                for entry in list_repo_tree(
                    self._data_repo_id, repo_type="dataset", path_in_repo="problems", recursive=True
                )
                if isinstance(entry, RepoFile) and Path(entry.path).match("problems/*/eval_config.json")
            )
            source_tasks = self._remote_tasks(islice(config_paths, self.limit))

        for source_task in islice(source_tasks, self.limit):
            task = HarborGeneBenchTask(
                source=source_task, docker_image=self.docker_image, data_path_prefix=self.data_path_prefix
            )
            if self._data_repo_id is not None and (
                self.overwrite or not task._is_complete(Path(self.output_dir), skip_data=self.skip_data)
            ):
                # Configs provide authoritative task IDs; fetch only listed public inputs,
                # keeping reports and reference answers out of the agent workspace.
                patterns = ["reference_grader.py", "LICENSE"]
                if not self.skip_data:
                    for filename in source_task.eval_config["data_files"]:
                        relative_path = Path(filename)
                        if relative_path.is_absolute() or ".." in relative_path.parts:
                            raise ValueError(f"Expected a problem-relative input path, got: {filename}")
                        patterns.append(f"problems/{source_task.task_dir.name}/{filename}")
                snapshot_download(
                    self._data_repo_id,
                    repo_type="dataset",
                    revision=source_task.task_dir.parent.parent.name,
                    allow_patterns=patterns,
                )
            yield task

    def _remote_tasks(self, config_paths: Iterator[str]) -> Iterator[GeneBenchTask]:
        for config_path in config_paths:
            snapshot_dir = Path(
                snapshot_download(self._data_repo_id, repo_type="dataset", allow_patterns=[config_path])
            )
            config_file = snapshot_dir / config_path
            config = json.loads(config_file.read_text())
            yield GeneBenchTask(id=config["id"], task_dir=config_file.parent)

    def write(self) -> None:
        """Prepare task folders without generating a Gym JSONL index."""
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        task_count = 0
        for task in tqdm(self.tasks, desc="genebench-pro"):
            task.write(output_dir, force=self.overwrite, skip_data=self.skip_data)
            task_count += 1

        print(f"Adapted {task_count} tasks to {output_dir}")


def prepare(
    *,
    output_dir: str | Path,
    docker_image: str,
    data_dir: str | Path = "openai/genebench-pro-public-package",
    limit: int | None = None,
    overwrite: bool = False,
    skip_data: bool = False,
    data_path_prefix: str | None = None,
) -> Path:
    """Prepare Harbor task folders and return their output directory."""
    benchmark = HarborGeneBench(
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
    tyro.cli(HarborGeneBench).write()
