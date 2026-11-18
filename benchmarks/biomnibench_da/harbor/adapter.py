# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import shutil
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Annotated

import tomli_w
import tyro
from huggingface_hub import RepoFolder, list_repo_tree, snapshot_download
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

    def environment_data(self, *, skip_data: bool) -> str:
        """Return the task data path visible to the agent."""
        if skip_data:
            return f"/mnt/s3-data/nemotron-life-science/harbor-data/biomnibench_da/{self.id}/environment/data"
        return "/app/data"

    @property
    def toml(self) -> str:
        task_config = self.source.toml_config

        task_config["task"]["name"] = f"phylo/{self.id}"

        task_config["environment"].update(
            {
                "docker_image": self.docker_image,
                "workdir": "/app",
            }
        )

        task_config["verifier"]["env"] = {
            "JUDGE_MODEL": r"${JUDGE_MODEL}",
            "JUDGE_MODEL_API_BASE": r"${JUDGE_MODEL_API_BASE}",
            "JUDGE_MODEL_API_KEY": r"${JUDGE_MODEL_API_KEY:-}",
        }

        return tomli_w.dumps(task_config)

    def _is_complete(self, output_dir: Path, *, skip_data: bool) -> bool:
        task_dir = output_dir / self.id
        return (task_dir / ".done").is_file() and (
            not (task_dir / "environment").exists() if skip_data else (task_dir / "environment/data").is_dir()
        )

    def write(self, output_dir: Path, force: bool = False, *, skip_data: bool = False) -> Path:
        """Reuse completed tasks unless forced, rebuilding incomplete task directories."""
        task_dir = output_dir / self.id
        done_marker = task_dir / ".done"

        if task_dir.exists():
            if not force and self._is_complete(output_dir, skip_data=skip_data):
                return task_dir
            shutil.rmtree(task_dir)

        shutil.copytree(
            self.source.task_dir,
            task_dir,
            dirs_exist_ok=True,
            ignore=lambda directory, _names: (
                ("environment",) if skip_data and Path(directory) == self.source.task_dir else ()
            ),
        )
        shutil.copytree(self.template_dir, task_dir, dirs_exist_ok=True)

        ## Do not need custom Dockerfile per task.
        (task_dir / "environment/Dockerfile").unlink(missing_ok=True)
        (task_dir / "task.toml").write_text(self.toml + "\n")
        if skip_data:
            instruction_path = task_dir / "instruction.md"
            instruction = instruction_path.read_text()
            instruction_path.write_text(
                instruction.replace(
                    "## Data Files\n",
                    f"## Data Files\n\nAll data files are available at `{self.environment_data(skip_data=skip_data)}`.\n",
                    1,
                )
            )

        done_marker.touch()

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
    """Rebuild completed tasks too; incomplete tasks are always rebuilt."""

    skip_data: bool = False
    """Skip downloading and copying environment/; supply task data separately at the path added to the instruction."""

    _data_repo_id: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.output_dir = Path(self.output_dir).expanduser()

        self.data_dir = Path(self.data_dir).expanduser()

        if not self.data_dir.is_dir():
            self._data_repo_id = str(self.data_dir)

    @property
    def tasks(self) -> Iterator[HarborBOBTask]:
        if self.limit is not None and self.limit <= 0:
            return
        if self._data_repo_id is None:
            source_tasks = BOBTask.from_dir(self.data_dir)
        else:
            task_names = sorted(
                entry.path
                for entry in list_repo_tree(self._data_repo_id, repo_type="dataset")
                if isinstance(entry, RepoFolder) and entry.path.startswith("da-")
            )
            source_tasks = (BOBTask(name, self.data_dir / name) for name in task_names)

        for source_task in islice(source_tasks, self.limit):
            task = HarborBOBTask(source_task, docker_image=self.docker_image)
            if self._data_repo_id is not None and (
                self.overwrite or not task._is_complete(self.output_dir, skip_data=self.skip_data)
            ):
                snapshot_dir = Path(
                    snapshot_download(
                        self._data_repo_id,
                        repo_type="dataset",
                        allow_patterns=[f"{source_task.id}/**"],
                        ignore_patterns=[f"{source_task.id}/environment/**"] if self.skip_data else None,
                    )
                )
                source_task.task_dir = snapshot_dir / source_task.id
            yield task

    def write(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        task_count = 0
        for task in tqdm(self.tasks, desc="bob"):
            task.write(self.output_dir, force=self.overwrite, skip_data=self.skip_data)
            task_count += 1

        print(f"Adapted {task_count} tasks to {self.output_dir}")


def prepare(
    *,
    output_dir: str | Path,
    docker_image: str,
    data_dir: str | Path = "phylobio/BiomniBench-DA",
    limit: int | None = None,
    overwrite: bool = False,
    skip_data: bool = False,
) -> Path:
    """Prepare Harbor tasks and return their output directory."""
    benchmark = HarborBOB(
        docker_image=docker_image,
        output_dir=output_dir,
        data_dir=data_dir,
        limit=limit,
        overwrite=overwrite,
        skip_data=skip_data,
    )
    benchmark.write()
    return Path(benchmark.output_dir)


if __name__ == "__main__":
    tyro.cli(HarborBOB).write()
