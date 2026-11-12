# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import shutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Annotated

import tyro
from datasets import Dataset, load_dataset
from huggingface_hub import sync_bucket
from tqdm.auto import tqdm


@dataclass(slots=True)
class BBHTask:
    """A BixBench-Hypothesis task from the source JSONL."""

    answer: bool
    hypothesis: str
    id: str
    input_data_path: str | Path
    max_points: int
    nb_primary_language: str
    protocol: str
    rubric: str

    ## External metadata.
    capsules_dir: str | Path

    def __post_init__(self):
        self.input_data_path = Path(self.input_data_path)
        while self.input_data_path.suffix:
            self.input_data_path = self.input_data_path.with_suffix("")

        if not str(self.capsules_dir).startswith("hf://buckets"):
            self.capsules_dir = Path(self.capsules_dir).expanduser()

    @classmethod
    def from_jsonl(cls, jsonl_file: str | Path, capsules_dir: str | Path) -> Iterator["BBHTask"]:
        with open(jsonl_file) as f:
            for line in f:
                record = json.loads(line)
                yield cls(**record, capsules_dir=capsules_dir)

    @classmethod
    def from_hf(cls, dataset: Dataset, capsules_dir: str | Path) -> Iterator["BBHTask"]:
        for record in dataset:
            yield cls(**record, capsules_dir=capsules_dir)

    def to_json(self) -> dict[str, object]:
        excluded = {"capsules_dir"}
        result = {}
        for fld in fields(self):
            if fld.name in excluded:
                continue

            value = getattr(self, fld.name)
            result[fld.name] = str(value) if isinstance(value, Path) else value

        return result

    @property
    def language(self) -> str:
        return self.nb_primary_language.lower()

    @property
    def capsule(self) -> Callable[[str | Path], None]:
        if str(self.capsules_dir).startswith("hf://buckets"):
            capsule_folder = f"{self.capsules_dir}/CapsuleData-{self.id}"
        else:
            capsule_folder = Path(self.input_data_path)
            while capsule_folder.suffix:
                capsule_folder = capsule_folder.with_suffix("")

            candidates = [
                self.capsules_dir / capsule_folder,
                self.capsules_dir / self.input_data_path,
                self.capsules_dir / f"CapsuleData-{self.id}",
                self.capsules_dir / f"capsule_{self.id}",
            ]
            for d in candidates:
                if d.is_dir():
                    capsule_folder = d
                    break

            if not capsule_folder.is_dir():
                raise FileNotFoundError(f"Input data directory does not exist: {candidates[0]}")

        def get_capsule(out: str | Path):
            if isinstance(capsule_folder, Path):
                shutil.copytree(capsule_folder, out, dirs_exist_ok=True)
            else:
                sync_bucket(source=capsule_folder, dest=str(out), delete=True)

        return get_capsule

    def get_system_prompt(
        self,
        template_path: str | Path,
        timeout: int,
        guidelines_template_path: str | Path | None = None,
    ) -> str:
        prompt_template = Path(template_path).read_text().strip()

        if self.language == "r":
            guidelines_template_path = (
                Path(guidelines_template_path or Path(__file__).parent) / "prompts/r-guidelines.md"
            )
            additional_guidelines = "\n" + guidelines_template_path.read_text().strip() + "\n"
        else:
            additional_guidelines = ""

        return prompt_template.format(
            language=self.language.capitalize(),
            timeout=timeout,
            additional_guidelines=additional_guidelines,
        )

    def get_user_prompt(self, template_path: str | Path) -> str:
        prompt_template = Path(template_path).read_text().strip()
        return prompt_template.format(
            language=self.language.capitalize(),
            hypothesis=self.hypothesis,
            protocol=self.protocol,
        )


@dataclass(slots=True)
class HarborBBHTask:
    source: BBHTask
    docker_image: str
    template_dir: str | Path = Path(__file__).parent / "task-template"
    timeout: int = 3600

    @property
    def id(self) -> str:
        return f"bbh-task__{self.source.id}"

    @property
    def toml(self) -> str:
        toml_template = (self.template_dir / "task.toml").read_text().strip()
        return toml_template.format(
            id=self.id,
            timeout=self.timeout,
            docker_image=self.docker_image,
        )

    @property
    def instruction(self) -> str:
        return self.source.get_user_prompt(self.template_dir / "steps/rollout/instruction.md")

    @property
    def opencode_agent(self) -> str:
        return self.source.get_system_prompt(
            self.template_dir / "environment/.opencode/agents/bbh.md",
            timeout=self.timeout,
        )

    @property
    def gym_input(self) -> dict:
        return {"task_name": self.id, "responses_create_params": {"input": []}}

    def write(self, output_dir: str | Path, force: bool = False) -> Path:
        output_dir = Path(output_dir).expanduser()
        task_dir = output_dir / self.id

        if task_dir.exists():
            if not force:
                raise FileExistsError(f"Harbor task output directory already exists: {task_dir}")
            shutil.rmtree(task_dir)

        shutil.copytree(self.template_dir, task_dir, dirs_exist_ok=True)

        self.source.capsule(task_dir / "environment/data")

        (task_dir / "task.toml").write_text(self.toml + "\n")
        (task_dir / "steps/rollout/instruction.md").write_text(self.instruction + "\n")
        (task_dir / "environment/task.jsonl").write_text(json.dumps(self.source.to_json()) + "\n")
        (task_dir / "environment/.opencode/agents/bbh.md").write_text(self.opencode_agent + "\n")

        return task_dir


@dataclass(slots=True)
class HarborBBH:
    docker_image: str
    """Docker image for task execution."""

    output_dir: Annotated[str | Path, tyro.conf.arg(aliases=["-o"])]
    """Directory for generated Harbor tasks."""

    data_dir: Annotated[str | Path, tyro.conf.arg(aliases=["-d"])] = "EdisonScientific/bixbench_hypothesis"
    """HuggingFace repo ID or path to benchmark data directory."""

    capsules_dir: Annotated[str | Path, tyro.conf.arg(aliases=["-c"])] = (
        "hf://buckets/EdisonScientific/bixbench-hypothesis-capsules"
    )
    """HuggingFace bucket ID or path to benchmark capsules directory."""

    limit: Annotated[int | None, tyro.conf.arg(aliases=["-l"])] = None
    """Maximum number of tasks to convert."""

    overwrite: bool = False
    """Whether existing generated tasks may be replaced."""

    def __post_init__(self):
        self.output_dir = Path(self.output_dir).expanduser()

    @property
    def tasks(self) -> Iterator[HarborBBHTask]:
        if Path(self.data_dir).is_file():
            for source_task in BBHTask.from_jsonl(self.data_dir, self.capsules_dir):
                yield HarborBBHTask(source_task, docker_image=self.docker_image)
        else:
            hf_dataset = load_dataset(self.data_dir, split="train")
            for source_task in BBHTask.from_hf(hf_dataset, self.capsules_dir):
                yield HarborBBHTask(source_task, docker_image=self.docker_image)

    def write(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        gym_rows: list[str] = []
        for task in tqdm(self.tasks, desc="bbh"):
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
    data_dir: str | Path = "EdisonScientific/bixbench_hypothesis",
    capsules_dir: str | Path = "hf://buckets/EdisonScientific/bixbench-hypothesis-capsules",
    limit: int | None = None,
    overwrite: bool = False,
) -> Path:
    """Prepare Harbor tasks and return their Gym JSONL index path."""
    benchmark = HarborBBH(
        docker_image=docker_image,
        output_dir=output_dir,
        data_dir=data_dir,
        capsules_dir=capsules_dir,
        limit=limit,
        overwrite=overwrite,
    )
    benchmark.write()
    return Path(benchmark.output_dir) / "gym.jsonl"


if __name__ == "__main__":
    tyro.cli(HarborBBH).write()
