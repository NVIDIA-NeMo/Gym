# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
import shutil
import zipfile
from ast import literal_eval
from pathlib import Path
from tempfile import mkdtemp
from typing import cast

import huggingface_hub
from datasets import Dataset, load_dataset
from pydantic import Field
from tqdm import tqdm

from aviary.core import EvalAnswerMode, Message, Messages, TaskDataset, Tool, eval_answer
from aviary.envs.notebook import NBEnvironment
from aviary.envs.notebook.utils import NBLanguage
from resources_servers.aviary.app import AviaryResourcesServer


logger = logging.getLogger(__name__)

CAPSULE_DATA_LOCATION = os.getenv("BIXBENCH_CAPSULE_DATA_LOCATION", "~/bixbench_capsules")


class BixBenchEnv(NBEnvironment):
    def __init__(self, *args, question: str, answer: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.question = question
        self.answer = answer

    # TODO: figure out why we can't pass optional args anymore
    async def edit_cell(self, contents: str, idx: int | None) -> str:
        """Edit the notebook by modifying a specific code cell.

        ONLY CODE CELLS ARE SUPPORTED. Do no attempt to write Markdown or raw text,
        though you are permitted (and encouraged) to write comments in the code cells.
        The cell will be automatically rerun if a successful edit is made.

        WRITE SHORT TO MEDIUM LENGTH CELLS FOR EASE OF DEBUGGING.

        Args:
            contents: Cell contents to insert. We assume the cell is a code block.
            idx: Index of the cell to edit. If None, appends a new cell.
        """
        return await super().edit_cell(contents, idx)

    async def reset(self) -> tuple[Messages, list[Tool]]:
        obs, tools = await super().reset()
        tools.append(Tool.from_function(self.submit_solution))

        obs = [
            Message(
                content="Using the available tools and data, write a jupyter notebook "
                f"to solve the following problem: {self.question!r}"
            ),
            *obs,
        ]

        return obs, tools

    async def submit_solution(self, solution: str) -> str:
        """Submit your solution to the given task.

        Note that this tool may only be called once and ends the episode.

        Args:
            solution: Your solution to the problem
        """
        # TODO: support various eval modes
        self.state.answer = solution
        self.state.done = True

        score = await eval_answer(
            proposed=solution,
            correct=self.answer,
            question=self.question,
            eval_mode=EvalAnswerMode.LLM,
            llm_eval_config={"name": "gpt-5-mini"},
        )
        correct = score == 1.0
        self.state.total_reward += 1.0 if correct else 0.0

        return "Correct" if correct else "Incorrect"


class BixBenchDataset(TaskDataset[BixBenchEnv]):
    def __init__(self, split: str = "train"):
        bixbench_repo_id = "futurehouse/BixBench"
        dataset = cast(Dataset, load_dataset(bixbench_repo_id, split=split))
        self.dataset: list[dict] = []
        for row in dataset:
            for question in literal_eval(row["questions"]):
                self.dataset.append(
                    {
                        "uuid": row["uuid"],
                        "question": question["question"],
                        "answer": question["ideal_answer"],
                    }
                )

        self.capsule_path = Path(CAPSULE_DATA_LOCATION).expanduser()
        if not self.capsule_path.exists():
            logger.warning(f"Downloading BixBench dataset to {self.capsule_path}, this may take a while...")
            self.capsule_path.mkdir(parents=True, exist_ok=True)
            huggingface_hub.snapshot_download(
                repo_id=bixbench_repo_id,
                repo_type="dataset",
                local_dir=self.capsule_path,
            )
            logger.warning(f"Extracting BixBench dataset to {self.capsule_path}, this may take a while...")
            for path in tqdm(list(self.capsule_path.glob("*.zip")), desc="Extracting"):
                with zipfile.ZipFile(path, "r") as zip_ref:
                    zip_ref.extractall(self.capsule_path)
                path.unlink()

                maybe_folder_dir = self.capsule_path / path.stem
                if maybe_folder_dir.is_dir():
                    # move all contents one level up and remove the folder
                    for item in maybe_folder_dir.glob("Capsule*"):
                        shutil.copytree(item, self.capsule_path / item.name)
                    shutil.rmtree(maybe_folder_dir)

    def __len__(self):
        return len(self.dataset)

    def get_new_env_by_idx(self, idx: int) -> BixBenchEnv:
        row = self.dataset[idx]
        uuid = row["uuid"]

        capsule_path = self.capsule_path / f"CapsuleData-{uuid}"

        notebook_name = NBEnvironment.NOTEBOOK_NAME
        # Define local capsule directory
        problem_dir = Path(mkdtemp())
        problem_dir.mkdir(parents=True, exist_ok=True)

        # Copy capsule contents to local directory
        for item in capsule_path.iterdir():
            if item.is_dir():
                shutil.copytree(item, problem_dir / item.name)
            else:
                shutil.copy(item, problem_dir)

        nb_path = problem_dir / notebook_name

        return BixBenchEnv(
            question=row["question"],
            answer=row["answer"],
            work_dir=problem_dir,
            nb_path=nb_path,
            language=NBLanguage.PYTHON,
            use_tmp_work_dir=False,
            use_docker=True,
        )


class NotebookResourcesServer(AviaryResourcesServer[BixBenchEnv, BixBenchDataset]):
    dataset: BixBenchDataset = Field(default_factory=lambda: BixBenchDataset())


if __name__ == "__main__":
    NotebookResourcesServer.run_webserver()
