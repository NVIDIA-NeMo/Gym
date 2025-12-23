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
"""
  - 0-198: LitQA2 (199 items) - text + paper search
  - 199-379: FigQA (181 items) - figure images
  - 380-623: TableQA (244 items) - table images
"""
import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import litellm
from aviary.core import TaskDataset
from lmi import LiteLLMModel
from paperqa.agents.search import get_directory_index
from paperqa.settings import Settings
from pydantic import model_validator

import aviary.utils
from aviary.envs.labbench import (
    GradablePaperQAEnvironment,
    ImageQATaskDataset,
    LABBenchDatasets,
    TextQATaskDataset,
    TextQATaskSplit,
)
from resources_servers.aviary.app import AviaryResourcesServer

logger = logging.getLogger(__name__)

DEFAULT_PAPER_DIRECTORY = Path(__file__).parent.parent.parent / "labbench_papers"

if not DEFAULT_PAPER_DIRECTORY.exists():
    DEFAULT_PAPER_DIRECTORY = Path(__file__).parent.parent.parent / "paper-qa" / "tests" / "stub_data"


class LABBenchDataset(TaskDataset[GradablePaperQAEnvironment]):
    """
    Aviary currently supports LitQA2, FigQA, TableQA.
    
    LAB-Bench also has SuppQA, ProtocolQA, CloningScenarios, I guess not yet in Aviary.
    """

    def __init__(
        self,
        litqa_dataset: TextQATaskDataset,
        figqa_dataset: ImageQATaskDataset,
        tableqa_dataset: ImageQATaskDataset,
    ):
        self.litqa_dataset = litqa_dataset
        self.figqa_dataset = figqa_dataset
        self.tableqa_dataset = tableqa_dataset

        self.litqa_size = len(litqa_dataset)
        self.figqa_size = len(figqa_dataset)
        self.tableqa_size = len(tableqa_dataset)

        self.figqa_start = self.litqa_size
        self.tableqa_start = self.litqa_size + self.figqa_size

        logger.info(
            f"LABBenchDataset: LitQA2[0-{self.litqa_size-1}], "
            f"FigQA[{self.figqa_start}-{self.figqa_start + self.figqa_size - 1}], "
            f"TableQA[{self.tableqa_start}-{self.tableqa_start + self.tableqa_size - 1}]"
        )

    def __len__(self) -> int:
        return self.litqa_size + self.figqa_size + self.tableqa_size

    def get_new_env_by_idx(self, idx: int) -> GradablePaperQAEnvironment:
        if idx < self.figqa_start:
            return self.litqa_dataset.get_new_env_by_idx(idx)
        elif idx < self.tableqa_start:
            return self.figqa_dataset.get_new_env_by_idx(idx - self.figqa_start)
        else:
            return self.tableqa_dataset.get_new_env_by_idx(idx - self.tableqa_start)

    def iter_batches(self, batch_size: int, shuffle: bool = False) -> Any:
        raise NotImplementedError("Use get_new_env_by_idx directly")


class LABBenchResourcesServer(AviaryResourcesServer[GradablePaperQAEnvironment, LABBenchDataset]):
    @model_validator(mode="before")
    @classmethod
    def create_dataset(cls, data: dict) -> dict:
        global_config = data.get("server_client").global_config_dict

        model_name = global_config.get("policy_model_name", "gpt-4o-mini")
        api_base = global_config.get("policy_base_url")
        api_key = global_config.get("policy_api_key", "dummy")

        litellm_model_name = f"openai/{model_name}"
        aviary.utils.DEFAULT_EVAL_MODEL_NAME = litellm_model_name
        aviary.utils.LLM_BOOL_EVAL_CONFIG["model"] = litellm_model_name
        aviary.utils.LLM_EXTRACT_CONFIG["model"] = litellm_model_name
        aviary.utils.LLM_SCORE_EVAL_CONFIG["model"] = litellm_model_name

        litellm.api_key = api_key
        litellm.api_base = api_base
        litellm.request_timeout = 300

        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_API_BASE"] = api_base

        eval_model = LiteLLMModel(
            name=model_name,
            config={
                "api_base": api_base,
                "api_key": api_key,
            },
        )

        paper_directory = global_config.get("paper_directory")
        if paper_directory:
            paper_directory = Path(paper_directory)
        else:
            paper_directory = DEFAULT_PAPER_DIRECTORY

        from paperqa.settings import AgentSettings, IndexSettings, ParsingSettings

        def make_llm_config(model: str) -> dict:
            return {
                "model_list": [
                    {
                        "model_name": model,
                        "litellm_params": {
                            "model": f"openai/{model}",
                            "api_base": api_base,
                            "api_key": api_key,
                            "timeout": 300,
                        },
                    }
                ]
            }

        settings = Settings(
            llm=model_name,
            llm_config=make_llm_config(model_name),
            summary_llm=model_name,
            summary_llm_config=make_llm_config(model_name),
            agent=AgentSettings(
                agent_llm=model_name,
                agent_llm_config=make_llm_config(model_name),
                index=IndexSettings(
                    paper_directory=paper_directory,
                ),
            ),
            parsing=ParsingSettings(
                defer_embedding=True,
                enrichment_llm=model_name,
                enrichment_llm_config=make_llm_config(model_name),
            ),
        )

        logger.info(f"Building paper index from directory: {paper_directory}")

        def build_index():
            return asyncio.run(get_directory_index(settings=settings, build=True))

        try:
            asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(build_index)
                index = future.result()
        except RuntimeError:
            index = build_index()

        logger.info(f"Paper index built successfully: {index.index_name}")

        logger.info("Creating LitQA2 dataset...")
        litqa_dataset = TextQATaskDataset(
            dataset=LABBenchDatasets.LIT_QA2,
            split=TextQATaskSplit.TRAIN,
            settings=settings,
            eval_model=eval_model,
        )

        logger.info("Creating FigQA dataset...")
        figqa_dataset = ImageQATaskDataset(
            dataset=LABBenchDatasets.FIG_QA,
            split=TextQATaskSplit.TRAIN,
        )

        logger.info("Creating TableQA dataset...")
        tableqa_dataset = ImageQATaskDataset(
            dataset=LABBenchDatasets.TABLE_QA,
            split=TextQATaskSplit.TRAIN,
        )

        data["dataset"] = LABBenchDataset(
            litqa_dataset=litqa_dataset,
            figqa_dataset=figqa_dataset,
            tableqa_dataset=tableqa_dataset,
        )
        return data


if __name__ == "__main__":
    LABBenchResourcesServer.run_webserver()
