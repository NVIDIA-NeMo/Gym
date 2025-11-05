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
import os
from pathlib import Path
from typing import Any, Optional

import datasets
import ray
import transformers
from fastapi import FastAPI
from metricx24.models import MT5ForRegression

from nemo_gym import CACHE_DIR
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.ray_utils import (
    spinup_ray_gpu_workers,
)


@ray.remote
class TranslationMetricxModelWorker:
    def __init__(self):
        self.model_name = None
        self.device_map = None
        self.output_dir = None
        self.model = None

    def _load_model(self, model_name, device_map, output_dir):
        self.model_name = model_name
        self.device_map = device_map
        self.output_dir = output_dir

        # Load model with device placement
        model = MT5ForRegression.from_pretrained(
            self.model_name, torch_dtype="auto", device_map=self.device_map
        )
        # Inputs should go to the device where the first layer is
        # Get device from the first model parameter
        self._inputs_device = next(model.parameters()).device

        model.eval()
        self.model = model

        # Create trainer
        training_args = transformers.TrainingArguments(
            output_dir=output_dir,
            per_device_eval_batch_size=1,
            dataloader_pin_memory=False,
        )
        trainer = transformers.Trainer(
            model=model,
            args=training_args,
        )
        self.trainer = trainer

        return self._inputs_device

    def predict(self, *args, **kwargs):
        return self.trainer.predict(*args, **kwargs)


class TranslationMetricxResourcesServerConfig(BaseResourcesServerConfig):
    """
    Configuration for the TranslationMetricxResourcesServer.

    Attributes:
        use_reference (bool): Whether to use a reference translation
        metricx_model_name (str): The MetricX model name to use. The default "google/metricx-24-hybrid-large-v2p6-bfloat16"
            is the smallest model at 1.2B parameters.
        tokenizer_name (str): The name of the mT5 tokenizer to use with the MetricX model. Size must match MetricX model.
            For the default model above, use "google/mt5-large".
        device_map (str): Device placement for the model. Options include "cpu", specific GPU (e.g., "cuda:1"),
            "auto", "balanced", "balanced_low_0", "sequential".
        max_input_length (int): Maximum input sequence length (see MetricX documentation, default 1536)
        output_dir (str): Output directory for Trainer class. Nothing is actually output during prediction, but it's mandatory to supply.
    """

    use_reference: bool = True
    metricx_model_name: str = "google/metricx-24-hybrid-large-v2p6-bfloat16"
    tokenizer_name: str = "google/mt5-large"
    device_map: str = "cpu"
    max_input_length: int = 1536
    output_dir: str = str(Path(CACHE_DIR) / "metricx_output")


class TranslationMetricxVerifyRequest(BaseVerifyRequest):
    src_txt: str
    trg_txt: Optional[str] = None


class TranslationMetricxVerifyResponse(BaseVerifyResponse):
    src_txt: str
    trg_txt: Optional[str] = None
    extracted_answer: str


class TranslationMetricxResourcesServer(SimpleResourcesServer):
    config: TranslationMetricxResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)

        # Load tokenizer (MetricX models use MT5 tokenizers, separate from the model name)
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        self._tokenizer = tokenizer

        # Ensure output directory exists (following predict.py lines 167-169)
        os.makedirs(self.config.output_dir, exist_ok=True)

        model_workers = spinup_ray_gpu_workers(TranslationMetricxModelWorker)
        inputs_device = None
        for model_worker in model_workers:
            # Load model with device placement
            inputs_device = ray.get(model_worker._load_model.remote(
                self.config.metricx_model_name,
                self.config.device_map,
                self.config.output_dir,
            ))
        self._model_workers = model_workers
        self._inputs_device = inputs_device

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        # Additional server routes go here! e.g.:
        # app.post("/get_weather")(self.get_weather)

        return app

    async def verify(self, body: TranslationMetricxVerifyRequest) -> TranslationMetricxVerifyResponse:
        assistant_responses = []
        for output_item in body.response.output:
            if output_item.type != "message":
                continue

            for content_item in output_item.content:
                if content_item.type != "output_text":
                    continue

                assistant_responses.append(content_item.text)

        combined_response = "".join(assistant_responses)

        (reward, extracted_answer) = self._verify_answer(
            model_response=combined_response, source_text=body.src_txt, target_text=body.trg_txt
        )

        return TranslationMetricxVerifyResponse(**body.model_dump(), extracted_answer=extracted_answer, reward=reward)

    def _verify_answer(
        self, model_response: str, source_text: str, target_text: Optional[str] = None
    ) -> tuple[float, str]:
        extracted_answer = self._extract_answer(model_response)
        ds = self._create_dataset_from_example(extracted_answer, source_text, target_text)
        predictions, _, _ = ray.get(self._model_workers[0].predict.remote(test_dataset=ds))
        score = float(predictions[0])

        # MetricX scores are between 0 and 25, where 25 is worst, so we normalize to 0 to 1 where 0 is worst
        reward = (25 - score) / 25.0
        return reward, extracted_answer

    def _create_dataset_from_example(
        self, hypothesis: str, source_text: str, reference_text: Optional[str] = None
    ) -> datasets.Dataset:
        """Create a dataset from a single example, following get_dataset logic from predict.py."""
        # Create input string based on reference ("QE") mode (QE is when use_reference is False)
        if not self.config.use_reference or reference_text is None:
            input_text = "source: " + source_text + " candidate: " + hypothesis
        else:
            input_text = "source: " + source_text + " candidate: " + hypothesis + " reference: " + reference_text

        # Tokenize (returns dict with lists)
        tokenized = self._tokenizer(
            input_text,
            max_length=self.config.max_input_length,
            truncation=True,
            padding=False,
        )

        # Create a single example dict (matching the format that predict.py creates)
        example_dict = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }

        # If last token is EOS, remove it (following predict.py _remove_eos function logic)
        eos_token_id = self._tokenizer.eos_token_id
        if eos_token_id is not None and example_dict["input_ids"][-1] == eos_token_id:
            example_dict["input_ids"] = example_dict["input_ids"][:-1]
            example_dict["attention_mask"] = example_dict["attention_mask"][:-1]

        # Create dataset from a list of examples (each example is one dict)
        # Following predict.py structure: ds.map() operations create per-example dicts
        ds = datasets.Dataset.from_list([example_dict])

        # Set format to torch and move to device (following predict.py line 119-124)
        ds.set_format(
            type="torch",
            columns=["input_ids", "attention_mask"],
            device=self._inputs_device,
            output_all_columns=True,
        )

        return ds

    def _extract_answer(self, model_response: str) -> str:
        # Strip any thinking
        no_think_response = model_response.split("</think>")[-1]
        no_think_response = no_think_response.strip()
        return no_think_response


if __name__ == "__main__":
    TranslationMetricxResourcesServer.run_webserver()
