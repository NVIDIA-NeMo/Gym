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
from typing import Any, List, Optional, Union

from comet.models import download_model, load_from_checkpoint
from fastapi import FastAPI

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class TranslationCometResourcesServerConfig(BaseResourcesServerConfig):
    use_reference: bool = True  # Must match model set in comet_model_name
    comet_model_name: str = "Unbabel/wmt22-comet-da"
    comet_gpu_count: int = None  # CPU only
    comet_gpu_devices: Union[List[int], str, int] = "auto"
    model_cache_dir: Optional[str] = None
    reasoning_split_word: str = "</think>"


class TranslationCometVerifyRequest(BaseVerifyRequest):
    src_txt: str
    trg_txt: Optional[str] = None


class TranslationCometVerifyResponse(BaseVerifyResponse):
    src_txt: str
    trg_txt: Optional[str] = None
    extracted_answer: str


class TranslationCometResourcesServer(SimpleResourcesServer):
    config: TranslationCometResourcesServerConfig
    batch_size: int = 1  # We only process one item at a time so this is always 1

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)

        # # Manually load the model without the Comet wrapper class so we can control the GPU allocation
        # # https://stackoverflow.com/questions/75879866/how-to-load-unbabel-comet-model-without-nested-wrapper-initialization

        # model_path = snapshot_download(repo_id=self.config.comet_model_name)
        # model_checkpoint_path = f'{model_path}/checkpoints/model.ckpt'
        # if self.config.use_reference:
        #     self._comet_model = RegressionMetric.load_from_checkpoint(model_checkpoint_path, layer_transformation='softmax')
        # else:
        #     self._comet_model = ReferencelessRegression.load_from_checkpoint(model_checkpoint_path, layer_transformation='softmax')

        model_path = download_model(model=self.config.comet_model_name, saving_directory=self.config.model_cache_dir)
        self._comet_model = load_from_checkpoint(model_path)

        # TODO long-term we want to pull logic out of predict in base.py in COMET
        # so that we keep one PTL Trainer for the whole server, and don't make a new one for every datapoint
        # since each PTL trainer moves the model weights from CPU to GPU each time which will be slow

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        # Additional server routes go here! e.g.:
        # app.post("/get_weather")(self.get_weather)

        return app

    async def verify(self, body: TranslationCometVerifyRequest) -> TranslationCometVerifyResponse:
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
            source_text=body.src_txt, target_text=body.trg_txt, model_response=combined_response
        )

        return TranslationCometVerifyResponse(**body.model_dump(), extracted_answer=extracted_answer, reward=reward)

    def _verify_answer(self, source_text: str, target_text: str, model_response: str) -> tuple[float, str]:
        extracted_answer = self._extract_answer(model_response)

        if self.config.use_reference:
            comet_data = [{"src": source_text, "mt": extracted_answer, "ref": target_text}]
        else:
            comet_data = [{"src": source_text, "mt": extracted_answer}]

        # TODO this is inefficent and sets up a new PTL Trainer each time
        # It's designed to be run on a whole dataset at once
        # This means the weights get moved from CPU to GPU (if applicable) each time this is called
        model_output = self._comet_model.predict(
            comet_data,
            batch_size=self.batch_size,
            gpus=self.config.comet_gpu_count,
            devices=self.config.comet_gpu_devices,
            progress_bar=False,
        )
        reward = model_output.system_score

        return reward, extracted_answer

    def _extract_answer(self, model_response: str) -> str:
        # Strip any thinking
        no_think_response = model_response.split(self.config.reasoning_split_word)[-1]
        no_think_response = no_think_response.strip()
        return no_think_response


if __name__ == "__main__":
    TranslationCometResourcesServer.run_webserver()
