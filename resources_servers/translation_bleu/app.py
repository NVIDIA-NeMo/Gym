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
import re
from typing import Any, Dict

from fastapi import FastAPI
from sacrebleu.metrics import BLEU

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class TranslationBleuResourcesServerConfig(BaseResourcesServerConfig):
    pass


class TranslationBleuRequest(BaseRunRequest):
    trg_text: str
    trg_lang: str


class TranslationBleuVerifyRequest(TranslationBleuRequest, BaseVerifyRequest):
    pass


class TranslationBleuVerifyResponse(BaseVerifyResponse):
    trg_text: str
    trg_lang: str
    extracted_answer: str


class TranslationBleuResourcesServer(SimpleResourcesServer):
    config: TranslationBleuResourcesServerConfig

    TOKENIZER_MAP: Dict[str, str] = {
        "zh": "zh",
        "zh-cn": "zh",
        "zh-tw": "zh",
        "zho-CN": "zh",
        "zho_simpl": "zh",
        "ja": "ja-mecab",
        "jpn": "ja-mecab",
        "th": "flores200",
        "ko": "ko-mecab",
    }

    def model_post_init(self, context: Any) -> None:
        # TODO can remove this, need to configure BLEU for every request
        # as it needs a different tokenizer for different languages
        super().model_post_init(context)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        # Additional server routes go here! e.g.:
        # app.post("/get_weather")(self.get_weather)

        return app

    async def verify(self, body: TranslationBleuVerifyRequest) -> TranslationBleuVerifyResponse:
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
            ground_truth=body.trg_text, target_lang=body.trg_lang, model_response=combined_response
        )

        return TranslationBleuVerifyResponse(**body.model_dump(), extracted_answer=extracted_answer, reward=reward)

    def _verify_answer(self, ground_truth: str, target_lang: str, model_response: str) -> tuple[float, str]:
        extracted_answer = self._extract_answer(model_response)

        if target_lang in self.TOKENIZER_MAP:
            tokenize = self.TOKENIZER_MAP[target_lang]
        else:
            tokenize = None
        # Use effective_order for sentence-level BLEU
        bleu = BLEU(trg_lang=target_lang, effective_order=True, tokenize=tokenize)

        # TODO how to handle multiple sentences? bleu.corpus_score expects a list of pre-split sentences
        bleu_output = bleu.sentence_score(extracted_answer, [ground_truth])
        # TODO Do we want to report any other BLEU outputs?
        bleu_score = bleu_output.score
        reward = bleu_score / 100.0

        return reward, extracted_answer

    def _extract_answer(self, model_response: str) -> str:
        # TODO is this necessary?
        # Strip <think> and </think> tags and their content
        no_think_response = re.sub(r"<think>.*?</think>", "", model_response)
        no_think_response = no_think_response.strip()
        return no_think_response


if __name__ == "__main__":
    TranslationBleuResourcesServer.run_webserver()
