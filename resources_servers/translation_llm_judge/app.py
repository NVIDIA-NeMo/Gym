"""
LLM-as-judge resources server.

Compares a model's generated answer to an expected answer using an LLM judge.
The judge prompt is fully configurable via server config.
"""

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
from __future__ import annotations

import re
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)


class TranslationLLMJudgeResourcesServerConfig(BaseResourcesServerConfig):
    """Configuration for the LLM judge server.

    - judge_model_server: target model server to use as the judge.
    - judge_responses_create_params: base create params; input will be set per request.
    - judge_system_message: optional custom system message for the judge.
    - judge_prompt_template: optional custom prompt template. Supported placeholders:
        {generated_text}, {src_text}, {src_lang}, {trg_lang}, {trg_text} (if use_reference is True, otherwise {trg_lang} and {src_lang} are required)
    """

    name: str = "translation_llm_judge"
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming

    judge_system_message: Optional[str] = None
    judge_prompt_template: str
    judge_score_extract_regex: str
    judge_max_score: int
    use_reference: bool = True  # If True, judge_prompt_template should include {trg_text}
    reasoning_split_word: str = "</think>"


class TranslationLLMJudgeRunRequest(BaseRunRequest):
    src_txt: str
    src_lang: str  # TODO should this be optional? Not all judge prompts will use it
    trg_lang: str  # TODO should this be optional? Not all judge prompts will use it
    trg_txt: Optional[str] = None


class TranslationLLMJudgeVerifyRequest(TranslationLLMJudgeRunRequest, BaseVerifyRequest):
    pass


class TranslationLLMJudgeEvaluation(BaseModel):
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse
    # Extracted score from judge output
    score: Optional[float] = None


class TranslationLLMJudgeVerifyResponse(BaseVerifyResponse):
    src_txt: str
    src_lang: str
    trg_lang: str
    trg_txt: Optional[str] = None
    judge_evaluation: TranslationLLMJudgeEvaluation


class TranslationLLMJudgeResourcesServer(SimpleResourcesServer):
    """Judge-only verifier using an LLM to evaluate translation quality."""

    config: TranslationLLMJudgeResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    def _extract_last_assistant_text(self, body: BaseVerifyRequest) -> str:
        """Extract the last assistant message text from the response.

        - If the assistant message has multiple text blocks, they are joined with newlines.
        - If ``extract_regex`` is provided, the last regex match is used; if capture
        groups exist, the first non-empty group is returned, otherwise the full match.
        - Returns an empty string when no assistant text is available.
        """
        # Return only the last assistant message's text content.
        for o in reversed(body.response.output):
            text = ""
            if getattr(o, "type", None) == "message" and getattr(o, "role", None) == "assistant":
                content = getattr(o, "content", None)
                if isinstance(content, list):
                    # Some providers split a single assistant message into multiple text blocks.
                    # Join all text blocks to reconstruct the full message text.
                    texts: list[str] = []
                    for c in content:
                        t = getattr(c, "text", None)
                        if isinstance(t, str):
                            texts.append(t)
                    text = "\n".join(texts).strip()
                elif isinstance(content, str):
                    text = content.strip()

        # Strip thinking if not already removed by reasoning parser
        text = self._strip_thinking(text)
        return text

    def _strip_thinking(self, model_response: str) -> str:
        # Strip any thinking
        no_think_response = model_response.split(self.config.reasoning_split_word)[-1]
        no_think_response = no_think_response.strip()
        return no_think_response

    async def verify(self, body: TranslationLLMJudgeVerifyRequest) -> TranslationLLMJudgeVerifyResponse:
        generated = self._extract_last_assistant_text(body)

        eval = await self._generate_judge_evaluation(
            generated_text=generated,
            src_txt=body.src_txt,
            src_lang=body.src_lang,
            trg_lang=body.trg_lang,
            trg_txt=body.trg_txt,
        )

        payload = body.model_dump()
        reward = eval.score / self.config.judge_max_score
        return TranslationLLMJudgeVerifyResponse(**payload, reward=reward, judge_evaluation=eval)

    async def _generate_judge_evaluation(
        self, *, generated_text: str, src_txt: str, src_lang: str, trg_lang: str, trg_txt: Optional[str] = None
    ) -> TranslationLLMJudgeEvaluation:
        cfg = self.config

        responses_create_params = cfg.judge_responses_create_params.model_copy(deep=True)
        prompt_template = cfg.judge_prompt_template
        system_message = cfg.judge_system_message

        if self.config.use_reference and trg_txt is not None:
            user_prompt = prompt_template.format(
                generated_text=generated_text, src_txt=src_txt, src_lang=src_lang, trg_lang=trg_lang, trg_txt=trg_txt
            )
        else:
            user_prompt = prompt_template.format(
                generated_text=generated_text, src_txt=src_txt, src_lang=src_lang, trg_lang=trg_lang
            )

        msgs: list[NeMoGymEasyInputMessage] = []
        if system_message is not None and system_message != "":
            msgs.append(NeMoGymEasyInputMessage(role="system", content=system_message))
        msgs.append(NeMoGymEasyInputMessage(role="user", content=user_prompt))
        responses_create_params.input = msgs

        response = await self.server_client.post(
            server_name=cfg.judge_model_server.name,
            url_path="/v1/responses",
            json=responses_create_params,
        )
        judge_response = NeMoGymResponse.model_validate(await response.json())
        eval_record = TranslationLLMJudgeEvaluation(
            responses_create_params=responses_create_params,
            response=judge_response,
            score=0.0,
        )

        # Parse the last output; fall back to a score of 0 if unexpected.
        try:
            last_output = judge_response.output[-1]
            if getattr(last_output, "type", None) != "message":
                return eval_record
            last_content = last_output.content[-1]
            text = getattr(last_content, "text", "")
        except Exception:
            return eval_record

        # Extract the score from the judge output
        match = re.search(cfg.judge_score_extract_regex, text)
        if match is None:
            return eval_record
        score = int(match.group(1))
        eval_record.score = score
        return eval_record


if __name__ == "__main__":
    TranslationLLMJudgeResourcesServer.run_webserver()
