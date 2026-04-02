# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Extends vllm_model with /v1/completions for Atropos envs that need completions"""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import Body, FastAPI, Request

from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


logger = logging.getLogger(__name__)


class AtroposModel(VLLMModel):
    config: VLLMModelConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/v1/completions")(self.completions)
        return app

    async def completions(self, request: Request, body: Dict[str, Any] = Body()) -> Dict[str, Any]:
        client = self._resolve_client(request)
        body["model"] = self.config.model
        body.setdefault("logprobs", 1)

        response_dict = await client.create_completion(**body)

        if self.config.return_token_id_information:
            prompt = body.get("prompt", "")
            tokenize_kwargs = {"model": self.config.model}
            if isinstance(prompt, str):
                tokenize_kwargs["prompt"] = prompt
            elif isinstance(prompt, dict) and "prompt_token_ids" in prompt:
                tokenize_kwargs["prompt"] = prompt
            else:
                tokenize_kwargs["prompt"] = str(prompt)

            tokenize_response = await client.create_tokenize(**tokenize_kwargs)
            prompt_token_ids = tokenize_response.get("tokens", [])

            for choice_dict in response_dict.get("choices", []):
                logprobs_data = choice_dict.get("logprobs") or {}
                token_logprobs = logprobs_data.get("token_logprobs") or []

                generation_log_probs = [lp if lp is not None else 0.0 for lp in token_logprobs]
                generation_token_ids = []

                text = choice_dict.get("text", "")
                if text:
                    full_prompt = (prompt if isinstance(prompt, str) else "") + text
                    full_response = await client.create_tokenize(**{"model": self.config.model, "prompt": full_prompt})
                    full_tokens = full_response.get("tokens", [])
                    if len(full_tokens) > len(prompt_token_ids):
                        generation_token_ids = full_tokens[len(prompt_token_ids) :]

                if len(generation_log_probs) > len(generation_token_ids):
                    generation_log_probs = generation_log_probs[: len(generation_token_ids)]
                elif len(generation_log_probs) < len(generation_token_ids):
                    generation_log_probs += [0.0] * (len(generation_token_ids) - len(generation_log_probs))

                choice_dict["prompt_token_ids"] = prompt_token_ids
                choice_dict["generation_token_ids"] = generation_token_ids
                choice_dict["generation_log_probs"] = generation_log_probs

        return response_dict


if __name__ == "__main__":
    AtroposModel.run_webserver()
