# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""NeMo-Gym Responses-API model server backed by an SGLang `/generate` endpoint.

Why this exists
---------------
Some SGLang-served models (e.g. custom forks) do NOT expose `token_id:NNN` logprob
tokens through the OpenAI `/v1/chat/completions` path, so the stock `vllm_model` server
cannot recover the exact generated token ids + logprobs that RL trainers (e.g. GRPO) need.

SGLang's *native* `/generate` (with `return_logprob=true`) returns the generated token
ids and their logprobs directly. This adapter subclasses `VLLMModel` and overrides ONLY
`chat_completions`:
  1. render the prompt with the model's local HF chat template -> prompt_token_ids,
  2. POST `{base_url}/generate` with those input_ids -> generation token ids + logprobs,
  3. attach prompt_token_ids / generation_token_ids / generation_log_probs to the
     assistant message, exactly as the vLLM path does.
Everything else (Responses<->ChatCompletions conversion, `responses()`, postprocess,
the training-class upgrade in `postprocess_assistant_message_dict`) is inherited.

The pure request/response transforms live in `_logic.py` (unit-tested in tests/).
"""

import os
import sys
from time import time
from typing import Any, Dict, List
from uuid import uuid4

from fastapi import Body, Request

from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
)
from nemo_gym.server_utils import (
    SESSION_ID_KEY,
    get_response_json,
    is_nemo_gym_fastapi_entrypoint,
)
from nemo_gym.server_utils import (
    request as ng_request,
)


try:
    from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig
except ImportError:  # ensure the Gym repo root is importable for the cross-server import
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig

from transformers import AutoTokenizer

from responses_api_models.sglang_model._logic import (
    build_sampling_params,
    cap_to_context,
    extract_generated_tokens_and_logprobs,
    normalize_token_ids,
)


class SGLangModelConfig(VLLMModelConfig):
    # Used only when the request carries no max_(completion_)tokens.
    default_max_new_tokens: int = 1024
    trust_remote_code: bool = True
    add_generation_prompt: bool = True
    # SGLang context window. Keep in sync with policy.max_total_sequence_length:
    # the prompt is truncated to ctx-1 and max_new_tokens shrunk so
    # input_len + max_new_tokens < ctx, else SGLang /generate returns 400.
    context_length: int = 4096


class SGLangModel(VLLMModel):
    config: SGLangModelConfig

    def _post_init(self) -> None:
        super()._post_init()
        # The model name is a local path with tokenizer + chat_template.jinja.
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model, trust_remote_code=self.config.trust_remote_code
        )
        # Bare SGLang server base url(s); we hit `{base}/generate`, not `/v1/...`.
        self._sglang_urls: List[str] = [u.rstrip("/") for u in self.config.base_url]

    async def chat_completions(
        self, request: Request, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        body_dict = body.model_dump(exclude_unset=True)
        body_dict = self._preprocess_chat_completion_create_params(request, body_dict)
        messages = body_dict["messages"]

        # 1) prompt token ids via the model's own chat template (local tokenizer).
        ct_kwargs = self.config.chat_template_kwargs or {}
        rendered = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=self.config.add_generation_prompt,
            tokenize=True,
            return_dict=False,
            **ct_kwargs,
        )
        prompt_token_ids = normalize_token_ids(rendered)

        # 2) cap to the context window, then generate via SGLang native /generate.
        sampling_params = build_sampling_params(body_dict, self.config.default_max_new_tokens)
        prompt_token_ids, sampling_params = cap_to_context(
            prompt_token_ids, sampling_params, self.config.context_length
        )
        payload = {
            "input_ids": prompt_token_ids,
            "sampling_params": sampling_params,
            "return_logprob": True,
            "logprob_start_len": -1,
        }
        sid = request.session.get(SESSION_ID_KEY, "") if hasattr(request, "session") else ""
        url = f"{self._sglang_urls[hash(sid) % len(self._sglang_urls)]}/generate"
        # Use NeMo-Gym's pooled aiohttp client. Raw aiohttp + native raise_for_status
        # trips the framework's exception_handling_middleware (it requires the escaping
        # exception to carry `response_content`).
        resp = await ng_request("POST", url, json=payload)
        if not resp.ok:
            content = await resp.read()
            print(
                f"[sglang_model] SGLang /generate -> {resp.status}; "
                f"input_len={len(prompt_token_ids)} max_new_tokens={sampling_params['max_new_tokens']}; "
                f"body={content[:800]!r}",
                flush=True,
            )
            try:
                resp.raise_for_status()
            except Exception as e:
                e.response_content = content  # satisfy nemo_gym exception middleware
                raise
        result = await get_response_json(resp)

        gen_token_ids, gen_log_probs = extract_generated_tokens_and_logprobs(result)
        # generation_token_ids stay RAW (incl. EOS/special — the policy generated them and
        # we train on them), but the assistant *content* the verifier grades must be clean,
        # matching vLLM's server-side decode. A trailing special token otherwise breaks
        # strict parsers (e.g. structured_outputs json.loads).
        gen_text = self._tokenizer.decode(gen_token_ids, skip_special_tokens=True)

        meta = result.get("meta_info", {}) or {}
        finish = meta.get("finish_reason")
        if isinstance(finish, dict):
            finish = finish.get("type")
        finish_reason = "length" if finish == "length" else "stop"

        # 3) OpenAI chat.completion dict + the training token fields on the assistant message.
        chat_completion_dict: Dict[str, Any] = {
            "id": f"chtcmpl-{uuid4().hex}",
            "object": "chat.completion",
            "created": int(time()),
            "model": self.config.model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": {"role": "assistant", "content": gen_text},
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_token_ids),
                "completion_tokens": len(gen_token_ids),
                "total_tokens": len(prompt_token_ids) + len(gen_token_ids),
            },
        }
        if self.config.return_token_id_information:
            chat_completion_dict["choices"][0]["message"].update(
                prompt_token_ids=prompt_token_ids,
                generation_token_ids=gen_token_ids,
                generation_log_probs=gen_log_probs,
            )
        return NeMoGymChatCompletion.model_validate(chat_completion_dict)


if __name__ == "__main__":
    SGLangModel.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = SGLangModel.run_webserver()  # noqa: F401
