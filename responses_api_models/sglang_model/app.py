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
"""NeMo-Gym Responses-API model server for SGLang-served policies.

Why this exists
---------------
RL trainers (e.g. GRPO) need the *exact* token ids the policy emitted and their logprobs.
`vllm_model` recovers those by parsing `token_id:NNN` logprob tokens out of
`/v1/chat/completions`, which is a vLLM-specific encoding that SGLang does not produce.

Two transports
--------------
``transport: chat`` (default, **requires sglang >= 0.5.13**)
    Drives SGLang's OpenAI-compatible ``/v1/chat/completions`` and asks for the training
    metadata via the native ``return_meta_info`` / ``return_prompt_token_ids`` request
    extensions (added by the TITO series, in the tree as of 0.5.13). Ids and logprobs come
    back on each choice as ``meta_info.output_token_logprobs`` / ``prompt_token_ids``.

    This is the preferred path: everything except the token extraction is the inherited
    ``VLLMModel`` behavior, so tool-call parsing, prompt templating, sampling params, auth
    and context-overflow handling are all done *server-side* by SGLang -- there is no local
    tokenizer to drift from the server's, and no client-side reimplementation to keep in sync.

``transport: generate`` (fallback)
    Drives SGLang's native ``/generate`` with ``return_logprob=true``, tokenizing the prompt
    locally. Needed only for SGLang builds/forks predating chat-side TITO. It cannot parse
    tool calls and it tokenizes with a *local* copy of the chat template, so it is off by
    construction if that copy differs from the server's. Prefer ``chat`` whenever possible.

The pure request/response transforms live in `_logic.py` (unit-tested in tests/).
"""

from time import time
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from fastapi import Body, Request

from nemo_gym.openai_utils import (
    NeMoGymAsyncOpenAI,
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
from responses_api_models.sglang_model._logic import (
    build_sampling_params,
    cap_to_context,
    extract_generated_tokens_and_logprobs,
    normalize_token_ids,
    unsupported_sampling_params,
    would_truncate,
)
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


# Chat-side TITO (`return_meta_info` / `return_prompt_token_ids` on /v1/chat/completions)
# landed in the sglang-miles sync series and is present in the 0.5.13 release tree.
MIN_SGLANG_VERSION_FOR_CHAT_TRANSPORT = "0.5.13"


class SGLangModelConfig(VLLMModelConfig):
    # `chat` requires sglang >= 0.5.13; `generate` is the fallback for older builds/forks.
    transport: Literal["chat", "generate"] = "chat"

    # --- `generate` transport only (ignored when transport == "chat") ---
    # Used only when the request carries no max_(completion_)tokens.
    default_max_new_tokens: int = 1024
    # Opt-in, matching the rest of the repo: executing model-repo code is the caller's decision.
    trust_remote_code: bool = False
    add_generation_prompt: bool = True
    # SGLang context window. Keep in sync with policy.max_total_sequence_length:
    # the prompt is truncated and max_new_tokens shrunk so input_len + max_new_tokens < ctx,
    # else SGLang /generate returns 400.
    context_length: int = 4096


class SGLangModel(VLLMModel):
    config: SGLangModelConfig

    def _post_init(self) -> None:
        super()._post_init()
        if self.config.transport != "generate":
            return
        # Local tokenization is only needed by the /generate fallback. Imported lazily so the
        # default `chat` transport does not pay for (or depend on) transformers at startup.
        from transformers import AutoTokenizer

        # The model name must be the exact path/revision the SGLang server was launched with;
        # any divergence silently conditions generation on ids from the wrong template.
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model, trust_remote_code=self.config.trust_remote_code
        )
        # Bare SGLang server base url(s); we hit `{base}/generate`, not `/v1/...`.
        self._sglang_urls: List[str] = [u.rstrip("/") for u in self.config.base_url]

    # ------------------------------------------------------------------
    # `chat` transport: inherit everything, override only token extraction
    # ------------------------------------------------------------------

    def _preprocess_chat_completion_create_params(self, request: Request, body_dict: Dict[str, Any]) -> Dict[str, Any]:
        body_dict = super()._preprocess_chat_completion_create_params(request, body_dict)
        if self.config.transport == "chat" and self.config.return_token_id_information:
            # vLLM-only knob: SGLang has no `token_id:NNN` logprob encoding, and leaving it set
            # would be a silent no-op that misrepresents where the ids come from.
            body_dict.pop("return_tokens_as_token_ids", None)
            # SGLang's native request extensions for the training metadata.
            body_dict["return_meta_info"] = True
            body_dict["return_prompt_token_ids"] = True
        return body_dict

    async def _attach_token_id_information(
        self, choice_dict: Dict[str, Any], body_dict: Dict[str, Any], client: NeMoGymAsyncOpenAI
    ) -> None:
        """Read the ids/logprobs SGLang already returned, instead of vLLM's `token_id:` parse.

        No `/tokenize` round-trip is needed: `return_prompt_token_ids` makes SGLang report the
        prompt ids it actually tokenized, which is authoritative in a way a local tokenizer
        cannot be.
        """
        # An aborted generation is a truncated fragment, not a completion. Letting it through
        # would put a poisoned rollout into the GRPO batch looking like a normal `stop`.
        if choice_dict.get("finish_reason") == "abort":
            raise RuntimeError(
                f"`{self.config.name}`: SGLang reported finish_reason='abort' (generation was "
                "cancelled server-side, e.g. preemption or a shutdown). Refusing to emit a "
                "partial rollout as a completed one."
            )

        generation_token_ids, generation_log_probs = extract_generated_tokens_and_logprobs(choice_dict)

        prompt_token_ids: Optional[List[int]] = choice_dict.get("prompt_token_ids")
        if prompt_token_ids is None:
            raise RuntimeError(
                f"`{self.config.name}` requested prompt token ids from SGLang "
                "(return_token_id_information=True, so return_prompt_token_ids=True was sent), "
                f"but the response carried none (choice keys={sorted(choice_dict)}). This server "
                f"requires sglang >= {MIN_SGLANG_VERSION_FOR_CHAT_TRANSPORT} for "
                "transport='chat'; set transport='generate' for older builds."
            )

        choice_dict["message"].update(
            dict(
                prompt_token_ids=[int(t) for t in prompt_token_ids],
                generation_token_ids=generation_token_ids,
                generation_log_probs=generation_log_probs,
            )
        )

        # Clean the duplicated / non-OpenAI information so the response validates.
        choice_dict.pop("logprobs", None)
        choice_dict.pop("prompt_token_ids", None)
        choice_dict.pop("meta_info", None)

    # ------------------------------------------------------------------
    # `generate` transport: fallback for builds without chat-side TITO
    # ------------------------------------------------------------------

    async def chat_completions(
        self, request: Request, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        if self.config.transport == "chat":
            return await super().chat_completions(request, body)
        return await self._chat_completions_via_generate(request, body)

    async def _chat_completions_via_generate(
        self, request: Request, body: NeMoGymChatCompletionCreateParamsNonStreaming
    ) -> NeMoGymChatCompletion:
        """Drive SGLang's native /generate, tokenizing the prompt locally.

        Only for SGLang builds without chat-side TITO. Known gaps vs. the `chat` transport:
        tool calls are not parsed (and tool schemas are rendered only if the local chat
        template does so), and the prompt is tokenized locally rather than by the server.
        """
        body_dict = body.model_dump(exclude_unset=True)
        body_dict = self._preprocess_chat_completion_create_params(request, body_dict)
        messages = body_dict["messages"]

        ignored = unsupported_sampling_params(body_dict)
        if ignored:
            print(
                f"[sglang_model] transport='generate' cannot honor {ignored}; the realized "
                "sampling distribution will differ from the configured recipe. "
                "Use transport='chat' (sglang >= 0.5.13) for full parameter support.",
                flush=True,
            )

        # 1) prompt token ids via the model's own chat template (local tokenizer). Use the
        # *merged* kwargs the inherited preprocess produced, so per-request overrides in
        # `metadata.chat_template_kwargs` (e.g. per-sample reasoning on/off) are honored --
        # reading self.config here would tokenize with the wrong template.
        ct_kwargs = body_dict.get("chat_template_kwargs") or {}
        rendered = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=self.config.add_generation_prompt,
            tokenize=True,
            return_dict=False,
            # Let the template render tool schemas into the prompt when the caller sent them.
            tools=body_dict.get("tools"),
            **ct_kwargs,
        )
        prompt_token_ids = normalize_token_ids(rendered)

        # 2) cap to the context window, then generate via SGLang native /generate.
        sampling_params = build_sampling_params(body_dict, self.config.default_max_new_tokens)
        truncated = would_truncate(prompt_token_ids, self.config.context_length)
        prompt_token_ids, sampling_params = cap_to_context(
            prompt_token_ids, sampling_params, self.config.context_length
        )
        if truncated:
            # Truncation keeps the prompt head, dropping the newest turn and the generation
            # cue. Match the inherited overflow behavior instead of emitting a rollout that
            # looks valid: an empty completion with finish_reason="length" is filterable.
            print(
                f"[sglang_model] prompt exceeded context_length={self.config.context_length}; "
                "returning an empty completion with finish_reason='length'.",
                flush=True,
            )
            res = self._create_empty_chat_completion()
            res.choices[0].finish_reason = "length"
            return res

        payload = {
            "input_ids": prompt_token_ids,
            "sampling_params": sampling_params,
            "return_logprob": True,
            "logprob_start_len": -1,
        }
        sid = request.session.get(SESSION_ID_KEY, "") if hasattr(request, "session") else ""
        url = f"{self._sglang_urls[hash(sid) % len(self._sglang_urls)]}/generate"
        # SGLang applies its --api-key middleware to /generate as well as /v1/*, so an
        # authenticated deployment 401s without this. Only set when non-empty: `request()`
        # does `kwargs.setdefault("headers", ...)`, which would keep an explicit None.
        extra_request_kwargs: Dict[str, Any] = {}
        if self.config.api_key:
            extra_request_kwargs["headers"] = {"Authorization": f"Bearer {self.config.api_key}"}
        # Use NeMo-Gym's pooled aiohttp client. Raw aiohttp + native raise_for_status
        # trips the framework's exception_handling_middleware (it requires the escaping
        # exception to carry `response_content`).
        resp = await ng_request("POST", url, json=payload, **extra_request_kwargs)
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

        meta = result.get("meta_info", {}) or {}
        finish = meta.get("finish_reason")
        if isinstance(finish, dict):
            finish = finish.get("type")
        if finish == "abort":
            raise RuntimeError(
                f"`{self.config.name}`: SGLang reported finish_reason='abort' (generation was "
                "cancelled server-side). Refusing to emit a partial rollout as a completed one."
            )
        finish_reason = "length" if finish == "length" else "stop"

        gen_token_ids, gen_log_probs = extract_generated_tokens_and_logprobs(result)
        # generation_token_ids stay RAW (incl. EOS/special — the policy generated them and
        # we train on them), but the assistant *content* the verifier grades must be clean,
        # matching vLLM's server-side decode. A trailing special token otherwise breaks
        # strict parsers (e.g. structured_outputs json.loads).
        gen_text = self._tokenizer.decode(gen_token_ids, skip_special_tokens=True)

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
