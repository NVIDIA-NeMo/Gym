# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Score continuation likelihoods through vLLM."""

import asyncio
import math

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


class LogLikelihoodRequest(BaseModel):
    context: str = Field(min_length=1)
    continuations: list[str] = Field(min_length=2, max_length=26)


class ContinuationScore(BaseModel):
    continuation: str
    logprob: float = Field(allow_inf_nan=False, strict=True)
    context_tokens: int
    continuation_token_ids: list[int]
    token_logprobs: list[float]


class LogLikelihoodResponse(BaseModel):
    model: str
    scores: list[ContinuationScore]


class VLLMLogLikelihoodConfig(VLLMModelConfig):
    max_context_tokens: int = Field(default=16384, ge=2)
    likelihood_seed: int = 42
    likelihood_add_special_tokens: bool = False


class VLLMLogLikelihoodModel(VLLMModel):
    """Extend the vLLM model server with continuation scoring."""

    config: VLLMLogLikelihoodConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/loglikelihood")(self.loglikelihood)
        return app

    async def loglikelihood(self, request: Request, body: LogLikelihoodRequest) -> LogLikelihoodResponse:
        client = self._resolve_client(request)
        prompt = body.context
        if not prompt.strip() or any(not continuation for continuation in body.continuations):
            raise ValueError("Likelihood needs a non-whitespace context and nonempty continuations")
        if self.config.render_chat_template:
            prompt = self._render_messages_via_chat_template({"messages": [{"role": "user", "content": prompt}]})
        context = prompt.rstrip()
        suffix = prompt[len(context) :]
        encoded = await client.create_tokenize(
            model=self.config.model, prompt=context, add_special_tokens=self.config.likelihood_add_special_tokens
        )
        context_ids = encoded["tokens"]
        if not context_ids:
            raise ValueError("Tokenizer returned an empty context")

        async def score(continuation: str) -> ContinuationScore:
            full = await client.create_tokenize(
                model=self.config.model,
                prompt=context + suffix + continuation,
                add_special_tokens=self.config.likelihood_add_special_tokens,
            )
            token_ids = full["tokens"]
            if token_ids[: len(context_ids)] != context_ids or len(token_ids) <= len(context_ids):
                raise ValueError("Context/continuation token boundary is ambiguous; refusing an incorrect score")
            if len(token_ids) + 1 > self.config.max_context_tokens:
                raise ValueError("Likelihood prompt exceeds max_context_tokens; increase the documented context limit")
            result = await client.create_completion(
                model=self.config.model,
                prompt=token_ids,
                temperature=0,
                max_tokens=1,
                logprobs=1,
                echo=True,
                seed=self.config.likelihood_seed,
            )
            choices = result.get("choices", [])
            if len(choices) != 1:
                raise ValueError("Expected exactly one echoed completion per continuation")
            probabilities = (choices[0].get("logprobs") or {}).get("token_logprobs", [])
            if len(probabilities) != len(token_ids) + 1:
                raise ValueError("vLLM did not return complete echoed prompt logprobs plus one generated token")
            selected = probabilities[len(context_ids) : len(token_ids)]
            if any(type(value) not in (float, int) or not math.isfinite(value) for value in selected):
                raise ValueError("Missing/nonfinite continuation logprobs cannot be scored")
            return ContinuationScore(
                continuation=continuation,
                logprob=sum(selected),
                context_tokens=len(context_ids),
                continuation_token_ids=token_ids[len(context_ids) :],
                token_logprobs=selected,
            )

        scores = await asyncio.gather(*(score(continuation) for continuation in body.continuations))
        return LogLikelihoodResponse(model=self.config.model, scores=scores)


if __name__ == "__main__":  # pragma: no cover
    VLLMLogLikelihoodModel.run_webserver()
