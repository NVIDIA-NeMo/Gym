<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# vLLM continuation likelihoods

This native model server subclasses `vllm_model`; standard Responses generation
is unchanged. `/loglikelihood` accepts `context` and ordered `continuations`, and
returns their summed conditional token log probabilities plus token-level audit
data. It uses Gym's shared `NeMoGymAsyncOpenAI`/aiohttp transport.

The supplied config scores raw prompts by default. Set `render_chat_template=true`
to render the context as one user message using the model's Hugging Face chat
template with `add_generation_prompt=True`; choices follow the assistant prefix.
In chat mode, the tokenizer loads from `tokenizer` or `model` and must match the
deployed model.

Tokenization uses the deployed model's `/tokenize` endpoint without added special
tokens by default. Set `likelihood_add_special_tokens=true` to match the official
MILU vLLM backend for Gemma. A completion with
`echo=true`, `logprobs=1`, and `max_tokens=1` measures each continuation; the one
generated token is discarded. Context scores are excluded, leading spaces belong
to the continuation, and multi-token labels are supported. Overflow, missing or
nonfinite scores, and ambiguous token boundaries raise errors, never fake zeros.
No character/token-length normalization is applied.

Likelihood requests send token IDs explicitly to avoid a second tokenization and
an implicit-BOS logprob indexing error. Generation sampling settings and stop
strings do not affect this route. The [MILU documentation](../../benchmarks/indic/milu/README.md)
records its protocol and model-specific special-token settings.
