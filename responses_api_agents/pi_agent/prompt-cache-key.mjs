// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Send one prompt_cache_key for every model call of a session, so a gateway that routes on it
// (OpenAI's cache-affinity field) keeps the conversation on one replica and its prefix cache.
// Pi sends its own key only to api.openai.com or with long cache retention.
export default function promptCacheKey(pi) {
  const key = process.env.NEMO_GYM_PI_PROMPT_CACHE_KEY;
  if (!key) throw new Error("NEMO_GYM_PI_PROMPT_CACHE_KEY must be set");
  pi.on("before_provider_request", (event, ctx) => {
    if (ctx.model?.provider !== "nemo" || ctx.model?.api !== "openai-completions") return;
    return { ...event.payload, prompt_cache_key: key };
  });
}
