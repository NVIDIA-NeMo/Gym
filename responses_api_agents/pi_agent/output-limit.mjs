// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Pi's model maxTokens is metadata; apply it to each serialized inference request.
export default function outputLimit(pi) {
  pi.on("before_provider_request", (event, ctx) => {
    if (ctx.model?.provider !== "nemo" || ctx.model?.api !== "openai-completions") return;
    const payload = { ...event.payload };
    const limits = [ctx.model.maxTokens, payload.max_tokens, payload.max_completion_tokens]
      .filter((value) => Number.isSafeInteger(value) && value > 0);
    // Preserve a smaller upstream cap, including requests made for compaction.
    payload.max_tokens = Math.min(...limits);
    delete payload.max_completion_tokens;
    return payload;
  });
}
