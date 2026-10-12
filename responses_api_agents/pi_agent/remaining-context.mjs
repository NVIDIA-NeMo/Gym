// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Let vLLM calculate the output budget using the fully serialized input.
export default function remainingContext(pi) {
  pi.on("before_provider_request", (event, ctx) => {
    if (ctx.model?.provider !== "nemo" || ctx.model?.api !== "openai-completions") return;
    const payload = { ...event.payload };
    delete payload.max_tokens;
    delete payload.max_completion_tokens;
    return payload;
  });
}
