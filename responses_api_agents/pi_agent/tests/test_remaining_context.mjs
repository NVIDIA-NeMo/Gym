// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import test from "node:test";
import remainingContext from "../remaining-context.mjs";

let handler;
remainingContext({ on(event, callback) {
  assert.equal(event, "before_provider_request");
  handler = callback;
} });

test("omits both caps on every turn while preserving history and sampling", () => {
  for (const length of [1, 2, 10]) {
    const messages = Array.from({ length }, () => ({ role: "user", content: "history" }));
    const payload = { messages, temperature: 0.7, max_tokens: 131072, max_completion_tokens: 131072 };
    const result = handler({ payload }, { model: { provider: "nemo", api: "openai-completions" } });
    assert.deepEqual(result, { messages, temperature: 0.7 });
    assert.equal(payload.max_tokens, 131072);
  }
});

test("leaves other providers and APIs unchanged", () => {
  for (const model of [undefined, { provider: "other", api: "openai-completions" },
                       { provider: "nemo", api: "openai-responses" }]) {
    assert.equal(handler({ payload: { max_tokens: 123 } }, { model }), undefined);
  }
});
