// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { correlationHeaders } from "../eval-correlation.mjs";

const input = { sessionID: "session", message: { id: "user" } };
const reply = { role: "assistant", sessionID: "session", parentID: "user", id: "reply", time: { created: 1 } };
const rows = [{ info: reply }, { info: { ...reply, id: "old", time: { created: 0, completed: 1 } } }];
assert.deepEqual(correlationHeaders(rows, input), {
  "x-opencode-session-id": "session", "x-opencode-assistant-message-id": "reply",
});
assert.deepEqual(correlationHeaders([...rows, { info: { ...reply, id: "ambiguous" } }], input), {});
assert.deepEqual(correlationHeaders([{ info: { ...reply, sessionID: "other" } }], input), {});
assert.deepEqual(correlationHeaders([{ info: { ...reply, error: { message: "failed" } } }], input), {});
assert.deepEqual(correlationHeaders([], input), {});
assert.deepEqual(correlationHeaders(undefined, input), {});

// Budget identity remains available even when assistant evidence is ambiguous or unavailable.
const { default: createPlugin } = await import("../eval-correlation.mjs");
for (const lookup of [
  async () => ({ data: [] }),
  async () => { throw new Error("lookup failed"); },
]) {
  const plugin = await createPlugin({ client: { session: { messages: lookup } } });
  for (const sessionID of ["root", "child"]) {
    const output = { headers: {} };
    await plugin["chat.headers"]({ ...input, sessionID }, output);
    assert.equal(output.headers["x-opencode-session-id"], sessionID);
    assert.equal(output.headers["x-opencode-assistant-message-id"], undefined);
  }
}
