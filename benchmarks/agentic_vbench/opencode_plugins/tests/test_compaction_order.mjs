// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";

import CompactionOrder, { orderCompaction } from "../compaction_order.mjs";

const msg = (id, role, parts, extra = {}) => ({ info: { id, role, sessionID: "ses_a", ...extra }, parts });
const tail = msg("tail", "assistant", [
  { type: "tool", callID: "call1", state: { status: "completed", output: "render complete" } },
]);
const marker = msg("marker", "user", [{ type: "compaction", tail_start_id: "tail" }]);
const summary = msg("summary", "assistant", [{ type: "text", text: "output not yet created" }], {
  parentID: "marker",
  summary: true,
  finish: "stop",
});
const next = msg("next", "user", [{ type: "text", text: "continue" }]);
const source = [tail, marker, summary, next];
const saved = JSON.stringify(source);

// The summary and its marker move before the retained tail; nothing else changes and the input is untouched.
const result = orderCompaction(source);
assert.deepEqual(result, [marker, summary, tail, next]);
assert.equal(JSON.stringify(source), saved);
assert.deepEqual(orderCompaction(result), result, "idempotent");
assert.deepEqual(
  [...result].sort((a, b) => a.info.id.localeCompare(b.info.id)),
  [...source].sort((a, b) => a.info.id.localeCompare(b.info.id)),
  "same messages, same contents and tool ids",
);
// Nothing to do without a compaction; a failed summary or an unknown tail start is left alone.
assert.deepEqual(orderCompaction([tail, next]), [tail, next]);
const failed = structuredClone(source);
failed[2].info.error = { message: "failed" };
assert.deepEqual(orderCompaction(failed), failed);
const missing = structuredClone(source);
missing[1].parts[0].tail_start_id = "missing";
assert.deepEqual(orderCompaction(missing), missing);
// Messages of another session in the moved range fail loudly.
const mixed = structuredClone(source);
mixed[0].info.sessionID = "ses_b";
assert.throws(() => orderCompaction(mixed), /mixed sessions/);
// A second compaction later in the same session orders too.
const tail2 = msg("tail2", "assistant", [{ type: "text", text: "new progress" }]);
const marker2 = msg("marker2", "user", [{ type: "compaction", tail_start_id: "tail2" }]);
const summary2 = msg("summary2", "assistant", [{ type: "text", text: "older summary2" }], {
  summary: true,
  finish: "stop",
  parentID: "marker2",
});
const twice = orderCompaction([...source, tail2, marker2, summary2]);
assert.deepEqual(twice, [marker, summary, tail, next, marker2, summary2, tail2]);
assert.deepEqual(orderCompaction(twice), twice);

// The exported hook reorders OpenCode's array in place.
const hook = await CompactionOrder();
const live = structuredClone(source);
const output = { messages: live };
await hook["experimental.chat.messages.transform"]({}, output);
assert.equal(output.messages, live);
assert.deepEqual(live, result);
console.log("ok compaction_order");
