// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";

import CompactionSummaryHeader, { HEADER } from "../compaction_summary_header.mjs";

const hooks = await CompactionSummaryHeader();
assert.equal(HEADER, "X-NeMo-Gym-OpenCode-Agent");
const output = { headers: { "X-Other": "kept" } };
hooks["chat.headers"]({ agent: "compaction", sessionID: "ses_a" }, output);
assert.deepEqual(output.headers, { "X-Other": "kept", [HEADER]: "compaction" });
const build = { headers: {} };
hooks["chat.headers"]({ agent: "build" }, build);
assert.deepEqual(build.headers, { [HEADER]: "build" });
// A call without an agent name gets no header rather than a guess.
const none = { headers: {} };
hooks["chat.headers"]({}, none);
hooks["chat.headers"]({ agent: "" }, none);
assert.deepEqual(none.headers, {});
console.log("ok compaction_summary_header");
