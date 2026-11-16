// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import test from "node:test";
import bashTimeout from "../bash-timeout.mjs";

test("supplies missing Bash deadlines, caps longer ones, and preserves shorter ones", () => {
  process.env.NEMO_GYM_PI_BASH_TIMEOUT = "120";
  let handler;
  bashTimeout({ on(event, callback) {
    assert.equal(event, "tool_call");
    handler = callback;
  } });
  for (const [requested, expected] of [[undefined, 120], [300, 120], [30, 30], [0, 120], [-1, 120], [null, 120], [NaN, 120], [Infinity, 120], ["30", 120]]) {
    const event = { toolName: "bash", input: { command: "sleep 1000", timeout: requested } };
    handler(event);
    assert.deepEqual(event.input, { command: "sleep 1000", timeout: expected });
  }
  const other = { toolName: "read", input: { path: "file.txt" } };
  handler(other);
  assert.deepEqual(other.input, { path: "file.txt" });
  delete process.env.NEMO_GYM_PI_BASH_TIMEOUT;
});

test("rejects absent or invalid configured limits", () => {
  for (const value of ["", "0", "-1", "NaN", "Infinity"]) {
    process.env.NEMO_GYM_PI_BASH_TIMEOUT = value;
    assert.throws(() => bashTimeout({}), /Invalid Gym Pi Bash timeout/);
  }
  delete process.env.NEMO_GYM_PI_BASH_TIMEOUT;
  assert.throws(() => bashTimeout({}), /Invalid Gym Pi Bash timeout/);
});
