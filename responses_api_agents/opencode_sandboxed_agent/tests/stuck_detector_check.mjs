// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Checks stuck_detector.mjs against OpenHands' scenarios; run by test_stuck_detector.py (needs node).
import assert from "node:assert/strict";
import { StuckDetector } from "../stuck_detector.mjs";

const { eventsFromParts, stuckScenario } = StuckDetector;
let n = 0;
const bash = (command, exit, output = "out") => ({
  id: `p${n++}`, type: "tool", tool: "bash",
  state: { status: "completed", input: { command }, output: `${output}${n}`, metadata: { exit } },
});
const failed = (filePath) => ({ id: `p${n++}`, type: "tool", tool: "edit", state: { status: "error", input: { filePath }, error: "Found no match" } });
const read = (filePath, output) => ({ id: `p${n++}`, type: "tool", tool: "read", state: { status: "completed", input: { filePath }, output } });
const step = (...items) => [{ type: "step-start" }, ...items.map((x) => (typeof x === "string" ? { type: "text", text: x } : x)), { type: "step-finish" }];
const run = (...steps) => stuckScenario(eventsFromParts(steps.flat()));

// 1: same command + exit code 4 times (output differs: OpenHands ignores it for commands)
assert.equal(run(step(bash("pytest", 1)), step(bash("pytest", 1)), step(bash("pytest", 1)), step(bash("pytest", 1))), "repeating_action_observation");
assert.equal(run(step(bash("pytest", 1)), step(bash("pytest", 1)), step(bash("pytest", 1))), null);
assert.equal(run(step(bash("pytest", 1)), step(bash("pytest", 1)), step(bash("pytest", 0)), step(bash("pytest", 1))), null);
// the step's text is the action's thought: different text, different action
assert.equal(run(step("a", bash("pytest", 1)), step("b", bash("pytest", 1)), step(bash("pytest", 1)), step(bash("pytest", 1))), null);
// an agent message (a step with no tool call) is an action, and breaks the run of equal actions
assert.equal(run(step(bash("ls", 0)), step(bash("ls", 0)), step("thinking"), step(bash("ls", 0)), step(bash("ls", 0))), null);
// non-command tools compare their output
assert.equal(run(step(read("a", "x")), step(read("a", "x")), step(read("a", "x")), step(read("a", "x"))), "repeating_action_observation");
assert.equal(run(step(read("a", "x")), step(read("a", "x")), step(read("a", "y")), step(read("a", "x"))), null);
// 2: same failing call 3 times
assert.equal(run(step(failed("f")), step(failed("f")), step(failed("f"))), "repeating_action_error");
assert.equal(run(step(failed("f")), step(failed("g")), step(failed("f"))), null);
// 4: alternating pair, 3 times
assert.equal(run(step(bash("a", 0)), step(read("b", "x")), step(bash("a", 0)), step(read("b", "x")), step(bash("a", 0)), step(read("b", "x"))), "action_observation_pattern");
assert.equal(run(step(bash("a", 0)), step(read("b", "x")), step(bash("a", 0)), step(read("b", "y")), step(bash("a", 0)), step(read("b", "x"))), null);
// parallel calls in one step: each is an action + observation; only the first carries the text
assert.equal(run(step("t", bash("x", 0), bash("x", 0)), step("t", bash("x", 0), bash("x", 0))), null);
assert.equal(run(step(bash("x", 0), bash("x", 0)), step(bash("x", 0), bash("x", 0))), "repeating_action_observation");
// an open step counts its finished calls, not its running ones
const running = { id: "r", type: "tool", tool: "bash", state: { status: "running", input: { command: "pytest" } } };
assert.equal(run(step(bash("pytest", 1)), step(bash("pytest", 1)), step(bash("pytest", 1)), [{ type: "step-start" }, running]), null);
assert.equal(run(step(bash("pytest", 1)), step(bash("pytest", 1)), step(bash("pytest", 1)), [{ type: "step-start" }, bash("pytest", 1)]), "repeating_action_observation");
// 5: 10 compactions in a row
assert.equal(stuckScenario(eventsFromParts(Array.from({ length: 10 }, () => ({ type: "compaction" })))), "context_window_error");
assert.equal(stuckScenario(eventsFromParts([...Array.from({ length: 9 }, () => ({ type: "compaction" })), ...step(bash("a", 0)), { type: "compaction" }])), null);

// The marker is only a diagnostic: when it cannot be written (a full /tmp, an obstruction) the session is still aborted,
// once. The plugin writes the marker next to itself, so it runs from a copy with a directory where the marker goes.
const { copyFileSync, mkdirSync, mkdtempSync } = await import("node:fs");
const { tmpdir } = await import("node:os");
const { join } = await import("node:path");
const dir = mkdtempSync(join(tmpdir(), "stuck-plugin-"));
copyFileSync(new URL("../stuck_detector.mjs", import.meta.url), join(dir, "stuck_detector.mjs"));
mkdirSync(join(dir, "stuck.json")); // EISDIR on write
const aborts = [];
const hooks = await (await import(join(dir, "stuck_detector.mjs"))).StuckDetector({
  client: { session: { abort: async (args) => { aborts.push(args); } } },
});
const sameCall = (i) => ({ id: `h${i}`, sessionID: "ses_h", messageID: "m1", type: "tool", tool: "bash", state: { status: "completed", input: { command: "pytest" }, output: "same", metadata: { exit: 1 } } });
const send = (part) => hooks.event({ event: { type: "message.part.updated", properties: { part: { sessionID: "ses_h", messageID: "m1", ...part } } } });
for (let i = 0; i < 6; i++) {
  await send({ id: `s${i}`, type: "step-start" });
  await send(sameCall(i));
  await send({ id: `f${i}`, type: "step-finish" });
}
assert.equal(aborts.length, 1, "the loop is aborted once although the marker could not be written");
assert.equal(aborts[0].path.id, "ses_h");
console.log("ok");
