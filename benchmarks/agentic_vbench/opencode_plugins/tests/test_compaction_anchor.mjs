// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

const root = await mkdtemp(join(tmpdir(), "agentic-vbench-anchor-tests-"));
process.env.XDG_STATE_HOME = root;
const { default: plugin } = await import("../compaction_anchor.mjs");

const original = (session, goal) => ({
  info: { id: "msg_" + session, sessionID: session, role: "user", time: { created: 1 } },
  parts: [{ type: "text", text: goal }],
});
const compacted = (session) => ({
  info: { id: "summary_" + session, sessionID: session, role: "assistant" },
  parts: [{ type: "text", text: "## Goal\nAn incomplete compressed description." }],
});
const goalA = 'Edit task A; write /workspace/output/solution.json EXACTLY as {"segments": [{"source": "1"}]}';
const goalB = 'Repair task B; write /workspace/output/cuts.json EXACTLY as {"cuts": []}';
try {
  const hook = (await plugin())["experimental.chat.messages.transform"];
  // The first call saves the prompt and changes nothing.
  const a = original("ses_A", goalA);
  const first = { messages: [structuredClone(a)] };
  await hook({}, first);
  assert.deepEqual(first.messages, [a]);
  // Two sessions in flight keep their own prompts.
  const b = original("ses_B", goalB);
  await Promise.all([hook({}, { messages: [b] }), hook({}, { messages: [a] })]);
  // After a compaction dropped the prompt it comes back verbatim, at the front, as many times as needed.
  for (let cycle = 0; cycle < 3; cycle++) {
    const next = { messages: [compacted("ses_A")] };
    await hook({}, next);
    assert.equal(next.messages[0].parts[0].text, goalA);
    assert.equal(next.messages[0].info.role, "user");
    assert(!JSON.stringify(next.messages).includes(goalB));
  }
  const other = { messages: [compacted("ses_B")] };
  await hook({}, other);
  assert.equal(other.messages[0].parts[0].text, goalB);
  // A new OpenCode process (a fresh plugin instance) reads the saved anchor back from disk.
  const restarted = (await plugin())["experimental.chat.messages.transform"];
  const resumed = { messages: [compacted("ses_A")] };
  await restarted({}, resumed);
  assert.equal(resumed.messages[0].parts[0].text, goalA);
  // Synthetic parts OpenCode appends to the prompt are not a change of the task.
  const remind = structuredClone(a);
  remind.info.time.updated = 100;
  remind.parts.push({ type: "text", text: "Synthetic runtime reminder", synthetic: true });
  await hook({}, { messages: [remind] });
  // A changed task, mixed sessions, an unknown session and a tampered file fail loudly.
  const tampered = structuredClone(a);
  tampered.parts[0].text = "Different task";
  await assert.rejects(hook({}, { messages: [tampered] }), /original task changed/);
  await assert.rejects(hook({}, { messages: [a, b] }), /mixed sessions/);
  await assert.rejects(hook({}, { messages: [compacted("ses_UNKNOWN")] }), /anchor missing/);
  const file = join(root, "opencode", "agentic-vbench-task-anchors", "ses_A.json");
  const saved = JSON.parse(await readFile(file, "utf8"));
  saved.message.parts[0].text = "Tampered persisted task";
  await writeFile(file, JSON.stringify(saved));
  const cold = (await plugin())["experimental.chat.messages.transform"];
  await assert.rejects(cold({}, { messages: [compacted("ses_A")] }), /integrity check/);
  console.log("ok compaction_anchor");
} finally {
  await rm(root, { recursive: true, force: true });
}
