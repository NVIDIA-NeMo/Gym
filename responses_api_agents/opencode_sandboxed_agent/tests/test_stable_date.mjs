// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { existsSync, rmSync } from "node:fs";
import test, { after } from "node:test";
import stableDate from "../stable-date.js";

const marker = "/tmp/nemo-gym-stable-date-active";
after(() => rmSync(marker, { force: true }));

const prompt = (date) => `You are opencode.\n<env>\n  Platform: linux\n  Today's date: ${date}\n</env>\nRules.`;

async function render(hook, sessionID, system) {
  // OpenCode passes a fresh `{ system }` and keeps reading its own array, so assert on that array.
  await hook["experimental.chat.system.transform"]({ sessionID, model: {} }, { system });
  return system;
}

test("a session keeps its first call's date across 00:00 UTC", async () => {
  const hook = await stableDate();
  const first = await render(hook, "s1", [prompt("Sun Nov 01 2026")]);
  const later = await render(hook, "s1", [prompt("Mon Nov 02 2026")]);
  assert.deepEqual(first, [prompt("Sun Nov 01 2026")]);
  assert.deepEqual(later, [prompt("Sun Nov 01 2026")]);
});

test("each session pins its own date", async () => {
  const hook = await stableDate();
  await render(hook, "main", [prompt("Sun Nov 01 2026")]);
  const subagent = await render(hook, "sub", [prompt("Mon Nov 02 2026")]);
  assert.deepEqual(subagent, [prompt("Mon Nov 02 2026")]);
});

test("only the date line changes", async () => {
  const hook = await stableDate();
  await render(hook, "s1", ["header", prompt("Sun Nov 01 2026")]);
  const later = await render(hook, "s1", ["header", prompt("Mon Nov 02 2026").replace("Rules.", "New rules.")]);
  assert.deepEqual(later, ["header", prompt("Sun Nov 01 2026").replace("Rules.", "New rules.")]);
});

test("prompts without a date or session pass through unmarked", async (t) => {
  rmSync(marker, { force: true });
  t.after(() => rmSync(marker, { force: true }));
  const hook = await stableDate();
  assert.deepEqual(await render(hook, "s1", ["no date here"]), ["no date here"]);
  assert.deepEqual(await render(hook, undefined, [prompt("Mon Nov 02 2026")]), [prompt("Mon Nov 02 2026")]);
  assert.equal(existsSync(marker), false);
});

test("pinning a date writes the host's marker", async (t) => {
  rmSync(marker, { force: true });
  t.after(() => rmSync(marker, { force: true }));
  const hook = await stableDate();
  await render(hook, "s1", [prompt("Sun Nov 01 2026")]);
  assert.equal(existsSync(marker), true);
});
