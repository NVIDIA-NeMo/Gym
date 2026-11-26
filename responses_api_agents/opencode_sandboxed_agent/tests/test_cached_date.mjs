// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import test from "node:test";
import cachedDate from "../cached-date.js";

const environment = (date, workdir = "/workspace/repo", newline = "\n") =>
  ["<env>", `  Working directory: ${workdir}`, `  Today's date: ${date}`, "</env>"].join(newline);

test("keeps the prompt prefix stable across midnight, including subagents", async () => {
  const hooks = await cachedDate();
  const transform = hooks["experimental.chat.system.transform"];
  const first = ["Instructions\n" + environment("Wed Nov 25 2026")];
  await transform({ sessionID: "parent" }, { system: first });
  for (const sessionID of ["parent", "child"]) {
    const system = ["Instructions\n" + environment("Thu Nov 26 2026")];
    const output = { system };
    await transform({ sessionID }, output);
    assert.strictEqual(output.system, system);
    assert.deepEqual(system, first);
  }
});

test("caches only the environment date and preserves other prompt changes", async () => {
  const hooks = await cachedDate();
  const transform = hooks["experimental.chat.system.transform"];
  const withoutDate = ["Today's date: custom instruction", "<env>\nPlatform: linux\n</env>"];
  await transform({}, { system: withoutDate });
  assert.deepEqual(withoutDate, ["Today's date: custom instruction", "<env>\nPlatform: linux\n</env>"]);

  await transform({}, { system: [environment("Wed Nov 25 2026")] });
  const system = ["Today's date: Thu Nov 26 2026", environment("Thu Nov 26 2026", "/workspace/other", "\r\n")];
  await transform({}, { system });
  assert.deepEqual(system, [
    "Today's date: Thu Nov 26 2026",
    environment("Wed Nov 25 2026", "/workspace/other", "\r\n"),
  ]);
});

test("a new runtime gets its own date", async () => {
  const first = await cachedDate();
  await first["experimental.chat.system.transform"]({}, { system: [environment("Wed Nov 25 2026")] });
  const second = await cachedDate();
  const system = [environment("Thu Nov 26 2026")];
  await second["experimental.chat.system.transform"]({}, { system });
  assert.deepEqual(system, [environment("Thu Nov 26 2026")]);
});
