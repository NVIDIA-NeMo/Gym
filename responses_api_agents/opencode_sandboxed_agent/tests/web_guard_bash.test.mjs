// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";

delete process.env.NEMO_GYM_BASH_GUARD;
delete process.env.NEMO_GYM_CONTAMINATION_GUARD;
const source = readFileSync(new URL("../plugins/nemo_gym_web_guard.js", import.meta.url));
const { default: webGuard } = await import(`data:text/javascript;base64,${source.toString("base64")}`);
const { bash } = (await webGuard()).tool;
const ctx = () => ({ abort: new AbortController().signal });

test("runs bash syntax, not dash", { skip: !existsSync("/bin/bash") }, async () => {
  const saved = process.env.SHELL;
  delete process.env.SHELL;
  try {
    assert.equal((await bash.execute({ command: "a=(x y); [[ $0 == *bash ]] && echo ${a[1]}" }, ctx())).trim(), "y");
  } finally {
    if (saved !== undefined) process.env.SHELL = saved;
  }
});

test("returns stdout and stderr of a failing command", async () => {
  const out = await bash.execute({ command: "echo out; echo err >&2; exit 3" }, ctx());
  assert.match(out, /out/);
  assert.match(out, /err/);
});

test("honours workdir", async () => {
  assert.equal((await bash.execute({ command: "pwd", workdir: "/" }, ctx())).trim(), "/");
});

test("timeout kills the whole process group", async () => {
  const start = Date.now();
  const out = await bash.execute({ command: "echo started; sleep 30 & sleep 30", timeout: 500 }, ctx());
  assert.ok(Date.now() - start < 5_000);
  assert.match(out, /started/);
  assert.match(out, /timed out after 500 ms/);
});

test("abort kills the command", async () => {
  const controller = new AbortController();
  setTimeout(() => controller.abort(), 300);
  const start = Date.now();
  const out = await bash.execute({ command: "sleep 30" }, { abort: controller.signal });
  assert.ok(Date.now() - start < 5_000);
  assert.match(out, /aborted/);
});

test("guard off leaves the builtin bash in place", async () => {
  process.env.NEMO_GYM_BASH_GUARD = "0";
  try {
    assert.equal((await webGuard()).tool.bash, undefined);
  } finally {
    delete process.env.NEMO_GYM_BASH_GUARD;
  }
});
