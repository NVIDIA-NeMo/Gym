// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

const PLUGIN = new URL("../opencode_bundle/nemo_gym_receipts.js", import.meta.url);
let imports = 0;

// Imports a fresh copy of the plugin, which wraps process.stderr.write as it loads, over a stream that records what
// reaches it, with every argument of each call, and returns `returned`. The real stream and environment are back once
// the test ends.
async function load(t, logPath) {
  const real = process.stderr.write;
  const received = [];
  const calls = [];
  const returned = Symbol("the stream's return value");
  const stream = (...args) => {
    calls.push(args);
    received.push(String(args[0]));
    return returned;
  };
  process.stderr.write = stream;
  if (logPath === undefined) delete process.env.NEMO_GYM_PLUGIN_LOG;
  else process.env.NEMO_GYM_PLUGIN_LOG = logPath;
  t.after(() => {
    process.stderr.write = real;
    delete process.env.NEMO_GYM_PLUGIN_LOG;
  });
  const plugin = (await import(`${PLUGIN.href}?import=${++imports}`)).default;
  return { plugin, received, calls, returned, stream };
}

function logIn(dir) {
  return join(mkdtempSync(join(tmpdir(), "receipts-")), dir, "plugins.log");
}

test("copies every plugin line to the log and lets every chunk through", async (t) => {
  const log = logIn("");
  const { plugin, received, calls, returned } = await load(t, log);
  const written = () => {};

  assert.deepEqual(await plugin(), {});
  // A write reaches the stream with every argument it was given, and returns what the stream returned.
  for (const line of ["[nemo-gym-plugin] loaded sentinel version=1 plugins=\n", "an OpenCode line\n"]) {
    assert.equal(process.stderr.write(line, "utf8", written), returned);
    assert.deepEqual(calls.at(-1), [line, "utf8", written]);
  }
  process.stderr.write(Buffer.from("[nemo-gym-plugin] terminal=max_iteration session=s calls=3 cap=2\n"));

  assert.equal(
    readFileSync(log, "utf8"),
    "[nemo-gym-plugin] loaded receipts version=1\n" +
      "[nemo-gym-plugin] loaded sentinel version=1 plugins=\n" +
      "[nemo-gym-plugin] terminal=max_iteration session=s calls=3 cap=2\n",
  );
  assert.deepEqual(received, [
    "[nemo-gym-plugin] loaded receipts version=1\n",
    "[nemo-gym-plugin] loaded sentinel version=1 plugins=\n",
    "an OpenCode line\n",
    "[nemo-gym-plugin] terminal=max_iteration session=s calls=3 cap=2\n",
  ]);
});

test("without NEMO_GYM_PLUGIN_LOG the stream is left as it was", async (t) => {
  const { plugin, received, stream } = await load(t, undefined);

  assert.equal(process.stderr.write, stream);
  await plugin();
  assert.deepEqual(received, ["[nemo-gym-plugin] loaded receipts version=1\n"]);
});

test("a log that cannot be written leaves the stream working", async (t) => {
  const log = logIn("missing-directory");
  const { plugin, received } = await load(t, log);

  await plugin();

  assert.deepEqual(received, ["[nemo-gym-plugin] loaded receipts version=1\n"]);
  assert.equal(existsSync(log), false);
});

// Every write to /dev/full fails as on a full disk.
test("a full disk leaves the stream working", { skip: !existsSync("/dev/full") && "needs /dev/full" }, async (t) => {
  const { plugin, received, returned } = await load(t, "/dev/full");
  const line = "[nemo-gym-plugin] loaded sentinel version=1 plugins=\n";

  await plugin();

  assert.equal(process.stderr.write(line), returned);
  assert.deepEqual(received, ["[nemo-gym-plugin] loaded receipts version=1\n", line]);
});
