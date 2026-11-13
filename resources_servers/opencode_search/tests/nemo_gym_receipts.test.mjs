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
// reaches it. The real stream and environment are back once the test ends.
async function load(t, logPath) {
  const real = process.stderr.write;
  const received = [];
  const stream = (chunk) => {
    received.push(String(chunk));
    return true;
  };
  process.stderr.write = stream;
  if (logPath === undefined) delete process.env.NEMO_GYM_PLUGIN_LOG;
  else process.env.NEMO_GYM_PLUGIN_LOG = logPath;
  t.after(() => {
    process.stderr.write = real;
    delete process.env.NEMO_GYM_PLUGIN_LOG;
  });
  const plugin = (await import(`${PLUGIN.href}?import=${++imports}`)).default;
  return { plugin, received, stream };
}

function logIn(dir) {
  return join(mkdtempSync(join(tmpdir(), "receipts-")), dir, "plugins.log");
}

test("copies every plugin line to the log and lets every chunk through", async (t) => {
  const log = logIn("");
  const { plugin, received } = await load(t, log);

  assert.deepEqual(await plugin(), {});
  process.stderr.write("[nemo-gym-plugin] loaded sentinel version=1 plugins=\n");
  process.stderr.write("an OpenCode line\n");
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
