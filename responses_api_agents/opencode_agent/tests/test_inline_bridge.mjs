// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, readdir, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { bridgeMessages, fixedDescriptorBytes, restoreToolMedia, TOOL_MEDIA_MIME_PREFIX } from "../media_bridge.mjs";
import { captureModelInput, restoreModelInput } from "../media_input_capture.mjs";

// The Python loader tests validate a real NPZ. This fixture exercises only
// the JS byte-preserving bridge and its integrity check.
const archive = Buffer.from([80, 75, 3, 4, 1, 2, 3, 4]);
const bytes = Buffer.concat([Buffer.from("NEMOTRON_FIXED_VIDEO_INLINE_V1\n"), createHash("sha256").update(archive).digest(), archive]);
const url = `data:video/mp4;base64,${bytes.toString("base64")}`;

test("inline RGB archive survives the native tool-role bridge byte for byte", () => {
  assert.deepEqual(fixedDescriptorBytes(url), bytes);
  const receipt = "Video: fixture.mp4\nFrames: 4 total, 4 unique source frames (0.0% repeated)";
  const source = [{ parts: [{ type: "tool", tool: "view_video", callID: "v1", state: {
    status: "completed", output: receipt, attachments: [{ type: "file", mime: "video/mp4", url }],
  } }] }];
  const bridged = bridgeMessages(source);
  assert.equal(bridged[0].parts[0].state.output, receipt);
  const restored = restoreToolMedia({ messages: [{ role: "tool", tool_call_id: "v1", content: JSON.stringify([
    { type: "text", text: receipt },
    { type: "file-data", mediaType: TOOL_MEDIA_MIME_PREFIX + "video+mp4", data: bytes.toString("base64") },
  ]) }] });
  assert.equal(restored.messages.length, 1);
  assert.equal(restored.messages[0].role, "tool");
  assert.equal(restored.messages[0].tool_call_id, "v1");
  assert.equal(restored.messages[0].content[0].text, receipt);
  assert.equal(restored.messages[0].content[1].video_url.url, url);
  assert.equal(JSON.stringify(restored).includes("archive_path"), false);
});

test("corrupt inline video is rejected instead of restored or counted", () => {
  const broken = Buffer.from(bytes);
  broken[broken.length - 1] ^= 1;
  assert.throws(() => fixedDescriptorBytes(broken), /archive\/hash/);
  assert.throws(() => fixedDescriptorBytes(bytes.subarray(0, 40)), /archive\/hash/);
});

test("captured inline media round-trips without a serving-side archive reference", async () => {
  const directory = await mkdtemp(path.join(os.tmpdir(), "inline-capture-"));
  try {
    const body = { messages: [
      { role: "assistant", tool_calls: [{ id: "v1", type: "function", function: { name: "view_video", arguments: "{}" } }] },
      { role: "tool", tool_call_id: "v1", content: [{ type: "video_url", video_url: { url } }] },
    ] };
    const tokens = [1, 2, 3], hash = (value) => createHash("sha256").update(value).digest("hex");
    for (const capture_id of ["one", "two"]) {
      const result = await captureModelInput({ directory, body, tokens, record: {
        capture_id, invocation_id: "test", agent: "build", input_tokens: tokens.length,
        token_ids_sha256: hash(JSON.stringify(tokens)), request_sha256: hash(JSON.stringify(body)),
        action: "send", started_at: "2026-10-13T00:00:00Z",
      } });
      const snapshot = JSON.parse(await readFile(result.path, "utf8"));
      assert.equal(snapshot.media_references[0].kind, "inline_video");
      assert.deepEqual(await restoreModelInput(snapshot), body);
      assert.equal(JSON.stringify(snapshot).includes("archive_path"), false);
      assert.deepEqual(await readFile(snapshot.media_references[0].path), bytes);
    }
    assert.equal((await readdir(path.join(directory, "video_blobs"))).length, 1);
  } finally { await rm(directory, { recursive: true, force: true }); }
});
