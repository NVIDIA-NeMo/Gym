// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { chmod, lstat, mkdtemp, mkdir, readFile, readdir, rm, symlink, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";
import { captureModelInput, restoreModelInput } from "../media_input_capture.mjs";

const hash = (value) => createHash("sha256").update(value).digest("hex");
const pixels = Buffer.from("immutable synthetic image payload");
const image = `data:image/png;base64,${pixels.toString("base64")}`;
const descriptor = 'NEMOTRON_FIXED_VIDEO_V1\n{"archive_path":"/external/frozen-video.npz","sha256":"' + "a".repeat(64) + '"}';
const video = `data:video/mp4;base64,${Buffer.from(descriptor).toString("base64")}`;
const tokens = [3, 42, 1];
const record = { capture_id: "capture-1", invocation_id: "session1", agent: "build", started_at: 17,
  action: "send", input_tokens: tokens.length, token_ids_sha256: hash(JSON.stringify(tokens)), request_sha256: "b".repeat(64) };
const body = { model: "nano", messages: [
  { role: "system", content: "PRIVATE SYSTEM" },
  { role: "assistant", tool_calls: [{ id: "c1", type: "function", function: { name: "view_video", arguments: '{"start":1}' } }] },
  { role: "tool", tool_call_id: "c1", content: "PRIVATE RECEIPT" },
  { role: "user", content: [
    { type: "text", text: "before" }, { type: "video_url", video_url: { url: video } },
    { type: "image_url", image_url: { url: image, detail: "high" } }, { type: "text", text: "after" },
  ] },
], tools: [{ type: "function", function: { name: "view_video", parameters: { type: "object" } } }],
chat_template_kwargs: { enable_thinking: false }, max_tokens: 128, stream: true };

async function withDirectory(run) {
  const directory = await mkdtemp(path.join(os.tmpdir(), "nemotron-input-test-"));
  try { await run(directory); }
  finally { await rm(directory, { recursive: true }); }
}
async function capture(directory, changes = {}) {
  const reference = await captureModelInput({ directory, body, tokens, record: { ...record, ...changes } });
  const raw = await readFile(reference.path, "utf8");
  assert.equal(hash(raw), reference.sha256);
  return { reference, raw, snapshot: JSON.parse(raw) };
}

test("full input round trip preserves tools, ordering, reasoning config and tokens without inline media", async () => withDirectory(async (directory) => {
  const original = structuredClone(body);
  const { snapshot, raw, reference } = await capture(directory);
  assert.equal(raw.includes(";base64,"), false);
  assert.ok(raw.includes("PRIVATE SYSTEM") && raw.includes("PRIVATE RECEIPT"));
  assert.deepEqual(snapshot.input_token_ids, tokens);
  assert.equal(snapshot.token_ids_source, "same_server_native_tokenize_preflight");
  assert.equal(snapshot.admission_action, "send");
  assert.deepEqual(await restoreModelInput(snapshot), body);
  assert.deepEqual(body, original); // The generation request is never mutated.
  assert.equal((await lstat(reference.path)).mode & 0o777, 0o600);
  assert.equal((await lstat(directory)).mode & 0o777, 0o700);
  assert.equal((await lstat(snapshot.media_references[1].path)).mode & 0o777, 0o600);
}));

test("tool media round trip retains each call identity without a synthetic user message", async () => withDirectory(async (directory) => {
  const toolBody = { model: "nano", messages: [
    { role: "user", content: "Inspect the two files." },
    { role: "assistant", tool_calls: [
      { id: "image-call", type: "function", function: { name: "view_image", arguments: "{}" } },
      { id: "video-call", type: "function", function: { name: "view_video", arguments: "{}" } },
    ] },
    { role: "tool", tool_call_id: "image-call", content: [{ type: "image_url", image_url: { url: image } }] },
    { role: "tool", tool_call_id: "video-call", content: [{ type: "video_url", video_url: { url: video } }] },
  ] };
  const reference = await captureModelInput({ directory, body: toolBody, tokens, record });
  const raw = await readFile(reference.path, "utf8");
  const snapshot = JSON.parse(raw);
  assert.equal(raw.includes(";base64,"), false);
  assert.deepEqual(await restoreModelInput(snapshot), toolBody);
  assert.deepEqual(snapshot.request.messages.map((message) => message.role), ["user", "assistant", "tool", "tool"]);
  assert.deepEqual(snapshot.request.messages.slice(2).map((message) => message.tool_call_id), ["image-call", "video-call"]);
  assert.deepEqual(snapshot.media_references.map((reference) => reference.message_index), [2, 3]);
}));

test("compact and rejected calls retain separate histories rather than being flattened", async () => withDirectory(async (directory) => {
  const denied = (await capture(directory, { action: "compact" })).snapshot;
  const compactBody = { model: "nano", messages: [{ role: "system", content: "summarize" }, { role: "user", content: "OLD HISTORY" }] };
  const compactRef = await captureModelInput({ directory, body: compactBody, tokens, record: { ...record, capture_id: "capture-2", agent: "compaction" } });
  const compact = JSON.parse(await readFile(compactRef.path, "utf8"));
  assert.equal(denied.admission_action, "compact");
  assert.equal(compact.agent, "compaction");
  assert.deepEqual(compact.media_references, []);
  assert.deepEqual(await restoreModelInput(compact), compactBody);
  const resumed = await capture(directory, { capture_id: "capture-3" });
  assert.equal(resumed.snapshot.media_references[0].sha256, denied.media_references[0].sha256);
  assert.equal((await readdir(path.join(directory, "image_blobs"))).length, 1);
}));

test("concurrent image deduplication publishes complete private blobs", async () => withDirectory(async (directory) => {
  const captures = await Promise.all(Array.from({ length: 16 }, (_, i) => capture(directory, { capture_id: `parallel-${i}` })));
  for (const { snapshot } of captures) assert.deepEqual(await restoreModelInput(snapshot), body);
  assert.equal((await readdir(path.join(directory, "image_blobs"))).length, 1);
}));

test("existing capture outputs are never overwritten", async () => withDirectory(async (directory) => {
  const { reference, raw } = await capture(directory);
  await assert.rejects(capture(directory), (error) => error.code === "EEXIST");
  assert.equal(await readFile(reference.path, "utf8"), raw);
}));

test("tampered token IDs, request text and media binding fail closed", async () => withDirectory(async (directory) => {
  const { snapshot } = await capture(directory);
  for (const mutate of [
    (value) => value.input_token_ids[0]++,
    (value) => value.request.messages[0].content = "changed",
    (value) => value.media_references[0].sha256 = "f".repeat(64),
    (value) => value.media_references[0].descriptor_utf8 += " ",
    (value) => value.media_references.push(value.media_references[0]),
  ]) {
    const changed = structuredClone(snapshot);
    mutate(changed);
    await assert.rejects(restoreModelInput(changed));
  }
}));

test("tampered or symlink image blobs are not accepted or overwritten", async () => withDirectory(async (directory) => {
  const { snapshot } = await capture(directory);
  const filename = snapshot.media_references[1].path;
  await writeFile(filename, "tampered");
  await assert.rejects(restoreModelInput(snapshot), /SHA256/);
  await assert.rejects(capture(directory, { capture_id: "capture-2" }), /SHA256/);
  assert.equal(await readFile(filename, "utf8"), "tampered");
  await rm(filename);
  const target = path.join(directory, "external.bin");
  await writeFile(target, pixels, { mode: 0o600 });
  await symlink(target, filename);
  await assert.rejects(restoreModelInput(snapshot), /private image/);
}));

test("public or symlink capture directories fail before recording content", async () => withDirectory(async (directory) => {
  const publicDirectory = path.join(directory, "public");
  await mkdir(publicDirectory);
  await chmod(publicDirectory, 0o755);
  await assert.rejects(capture(publicDirectory), /mode 0700/);
  assert.deepEqual(await readdir(publicDirectory), []);
  const indirect = path.join(directory, "indirect");
  await symlink(directory, indirect);
  await assert.rejects(capture(indirect), /non-symlink/);
}));

test("unknown, remote and malformed media cannot be dumped inline", async () => withDirectory(async (directory) => {
  for (const part of [
    { type: "image_url", image_url: { url: "https://invalid.test/image.png" } },
    { type: "image_url", image_url: { url: "data:image/png;base64,YWJj=" } },
    { type: "video_url", video_url: { url: "data:video/mp4;base64,YWJj" } },
    { type: "input_audio", input_audio: { data: "YWJj" } },
  ]) {
    await assert.rejects(captureModelInput({ directory, body: { messages: [{ role: "user", content: [part] }] }, tokens, record }));
  }
  await assert.rejects(capture(directory, { capture_id: "../escape" }), /identity/);
  assert.deepEqual(await readdir(directory), []);
}));
