// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { test } from "node:test";
import { mkdtemp, writeFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { bridgeMessages, bridgePrompt, fixedDescriptorBytes, restoreNativeVideo, restoreToolMedia, TOOL_MEDIA_MIME_PREFIX, VIDEO_BRIDGE_MIME } from "../media_bridge.mjs";

// The provider and plugin load OpenCode's pinned SDK from the media runtime when imported.
const runtime = { skip: process.env.NEMOTRON_OPENCODE_DEPENDENCY_ROOT ? false : "NEMOTRON_OPENCODE_DEPENDENCY_ROOT is unset" };
const loadProvider = async () => (await import("../nemotron_media_provider.mjs")).createNemotronMedia;
const loadPlugin = async () => (await import("../nemotron_media_plugin.mjs")).default;

const bytes = Buffer.from("NEMOTRON_FIXED_VIDEO_V1\n" + JSON.stringify({ archive_path: "/external/a.npz", sha256: "a".repeat(64) }));
const url = `data:video/mp4;base64,${bytes.toString("base64")}`;

test("message bridge leaves the persisted video artifact unchanged", () => {
  const original = [{ info: { role: "assistant" }, parts: [{ type: "tool", callID: "v1", state: { status: "completed", output: "video", attachments: [{ type: "file", mime: "video/mp4", url }] } }] }];
  const result = bridgeMessages(original);
  assert.equal(result[0].parts[0].state.attachments[0].mime, TOOL_MEDIA_MIME_PREFIX + "video+mp4");
  assert.equal(original[0].parts[0].state.attachments[0].mime, "video/mp4");
  assert.equal(result[0].parts[0].callID, "v1");
  assert.equal(result[0].parts[0].state.attachments[0].url, url);
});

test("one native video and normal image preserve bytes and content order", () => {
  const prompt = [{ role: "user", content: [{ type: "text", text: "before" }, { type: "file", mediaType: "video/mp4", data: bytes }, { type: "file", mediaType: "image/png", data: Buffer.from("png") }, { type: "text", text: "after" }] }];
  const bridged = bridgePrompt(prompt);
  assert.equal(bridged[0].content[1].mediaType, VIDEO_BRIDGE_MIME);
  assert.equal(prompt[0].content[1].mediaType, "video/mp4");
  assert.deepEqual(bridged[0].content[1].data, bytes);
  const body = restoreNativeVideo({ messages: [{ role: "user", content: [{ type: "text", text: "before" }, { type: "image_url", image_url: { url: `data:${VIDEO_BRIDGE_MIME};base64,${bytes.toString("base64")}` } }, { type: "image_url", image_url: { url: "data:image/png;base64,cG5n" } }, { type: "text", text: "after" }] }] });
  assert.deepEqual(body.messages[0].content.map((p) => p.type), ["text", "video_url", "image_url", "text"]);
  assert.equal(body.messages[0].content[1].video_url.url, url);
});

test("malformed descriptors and remote videos fail closed", () => {
  for (const data of ["https://invalid.test/a.mp4", "data:video/mp4;base64,YWJj", Buffer.from("ordinary video"), Buffer.from("NEMOTRON_FIXED_VIDEO_V1\n{}")]) assert.throws(() => fixedDescriptorBytes(data));
});

test("the actual pinned SDK puts native video inside its bound tool result", runtime, async () => {
  const createNemotronMedia = await loadProvider();
  let sent;
  const provider = createNemotronMedia({ name: "nemo", baseURL: "http://model.test/v1", fetch: async (_, init) => {
    sent = JSON.parse(init.body);
    return new Response(JSON.stringify({ id: "test", object: "chat.completion", created: 1, model: "nano", choices: [{ index: 0, message: { role: "assistant", content: "seen" }, finish_reason: "stop" }], usage: { prompt_tokens: 600, completion_tokens: 1, total_tokens: 601 } }), { headers: { "Content-Type": "application/json" } });
  } });
  const response = await provider.languageModel("nano").doGenerate({ prompt: [
    { role: "assistant", content: [{ type: "tool-call", toolCallId: "v1", toolName: "watch_video", input: { num_frames: 4 } }] },
    { role: "tool", content: [{ type: "tool-result", toolCallId: "v1", toolName: "watch_video", output: { type: "content", value: [{ type: "file-data", mediaType: TOOL_MEDIA_MIME_PREFIX + "video+mp4", data: bytes.toString("base64") }] } }] },
  ], maxOutputTokens: 16 });
  assert.equal(response.content[0].text, "seen");
  assert.equal(sent.messages[0].tool_calls[0].id, "v1");
  assert.equal(sent.messages[0].tool_calls[0].function.name, "watch_video");
  assert.equal(sent.messages[1].tool_call_id, "v1");
  assert.equal(sent.messages.length, 2);
  assert.equal(sent.messages[1].role, "tool");
  assert.equal(sent.messages[1].content[0].type, "video_url");
  assert.equal(sent.messages[1].content[0].video_url.url, url);
  assert.equal(JSON.stringify(sent).includes(TOOL_MEDIA_MIME_PREFIX), false);
  assert.equal(JSON.stringify(sent).includes(VIDEO_BRIDGE_MIME), false);
});

test("streaming uses the same native-video transport", runtime, async () => {
  const createNemotronMedia = await loadProvider();
  let sent;
  const provider = createNemotronMedia({ name: "nemo", baseURL: "http://model.test/v1", fetch: async (_, init) => {
    sent = JSON.parse(init.body);
    const chunks = [
      { id: "test", object: "chat.completion.chunk", created: 1, model: "nano", choices: [{ index: 0, delta: { role: "assistant", content: "seen" }, finish_reason: null }] },
      { id: "test", object: "chat.completion.chunk", created: 1, model: "nano", choices: [{ index: 0, delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 600, completion_tokens: 1, total_tokens: 601 } },
    ];
    return new Response(chunks.map((c) => `data: ${JSON.stringify(c)}\n\n`).join("") + "data: [DONE]\n\n", { headers: { "Content-Type": "text/event-stream" } });
  } });
  const result = await provider.languageModel("nano").doStream({ prompt: [{ role: "user", content: [{ type: "file", mediaType: "video/mp4", data: bytes }] }], maxOutputTokens: 16 });
  const events = [];
  for await (const event of result.stream) events.push(event);
  assert.ok(events.some((event) => event.type === "text-delta" && event.delta === "seen"));
  assert.equal(sent.messages[0].content[0].video_url.url, url);
  assert.equal(sent.stream, true);
  assert.equal(sent.stream_options.include_usage, true);
});

test("initial videos are recognized as media for native compaction without changing the artifact", () => {
  const original = [{ info: { role: "user" }, parts: [{ type: "file", mime: "video/mp4", url }] }];
  assert.equal(bridgeMessages(original)[0].parts[0].mime, VIDEO_BRIDGE_MIME);
  assert.equal(original[0].parts[0].mime, "video/mp4");
});

test("OpenCode hooks mutate caller arrays in place and seed only once", runtime, async () => {
  const NemotronMediaPlugin = await loadPlugin();
  const dir = await mkdtemp(path.join(os.tmpdir(), "nemotron-media-hook-"));
  const previous = process.env.NEMOTRON_OPENCODE_MEDIA_LAUNCHER;
  try {
    const profile = { patch_size: 16, temporal_patch_size: 2, downsample_ratio: 0.5,
      max_context_tokens: 262144, reserved_output_tokens: 32768, safety_tokens: 4096, video_max_pixels: 67108864 };
    const configPath = path.join(dir, "config.json");
    await writeFile(configPath, JSON.stringify({ profile }), { flag: "wx", mode: 0o600 });
    const launcherPath = path.join(dir, "launcher.json");
    await writeFile(launcherPath, JSON.stringify({ config_path: configPath, seed_parts: [{ type: "file", mime: "video/mp4", url }] }), { flag: "wx", mode: 0o600 });
    process.env.NEMOTRON_OPENCODE_MEDIA_LAUNCHER = launcherPath;
    const plugin = await NemotronMediaPlugin();
    const originalPart = { type: "tool", tool: "watch_video", callID: "v1", state: { status: "completed", output: "Video: a.mp4", attachments: [{ type: "file", mime: "video/mp4", url }] } };
    const messages = [{ info: {}, parts: [originalPart] }];
    const output = { messages };
    await plugin["experimental.chat.messages.transform"]({}, output);
    assert.equal(output.messages, messages);
    assert.equal(messages[0].parts[0].state.attachments[0].mime, TOOL_MEDIA_MIME_PREFIX + "video+mp4");
    assert.equal(messages[0].parts[0].state.output, "Video: a.mp4");
    assert.equal(originalPart.state.output, "Video: a.mp4");
    assert.equal(originalPart.state.attachments[0].mime, "video/mp4");
    const parts = [{ type: "text", text: "placeholder" }];
    await plugin["chat.message"]({}, { message: { id: "msg_1", sessionID: "ses_1" }, parts });
    assert.equal(parts[0].type, "file");
    assert.equal(parts[0].mime, "video/mp4");
    assert.equal(parts[0].url, url);
    assert.equal(parts[0].sessionID, "ses_1");
    const later = [{ type: "text", text: "later" }];
    await plugin["chat.message"]({}, { message: { id: "msg_2", sessionID: "ses_1" }, parts: later });
    assert.deepEqual(later, [{ type: "text", text: "later" }]);
  } finally {
    if (previous === undefined) delete process.env.NEMOTRON_OPENCODE_MEDIA_LAUNCHER;
    else process.env.NEMOTRON_OPENCODE_MEDIA_LAUNCHER = previous;
    await rm(dir, { recursive: true });
  }
});

test("media plugin leaves native compaction prompts unmodified and keeps visual tools", runtime, async () => {
  const NemotronMediaPlugin = await loadPlugin();
  const dir = await mkdtemp(path.join(os.tmpdir(), "nemotron-compaction-hook-"));
  const previous = process.env.NEMOTRON_OPENCODE_MEDIA_LAUNCHER;
  try {
    const profile = { schema_version: 2, patch_size: 16, temporal_patch_size: 2, downsample_ratio: 0.5,
      max_context_tokens: 262144, reserved_output_tokens: 32768, safety_tokens: 4096, video_max_pixels: 67108864 };
    const configPath = path.join(dir, "config.json"), launcherPath = path.join(dir, "launcher.json");
    await writeFile(configPath, JSON.stringify({ profile }), { flag: "wx", mode: 0o600 });
    await writeFile(launcherPath, JSON.stringify({ config_path: configPath }), { flag: "wx", mode: 0o600 });
    process.env.NEMOTRON_OPENCODE_MEDIA_LAUNCHER = launcherPath;
    const plugin = await NemotronMediaPlugin();
    assert.equal(Object.hasOwn(plugin, "experimental.session.compacting"), false);
    assert.equal(Object.hasOwn(plugin, "experimental.chat.system.transform"), false);
    assert.deepEqual(Object.keys(plugin), ["chat.headers", "tool", "chat.message", "experimental.chat.messages.transform"]);
    assert.deepEqual(Object.keys(plugin.tool), ["view_image", "watch_video"]);
    assert.deepEqual(Object.keys(plugin.tool.view_image.args), ["path"]);
    assert.deepEqual(Object.keys(plugin.tool.watch_video.args), ["video_path", "start", "end", "num_frames", "resize_long_edge"]);
  } finally {
    if (previous === undefined) delete process.env.NEMOTRON_OPENCODE_MEDIA_LAUNCHER;
    else process.env.NEMOTRON_OPENCODE_MEDIA_LAUNCHER = previous;
    await rm(dir, { recursive: true });
  }
});

test("multiple images remain grouped inside one call, while user images are unchanged", () => {
  const png = "data:image/png;base64,cG5n", jpeg = "data:image/jpeg;base64,anBlZw==";
  const original = [{ info: { role: "assistant" }, parts: [{ type: "tool", tool: "view_image", callID: "i1",
    state: { status: "completed", output: "Image: a.png", attachments: [
      { type: "file", mime: "image/png", url: png }, { type: "file", mime: "image/jpeg", url: jpeg },
    ] } }] }];
  const copy = bridgeMessages(original)[0].parts[0];
  assert.equal(copy.state.output, "");
  const transported = copy.state.attachments.map((a) => ({ type: "file-data", mediaType: a.mime, data: a.url.split(",")[1] }));
  const body = { messages: [{ role: "tool", tool_call_id: "i1", content: JSON.stringify(transported) },
    { role: "user", content: [{ type: "image_url", image_url: { url: png } }] }] };
  const restored = restoreToolMedia(body);
  assert.deepEqual(restored.messages[0].content.map((a) => a.image_url.url), [png, jpeg]);
  assert.equal(restored.messages[0].tool_call_id, "i1");
  assert.deepEqual(restored.messages[1], body.messages[1]);
  assert.equal(typeof body.messages[0].content, "string");
  assert.equal(original[0].parts[0].state.attachments[0].mime, "image/png");
});

test("pinned SDK preserves parallel mixed-media call order, including failures", runtime, async () => {
  const createNemotronMedia = await loadProvider();
  let sent;
  const provider = createNemotronMedia({ name: "nemo", baseURL: "http://model.test/v1", fetch: async (_, init) => {
    sent = JSON.parse(init.body);
    return new Response(JSON.stringify({ id: "test", object: "chat.completion", created: 1, model: "nano",
      choices: [{ index: 0, message: { role: "assistant", content: "seen" }, finish_reason: "stop" }] }),
      { headers: { "Content-Type": "application/json" } });
  } });
  const calls = [["i1", "view_image"], ["v1", "watch_video"], ["err1", "view_image"]];
  await provider.languageModel("nano").doGenerate({ prompt: [
    { role: "assistant", content: calls.map(([id, name]) => ({ type: "tool-call", toolCallId: id, toolName: name, input: {} })) },
    { role: "tool", content: [
      { type: "tool-result", toolCallId: "i1", toolName: "view_image", output: { type: "content", value: [{ type: "file-data", mediaType: TOOL_MEDIA_MIME_PREFIX + "image+png", data: "cG5n" }] } },
      { type: "tool-result", toolCallId: "v1", toolName: "watch_video", output: { type: "content", value: [{ type: "file-data", mediaType: TOOL_MEDIA_MIME_PREFIX + "video+mp4", data: bytes.toString("base64") }] } },
      { type: "tool-result", toolCallId: "err1", toolName: "view_image", output: { type: "error-text", value: "File not found" } },
    ] },
  ], maxOutputTokens: 16 });
  assert.deepEqual(sent.messages.slice(1).map((m) => m.tool_call_id), ["i1", "v1", "err1"]);
  assert.deepEqual(sent.messages.map((m) => m.role), ["assistant", "tool", "tool", "tool"]);
  assert.equal(sent.messages[1].content[0].image_url.url, "data:image/png;base64,cG5n");
  assert.equal(sent.messages[2].content[0].video_url.url, url);
  assert.equal(sent.messages[3].content, "File not found");
  assert.equal(JSON.stringify(sent).includes(TOOL_MEDIA_MIME_PREFIX), false);
});

test("transport fails closed for missing identity, malformed media and unknown MIME", () => {
  const media = { type: "media", mediaType: TOOL_MEDIA_MIME_PREFIX + "image+png", data: "cG5n" };
  for (const message of [
    { role: "tool", content: JSON.stringify([media]) },
    { role: "tool", tool_call_id: "i1", content: JSON.stringify([{ ...media, data: "https://bad.test" }]) },
    { role: "tool", tool_call_id: "i1", content: JSON.stringify([{ ...media, mediaType: TOOL_MEDIA_MIME_PREFIX + "audio+wav" }]) },
    { role: "tool", tool_call_id: "i1", content: JSON.stringify([media, { type: "unknown" }]) },
    { role: "tool", tool_call_id: "i1", content: JSON.stringify([{ ...media, mediaType: TOOL_MEDIA_MIME_PREFIX + "video+mp4" }]) },
    { role: "tool", tool_call_id: "i1", content: JSON.stringify([{ ...media, type: "unknown" }]) },
  ]) assert.throws(() => restoreToolMedia({ messages: [message] }));
  const normal = { messages: [{ role: "tool", tool_call_id: "i1", content: '[{"type":"text","text":"normal JSON"}]' }] };
  assert.deepEqual(restoreToolMedia(normal), normal);
});

test("the observed OpenCode file-data wrapper and unadapted media both restore", () => {
  for (const type of ["file-data", "media"]) {
    const body = { messages: [{ role: "tool", tool_call_id: "v1", content: JSON.stringify([
      { type, mediaType: TOOL_MEDIA_MIME_PREFIX + "video+mp4", data: bytes.toString("base64") },
    ]) }] };
    const restored = restoreToolMedia(body);
    assert.equal(restored.messages[0].content[0].video_url.url, url);
    assert.equal(JSON.stringify(restored).includes(TOOL_MEDIA_MIME_PREFIX), false);
    assert.equal(typeof body.messages[0].content, "string");
  }
});

test("English video receipt stays with the native attachment and the original call ID", runtime, async () => {
  const createNemotronMedia = await loadProvider();
  const receipt = "Frames: 120 total, 108 unique source frames (10.0% repeated)\nVisual tokens: 8640\nSampling interval: 0.5 s";
  const original = [{ info: { role: "assistant" }, parts: [{ type: "tool", tool: "watch_video", callID: "receipt1",
    state: { status: "completed", output: receipt, attachments: [{ type: "file", mime: "video/mp4", url }] } }] }];
  const bridged = bridgeMessages(original)[0].parts[0];
  assert.equal(bridged.state.output, receipt);
  assert.equal(original[0].parts[0].state.attachments[0].mime, "video/mp4");
  let sent;
  const provider = createNemotronMedia({ name: "nemo", baseURL: "http://model.test/v1", fetch: async (_, init) => {
    sent = JSON.parse(init.body);
    return new Response(JSON.stringify({ id: "receipt", object: "chat.completion", created: 1, model: "nano",
      choices: [{ index: 0, message: { role: "assistant", content: "seen" }, finish_reason: "stop" }] }),
      { headers: { "Content-Type": "application/json" } });
  } });
  await provider.languageModel("nano").doGenerate({ prompt: [
    { role: "assistant", content: [{ type: "tool-call", toolCallId: "receipt1", toolName: "watch_video", input: {} }] },
    { role: "tool", content: [{ type: "tool-result", toolCallId: "receipt1", toolName: "watch_video", output: { type: "content", value: [
      { type: "text", text: bridged.state.output },
      { type: "file-data", mediaType: bridged.state.attachments[0].mime, data: bytes.toString("base64") },
    ] } }] },
  ], maxOutputTokens: 16 });
  assert.equal(sent.messages.length, 2);
  assert.equal(sent.messages[1].role, "tool");
  assert.equal(sent.messages[1].tool_call_id, "receipt1");
  assert.equal(sent.messages[1].content[0].text, receipt);
  assert.equal(sent.messages[1].content[1].video_url.url, url);
});
