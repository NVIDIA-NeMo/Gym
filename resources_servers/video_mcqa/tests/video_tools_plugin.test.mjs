// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The video tools plugin (sandbox_image/video_tools.mjs) driven through the hooks OpenCode calls, with a stand-in
// video_tools program. test_video_tools_plugin.py stages the plugin in VIDEO_TOOLS_DIR and runs this script.
import assert from "node:assert/strict";
import { writeFileSync } from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const dir = process.env.VIDEO_TOOLS_DIR;
const pluginURL = pathToFileURL(path.join(dir, "video_tools.mjs")).href;
const configPath = path.join(dir, "tools.json");
const program = path.join(dir, "program.mjs");
// The stand-in program answers by the requested path, as video_tools/cli.py would answer.
writeFileSync(program, `
let input = "";
process.stdin.on("data", (chunk) => { input += chunk; });
process.stdin.on("end", async () => {
  const request = JSON.parse(input);
  const name = request.arguments.video_path ?? request.arguments.path;
  if (!process.argv.includes("--config")) process.exit(9);
  if (name === "exit.mp4") { process.stderr.write("decoder crashed"); process.exit(3); }
  if (name === "slow.mp4") await new Promise((resolve) => setTimeout(resolve, 10000));
  if (name === "error.mp4") {
    process.stdout.write(JSON.stringify({ error: { code: "invalid_source_range", message: "end exceeds", details: {} }, output: "end exceeds" }));
    return;
  }
  const mime = name === "text.mp4" ? "text/plain" : "image/jpeg";
  const image = (byte) => ({ mime, url: "data:" + mime + ";base64," + Buffer.from([byte]).toString("base64") });
  const padding = name === "big.mp4" ? "x".repeat(4096) : "";
  process.stdout.write(JSON.stringify({ output: "Video: " + name + "\\nThis is a video:\\nFrame 1 sampled at 0.25 seconds:\\nFrame 2 sampled at 0.75 seconds:" + padding,
    attachments: [image(1), image(2)], metadata: { tool: request.tool } }));
});
`);

function configure(updates = {}) {
  writeFileSync(configPath, JSON.stringify({
    workspace_root: "/testbed", cache_dir: "/tmp/video-mcqa/index",
    profile: { patch_size: 14, downsample_ratio: 0.5, video_max_pixels: 67108864, decode_timeout_seconds: 120, max_materialized_bytes: 8589934592 },
    command: [process.execPath, program], timeout_seconds: 30, max_output_bytes: 1 << 20, max_tool_media_bytes: 419430400,
    ...updates,
  }));
}

const call = (plugin, name, args, abort = new AbortController().signal) => plugin.tool[name].execute(args, { abort });
const video = (videoPath) => ({ video_path: videoPath, start: 0, end: 1, num_frames: 2, resize_long_edge: 448 });

configure();
const mod = await import(pluginURL);
// OpenCode calls every export of a plugin module as a plugin, so the module exports only its plugin.
assert.deepEqual(Object.keys(mod), ["default"]);
const plugin = await mod.default();
assert.deepEqual(Object.keys(plugin.tool).sort(), ["view_image", "watch_video"]);
const watch = plugin.tool.watch_video;
assert.match(watch.description, /actual H\/W round down to multiples of 28, with each dimension at least 28/);
assert.match(watch.description, /num_frames is the total sampled frame count, not FPS: at least 1, divisible by 1\./);
assert.match(watch.description, /Per call, num_frames\*H\*W <= 67108864 pixels \(video_max_pixels\)/);
assert.deepEqual(Object.keys(watch.args), ["video_path", "start", "end", "num_frames", "resize_long_edge"]);
assert.deepEqual(watch.args.num_frames, { type: "integer", exclusiveMinimum: 0, maximum: Number.MAX_SAFE_INTEGER });
assert.deepEqual(plugin.tool.view_image.args, { path: { type: "string", minLength: 1 } });

// Frames: the text as the program wrote it, the images as attachments in order.
const frames = await call(plugin, "watch_video", video("videos/a.mp4"));
assert.equal(frames.title, "watch_video");
assert.match(frames.output, /^Video: videos\/a\.mp4\nThis is a video:\nFrame 1 sampled at 0\.25 seconds:/);
assert.deepEqual(frames.attachments, [
  { type: "file", mime: "image/jpeg", url: "data:image/jpeg;base64,AQ==", filename: "videos/a.mp4" },
  { type: "file", mime: "image/jpeg", url: "data:image/jpeg;base64,Ag==", filename: "videos/a.mp4" },
]);
assert.deepEqual(frames.metadata, { video_tools: { tool: "watch_video" } });

// A tool error is the model's to read, not a failed call.
const error = await call(plugin, "watch_video", video("error.mp4"));
assert.deepEqual(JSON.parse(error.output), { code: "invalid_source_range", message: "end exceeds", details: {} });
assert.equal(error.attachments, undefined);
assert.equal(error.metadata.video_tools_error.code, "invalid_source_range");

// A program that fails, a result that is not an image, and an aborted call fail the call.
await assert.rejects(call(plugin, "watch_video", video("exit.mp4")), /video tool exited 3: decoder crashed/);
await assert.rejects(call(plugin, "view_image", { path: "text.mp4" }), /invalid video tool image/);
const controller = new AbortController();
const aborted = call(plugin, "watch_video", video("slow.mp4"), controller.signal);
setTimeout(() => controller.abort(), 200);
await assert.rejects(aborted, /aborted/);

// The deadline and the output limit come from tools.json.
configure({ timeout_seconds: 0.3 });
const strict = await mod.default();
await assert.rejects(call(strict, "watch_video", video("slow.mp4")), /exceeded its deadline/);
configure({ max_output_bytes: 1024 });
await assert.rejects(call(await mod.default(), "watch_video", video("big.mp4")), /exceeded its output limit/);

// Every request sends earlier tool images again: from the result that would pass the episode's budget on, results
// show a notice, the same in every request, and the stored messages stay unchanged.
configure({ max_tool_media_bytes: 100 });
const budgeted = await mod.default();
const image = (n) => ({ type: "file", mime: "image/jpeg", url: "data:image/jpeg;base64," + "A".repeat(n) });
const part = (id, tool, attachments) => ({ type: "tool", tool, callID: id, state: { status: "completed", output: id, attachments } });
const size = image(20).url.length;
const history = () => [
  { info: { role: "assistant" }, parts: [part("a", "watch_video", [image(20)]), { type: "text", text: "between" }] },
  { info: { role: "assistant" }, parts: [part("b", "read", [image(20)])] },
  { info: { role: "assistant" }, parts: [part("c", "view_image", [image(100)])] },
  { info: { role: "assistant" }, parts: [part("d", "watch_video", [image(1)])] },
];
assert.ok(2 * size <= 100 && 3 * size > 100);
const original = history();
const stored = original.map((message) => message.parts[0]);
const first = { messages: original };
await budgeted["experimental.chat.messages.transform"]({}, first);
const second = { messages: history() };
await budgeted["experimental.chat.messages.transform"]({}, second);
assert.deepEqual(first.messages, second.messages);
assert.equal(first.messages, original, "OpenCode keeps its own reference to the array");
assert.equal(first.messages[0].parts[0].state.attachments.length, 1);
assert.equal(first.messages[1].parts[0].state.attachments.length, 1);
for (const index of [2, 3]) {
  // "d" alone would fit what is left, but nothing is sent after the first result over the budget.
  assert.deepEqual(first.messages[index].parts[0].state.attachments, []);
  assert.match(first.messages[index].parts[0].state.output, /^Tool result images not shown: this episode's tool images reached their 0 MiB limit/);
}
assert.deepEqual(first.messages[0].parts[1], { type: "text", text: "between" });
assert.equal(stored[2].state.attachments.length, 1, "the stored tool part keeps its images");

console.log("video tools plugin: all checks passed");
