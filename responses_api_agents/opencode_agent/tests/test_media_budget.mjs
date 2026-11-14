// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { test } from "node:test";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { IMAGE_MAX_PIXELS, VIDEO_MAX_FRAME_PIXELS, imageToolDescription, videoToolDescription } from "../media_budget.mjs";

// The media plugin loads OpenCode's pinned SDK from the media runtime when imported.
const runtime = { skip: process.env.NEMOTRON_OPENCODE_DEPENDENCY_ROOT ? false : "NEMOTRON_OPENCODE_DEPENDENCY_ROOT is unset" };

const profile = { schema_version: 1, patch_size: 16, temporal_patch_size: 2, downsample_ratio: 0.5,
  max_context_tokens: 262144, reserved_output_tokens: 32768, safety_tokens: 4096,
  video_max_pixels: 67108864, max_materialized_bytes: 8589934592 };

test("image description states the enforced path, byte and area limits without changing video policy", () => {
  const text = imageToolDescription();
  assert.equal(IMAGE_MAX_PIXELS, 4194304);
  for (const expected of ["known dimensions and file size", "absolute or workspace-relative",
    "regular image file inside the configured workspace", "URLs and paths escaping it are not supported",
    "H*W <= 4194304 pixels (2048*2048)", "area limit, not a long-edge limit",
    "<= 33554432 bytes (32 MiB)", "Returns original image bytes", "this tool does not resize",
    "create a separate smaller image"]) assert.ok(text.includes(expected), expected);
  assert.doesNotMatch(text, /num_frames|video_max_pixels|compact|225280|ffprobe|ffmpeg/);
  assert.ok(text.split(/\s+/).length < 130);
});

test("metadata, window and exact geometry constraints are visible without prescribing a utility", () => {
  const text = videoToolDescription(profile);
  for (const expected of ["known source metadata", "If duration or dimensions are unknown",
    "available tools first; do not guess", "absolute source seconds [start,end)",
    "0 <= start < end <= source duration", "unique sampled frames, not FPS",
    "at least 2, divisible by 2", "no more than available frames in the window",
    "requested long edge in pixels", "actual H/W round down to multiples of 32",
    "each dimension at least 32"]) assert.ok(text.includes(expected), expected);
  assert.doesNotMatch(text, /ffprobe|ffmpeg|probe_video|shell command/i);
});

test("total pixels and per-frame area have separate exact ceilings", () => {
  const text = videoToolDescription(profile);
  assert.equal(VIDEO_MAX_FRAME_PIXELS, 4194304);
  for (const expected of ["num_frames*H*W <= 67108864", "video_max_pixels",
    "H*W <= 4194304", "area limit, not a long-edge limit",
    "Use actual H/W, not resize_long_edge squared", "No silent frame reduction or resizing"])
    assert.ok(text.includes(expected), expected);
  assert.doesNotMatch(text, /ViT|target=|video_max_target_token|native compaction/);
  assert.ok(text.split(/\s+/).length < 260);
});

test("uniform-time sampling permits repeated source frames without an implicit FPS", () => {
  const text = videoToolDescription({ ...profile, schema_version: 2 });
  for (const expected of ["total sampled frame count, not FPS", "at least 2, divisible by 2",
    "Sampling times are uniform across the window", "source frame may repeat",
    "temporal labels are sampling times"]) assert.ok(text.includes(expected), expected);
  assert.doesNotMatch(text, /unique sampled frames|no more than available frames/);
  assert.ok(text.split(/\s+/).length < 260);
});

test("an omitted profile version follows the fresh Python profile semantics", () => {
  const { schema_version, ...unversioned } = profile;
  assert.equal(videoToolDescription(unversioned), videoToolDescription({ ...profile, schema_version: 2 }));
});

test("pixel ceiling stays fixed at 256K and 1M unless explicitly overridden", () => {
  const big = { ...profile, max_context_tokens: 1048576, reserved_output_tokens: 131072, safety_tokens: 16384 };
  for (const setting of [profile, big]) assert.ok(videoToolDescription(setting).includes("num_frames*H*W <= 67108864"));
  assert.equal(videoToolDescription(profile), videoToolDescription(big));
  assert.ok(videoToolDescription({ ...big, video_max_pixels: 134217728 }).includes("num_frames*H*W <= 134217728"));
  assert.equal(profile.video_max_pixels, 67108864);
});

test("tool text contains only tool constraints, not whole-request admission policy", () => {
  const text = videoToolDescription(profile);
  assert.doesNotMatch(text, /including history|complete next model input|225280|901120|terminates the trajectory/);
  assert.match(text, /Invalid arguments return a tool error/);
  assert.doesNotMatch(text, /compact|replay|per-video token cap/);
});

test("grid uses actual selected model geometry", () => {
  const text = videoToolDescription({ ...profile, patch_size: 14, temporal_patch_size: 4 });
  assert.match(text, /at least 4, divisible by 4/);
  assert.match(text, /multiples of 28/);
});

test("description does not promise full-context diagnostics in the receipt", () => {
  assert.doesNotMatch(videoToolDescription(profile), /Results report|exact token counts|returns? dimensions/i);
});

test("invalid geometry and pixel caps fail before advertising the tool", () => {
  for (const changes of [{ video_max_pixels: undefined }, { video_max_pixels: 0 },
    { video_max_pixels: -1 }, { video_max_pixels: 1.5 }, { video_max_pixels: 2**53 },
    { video_max_pixels: true }, { video_max_pixels: "67108864" },
    { patch_size: 16.5 }, { temporal_patch_size: 0 },
    { downsample_ratio: 0 }, { downsample_ratio: 0.3 }])
    assert.throws(() => videoToolDescription({ ...profile, ...changes }));
  assert.throws(() => videoToolDescription(undefined));
  for (const schema_version of [0, 3, true, "2", null])
    assert.throws(() => videoToolDescription({ ...profile, schema_version }), /schema_version/);
  assert.throws(() => videoToolDescription({ ...profile, video_max_target_token: 655360 }), /was removed/);
});

test("model-facing schema excludes trusted budget overrides", runtime, async () => {
  const directory = await mkdtemp(path.join(os.tmpdir(), "nemotron-budget-test-"));
  const prior = process.env.NEMOTRON_OPENCODE_MEDIA_LAUNCHER;
  try {
    const configPath = path.join(directory, "config.json"), launcherPath = path.join(directory, "launcher.json");
    await writeFile(configPath, JSON.stringify({ profile }));
    await writeFile(launcherPath, JSON.stringify({ config_path: configPath }));
    process.env.NEMOTRON_OPENCODE_MEDIA_LAUNCHER = launcherPath;
    const { default: plugin } = await import("../nemotron_media_plugin.mjs");
    const hooks = await plugin();
    assert.equal(hooks.tool.view_image.description, imageToolDescription());
    assert.equal(hooks.tool.watch_video.description, videoToolDescription(profile));
    assert.deepEqual(Object.keys(hooks.tool.watch_video.args), ["video_path", "start", "end", "num_frames", "resize_long_edge"]);
    assert.deepEqual(Object.keys(hooks.tool.view_image.args), ["path"]);
  } finally {
    if (prior === undefined) delete process.env.NEMOTRON_OPENCODE_MEDIA_LAUNCHER;
    else process.env.NEMOTRON_OPENCODE_MEDIA_LAUNCHER = prior;
    await rm(directory, { recursive: true });
  }
});
