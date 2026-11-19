// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// OpenCode plugin with the video tools watch_video and view_image, for OpenCode 1.17.11 in the video MCQA sandbox
// image (Dockerfile beside this file). Each call runs the video_tools program (tools.json's command, the image's
// `python3 -I -m video_tools.cli`) with the configuration that the resources server writes beside this file at each
// seed (tools.json, video_tools/contract.py ToolConfig).
//
// A tool returns its images as attachments. For an OpenAI-compatible provider, OpenCode keeps a tool result's text in
// the tool message and moves the images of all the tool results of one response, in call order, into one user message
// after them ("Attached media from tool result:"). So watch_video's text says that its frames are attached after the
// tool results and lists each frame's time in the order of the images, and its description gives the call order.
//
// Every request sends the episode's earlier tool images again. From the tool result whose images would pass
// max_tool_media_bytes (base64 bytes, all tools together) on, each result shows a notice in place of its images, the
// same in every request, so OpenCode's request stays under Node's string limit.
//
// The tools take plain JSON Schema arguments (OpenCode builds a tool's parameters from them), so the plugin imports
// nothing outside Node's standard library. The video_tools program validates every argument.

import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const CONFIG_PATH = path.join(ROOT, "tools.json");
const MAX_STDERR_BYTES = 64 * 1024;
// Same as video_tools/contract.py MAX_FRAME_PIXELS and MAX_IMAGE_BYTES.
const MAX_FRAME_PIXELS = 2048 * 2048;
const MAX_IMAGE_BYTES = 32 * 1024 * 1024;

function imageToolDescription() {
  return "View one local image file based on known dimensions and file size. " +
    "path may be absolute or workspace-relative, but must resolve to a regular image file inside the configured workspace; URLs and paths escaping it are not supported. " +
    `H*W <= ${MAX_FRAME_PIXELS} pixels (2048*2048), an area limit, not a long-edge limit. ` +
    `File size must be <= ${MAX_IMAGE_BYTES} bytes (32 MiB). Returns original image bytes, not a caption; this tool does not resize. ` +
    "If too large, create a separate smaller image inside the workspace and pass its path.";
}

function videoToolDescription(profile) {
  const { patch_size: patch, downsample_ratio: ratio, video_max_pixels: maxPixels } = profile ?? {};
  if (!Number.isSafeInteger(patch) || patch <= 0 || typeof ratio !== "number" || !(ratio > 0 && ratio <= 1) ||
      !Number.isSafeInteger(1 / ratio) || !Number.isSafeInteger(maxPixels) || maxPixels <= 0) {
    throw new Error("tools.json must contain a valid frame profile");
  }
  const alignment = patch / ratio;
  // One frame per temporal group: every frame is one image.
  const temporal = 1;
  return "Watch one native video based on known source metadata. If duration or dimensions are unknown, obtain them with available tools first; do not guess. " +
    "Choose absolute source seconds [start,end), with 0 <= start < end <= source duration. " +
    `num_frames is the total sampled frame count, not FPS: at least ${temporal}, divisible by ${temporal}. Sampling times are uniform across the window. A source frame may repeat while held until the next original PTS; temporal labels are sampling times. ` +
    `resize_long_edge is the requested long edge in pixels. Preserve source aspect ratio; actual H/W round down to multiples of ${alignment}, with each dimension at least ${alignment}. ` +
    `Per call, num_frames*H*W <= ${maxPixels} pixels (video_max_pixels). Per frame, H*W <= ${MAX_FRAME_PIXELS} pixels (2048*2048), an area limit, not a long-edge limit. Use actual H/W, not resize_long_edge squared. ` +
    "Invalid arguments return a tool error; choose fewer frames or a smaller resize. No silent frame reduction or resizing. " +
    "The frames are attached as images after the tool results, in the order the result lists them; images from several tool calls in one response follow in call order.";
}

// The JSON Schema of each tool's arguments; OpenCode sends it as the tool's parameters.
const VIDEO_ARGS = {
  video_path: { type: "string", minLength: 1 },
  start: { type: "number", minimum: 0 },
  end: { type: "number", exclusiveMinimum: 0 },
  num_frames: { type: "integer", exclusiveMinimum: 0, maximum: Number.MAX_SAFE_INTEGER },
  resize_long_edge: { type: "integer", exclusiveMinimum: 0, maximum: Number.MAX_SAFE_INTEGER },
};
const IMAGE_ARGS = { path: { type: "string", minLength: 1 } };

function attachmentBytes(part) {
  if (part.type !== "tool" || part.state?.status !== "completed") return 0;
  return (part.state.attachments ?? []).reduce((total, attachment) => total + (attachment.url?.length ?? 0), 0);
}

/** Drop the images of every tool result from the one that passes the budget on, in history order, so that each
 * request shows the same earlier history. The stored messages stay unchanged. */
function limitToolMedia(messages, maxBytes) {
  if (!Number.isSafeInteger(maxBytes) || maxBytes <= 0) throw new Error("tool media budget must be a positive integer");
  const limit = `${Math.floor(maxBytes / (1024 * 1024))} MiB`;
  let used = 0, exhausted = false;
  return messages.map((message) => ({
    ...message,
    parts: message.parts.map((part) => {
      const bytes = attachmentBytes(part);
      if (!bytes) return part;
      if (!exhausted && used + bytes <= maxBytes) {
        used += bytes;
        return part;
      }
      exhausted = true;
      return { ...part, state: { ...part.state, attachments: [],
        output: `Tool result images not shown: this episode's tool images reached their ${limit} limit, ` +
          "so no further images can be sent. Continue with the evidence you have." } };
    }),
  }));
}

function invoke(config, request, abort) {
  return new Promise((resolve, reject) => {
    if (abort?.aborted) return reject(new Error("video tool was aborted"));
    const [executable, ...argv] = config.command;
    const child = spawn(executable, [...argv, "--config", CONFIG_PATH], { stdio: ["pipe", "pipe", "pipe"], detached: true });
    const chunks = [], errors = [];
    let size = 0, errorSize = 0, failure;
    const stop = (message) => {
      failure ??= new Error(message);
      try {
        process.kill(-child.pid, "SIGKILL");
      } catch (error) { if (error.code !== "ESRCH") failure = error; }
    };
    const onAbort = () => stop("video tool was aborted");
    const timer = setTimeout(() => stop("video tool exceeded its deadline"), config.timeout_seconds * 1000);
    abort?.addEventListener("abort", onAbort, { once: true });
    child.stdout.on("data", (chunk) => {
      size += chunk.length;
      if (size > config.max_output_bytes) stop("video tool exceeded its output limit");
      else chunks.push(chunk);
    });
    child.stderr.on("data", (chunk) => {
      errorSize += chunk.length;
      if (errorSize > MAX_STDERR_BYTES) stop("video tool exceeded its stderr limit");
      else errors.push(chunk);
    });
    child.stdin.on("error", (error) => { failure ??= error; });
    child.on("error", (error) => { failure ??= error; });
    child.on("close", (code) => {
      clearTimeout(timer);
      abort?.removeEventListener("abort", onAbort);
      if (failure) return reject(failure);
      if (code !== 0) return reject(new Error(`video tool exited ${code}: ${Buffer.concat(errors).toString("utf8").slice(0, 4096)}`));
      try { resolve(JSON.parse(Buffer.concat(chunks).toString("utf8"))); }
      catch (error) { reject(new Error("video tool did not return valid JSON", { cause: error })); }
    });
    child.stdin.end(JSON.stringify(request));
  });
}

export default async function VideoToolsPlugin() {
  const config = JSON.parse(readFileSync(CONFIG_PATH, "utf8"));
  if (!Array.isArray(config.command) || !config.command.length || !(config.timeout_seconds > 0) ||
      !Number.isSafeInteger(config.max_output_bytes) || config.max_output_bytes <= 0) {
    throw new Error("tools.json must set command, timeout_seconds and max_output_bytes");
  }
  const videoDescription = videoToolDescription(config.profile);
  const execute = (name) => async (args, context) => {
    const result = await invoke(config, { tool: name, arguments: args }, context.abort);
    if (result.error) return { title: name, output: JSON.stringify(result.error), metadata: { video_tools_error: result.error } };
    if (typeof result.output !== "string" || !Array.isArray(result.attachments) || !result.attachments.length) {
      throw new Error("video tool result has no image");
    }
    const attachments = result.attachments.map((attachment) => {
      if (typeof attachment?.url !== "string" || !attachment.url.startsWith(`data:${attachment.mime};base64,`) ||
          !attachment.mime.startsWith("image/")) {
        throw new Error("invalid video tool image");
      }
      return { type: "file", mime: attachment.mime, url: attachment.url, filename: args.video_path ?? args.path };
    });
    return { title: name, output: result.output, metadata: { video_tools: result.metadata }, attachments };
  };
  return {
    tool: {
      view_image: { description: imageToolDescription(), args: IMAGE_ARGS, execute: execute("view_image") },
      watch_video: { description: videoDescription, args: VIDEO_ARGS, execute: execute("watch_video") },
    },
    "experimental.chat.messages.transform": async (_, output) => {
      // OpenCode keeps its own reference to this array, so the transform edits it in place.
      output.messages.splice(0, output.messages.length, ...limitToolMedia(output.messages, config.max_tool_media_bytes));
    },
  };
}
