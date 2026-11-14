// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import { readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import path from "node:path";
import { pathToFileURL } from "node:url";

// OpenCode 1.17.11 extracts only image/PDF tool attachments, and its compatible
// SDK rejects video file parts. This private transport MIME is used only in a
// model-input copy; the persisted tool attachment remains video/mp4. The wire
// request is restored to ONE native video_url, never a gallery of stills.
export const VIDEO_BRIDGE_MIME = "image/x-nemotron-fixed-video";
// A non-media MIME keeps tool attachments inside their bound tool result in
// unmodified OpenCode. The compatible SDK transports that result as JSON text;
// the provider restores real media before counting, capture or inference.
export const TOOL_MEDIA_MIME_PREFIX = "application/x-nemotron-tool-";
const MAGIC = Buffer.from("NEMOTRON_FIXED_VIDEO_V1\n");
const INLINE_MAGIC = Buffer.from("NEMOTRON_FIXED_VIDEO_INLINE_V1\n");
// Match the shared codec's bounded v2 metadata allowance; it enforces v1 limits separately.
const MAX_INLINE_BYTES = 8589934592 + 4 * 1024 * 1024 + 64000 + 4096 + INLINE_MAGIC.length + 32;

export async function pinnedDependency(name, version, moduleFile) {
  const root = process.env.NEMOTRON_OPENCODE_DEPENDENCY_ROOT;
  if (!root || !path.isAbsolute(root)) throw new Error("absolute media dependency root is required");
  const pkg = path.join(root, name);
  const metadata = JSON.parse(await readFile(path.join(pkg, "package.json"), "utf8"));
  if (metadata.version !== version) throw new Error(`${name} must be pinned to ${version}`);
  return import(pathToFileURL(path.join(pkg, moduleFile)).href);
}

export function fixedDescriptorBytes(data) {
  let bytes;
  if (data instanceof Uint8Array) bytes = Buffer.from(data);
  else {
    const value = data instanceof URL ? data.href : data;
    if (typeof value !== "string") throw new Error("invalid fixed-video attachment data");
    const match = /^data:(?:video\/mp4|image\/x-nemotron-fixed-video);base64,([A-Za-z0-9+/]*={0,2})$/.exec(value);
    if (match) bytes = Buffer.from(match[1], "base64");
    else if (/^[A-Za-z0-9+/]*={0,2}$/.test(value)) bytes = Buffer.from(value, "base64");
    else throw new Error("only inline fixed-video data are supported");
  }
  if (bytes.subarray(0, INLINE_MAGIC.length).equals(INLINE_MAGIC)) {
    const offset = INLINE_MAGIC.length + 32;
    if (bytes.length <= offset || bytes.length > MAX_INLINE_BYTES ||
        !bytes.subarray(offset, offset + 4).equals(Buffer.from([80, 75, 3, 4])) ||
        !createHash("sha256").update(bytes.subarray(offset)).digest().equals(bytes.subarray(INLINE_MAGIC.length, offset))) {
      throw new Error("invalid inline fixed-video archive/hash");
    }
    return bytes;
  }
  if (bytes.length > 16384 || !bytes.subarray(0, MAGIC.length).equals(MAGIC)) {
    throw new Error("video is not a bounded NEMOTRON_FIXED_VIDEO_V1 descriptor");
  }
  const record = JSON.parse(bytes.subarray(MAGIC.length).toString("utf8"));
  if (Object.keys(record).sort().join(",") !== "archive_path,sha256" ||
      typeof record.archive_path !== "string" || !path.isAbsolute(record.archive_path) ||
      !/^[a-f0-9]{64}$/.test(record.sha256)) throw new Error("invalid fixed-video descriptor record");
  return bytes;
}

export function bridgeMessages(messages) {
  return messages.map((message) => ({
    ...message,
    parts: message.parts.map((part) => {
      if (part.type === "file" && part.mime === "video/mp4") {
        fixedDescriptorBytes(part.url);
        return { ...part, mime: VIDEO_BRIDGE_MIME };
      }
      if (part.type !== "tool" || part.state?.status !== "completed" || !part.state.attachments?.length) return part;
      const attachments = part.state.attachments.map((attachment) => {
        if (attachment.mime !== "video/mp4" && !attachment.mime?.startsWith("image/")) return attachment;
        if (typeof attachment.url !== "string" ||
            !attachment.url.startsWith(`data:${attachment.mime};base64,`)) {
          throw new Error("tool media must be materialized inline before model replay");
        }
        if (attachment.mime === "video/mp4") fixedDescriptorBytes(attachment.url);
        return { ...attachment, mime: TOOL_MEDIA_MIME_PREFIX + attachment.mime.replace("/", "+") };
      });
      const visualOnly = part.tool === "view_image" &&
        attachments.some((attachment) => attachment.mime.startsWith(TOOL_MEDIA_MIME_PREFIX));
      return { ...part, state: { ...part.state, output: visualOnly ? "" : part.state.output, attachments } };
    }),
  }));
}

export function restoreToolMedia(body) {
  return { ...body, messages: body.messages.map((message) => {
    if (message.role !== "tool" || typeof message.content !== "string") return message;
    let content;
    try { content = JSON.parse(message.content); }
    catch { return message; }
    if (!Array.isArray(content) || !content.some((part) =>
      part?.mediaType?.startsWith(TOOL_MEDIA_MIME_PREFIX))) return message;
    if (typeof message.tool_call_id !== "string" || !message.tool_call_id) {
      throw new Error("tool media has no bound call identity");
    }
    return { ...message, content: content.map((part) => {
      if (part?.type === "text" && typeof part.text === "string") return { type: "text", text: part.text };
      // Stock OpenCode's SDK-version adapter uses file-data on its V2 model
      // boundary. Also accept media when the same content has not been adapted.
      if (!["file-data", "media"].includes(part?.type) || !part.mediaType?.startsWith(TOOL_MEDIA_MIME_PREFIX) ||
          typeof part.data !== "string" || !part.data || !/^[A-Za-z0-9+/]*={0,2}$/.test(part.data)) {
        throw new Error("malformed or mixed tool-media transport content");
      }
      const mime = part.mediaType.slice(TOOL_MEDIA_MIME_PREFIX.length).replace("+", "/");
      const url = `data:${mime};base64,${part.data}`;
      if (mime === "video/mp4") {
        fixedDescriptorBytes(url);
        return { type: "video_url", video_url: { url } };
      }
      if (!/^image\/[a-zA-Z0-9.+-]+$/.test(mime)) throw new Error("unsupported tool-media MIME");
      return { type: "image_url", image_url: { url } };
    }) };
  }) };
}

export function bridgePrompt(prompt) {
  return prompt.map((message) => {
    if (message.role !== "user" || !Array.isArray(message.content)) return message;
    return { ...message, content: message.content.map((part) => {
      if (part.type !== "file" || !["video/mp4", VIDEO_BRIDGE_MIME].includes(part.mediaType)) return part;
      return { ...part, mediaType: VIDEO_BRIDGE_MIME, data: fixedDescriptorBytes(part.data) };
    }) };
  });
}

export function restoreNativeVideo(body) {
  return { ...body, messages: body.messages.map((message) => {
    if (!Array.isArray(message.content)) return message;
    return { ...message, content: message.content.map((part) => {
      if (part.type !== "image_url" || !part.image_url?.url?.startsWith(`data:${VIDEO_BRIDGE_MIME};base64,`)) return part;
      const data = fixedDescriptorBytes(part.image_url.url);
      const { image_url: ignored, ...fields } = part;
      return { ...fields, type: "video_url", video_url: { url: `data:video/mp4;base64,${data.toString("base64")}` } };
    }) };
  }) };
}
