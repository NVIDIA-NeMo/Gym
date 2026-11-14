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

import { randomUUID } from "node:crypto";
import { link, lstat, readFile, unlink, writeFile } from "node:fs/promises";
import path from "node:path";
import { fixedDescriptorBytes } from "./media_bridge.mjs";
import { privateDirectory, sha256 } from "./media_capture_io.mjs";

const MAX_IMAGE_BYTES = 32 << 20;
const MAX_VIDEO_BYTES = 193 << 20;
const INLINE_VIDEO_MAGIC = Buffer.from("NEMOTRON_FIXED_VIDEO_INLINE_V1\n");

async function verifiedImage(filename, digest, maxBytes = MAX_IMAGE_BYTES) {
  const info = await lstat(filename);
  if (!info.isFile() || info.isSymbolicLink() || info.size > maxBytes ||
      (info.mode & 0o077) !== 0 || info.uid !== process.getuid()) throw new Error("invalid private image snapshot");
  const bytes = await readFile(filename);
  if (sha256(bytes) !== digest) throw new Error("image snapshot SHA256 mismatch");
  return bytes;
}

async function immutableImage(filename, bytes, digest, maxBytes = MAX_IMAGE_BYTES) {
  const pending = path.join(path.dirname(filename), `.pending-${randomUUID()}`);
  await writeFile(pending, bytes, { flag: "wx", mode: 0o600 });
  try {
    try { await link(pending, filename); }
    catch (error) {
      if (error.code !== "EEXIST") throw error;
      await verifiedImage(filename, digest, maxBytes); // Published blobs are complete, even under concurrency.
    }
  } finally { await unlink(pending); }
}

function inlineImage(url) {
  if (typeof url !== "string" || url.length > Math.ceil(MAX_IMAGE_BYTES / 3) * 4 + 128) {
    throw new Error("image input exceeds the capture byte limit");
  }
  const match = /^data:(image\/[A-Za-z0-9.+-]+);base64,([A-Za-z0-9+/]+={0,2})$/.exec(url);
  if (!match) throw new Error("input capture requires a materialized inline image");
  const bytes = Buffer.from(match[2], "base64");
  if (bytes.length > MAX_IMAGE_BYTES || bytes.toString("base64") !== match[2]) {
    throw new Error("invalid or oversized inline image payload");
  }
  return { mime: match[1], bytes };
}

/** Snapshot one effective request, including compactor calls and backend failures.
 * Token IDs come from the generation response; unavailable IDs stay null. Token
 * IDs of a record without token_ids_source are labelled as a same-server tokenize
 * preflight (schema 1); restoreModelInput reads schema 1 and 2 snapshots.
 */
export async function captureModelInput({ directory, body, tokens, record }) {
  if (!/^[A-Za-z0-9_-]{1,128}$/.test(record.capture_id)) throw new Error("invalid capture identity");
  const unavailable = tokens === null;
  if (unavailable ? record.token_ids_sha256 !== null :
      !Array.isArray(tokens) || !tokens.every((id) => Number.isSafeInteger(id) && id >= 0) ||
      record.input_tokens !== tokens.length || sha256(JSON.stringify(tokens)) !== record.token_ids_sha256) {
    throw new Error("invalid capture token IDs");
  }
  await privateDirectory(directory, "input");
  const request = structuredClone(body), media = [];
  for (const [messageIndex, message] of request.messages.entries()) {
    if (!Array.isArray(message.content)) continue;
    for (const [partIndex, part] of message.content.entries()) {
      if (!["image_url", "video_url", "text"].includes(part.type)) {
        throw new Error("unsupported media content cannot be persisted safely");
      }
      if (part.type === "text") continue;
      const container = part[part.type], url = container?.url;
      let reference;
      if (part.type === "video_url") {
        if (typeof url !== "string" || !url.startsWith("data:video/mp4;base64,")) {
          throw new Error("input capture requires fixed-video data");
        }
        const bytes = fixedDescriptorBytes(url);
        if (bytes.subarray(0, INLINE_VIDEO_MAGIC.length).equals(INLINE_VIDEO_MAGIC)) {
          if (bytes.length > MAX_VIDEO_BYTES) throw new Error("inline video exceeds capture byte limit");
          const digest = sha256(bytes), blobs = path.join(directory, "video_blobs");
          await privateDirectory(blobs, "input");
          const filename = path.join(blobs, `${digest}.bin`);
          await immutableImage(filename, bytes, digest, MAX_VIDEO_BYTES);
          reference = { kind: "inline_video", path: filename, sha256: digest };
        } else {
          const descriptor = bytes.toString("utf8");
          const fixed = JSON.parse(descriptor.slice(descriptor.indexOf("\n") + 1));
          reference = { kind: "fixed_video", path: fixed.archive_path, sha256: fixed.sha256,
            descriptor_utf8: descriptor, descriptor_sha256: sha256(bytes) };
        }
      } else {
        const { mime, bytes } = inlineImage(url), digest = sha256(bytes);
        const blobs = path.join(directory, "image_blobs");
        await privateDirectory(blobs, "input");
        const filename = path.join(blobs, `${digest}.bin`);
        await immutableImage(filename, bytes, digest);
        reference = { kind: "image", path: filename, sha256: digest, mime };
      }
      const id = `media_${media.length}`;
      media.push({ ...reference, id, message_index: messageIndex, part_index: partIndex, field: part.type,
        url_sha256: sha256(url) });
      container.url = { media_reference: id };
    }
  }
  const tokenSource = record.token_ids_source ?? (unavailable ? null : "same_server_native_tokenize_preflight");
  const snapshot = { schema_version: tokenSource === "same_server_native_tokenize_preflight" ? 1 : 2, kind: "native_model_input_snapshot",
    capture_id: record.capture_id, invocation_id: record.invocation_id, agent: record.agent,
    parent_message_id: record.parent_message_id ?? null,
    captured_at: record.started_at, admission_action: record.action,
    token_ids_source: tokenSource,
    input_tokens: record.input_tokens, input_token_ids: tokens, token_ids_sha256: record.token_ids_sha256,
    wire_request_sha256: record.request_sha256, canonical_request_sha256: sha256(JSON.stringify(body)),
    request, media_references: media };
  const filename = path.join(directory, `${record.capture_id}.json`);
  await writeFile(filename, JSON.stringify(snapshot), { flag: "wx", mode: 0o600 });
  return { path: filename, sha256: sha256(JSON.stringify(snapshot)) };
}

/** Rehydrate a private snapshot without flattening the pre/post-compact history. */
export async function restoreModelInput(snapshot) {
  const unavailable = snapshot.schema_version === 2 && snapshot.input_token_ids === null &&
    snapshot.token_ids_source === null && snapshot.token_ids_sha256 === null;
  if (![1, 2].includes(snapshot.schema_version) || snapshot.kind !== "native_model_input_snapshot" ||
      (!unavailable && (!Array.isArray(snapshot.input_token_ids) ||
      !snapshot.input_token_ids.every((id) => Number.isSafeInteger(id) && id >= 0) ||
      snapshot.input_tokens !== snapshot.input_token_ids.length ||
      sha256(JSON.stringify(snapshot.input_token_ids)) !== snapshot.token_ids_sha256))) {
    throw new Error("invalid model input snapshot or token IDs");
  }
  const request = structuredClone(snapshot.request), seen = new Set();
  for (const reference of snapshot.media_references) {
    const container = request.messages[reference.message_index]?.content[reference.part_index]?.[reference.field];
    if (!container || container.url?.media_reference !== reference.id || seen.has(reference.id)) {
      throw new Error("unbound or duplicate media snapshot reference");
    }
    seen.add(reference.id);
    let url;
    if (reference.kind === "image") {
      const bytes = await verifiedImage(reference.path, reference.sha256);
      url = `data:${reference.mime};base64,${bytes.toString("base64")}`;
    } else if (reference.kind === "inline_video") {
      const bytes = await verifiedImage(reference.path, reference.sha256, MAX_VIDEO_BYTES);
      if (!bytes.subarray(0, INLINE_VIDEO_MAGIC.length).equals(INLINE_VIDEO_MAGIC)) {
        throw new Error("inline video snapshot has no self-contained media");
      }
      fixedDescriptorBytes(bytes);
      url = `data:video/mp4;base64,${bytes.toString("base64")}`;
    } else if (reference.kind === "fixed_video") {
      const bytes = Buffer.from(reference.descriptor_utf8, "utf8");
      const descriptor = JSON.parse(bytes.subarray(bytes.indexOf(10) + 1).toString("utf8"));
      if (sha256(bytes) !== reference.descriptor_sha256 || descriptor.archive_path !== reference.path ||
          descriptor.sha256 !== reference.sha256) throw new Error("fixed-video snapshot descriptor mismatch");
      fixedDescriptorBytes(bytes);
      url = `data:video/mp4;base64,${bytes.toString("base64")}`;
    } else throw new Error("unsupported media snapshot reference");
    if (sha256(url) !== reference.url_sha256) throw new Error("media wire URL SHA256 mismatch");
    container.url = url;
  }
  if (sha256(JSON.stringify(request)) !== snapshot.canonical_request_sha256) {
    throw new Error("restored model request SHA256 mismatch");
  }
  return request;
}
