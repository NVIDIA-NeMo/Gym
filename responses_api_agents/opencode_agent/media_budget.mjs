// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A model-visible explanation, derived from the same trusted profile as decoding.
// This is not an additional budget, resize policy, or admission implementation.

// Per-frame area policy; keep aligned with video_media_contract.py.
export const VIDEO_MAX_FRAME_PIXELS = 2048 * 2048;
export const IMAGE_MAX_PIXELS = 2048 * 2048;

export function imageToolDescription() {
  return "View one local image file based on known dimensions and file size. " +
    "path may be absolute or workspace-relative, but must resolve to a regular image file inside the configured workspace; URLs and paths escaping it are not supported. " +
    `H*W <= ${IMAGE_MAX_PIXELS} pixels (2048*2048), an area limit, not a long-edge limit. ` +
    "File size must be <= 33554432 bytes (32 MiB). Returns original image bytes, not a caption; this tool does not resize. " +
    "If too large, create a separate smaller image inside the workspace and pass its path.";
}

export function videoToolDescription(profile) {
  const { patch_size: patch, temporal_patch_size: temporal,
    downsample_ratio: ratio } = profile ?? {};
  if (![patch, temporal].every((value) => Number.isSafeInteger(value) && value > 0) ||
      typeof ratio !== "number" || !Number.isFinite(ratio) || ratio <= 0 || ratio > 1 ||
      !Number.isSafeInteger(1 / ratio)) {
    throw new Error("media config must contain the fully resolved NemotronMediaProfile");
  }
  if (Object.hasOwn(profile, "video_max_target_token")) {
    throw new Error("video_max_target_token was removed; use complete-request context admission");
  }
  if (profile.schema_version !== undefined && ![1, 2].includes(profile.schema_version)) {
    throw new Error("unsupported NemotronMediaProfile schema_version");
  }
  if (!Number.isSafeInteger(profile.video_max_pixels) || profile.video_max_pixels <= 0) {
    throw new Error("media profile video_max_pixels must be a positive exact integer");
  }
  const merge = 1 / ratio, alignment = patch * merge;
  if (![alignment, merge * merge].every(Number.isSafeInteger)) {
    throw new Error("media profile geometry exceeds exact integer range");
  }
  const sampling = profile.schema_version === 1
    ? `num_frames is a count of unique sampled frames, not FPS: at least ${temporal}, divisible by ${temporal}, and no more than available frames in the window. `
    : `num_frames is the total sampled frame count, not FPS: at least ${temporal}, divisible by ${temporal}. Sampling times are uniform across the window. A source frame may repeat while held until the next original PTS; temporal labels are sampling times. `;
  return "Watch one native video based on known source metadata. If duration or dimensions are unknown, obtain them with available tools first; do not guess. " +
    "Choose absolute source seconds [start,end), with 0 <= start < end <= source duration. " + sampling +
    `resize_long_edge is the requested long edge in pixels. Preserve source aspect ratio; actual H/W round down to multiples of ${alignment}, with each dimension at least ${alignment}. ` +
    `Per call, num_frames*H*W <= ${profile.video_max_pixels} pixels (video_max_pixels). Per frame, H*W <= ${VIDEO_MAX_FRAME_PIXELS} pixels (2048*2048), an area limit, not a long-edge limit. Use actual H/W, not resize_long_edge squared. ` +
    "Invalid arguments return a tool error; choose fewer frames or a smaller resize. No silent frame reduction or resizing.";
}
