// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createHash, randomUUID } from "node:crypto";
import { readFileSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { captureModelInput } from "./media_input_capture.mjs";
import { captureModelOutput, outputCaptureRequest } from "./media_output_capture.mjs";
import { privateDirectory } from "./media_capture_io.mjs";

const sha256 = (value) => createHash("sha256").update(value).digest("hex");

function modelOrigin(baseURL) {
  const base = new URL(baseURL);
  if (!["http:", "https:"].includes(base.protocol) || base.username || base.password || base.search || base.hash ||
      !/\/v1\/?$/.test(base.pathname)) throw new Error("native model base URL must end in /v1 without credentials or query parameters");
  return base.origin;
}

export function launcherContext(options) {
  const filename = process.env.NEMOTRON_OPENCODE_MEDIA_LAUNCHER;
  if (!filename) return null;
  const launcher = JSON.parse(readFileSync(filename, "utf8"));
  const config = JSON.parse(readFileSync(launcher.config_path, "utf8"));
  const profile = config.profile;
  if (![profile.max_context_tokens, profile.reserved_output_tokens].every((n) => Number.isSafeInteger(n) && n > 0) ||
      profile.max_context_tokens <= profile.reserved_output_tokens ||
      profile.max_context_tokens !== launcher.context_window || profile.reserved_output_tokens !== launcher.max_output_tokens) {
    throw new Error("OpenCode context/output limits must match the media profile exactly");
  }
  modelOrigin(options.baseURL);
  if (!path.isAbsolute(config.artifact_root) || !/^opencode_[a-f0-9]+$/.test(launcher.context_namespace)) {
    throw new Error("invalid private context-statistics workspace");
  }
  if (typeof launcher.capture_model_inputs !== "boolean") {
    throw new Error("trusted launcher must explicitly declare model-input capture permission");
  }
  return { profile, baseURL: options.baseURL,
    directory: path.join(config.artifact_root, "context_stats", launcher.context_namespace),
    captureDirectory: launcher.capture_model_inputs
      ? path.join(config.artifact_root, "model_inputs", launcher.context_namespace) : null };
}

export function assertToolBundles(messages) {
  const outstanding = [], seen = new Set();
  for (const message of messages) {
    if (message.role === "tool") {
      if (!outstanding.length || outstanding.shift() !== message.tool_call_id) throw new Error("unbound, duplicate or out-of-order tool result");
      continue;
    }
    if (outstanding.length) throw new Error("incomplete tool-result bundle before the next message/media");
    for (const call of message.tool_calls ?? []) {
      if (message.role !== "assistant" || typeof call.id !== "string" || !call.id || seen.has(call.id)) throw new Error("missing, misplaced or duplicate tool-call identity");
      seen.add(call.id); outstanding.push(call.id);
    }
  }
  if (outstanding.length) throw new Error("incomplete final tool-result bundle");
}

function failure(code, message, details = {}) {
  return new Response(JSON.stringify({ error: { type: "invalid_request_error", code, message, details } }),
    { status: 400, headers: { "Content-Type": "application/json" } });
}

async function backendContextFailure(response) {
  if (![400, 413, 422].includes(response.status)) return false;
  try {
    const { error } = await response.clone().json();
    return error?.code === "context_length_exceeded" ||
      /\bcontext_length_exceeded\b|maximum context length|input length .*exceeds|input tokens .*exceed/i.test(error?.message ?? "");
  } catch (error) {
    if (error instanceof SyntaxError) return false;
    throw error;
  }
}

/** Forward one complete request. Backend admission and native response IDs only.
 * There is no counting API, custom compact trigger or truncation, and sampling
 * parameters are forwarded as sent.
 */
export function contextFetch(fetcher, context) {
  const terminated = new Map();
  const origin = context && modelOrigin(context.baseURL);
  return async (url, init) => {
    if (!context || !String(url).endsWith("/chat/completions")) return fetcher(url, init);
    if (new URL(url).origin !== origin) throw new Error("model destinations differ");
    const body = JSON.parse(init.body);
    assertToolBundles(body.messages);
    const headers = new Headers(init.headers);
    const invocation = headers.get("x-session-id"), agent = headers.get("x-nemotron-agent");
    if (!invocation || !agent) return failure("media_context_owner_missing", "Context statistics require explicit session and agent ownership.");
    if (terminated.has(invocation)) return failure("input_budget_exceeded", "The inference backend rejected this trajectory's context.", terminated.get(invocation));
    const output = body.max_completion_tokens ?? body.max_tokens;
    if (output !== undefined && (!Number.isSafeInteger(output) || output <= 0 || output > context.profile.reserved_output_tokens)) {
      return failure("media_output_reserve_mismatch", "Requested output exceeds the configured generation reserve.");
    }
    await mkdir(context.directory, { recursive: true, mode: 0o700 });
    if (context.captureDirectory) await privateDirectory(context.captureDirectory, "input");
    const record = { schema_version: 2, kind: "model_request_statistics", capture_id: randomUUID(),
      invocation_id: invocation, agent, parent_message_id: headers.get("x-nemotron-parent-message"),
      started_at: Date.now() / 1000, action: "send", context_enforcement: "inference_backend",
      input_tokens: null, input_limit: context.profile.max_context_tokens - (output ?? context.profile.reserved_output_tokens),
      configured_max_model_len: context.profile.max_context_tokens, active_max_model_len: null,
      token_ids_sha256: null, token_ids_source: null, request_sha256: sha256(init.body),
      reserved_output_tokens: context.profile.reserved_output_tokens, generation_sent: false,
      terminal: false, error_code: null, preflight_requests: 0 };
    const statFile = path.join(context.directory, `${record.capture_id}.json`);
    const persist = async ({ promptTokenIds = null, usage = null } = {}) => {
      record.input_tokens = promptTokenIds?.length ?? usage?.prompt_tokens ?? null;
      record.token_ids_sha256 = promptTokenIds === null ? null : sha256(JSON.stringify(promptTokenIds));
      record.token_ids_source = promptTokenIds === null ? null : "vllm_native_generation_response";
      if (context.captureDirectory) record.model_input_snapshot = await captureModelInput({
        directory: context.captureDirectory, body, tokens: promptTokenIds, record });
      await writeFile(statFile, JSON.stringify(record), { flag: "wx", mode: 0o600 });
    };
    const generationInit = context.captureDirectory ? { ...init, body: JSON.stringify(outputCaptureRequest(body)) } : init;
    record.server_request_sha256 = sha256(generationInit.body);
    let response;
    try { response = await fetcher(url, generationInit); }
    catch (error) {
      record.error_code = "model_transport_error"; record.terminal = true;
      await persist(); throw error;
    }
    record.status_code = response.status;
    record.generation_sent = true;
    if (await backendContextFailure(response)) {
      record.error_code = "input_budget_exceeded"; record.terminal = true;
      const details = { backend_status: response.status, capture_id: record.capture_id, terminal: true };
      terminated.set(invocation, details);
      response = failure("input_budget_exceeded", "The inference backend rejected the complete input/output context budget.", details);
    }
    if (context.captureDirectory) return captureModelOutput({ response, body, record,
      directory: path.join(context.captureDirectory, "outputs"), onFinalize: persist });
    await persist();
    return response;
  };
}
