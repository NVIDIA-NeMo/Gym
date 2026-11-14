// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { writeFile } from "node:fs/promises";
import path from "node:path";
import { privateDirectory, sha256 } from "./media_capture_io.mjs";

const tokenIDs = (ids) => Array.isArray(ids) && ids.every((id) => Number.isSafeInteger(id) && id >= 0);
const MAX_EVENT_CHARS = 8 << 20;
const MAX_OUTPUT_CHARS = 32 << 20;

/** Add capture-only response metadata, without changing sampling or input content. */
export function outputCaptureRequest(body) {
  if ((body.n !== undefined && body.n !== 1) || body.echo === true || body.include_reasoning === false) {
    throw new Error("exact output capture requires n=1, no echo, and no hidden reasoning tokens");
  }
  if (body.logprobs === true || body.return_token_ids === true) {
    throw new Error("capture transport owns token/logprob fields; do not request them on the agent hop");
  }
  return { ...body, return_token_ids: true, logprobs: true, top_logprobs: 0,
    return_tokens_as_token_ids: true,
    ...(body.stream ? { stream_options: { ...body.stream_options, include_usage: true } } : {}) };
}

/** Observe the same native response the SDK consumes; never re-tokenize text.
 * Capture integrity failures exclude the segment, not an otherwise valid answer.
 * Cancellation, incomplete streams and HTTP failures retain explicit outcomes.
 */
export async function captureModelOutput({ response, directory, body, tokens = null, record, onFinalize }) {
  if (!path.isAbsolute(directory) || !/^[A-Za-z0-9_-]{1,128}$/.test(record.capture_id)) {
    throw new Error("invalid private output capture path or identity");
  }
  await privateDirectory(directory, "output");
  const filename = path.join(directory, `${record.capture_id}.json`);
  record.model_output_snapshot_path = filename;
  const errors = new Set(), generated = [], logprobs = [], tools = new Map();
  let prompt = null, responseId = null, model = null, usage = null, finish = null;
  let content = "", reasoning = "", finished = false, outputChars = 0;
  const budget = body.max_completion_tokens ?? body.max_tokens ?? record.reserved_output_tokens;

  function consume(chunk, streaming) {
    if (chunk.error) errors.add("server_stream_error");
    if (typeof chunk.id !== "string" || !chunk.id || (responseId && chunk.id !== responseId)) errors.add("response_identity_mismatch");
    else responseId = chunk.id;
    if (typeof chunk.model !== "string" || !chunk.model || (model && chunk.model !== model)) errors.add("response_model_mismatch");
    else model = chunk.model;
    if (chunk.prompt_token_ids != null) {
      if (!tokenIDs(chunk.prompt_token_ids) ||
          (tokens !== null && sha256(JSON.stringify(chunk.prompt_token_ids)) !== record.token_ids_sha256) ||
          (prompt !== null && sha256(JSON.stringify(chunk.prompt_token_ids)) !== sha256(JSON.stringify(prompt)))) errors.add("engine_prompt_token_ids_mismatch");
      else prompt = chunk.prompt_token_ids;
    }
    if (chunk.usage != null) usage = chunk.usage;
    delete chunk.prompt_token_ids;
    if (!Array.isArray(chunk.choices)) errors.add("choices_missing");
    for (const choice of chunk.choices ?? []) {
      if (choice.index !== 0) errors.add("multiple_choices_unsupported");
      const ids = choice.token_ids;
      const probabilities = choice.logprobs?.content;
      if (ids != null) {
        if (!tokenIDs(ids) || generated.length + ids.length > budget) {
          errors.add("invalid_or_oversized_generation_token_ids");
        } else {
          for (const id of ids) generated.push(id);
          if (!Array.isArray(probabilities) || probabilities.length !== ids.length ||
              probabilities.some((entry, index) => !Number.isFinite(entry.logprob) || entry.logprob > 0 ||
                entry.logprob <= -9999 || entry.token !== `token_id:${ids[index]}`)) {
            if (ids.length) errors.add("generation_logprobs_mismatch");
          } else for (const entry of probabilities) logprobs.push(entry.logprob);
        }
      } else if (probabilities?.length) errors.add("generation_token_ids_missing");
      delete choice.token_ids;
      delete choice.logprobs;
      if (choice.finish_reason != null) {
        if (finish !== null) errors.add("duplicate_finish_reason");
        finish = choice.finish_reason;
      }
      const message = streaming ? choice.delta : choice.message;
      if (!message) continue;
      if (typeof message.content === "string") content += message.content;
      if (typeof message.reasoning === "string") reasoning += message.reasoning;
      for (const call of message.tool_calls ?? []) {
        const index = streaming ? call.index : tools.size;
        if (!Number.isSafeInteger(index) || index < 0 || index > 1024) {
          errors.add("invalid_tool_call_index"); continue;
        }
        const item = tools.get(index) ?? { id: "", type: "function", function: { name: "", arguments: "" } };
        if (call.id) {
          if (item.id && item.id !== call.id) errors.add("tool_call_identity_mismatch");
          item.id = call.id;
        }
        item.function.name += call.function?.name ?? "";
        item.function.arguments += call.function?.arguments ?? "";
        tools.set(index, item);
      }
    }
    return chunk;
  }

  async function finalize(status) {
    if (finished) return;
    finished = true;
    if (status !== "completed") errors.add(status);
    if (prompt === null) errors.add("engine_prompt_token_ids_missing");
    if (!generated.length) errors.add("empty_generation");
    if (generated.length !== logprobs.length) errors.add("generation_logprobs_mismatch");
    if (finish === null) errors.add("finish_reason_missing");
    const promptLength = prompt?.length ?? tokens?.length;
    if (!usage || promptLength === undefined || usage.prompt_tokens !== promptLength || usage.completion_tokens !== generated.length ||
        usage.total_tokens !== promptLength + generated.length) errors.add("usage_token_ids_mismatch");
    if (onFinalize) await onFinalize({ promptTokenIds: prompt, usage, status });
    const exclusions = [...errors];
    if (!["stop", "tool_calls"].includes(finish)) exclusions.push("unfinished_or_filtered_generation");
    if (["compaction", "title", "summary"].includes(record.agent)) exclusions.push(`auxiliary_${record.agent}`);
    const snapshot = { schema_version: 1, kind: "native_model_output_snapshot",
      capture_id: record.capture_id, invocation_id: record.invocation_id, agent: record.agent,
      parent_message_id: record.parent_message_id, completed_at: Date.now() / 1000,
      input_snapshot: record.model_input_snapshot, server_request_sha256: record.server_request_sha256,
      status, response_id: responseId, model, finish_reason: finish, usage,
      token_ids_source: "vllm_native_generation_response", prompt_token_ids: prompt,
      generation_token_ids: generated, generation_log_probs: logprobs,
      generation_log_probs_source: "vllm_chat_logprobs_unclamped",
      generation_token_ids_sha256: sha256(JSON.stringify(generated)),
      generation_log_probs_sha256: sha256(JSON.stringify(logprobs)),
      message: { role: "assistant", content, ...(reasoning ? { reasoning } : {}),
        ...(tools.size ? { tool_calls: [...tools.entries()].sort((a, b) => a[0] - b[0]).map((item) => item[1]) } : {}) },
      integrity_verified: errors.size === 0, segment_eligible: exclusions.length === 0,
      exclusions, independent_segment: true };
    await writeFile(filename, JSON.stringify(snapshot), { flag: "wx", mode: 0o600 });
  }

  if (!response.ok) { await finalize("http_error"); return response; }
  if (!body.stream) {
    let chunk;
    try { chunk = consume(await response.json(), false); }
    catch (error) { await finalize("response_error"); throw error; }
    await finalize("completed");
    const headers = new Headers(response.headers);
    headers.delete("content-length"); headers.delete("content-encoding");
    return new Response(JSON.stringify(chunk), { status: response.status, headers });
  }
  if (!response.body || !response.headers.get("content-type")?.includes("text/event-stream")) {
    await finalize("invalid_stream_content_type"); return response;
  }
  const reader = response.body.getReader(), decoder = new TextDecoder("utf-8", { fatal: true }), encoder = new TextEncoder();
  let buffer = "", doneMarker = false;
  const stream = new ReadableStream({
    async pull(controller) {
      try {
        while (true) {
          const match = /\r?\n\r?\n/.exec(buffer);
          if (!match) {
            const received = await reader.read();
            if (received.done) {
              buffer += decoder.decode();
              if (buffer.trim()) errors.add("incomplete_sse_event");
              await finalize(doneMarker ? "completed" : "stream_incomplete");
              controller.close(); return;
            }
            buffer += decoder.decode(received.value, { stream: true });
            if (buffer.length > MAX_EVENT_CHARS && !/\r?\n\r?\n/.test(buffer)) throw new Error("output capture SSE event exceeds byte limit");
            continue;
          }
          const packet = buffer.slice(0, match.index);
          buffer = buffer.slice(match.index + match[0].length);
          if (packet.length > MAX_EVENT_CHARS) throw new Error("output capture SSE event exceeds byte limit");
          outputChars += packet.length;
          if (outputChars > MAX_OUTPUT_CHARS) throw new Error("output capture stream exceeds byte limit");
          const lines = packet.split(/\r?\n/), data = lines.filter((line) => line.startsWith("data:"))
            .map((line) => line.slice(5).replace(/^ /, "")).join("\n");
          let forwarded = packet;
          if (data === "[DONE]") {
            if (doneMarker) errors.add("duplicate_done_marker");
            doneMarker = true;
            // Durable capture precedes delivery of the native terminal marker.
            await finalize("completed");
            controller.enqueue(encoder.encode(`${packet}\n\n`));
            controller.close();
            await reader.cancel();
            return;
          } else if (data) {
            if (doneMarker) throw new Error("model data after terminal SSE marker");
            const chunk = consume(JSON.parse(data), true);
            forwarded = [...lines.filter((line) => !line.startsWith("data:")), `data: ${JSON.stringify(chunk)}`].join("\n");
          }
          controller.enqueue(encoder.encode(`${forwarded}\n\n`)); return;
        }
      } catch (error) {
        await finalize("stream_error");
        await reader.cancel().catch(() => {}); // Preserve the primary stream/capture failure.
        controller.error(error);
      }
    },
    async cancel(reason) {
      try { await reader.cancel(reason); }
      finally { await finalize("cancelled"); }
    },
  });
  const headers = new Headers(response.headers);
  headers.delete("content-length"); headers.delete("content-encoding");
  return new Response(stream, { status: response.status, headers });
}
