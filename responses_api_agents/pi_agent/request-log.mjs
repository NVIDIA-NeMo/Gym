// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createHash } from "node:crypto";
import { appendFileSync, writeFileSync } from "node:fs";

// Record what each model request carries, after every other request hook (load this extension
// last): one JSON line per request with every top-level field except the conversation and tool
// schemas, which are summarized. The first request is also written in full beside the log.
// Headers, and so the API key, are never recorded.
export default function requestLog(pi) {
  const path = process.env.NEMO_GYM_PI_REQUEST_LOG;
  if (!path) throw new Error("NEMO_GYM_PI_REQUEST_LOG must be set");
  let count = 0;
  pi.on("before_provider_request", (event, ctx) => {
    const payload = event.payload ?? {};
    const { messages, tools, ...fields } = payload;
    count += 1;
    if (count === 1) {
      writeFileSync(path.replace(/\.jsonl$/, "") + ".first.json", JSON.stringify(payload, null, 1) + "\n");
    }
    const record = {
      request: count,
      time: Date.now() / 1000,
      provider: ctx.model?.provider,
      api: ctx.model?.api,
      ...fields,
      messages: Array.isArray(messages) ? messages.length : messages,
      last_role: Array.isArray(messages) ? messages.at(-1)?.role : undefined,
      tools: Array.isArray(tools) ? tools.map((tool) => tool?.function?.name ?? tool?.name) : tools,
      tools_sha256: tools === undefined ? undefined : createHash("sha256").update(JSON.stringify(tools)).digest("hex"),
    };
    appendFileSync(path, JSON.stringify(record) + "\n");
  });
}
