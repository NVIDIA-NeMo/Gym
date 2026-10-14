// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Writable } from "node:stream";
import { isContextOverflow } from "@earendil-works/pi-ai";

// Use Pi's own context-limit classifier; transport/provider failures must not become reward zero.
// This is an adapter event in JSON mode, never a message sent to the model. Pi's JSON mode
// replaces console.log and process.stdout.write with writers to stderr. The stream's own write
// still reaches stdout, where Gym's runner records Pi's events, and queues the line whole behind
// Pi's own output; a direct fd write could split one of Pi's long lines or fail when the pipe is full.
export default function outcome(pi) {
  pi.on("message_end", (event, ctx) => {
    if (event.message.role !== "assistant") return;
    Writable.prototype.write.call(process.stdout, JSON.stringify({
      type: "ng_pi_outcome",
      context_overflow: isContextOverflow(event.message, ctx.model?.contextWindow),
    }) + "\n");
  });
}
