// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Writable } from "node:stream";
import { isContextOverflow } from "@earendil-works/pi-ai";

// Use Pi's own context-limit classifier; transport/provider failures must not become reward zero.
// This is an adapter event in JSON mode, never a message sent to the model. That mode points
// process.stdout.write, which console.log uses, at stderr; the stream's own write still reaches Gym's
// runner on stdout as one whole line. The line comes before this message's message_end and can pass lines
// Pi queued earlier, so read it by type. A direct fd write could split a long Pi line or fail on a full pipe.
export default function outcome(pi) {
  pi.on("message_end", (event, ctx) => {
    if (event.message.role !== "assistant") return;
    // Pi before 0.86.0 also counts a 400/413 whose body has no `error` object ("(no body)") as an overflow. The
    // rule is for Cerebras, and 0.86.0 applies it only there. Gym's model servers can pass such a 400 through
    // from any provider and for any cause (SGLang's flat errors, overflows included), so Gym does not count it.
    const noErrorBody = /^4(?:00|13)\s*(?:status code)?\s*\(no body\)/i.test(event.message.errorMessage ?? "");
    Writable.prototype.write.call(process.stdout, JSON.stringify({
      type: "ng_pi_outcome",
      context_overflow: !noErrorBody && isContextOverflow(event.message, ctx.model?.contextWindow),
    }) + "\n");
  });
}
