// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isContextOverflow } from "@earendil-works/pi-ai";

// Use Pi's own context-limit classifier; transport/provider failures must not become reward zero.
// This is an adapter event in JSON mode, never a message sent to the model.
export default function outcome(pi) {
  pi.on("message_end", (event, ctx) => {
    if (event.message.role !== "assistant") return;
    console.log(JSON.stringify({
      type: "ng_pi_outcome",
      context_overflow: isContextOverflow(event.message, ctx.model?.contextWindow),
    }));
  });
}
