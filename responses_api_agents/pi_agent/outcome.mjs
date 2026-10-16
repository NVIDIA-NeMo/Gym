// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { appendFileSync } from "node:fs";
import { isContextOverflow } from "@earendil-works/pi-ai";

// Use Pi's own context-limit classifier; transport/provider failures must not become reward zero.
// This is an adapter event in JSON mode, never a message sent to the model. In print and JSON modes
// Pi sends extension console output to stderr (only its own events reach stdout), so the sandbox
// runner names a file for these records in NEMO_GYM_PI_OUTCOME_LOG and adds them to its event log.
export default function outcome(pi) {
  const path = process.env.NEMO_GYM_PI_OUTCOME_LOG;
  pi.on("message_end", (event, ctx) => {
    if (event.message.role !== "assistant") return;
    const line = JSON.stringify({
      type: "ng_pi_outcome",
      context_overflow: isContextOverflow(event.message, ctx.model?.contextWindow),
    });
    if (path) appendFileSync(path, line + "\n");
    else console.log(line);
  });
}
