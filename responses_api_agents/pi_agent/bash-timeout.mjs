// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
export default function bashTimeout(pi) {
  const limit = Number(process.env.NEMO_GYM_PI_BASH_TIMEOUT);
  if (!Number.isFinite(limit) || limit <= 0) throw new Error("Invalid Gym Pi Bash timeout");
  pi.on("tool_call", (event) => {
    if (event.toolName !== "bash") return;
    // Pi executes the mutated arguments using its native process-tree timeout.
    // Preserve shorter model deadlines; cap longer ones and supply omitted ones.
    const requested = event.input.timeout;
    event.input.timeout = Number.isFinite(requested) && requested > 0
      ? Math.min(requested, limit) : limit;
  });
}
