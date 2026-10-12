// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export default function runtimeGuards(pi) {
  const timeout = Number(process.env.NEMO_GYM_PI_BASH_TIMEOUT);
  if (!Number.isSafeInteger(timeout) || timeout <= 0) {
    throw new Error("NEMO_GYM_PI_BASH_TIMEOUT must be a positive integer");
  }
  pi.on("tool_call", (event) => {
    if (event.toolName !== "bash") return;
    // Pi explicitly supports mutating tool input before execution. Preserve
    // shorter requested deadlines; bound commands with no deadline as well.
    const requested = event.input.timeout;
    event.input.timeout = Number.isFinite(requested) && requested > 0
      ? Math.min(requested, timeout) : timeout;
  });
  pi.on("before_agent_start", (event) => ({
    systemPrompt: event.systemPrompt + "\n\nSandbox execution: search within the repository or specific local " +
      "dependency directories. Avoid whole-filesystem searches such as `find /`; shared mounts " +
      "under /mnt can stall. Each bash command has a maximum runtime of " + timeout +
      " seconds. Use focused commands and inspect timeout errors before retrying.",
  }));
}
