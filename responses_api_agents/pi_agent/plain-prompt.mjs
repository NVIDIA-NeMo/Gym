// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// plain_system_prompt: Pi's default system prompt without the block that points at Pi's own documentation, whose
// paths name the runtime's folder, and without the machine's date, for tasks whose prompt sets their own time.
const DOCUMENTATION = /\n\nPi documentation \([^\n]*\):(?:\n- [^\n]*)*/;
const DATE = /\nCurrent date: \d{4}-\d{2}-\d{2}(?=\n|$)/;

export default function plainPrompt(pi) {
  pi.on("before_agent_start", (event) => ({
    systemPrompt: event.systemPrompt.replace(DOCUMENTATION, "").replace(DATE, ""),
  }));
}
