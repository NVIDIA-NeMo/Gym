// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Name the OpenCode agent that makes each model call in a request header, so Gym's vllm_model can tell the
// compaction summary call from the build agent's calls: with `compaction_summary_max_tokens` set it bounds the
// summary's output and retries a summary it cannot use. OpenCode runs the summary under its `compaction` agent.

export const HEADER = "X-NeMo-Gym-OpenCode-Agent";

export default async function CompactionSummaryHeader() {
  return {
    "chat.headers"(input, output) {
      if (typeof input.agent === "string" && input.agent) {
        output.headers[HEADER] = input.agent;
      }
    },
  };
}
