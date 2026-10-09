// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Written by NeMo Gym into $HOME/.config/opencode/plugins/ when the agent's `latency_metrics` flag is on.
// OpenCode calls `chat.headers` right before every model request; the headers reach Gym's model-server
// capture and stop there. Nothing here changes the request body, so model behaviour is unchanged.
export const NemoGymLatency = async () => {
  let counter = 0
  return {
    "chat.headers": async (input, output) => {
      counter += 1
      const sessionID = input && input.sessionID ? String(input.sessionID) : ""
      const agent = input && input.agent ? String(input.agent) : ""
      output.headers["x-session-id"] = sessionID
      output.headers["x-ng-agent"] = agent
      output.headers["x-ng-client-call-id"] = `oc-${sessionID}-${counter}-${Math.random().toString(16).slice(2, 10)}`
      output.headers["x-ng-client-sent-at"] = String(Date.now())
    },
  }
}
