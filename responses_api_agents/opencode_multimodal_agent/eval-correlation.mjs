// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Stock 1.17.11 exposes the user message to chat.headers. Resolve the already
// persisted, unfinished assistant reply instead of labeling a call with a user ID.
export function correlationHeaders(messages, input) {
  if (!Array.isArray(messages) || !input.message?.id) return {};
  const candidates = messages.map(row => row.info).filter(info =>
    info?.role === "assistant" && info.sessionID === input.sessionID &&
    info.parentID === input.message.id && info.time?.completed == null && !info.error
  );
  if (candidates.length !== 1 || !candidates[0].id) return {};
  return {
    "x-opencode-session-id": input.sessionID,
    "x-opencode-assistant-message-id": candidates[0].id,
  };
}

export default async ({ client }) => ({
  "chat.headers": async (input, output) => {
    try {
      const result = await client.session.messages({ path: { id: input.sessionID } });
      const headers = correlationHeaders(result.data, input);
      if (!headers["x-opencode-assistant-message-id"]) {
        console.warn("Gym correlation: no unique persisted assistant reply; leaving call unlinked");
      }
      Object.assign(output.headers, headers);
    } catch (error) {
      // Observability must not change the answer. Missing evidence fails the quality gate.
      console.warn("Gym correlation: session lookup failed; leaving call unlinked");
    }
  },
});
