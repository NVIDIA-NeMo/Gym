// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// OpenCode 1.17's classic runner publishes the saved assistant before calling
// chat.headers, whose input.message is the user message rather than the reply.
export const AssistantMessageHeader = () => {
  const active = new Map();
  return {
    // OpenCode does not await event hooks; update this map before any await.
    event({ event }) {
      const p = event.properties;
      if (event.type === "session.idle" || event.type === "session.deleted" ||
          (event.type === "session.status" && p.status?.type === "idle")) {
        active.delete(p.sessionID);
        return;
      }
      if (event.type === "message.removed") {
        active.get(p.sessionID)?.delete(p.messageID);
        return;
      }
      if (event.type !== "message.updated" || p.info?.role !== "assistant") return;
      const m = p.info;
      if (m.time?.completed !== undefined) {
        active.get(m.sessionID)?.delete(m.id);
        return;
      }
      let replies = active.get(m.sessionID);
      if (!replies) active.set(m.sessionID, replies = new Map());
      replies.set(m.id, {
        id: m.id, parentID: m.parentID, agent: m.agent,
        modelID: m.modelID, providerID: m.providerID,
      });
    },
    "chat.headers"(input, output) {
      const candidates = [...(active.get(input.sessionID)?.values() ?? [])].filter(m =>
        m.parentID === input.message.id && m.agent === input.agent &&
        m.modelID === input.model.id && m.providerID === input.model.providerID);
      // Title calls have no saved assistant; never guess when ownership is ambiguous.
      if (candidates.length === 1) {
        output.headers["X-OpenCode-Assistant-Message-Id"] = candidates[0].id;
      }
    },
  };
};
