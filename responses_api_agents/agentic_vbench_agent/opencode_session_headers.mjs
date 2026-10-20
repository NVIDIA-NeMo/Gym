// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Observe native OpenCode sessions without changing messages, tools, or sampling.
export const SessionHeaders = async ({ client }) => {
  const sessions = new Map();

  const checkedData = (response, label) => {
    if (response.error || response.data === undefined) {
      throw new Error(`Cannot record OpenCode session provenance: ${label}`);
    }
    return response.data;
  };

  const identify = async (id) => {
    if (sessions.has(id)) return sessions.get(id);
    const info = checkedData(await client.session.get({ path: { id } }), "session lookup failed");
    if (info.id !== id) throw new Error("OpenCode session lookup returned a different session");
    if (!info.parentID) {
      const identity = { id };
      sessions.set(id, identity);
      return identity;
    }
    const messages = checkedData(
      await client.session.messages({ path: { id: info.parentID } }),
      "parent message lookup failed",
    );
    if (!Array.isArray(messages)) throw new Error("OpenCode parent messages are not an array");
    // Task sets metadata.sessionId before invoking the child. Read the actual
    // tool part, so identical Task prompts cannot alias distinct children.
    const calls = messages.flatMap((message) => message.parts ?? []).filter(
      (part) => part.type === "tool" && part.tool === "task" && part.state?.metadata?.sessionId === id,
    );
    const birth = calls[0];
    const current = calls.findLast((part) => part.state?.status === "running");
    if (!birth?.callID || !current?.callID) {
      throw new Error("Cannot bind OpenCode child to its spawning and current Task calls");
    }
    const identity = { id, parent: info.parentID, spawn: birth.callID, call: current.callID };
    sessions.set(id, identity);
    return identity;
  };

  return {
    "tool.execute.before": async (input, output) => {
      if (input.tool.toLowerCase() !== "task") return;
      const id = output.args?.task_id;
      const identity = sessions.get(id);
      if (identity) {
        if (identity.parent !== input.sessionID) {
          throw new Error("A Task follow-up targeted a child from another parent session");
        }
        identity.call = input.callID;
      }
    },
    "chat.headers": async (input, output) => {
      const identity = await identify(input.sessionID);
      output.headers["x-opencode-session"] = identity.id;
      output.headers["x-opencode-agent"] = input.agent;
      if (identity.parent) {
        output.headers["x-opencode-parent-session"] = identity.parent;
        output.headers["x-opencode-spawn-call"] = identity.spawn;
        output.headers["x-opencode-task-call"] = identity.call;
      }
    },
  };
};
