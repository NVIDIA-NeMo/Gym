// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Observe native OpenCode sessions without changing messages, tools, or sampling.
export const SessionHeaders = async ({ client }) => {
  const sessions = new Map();
  let bindingURL;

  const identify = async (id) => {
    if (typeof id !== "string" || !id) throw new Error("OpenCode returned no native session ID");
    if (sessions.has(id)) return sessions.get(id);
    const response = await client.session.get({ path: { id } });
    if (response.error || response.data?.id !== id) {
      throw new Error("Cannot record OpenCode session provenance: session lookup failed");
    }
    const identity = { id, parent: response.data.parentID };
    sessions.set(id, identity);
    return identity;
  };

  return {
    "tool.execute.before": async (input, output) => {
      if (input.tool.toLowerCase() !== "task") return;
      const identity = sessions.get(output.args?.task_id);
      if (identity && identity.parent !== input.sessionID) {
        throw new Error("A Task follow-up targeted a child from another parent session");
      }
    },
    "tool.execute.after": async (input, output) => {
      if (input.tool.toLowerCase() !== "task") return;
      const identity = await identify(output.metadata?.sessionId);
      if (!bindingURL || identity.parent !== input.sessionID) {
        throw new Error("Cannot record OpenCode Task provenance: child belongs to another parent");
      }
      // The running parent tool part is not reliably visible through the SDK
      // before the child's first call. The native Task result supplies its exact
      // child ID, including invalid-ID fallback and resumed sessions. Record that
      // edge after execution; the proxy refuses to publish unresolved captures.
      const response = await fetch(bindingURL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: identity.id,
          parent_session_id: identity.parent,
          delegation_call_id: input.callID,
        }),
      });
      const receipt = await response.text();
      if (!response.ok) {
        throw new Error(`Cannot record OpenCode Task provenance: ${response.status} ${receipt}`);
      }
    },
    "chat.headers": async (input, output) => {
      const endpoint = new URL(`${input.provider.options.baseURL.replace(/\/$/, "")}/session-bindings`);
      if (bindingURL && bindingURL !== endpoint.href) {
        throw new Error("OpenCode session changed capture endpoints");
      }
      bindingURL = endpoint.href;
      const identity = await identify(input.sessionID);
      output.headers["x-opencode-session"] = identity.id;
      output.headers["x-opencode-agent"] = input.agent;
      if (identity.parent) {
        output.headers["x-opencode-parent-session"] = identity.parent;
        output.headers["x-opencode-task-binding"] = "deferred";
      }
    },
  };
};
