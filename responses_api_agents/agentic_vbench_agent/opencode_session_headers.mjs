// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Observe native OpenCode sessions without changing messages, tools, or sampling.
export const SessionHeaders = async ({ client }) => {
  const sessions = new Map();
  const invocations = new Map();
  const tasks = new Map();
  const waiting = new Map();
  const running = new Map();
  const starting = new Map();
  let captureURL;
  let provenanceError;

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

  const taskFor = async (id) => {
    if (!tasks.has(id) && !provenanceError) {
      // OpenCode publishes running Task metadata before child inference, but
      // dispatches plugin events asynchronously. Never infer an edge from text.
      await new Promise((resolve) => {
        const timer = setTimeout(resolve, 5000);
        const ready = () => { clearTimeout(timer); resolve(); };
        waiting.set(id, ready);
      });
      waiting.delete(id);
    }
    if (provenanceError) throw new Error(provenanceError);
    const task = tasks.get(id);
    if (!task) throw new Error("Cannot record OpenCode child: native running Task metadata is missing");
    return task;
  };

  return {
    event: async ({ event }) => {
      const part = event.properties?.part;
      if (event.type !== "message.part.updated" || part?.type !== "tool" || part.tool !== "task") return;
      if (["completed", "error"].includes(part.state?.status)) {
        const invocation = invocations.get(part.callID);
        if (invocation?.parent === part.sessionID) invocation.complete = true;
        running.delete(part.callID);
        return;
      }
      if (part.state?.status === "running") {
        const previous = running.get(part.callID);
        if (previous && previous !== part.sessionID) {
          provenanceError = "Native running Task part belongs to another parent session";
          for (const ready of starting.values()) ready();
          for (const ready of waiting.values()) ready();
          return;
        }
        running.set(part.callID, part.sessionID);
        starting.get(part.callID)?.();
      }
      const child = part.state?.metadata?.sessionId;
      if (part.state?.status !== "running" || typeof child !== "string" || !child) return;
      const invocation = invocations.get(part.callID);
      const previous = tasks.get(child);
      if (!invocation || invocation.parent !== part.sessionID ||
          (invocation.child && invocation.child !== child) ||
          (previous && (previous.parent !== part.sessionID ||
            (previous.call !== part.callID && !invocations.get(previous.call)?.complete)))) {
        // Event hooks are fire-and-forget in OpenCode. Propagate a failure
        // through the awaited model hook instead of an unhandled rejection.
        provenanceError = "Native running Task metadata has inconsistent child ownership";
        for (const ready of starting.values()) ready();
        for (const ready of waiting.values()) ready();
        return;
      }
      invocation.child = child;
      tasks.set(child, { parent: part.sessionID, spawn: previous?.spawn ?? part.callID, call: part.callID });
      waiting.get(child)?.();
    },
    "tool.execute.before": async (input, output) => {
      if (input.tool.toLowerCase() !== "task") return;
      const identity = sessions.get(output.args?.task_id);
      if (identity && identity.parent !== input.sessionID) {
        throw new Error("A Task follow-up targeted a child from another parent session");
      }
      invocations.set(input.callID, { parent: input.sessionID, complete: false });
      // OpenCode can start an AI SDK tool before its stream processor has
      // registered that tool's part. Task's ctx.metadata then silently drops
      // the native child ID. Wait for the processor's own running event before
      // allowing Task to create its child; never manufacture an ownership edge.
      if (!running.has(input.callID) && !provenanceError) {
        await new Promise((resolve) => {
          const timer = setTimeout(resolve, 5000);
          starting.set(input.callID, () => { clearTimeout(timer); resolve(); });
        });
        starting.delete(input.callID);
      }
      if (provenanceError) throw new Error(provenanceError);
      if (running.get(input.callID) !== input.sessionID) {
        throw new Error("Cannot execute OpenCode Task before its native tool part is running");
      }
    },
    "tool.execute.after": async (input, output) => {
      if (input.tool.toLowerCase() !== "task") return;
      const identity = await identify(output.metadata?.sessionId);
      const task = await taskFor(identity.id);
      if (identity.parent !== input.sessionID || task.parent !== input.sessionID || task.call !== input.callID) {
        throw new Error("Cannot record OpenCode Task provenance: child belongs to another parent");
      }
      invocations.get(input.callID).complete = true;
    },
    "chat.headers": async (input, output) => {
      const endpoint = new URL(`${input.provider.options.baseURL.replace(/\/$/, "")}/session-bindings`);
      if (captureURL && captureURL !== endpoint.href) {
        throw new Error("OpenCode session changed capture endpoints");
      }
      captureURL = endpoint.href;
      const identity = await identify(input.sessionID);
      output.headers["x-opencode-session"] = identity.id;
      output.headers["x-opencode-agent"] = input.agent;
      if (identity.parent) {
        const task = await taskFor(identity.id);
        if (task.parent !== identity.parent) throw new Error("Native Task child belongs to another parent session");
        output.headers["x-opencode-parent-session"] = identity.parent;
        // Each captured call owns its exact native edge immediately, even if
        // Harbor later times out before Task can return or run its after hook.
        output.headers["x-opencode-spawn-call"] = task.spawn;
        output.headers["x-opencode-task-call"] = task.call;
      }
    },
  };
};
