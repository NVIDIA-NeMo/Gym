// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { randomUUID } from "node:crypto";
import { pathToFileURL } from "node:url";

// Pi 0.80.2 routes summarization through agent.streamFn, but does not emit
// message_end for those responses. Observe that stream only between its own
// compaction_start/end events; leave triggering, summarization and retry intact.
// The SDK's extension hooks do not provide a compaction-failure/end boundary.
export function observeCompaction(AgentSession) {
  const emit = AgentSession.prototype._emit;
  const active = new WeakMap();
  AgentSession.prototype._emit = function (event) {
    if (event.type === "compaction_start" && !active.has(this)) {
      const session = this;
      const operation = { id: randomUUID(), stream: this.agent.streamFn };
      active.set(this, operation);
      this.agent.streamFn = async (...args) => {
        const stream = await operation.stream.apply(this.agent, args);
        const result = stream.result.bind(stream);
        let observed;
        // compact() awaits result(). Publish before returning it so even a
        // failed or interrupted compaction retains every completed helper.
        stream.result = () => observed ??= result().then((message) => {
          emit.call(session, {
            type: "ng_pi_compaction_call",
            compaction_id: operation.id,
            message: {
              role: "assistant",
              content: [],
              responseId: message.responseId,
              usage: message.usage,
              stopReason: message.stopReason,
            },
          });
          return message;
        });
        return stream;
      };
    }
    const operation = active.get(this);
    if (operation && (event.type === "compaction_start" || event.type === "compaction_end")) {
      event = { ...event, ng_compaction_id: operation.id };
      if (event.type === "compaction_end") {
        this.agent.streamFn = operation.stream;
        active.delete(this);
      }
    }
    return emit.call(this, event);
  };
}

// Loaded by Node before the pinned CLI. Tests import the observer without
// importing Pi or replacing the real CLI's execution path.
if (process.argv[1]?.endsWith("/@earendil-works/pi-coding-agent/dist/cli.js")) {
  const { AgentSession } = await import(new URL("./core/agent-session.js", pathToFileURL(process.argv[1])));
  observeCompaction(AgentSession);
}
