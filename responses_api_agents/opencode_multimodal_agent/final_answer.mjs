// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// The agent prepends a request-local `policy` constant when staging this plugin.
export default async () => {
  const sessions = new Map();
  function state(sessionID) {
    const value = sessions.get(sessionID);
    if (!value) throw new Error("Missing visual final-answer session state");
    return value;
  }
  return {
    "experimental.chat.messages.transform": async (_input, output) => {
      const sessionID = output.messages.at(-1)?.info.sessionID;
      if (!sessionID) throw new Error("Missing visual session identity");
      const assistants = output.messages.filter(m => m.info.role === "assistant" && m.info.time.completed);
      const previous = assistants.at(-1)?.info.tokens;
      const used = previous?.total ?? ((previous?.input ?? 0) + (previous?.output ?? 0)
        + (previous?.cache?.read ?? 0) + (previous?.cache?.write ?? 0));
      const turn = assistants.length + 1;
      const old = sessions.get(sessionID);
      if (old?.final && turn > old.turn) throw new Error("Visual final-answer turn already consumed");
      sessions.set(sessionID, {
        turn, used,
        final: turn >= policy.max_turns || used >= policy.context_window - policy.reserve,
      });
    },
    "chat.params": async (input, output) => {
      const value = state(input.sessionID);
      const available = policy.context_window - value.used - (value.final ? 0 : policy.reserve);
      if (available <= 0) throw new Error("No remaining visual answer context");
      output.maxOutputTokens = Math.min(output.maxOutputTokens ?? policy.max_output_tokens, available);
    },
    "chat.headers": async (input, output) => {
      state(input.sessionID);
      output.headers["x-nemo-visual-session"] = input.sessionID;
    },
    "tool.execute.before": async (input) => {
      if (state(input.sessionID).final) throw new Error("No tools are available on the final-answer turn");
    },
    config: async (config) => {
      const options = config.provider.nemo.options;
      const upstreamFetch = options.fetch ?? globalThis.fetch;
      options.fetch = async (url, init) => {
        const headers = new Headers(init?.headers);
        const value = state(headers.get("x-nemo-visual-session"));
        if (typeof init?.body !== "string") throw new Error("Expected JSON visual model request");
        const body = JSON.parse(init.body);
        const extra = JSON.parse(body.metadata?.extra_body ?? "{}");
        extra.nemo_final_answer_reserve = value.final ? 0 : policy.reserve;
        body.metadata = {...body.metadata, extra_body: JSON.stringify(extra)};
        if (value.final) {
          const messages = [...body.messages];
          const last = messages.at(-1);
          if (last?.role === "assistant" && last.content === policy.stock_max_steps) messages.pop();
          messages.push({role: "user", content: policy.prompt});
          body.messages = messages;
          // Preserve tool schemas so exact carried prompt history remains stable.
          body.tool_choice = "none";
        }
        return upstreamFetch(url, {...init, body: JSON.stringify(body)});
      };
    },
  };
};
