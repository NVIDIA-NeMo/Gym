// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// After an automatic compaction OpenCode keeps the summary where the compaction happened, after the recent
// messages it retained, so the model reads the summary of the older conversation after the newer messages. This
// hook moves each finished summary (with its compaction marker) to where the retained tail starts, so the
// conversation reads in order: summary, then the retained messages, then what followed. Messages are moved, never
// changed: their contents and tool ids stay as they were, and a summary whose tail start is unknown or that failed
// is left where it is.

export function orderCompaction(messages) {
  const result = structuredClone(messages);
  const summaries = result.filter(
    (m) => m.info.role === "assistant" && m.info.summary && m.info.finish && !m.info.error,
  );
  for (const summary of summaries) {
    const summaryIndex = result.indexOf(summary);
    const markerIndex = result.findIndex((m) => m.info.role === "user" && m.info.id === summary.info.parentID);
    if (markerIndex < 0) continue;
    const marker = result[markerIndex];
    const part = marker.parts.find((p) => p.type === "compaction");
    if (!part?.tail_start_id) continue;
    const tailIndex = result.findIndex((m) => m.info.id === part.tail_start_id);
    if (tailIndex < 0 || tailIndex >= markerIndex) continue;
    if (summaryIndex !== markerIndex + 1) throw new Error("Compaction order: nonadjacent marker and summary");
    if (result.slice(tailIndex, summaryIndex + 1).some((m) => m.info.sessionID !== summary.info.sessionID)) {
      throw new Error("Compaction order: mixed sessions");
    }
    result.splice(markerIndex, 2);
    result.splice(tailIndex, 0, marker, summary);
  }
  return result;
}

export default async function CompactionOrder() {
  return {
    "experimental.chat.messages.transform": async (_input, output) => {
      const ordered = orderCompaction(output.messages);
      output.messages.splice(0, output.messages.length, ...ordered);
    },
  };
}
