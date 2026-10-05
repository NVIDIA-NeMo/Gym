// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// OpenCode regenerates its environment prompt for every call. Changing the date
// at midnight invalidates the exact history used by token-capture lineage.
// Keep only that field at its first value for this OpenCode process/rollout.
export default async function stablePromptDate() {
  let initialDate;
  return {
    "experimental.chat.system.transform": async (_input, output) => {
      let matches = 0;
      const transformed = output.system.map((text) =>
        text.replace(/<env>[\s\S]*?<\/env>/g, (env) =>
          env.replace(/^([ \t]*Today's date: )([^\r\n]+)$/gm, (_line, prefix, date) => {
            matches += 1;
            initialDate ??= date;
            return prefix + initialDate;
          }),
        ),
      );
      // Auxiliary calls, including compaction, have no environment prompt.
      if (matches === 0 && !output.system.some((text) => text.includes("<env>"))) return;
      if (matches !== 1) {
        throw new Error(`Expected one OpenCode environment date; found ${matches}`);
      }
      output.system.splice(0, output.system.length, ...transformed);
    },
  };
}
