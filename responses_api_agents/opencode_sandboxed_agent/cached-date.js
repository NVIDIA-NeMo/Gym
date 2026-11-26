// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export default async function cachedDate() {
  // Keep OpenCode's first rendered date (including its timezone) for this runtime.
  // A midnight change near the start of the prompt invalidates the whole cached prefix.
  let date;
  return {
    "experimental.chat.system.transform": async (_input, output) => {
      // OpenCode retains this array, so mutate its entries rather than replacing it.
      for (let i = 0; i < output.system.length; i++) {
        output.system[i] = output.system[i].replace(/<env>[\s\S]*?<\/env>/g, (environment) =>
          environment.replace(/^([ \t]*Today's date: )([^\r\n]+)/gm, (_line, prefix, current) => {
            date ??= current;
            return prefix + date;
          }),
        );
      }
    },
  };
}
