// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Let vLLM choose the output budget from the remaining context.
export const RemainingContext = async () => ({
  "chat.params": async (input, output) => {
    if (input.model.providerID === "nemo_gym") {
      delete output.maxOutputTokens;
    }
  },
});
