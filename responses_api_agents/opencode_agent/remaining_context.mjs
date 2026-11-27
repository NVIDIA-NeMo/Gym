// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Send no max_tokens on the session provider's requests, so the endpoint chooses the output
// budget from the context left (output_token_policy: remaining_context).
export const RemainingContext = async () => ({
  "chat.params": async (input, output) => {
    if (input.model.providerID === "nemo_gym") {
      delete output.maxOutputTokens;
    }
  },
});
