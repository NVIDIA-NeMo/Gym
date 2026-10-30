// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { writeFileSync } from "node:fs";

// Gym accepts the provider's tool IDs unchanged. OpenClaw's generic custom-provider
// policy strips punctuation, breaking the join to the captured model response.
export default {
  id: "nemo-gym",
  name: "NeMo Gym model identity",
  register(api) {
    api.registerProvider({
      id: "nemo",
      label: "NeMo Gym",
      auth: [],
      buildReplayPolicy() {
        writeFileSync(new URL("policy.ready", import.meta.url), "nemo-gym-replay-v1\n");
        return { sanitizeToolCallIds: false, dropReasoningFromHistory: false };
      },
    });
  },
};
