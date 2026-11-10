// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
export default async () => ({
  "chat.headers": async (input, output) => {
    output.headers["x-opencode-session-id"] = input.sessionID;
  },
});
