/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/** Author data — keep in sync with .authors.yml */

export interface Author {
  name: string;
  description: string;
  avatar: string;
}

export const authors: Record<string, Author> = {
  nvidia: {
    name: "NVIDIA NeMo Gym Team",
    description: "NeMo Gym Core Team",
    avatar: "https://avatars.githubusercontent.com/u/1728152?s=200&v=4",
  },
};
