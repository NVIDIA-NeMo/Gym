// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { gzipFetch } from "./request_gzip.mjs";

export const RequestGzip = async () => ({
    config: async (config) => {
        const options = config.provider.nemo_gym.options;
        options.fetch = gzipFetch(options.fetch ?? globalThis.fetch.bind(globalThis), options.baseURL);
    },
});
