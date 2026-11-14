// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import { bridgePrompt, pinnedDependency, restoreNativeVideo, restoreToolMedia } from "./media_bridge.mjs";
import { assertToolBundles, contextFetch, launcherContext } from "./media_context.mjs";

const { createOpenAICompatible } = await pinnedDependency("@ai-sdk/openai-compatible", "2.0.41", "dist/index.mjs");

export function createNemotronMedia(options) {
  const transform = options.transformRequestBody;
  const base = createOpenAICompatible({
    ...options,
    includeUsage: options.includeUsage ?? true,
    fetch: contextFetch(options.fetch ?? globalThis.fetch, options.mediaContext ?? launcherContext(options)),
    transformRequestBody(body) {
      const native = restoreToolMedia(restoreNativeVideo(body));
      const result = transform ? transform(native) : native;
      // Keep the call order OpenCode produced; there is no second sorter.
      assertToolBundles(result.messages);
      return result;
    },
  });
  const languageModel = (id) => {
    const model = base.languageModel(id);
    return {
      specificationVersion: model.specificationVersion,
      provider: model.provider,
      modelId: model.modelId,
      supportedUrls: model.supportedUrls,
      doGenerate: (args) => model.doGenerate({ ...args, prompt: bridgePrompt(args.prompt) }),
      doStream: (args) => model.doStream({ ...args, prompt: bridgePrompt(args.prompt) }),
    };
  };
  const provider = (id) => languageModel(id);
  Object.assign(provider, base, { languageModel, chatModel: languageModel });
  return provider;
}
