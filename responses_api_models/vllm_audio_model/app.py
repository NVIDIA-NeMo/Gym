# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""vllm_model variant that carries audio over a Responses-API-compatible sidechannel.

Why this exists
---------------
OpenAI's Responses API content union (``ResponseInputContentParam``) only
allows ``input_text``, ``input_image``, ``input_file`` content blocks — there
is no audio variant in the schema, even though ``ResponseInputAudioParam``
exists in the SDK as an orphan type. Gym's simple_agent strictly validates
incoming requests against this Responses-API schema, so audio content blocks
in ``responses_create_params.input.content`` get rejected at the agent layer
before they ever reach the model server.

vLLM's underlying ``/v1/chat/completions`` endpoint, on the other hand, accepts
``audio_url`` (data-URI) and ``input_audio`` (base64+format) content blocks
for any audio-multimodal model. NeMo-Skills' ``VLLMMultimodalModel`` exploits
this by intercepting at the model-runner layer: it reads audio from per-row
metadata, base64-encodes it, and prepends an audio content block to the user
message in the Chat Completions request.

This wrapper is the Gym equivalent of that pattern. It runs on the model side
(no agent-layer schema change required), pulls the audio data URL out of
``responses_create_params.metadata["audio_url"]``, and splices an
``audio_url`` block into the most recent user message in the Chat Completions
request that vllm_model produces. ``metadata`` is the only opaque
string→string passthrough field the Responses-API schema permits, so it's
where audio data has to ride for now.

Once OpenAI extends ``ResponseInputContentParam`` to include audio (or Gym
adopts a local extension — Option B from the migration writeup), this wrapper
should be deprecated in favor of native audio content blocks.
"""

from typing import Any, Dict

from fastapi import Request

from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


METADATA_AUDIO_URL_KEY = "audio_url"


class VLLMAudioModel(VLLMModel):
    """``VLLMModel`` plus a metadata-sidechannel audio splicer.

    Behavior: identical to ``VLLMModel`` for non-audio rows. When a request's
    ``metadata`` carries an ``audio_url`` data-URI string, this model strips
    that key and prepends an ``audio_url`` content block to the most recent
    user message before forwarding to vLLM Chat Completions.
    """

    config: VLLMModelConfig

    def _preprocess_chat_completion_create_params(self, request: Request, body_dict: Dict[str, Any]) -> Dict[str, Any]:
        body_dict = super()._preprocess_chat_completion_create_params(request, body_dict)

        metadata = body_dict.get("metadata") or {}
        audio_url = metadata.get(METADATA_AUDIO_URL_KEY)
        if not audio_url:
            return body_dict

        # vLLM doesn't need to see audio_url in metadata once we've spliced
        # the corresponding content block. Strip it to keep the upstream
        # request clean.
        metadata.pop(METADATA_AUDIO_URL_KEY, None)
        if not metadata:
            body_dict.pop("metadata", None)

        # Find the most recent user message and prepend the audio content
        # block to its content. Mirrors NeMo-Skills' content_text_to_list
        # placement (audio BEFORE text) — required by some audio models.
        audio_block = {"type": "audio_url", "audio_url": {"url": audio_url}}
        for msg in reversed(body_dict.get("messages", []) or []):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = [
                    audio_block,
                    {"type": "text", "text": content},
                ]
            elif isinstance(content, list):
                msg["content"] = [audio_block] + list(content)
            else:
                # ``None`` / other -> create a fresh content list with just the audio block
                msg["content"] = [audio_block]
            break
        else:
            # No user message found — create one with just the audio block.
            # Unusual shape, but don't drop the audio silently.
            body_dict.setdefault("messages", []).append({"role": "user", "content": [audio_block]})

        return body_dict


if __name__ == "__main__":
    VLLMAudioModel.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = VLLMAudioModel.run_webserver()  # noqa: F401
