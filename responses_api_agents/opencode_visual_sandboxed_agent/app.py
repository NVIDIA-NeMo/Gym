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
"""OpenCode sandboxed agent with image input and output support.

The base agent sends only text to the model. Two things stop images from reaching it:

1. OpenCode drops images for models not declared image-capable. The read tool then returns
   "ERROR: Cannot read image (this model does not support image input)" instead of the
   picture. Here the model config declares ``modalities.input = [text, image]``. OpenCode
   then sends an image the model reads as a follow-up user message with ``image_url``
   parts, because ``@ai-sdk/openai-compatible`` cannot put media in tool results.
2. The base agent keeps only text from the task prompt. Here ``input_image`` parts are
   written into the sandbox and passed to ``opencode run --file``, which puts them in
   the first user turn.

The rebuilt rollout output keeps the images the model saw: tool-result attachments and
``--file`` attachments. They are stored inline or as a sha256 reference (see
``rollout_image_mode``).
"""

import base64
import binascii
import hashlib
import json
from pathlib import Path
from shlex import quote
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import Request
from openai.types.responses import ResponseInputTextParam

from nemo_gym.base_responses_api_agent import Body
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputItem,
)
from nemo_gym.responses_converter import ResponsesConverter
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.opencode_sandboxed_agent.app import (
    OpenCodeSandboxedAgent,
    OpenCodeSandboxedAgentConfig,
)


# OpenCode prefixes images it moves out of a tool result into a user message with this text.
TOOL_MEDIA_PREAMBLE = "Attached media from tool result:"

_MIME_TO_EXTENSION = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
}


class OpenCodeVisualSandboxedAgentConfig(OpenCodeSandboxedAgentConfig):
    # Declare image input to OpenCode. Without it OpenCode swaps every image for an error string.
    enable_image_input: bool = True
    # Sandbox directory that receives the task prompt's input_image parts before `opencode run --file`.
    prompt_image_dir: str = "/tmp/nemo_gym_prompt_images"
    # Decoded-size cap per prompt image; larger images fail the request instead of bloating the prompt.
    max_prompt_image_bytes: int = 20 * 1024 * 1024
    # How images appear in the rebuilt rollout output. `inline` keeps the data URL the model received;
    # `reference` replaces it with its sha256 so rollouts with many screenshots stay small.
    rollout_image_mode: Literal["inline", "reference"] = "reference"


def decode_image_data_url(url: str) -> Tuple[str, bytes]:
    """Return ``(mime, bytes)`` for a base64 ``data:image/...`` URL.

    Raises ValueError for anything else. Remote URLs are rejected on purpose: a rollout's
    input should not depend on fetching from the network at run time.
    """
    if not url.startswith("data:"):
        raise ValueError(f"input_image.image_url must be a base64 data URL, got {url[:64]!r}")
    header, sep, payload = url.partition(",")
    if not sep or ";base64" not in header:
        raise ValueError("input_image.image_url must be a base64-encoded data URL")
    mime = header[len("data:") :].split(";", 1)[0].strip().lower() or "image/png"
    if not mime.startswith("image/"):
        raise ValueError(f"input_image data URL has non-image MIME type {mime!r}")
    try:
        data = base64.b64decode(payload, validate=True)
    except binascii.Error as exc:
        raise ValueError("input_image data URL is not valid base64") from exc
    return mime, data


def split_prompt_images(
    input_items: List[Any],
) -> Tuple[List[Any], List[str]]:
    """Separate the user turn's ``input_image`` parts from its text.

    Returns the input items with the user content reduced to one text string (the form the base
    agent accepts) and the image data URLs in prompt order.
    """
    rewritten: List[Any] = []
    image_urls: List[str] = []
    for item in input_items:
        role = item.get("role") if isinstance(item, dict) else getattr(item, "role", None)
        content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
        if role != "user" or not isinstance(content, list):
            rewritten.append(item)
            continue

        texts: List[str] = []
        for part in content:
            part_type = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
            if part_type == "input_text":
                texts.append(part["text"] if isinstance(part, dict) else part.text)
            elif part_type == "input_image":
                url = part.get("image_url") if isinstance(part, dict) else getattr(part, "image_url", None)
                if not url:
                    raise ValueError("input_image parts must carry an image_url (file_id is not supported)")
                image_urls.append(url)
            else:
                raise ValueError(f"Unsupported user content part type for OpenCode: {part_type!r}")
        rewritten.append(NeMoGymEasyInputMessage(role="user", content="\n\n".join(texts)))
    return rewritten, image_urls


class OpenCodeVisualSandboxedAgent(OpenCodeSandboxedAgent):
    config: OpenCodeVisualSandboxedAgentConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._sandbox_id_to_prompt_files: Dict[str, List[str]] = dict()

    async def _create_opencode_config(self, request: Request) -> Dict[str, Any]:
        opencode_config = await super()._create_opencode_config(request)
        if self.config.enable_image_input:
            for provider in opencode_config["provider"].values():
                for model in provider["models"].values():
                    model["modalities"] = {"input": ["text", "image"], "output": ["text"]}
                    model["attachment"] = True
        return opencode_config

    def _opencode_run_extra_args(self, request: Request) -> str:
        files = self._sandbox_id_to_prompt_files.get(request.cookies["sandbox_id"], [])
        return " ".join(f"--file {quote(path)}" for path in files)

    async def _upload_prompt_images(self, sandbox: Any, image_urls: List[str]) -> List[str]:
        remote_paths: List[str] = []
        with TemporaryDirectory() as tmp_dir:
            for index, url in enumerate(image_urls):
                mime, data = decode_image_data_url(url)
                if len(data) > self.config.max_prompt_image_bytes:
                    raise ValueError(
                        f"Prompt image {index} is {len(data)} bytes, above max_prompt_image_bytes="
                        f"{self.config.max_prompt_image_bytes}"
                    )
                filename = f"prompt_image_{index}{_MIME_TO_EXTENSION.get(mime, '.png')}"
                local_path = Path(tmp_dir) / filename
                local_path.write_bytes(data)
                remote_path = f"{self.config.prompt_image_dir}/{filename}"
                await sandbox.upload(local_path, remote_path)
                remote_paths.append(remote_path)
        return remote_paths

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        sandbox_key = request.cookies["sandbox_id"]
        input_items, image_urls = split_prompt_images(list(body.input))
        if image_urls:
            if not self.config.enable_image_input:
                raise ValueError("The task prompt has input_image parts but enable_image_input is false")
            sandbox = self._sandbox_id_to_sandbox[sandbox_key]
            mkdir_result = await sandbox.exec(f"mkdir -p {quote(self.config.prompt_image_dir)}")
            if mkdir_result.return_code != 0:
                raise RuntimeError(f"Could not create {self.config.prompt_image_dir} in the sandbox: {mkdir_result}")
            self._sandbox_id_to_prompt_files[sandbox_key] = await self._upload_prompt_images(sandbox, image_urls)
        try:
            return await super().responses(request, body.model_copy(update={"input": input_items}))
        finally:
            self._sandbox_id_to_prompt_files.pop(sandbox_key, None)

    def _rollout_image_part(self, url: str, mime: Optional[str] = None) -> Dict[str, Any]:
        if self.config.rollout_image_mode == "inline":
            return {"type": "input_image", "image_url": url, "detail": "auto"}
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        mime_suffix = f";mime={mime}" if mime else ""
        return {"type": "input_image", "image_url": f"sha256:{digest}{mime_suffix}", "detail": "auto"}

    def _opencode_export_to_output_items(self, opencode_export: Dict[str, Any]) -> List[NeMoGymResponseOutputItem]:
        """Rebuild the conversation OpenCode sent, including image parts.

        Mirrors the base conversion and adds images. User ``file`` parts become ``input_image``
        parts. Image attachments on a tool result become the user message OpenCode sends after it.
        """
        messages: List[NeMoGymResponseOutputItem] = []
        for message in opencode_export["messages"]:
            role = message["info"]["role"]
            if role == "user":
                parts: List[Any] = []
                for part in message["parts"]:
                    if part["type"] == "text":
                        parts.append(ResponseInputTextParam(text=part["text"], type="input_text"))
                    elif part["type"] == "file" and str(part.get("mime", "")).startswith("image/"):
                        parts.append(self._rollout_image_part(part["url"], part.get("mime")))
                messages.append(NeMoGymEasyInputMessage(content=parts, role="user"))
            elif role == "assistant":
                converter = ResponsesConverter(return_token_id_information=True)
                for part in message["parts"]:
                    if part["type"] == "text":
                        messages.extend(
                            converter.postprocess_assistant_message_dict(
                                message_dict={"content": part["text"], "role": "assistant"}
                            )
                        )
                    elif part["type"] == "reasoning":
                        messages.extend(
                            converter.postprocess_assistant_message_dict(
                                message_dict={
                                    "content": converter._wrap_reasoning_in_think_tags([part["text"]]),
                                    "role": "assistant",
                                }
                            )
                        )
                    elif part["type"] == "tool":
                        state = part["state"]
                        messages.append(
                            NeMoGymResponseFunctionToolCall(
                                arguments=json.dumps(state["input"]),
                                call_id=part["callID"],
                                name=part["tool"],
                            )
                        )
                        messages.append(
                            NeMoGymFunctionCallOutput(call_id=part["callID"], output=state.get("output", ""))
                        )
                        images = [
                            attachment
                            for attachment in state.get("attachments") or []
                            if str(attachment.get("mime", "")).startswith("image/") and attachment.get("url")
                        ]
                        if images:
                            messages.append(
                                NeMoGymEasyInputMessage(
                                    role="user",
                                    content=[
                                        ResponseInputTextParam(text=TOOL_MEDIA_PREAMBLE, type="input_text"),
                                        *(self._rollout_image_part(a["url"], a.get("mime")) for a in images),
                                    ],
                                )
                            )
                    elif part["type"] in ("step-finish", "step-start", "patch"):
                        pass
                    else:
                        raise NotImplementedError(part)
            else:
                raise NotImplementedError(message)
        return messages


if __name__ == "__main__":
    OpenCodeVisualSandboxedAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = OpenCodeVisualSandboxedAgent.run_webserver()  # noqa: F401
