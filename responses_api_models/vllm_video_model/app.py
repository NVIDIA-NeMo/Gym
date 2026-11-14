# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in tool image/video transport over the stock vLLM model server.

Also sends local video paths, which NeMo-RL Gym rows use, as file:// URLs: vLLM reads
local media only from file:// URLs under --allowed-local-media-path.
"""

from pathlib import Path
from typing import Any

from nemo_gym.responses_converter import ResponsesConverter, ResponsesConverterState
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from responses_api_models.vllm_model.app import VLLMModel


def _local_video_as_file_url(part: Any) -> Any:
    """Return an ``input_video`` part with an absolute local path as a file:// URL."""
    if not isinstance(part, dict) or part.get("type") != "input_video":
        return part
    key = "video_url" if "video_url" in part else "video"
    source = part.get(key)
    if isinstance(source, str) and source.startswith("/"):
        return {**part, key: Path(source).as_uri()}
    if isinstance(source, dict) and str(source.get("url", "")).startswith("/"):
        return {**part, key: {**source, "url": Path(source["url"]).as_uri()}}
    return part


class ToolMediaConverter(ResponsesConverter):
    """Keep media in the tool result that returned it, without synthetic user turns.

    Other model servers use the shared converter. This adapter is for a vLLM
    backend explicitly validated for tool-role media.
    """

    def _format_message(
        self,
        m: dict,
        state: ResponsesConverterState,
        *,
        drop_unrepresentable_request_fields: bool = False,
    ) -> None:
        content = m.get("content")
        if isinstance(content, list):
            m = {**m, "content": [_local_video_as_file_url(part) for part in content]}
        super()._format_message(m, state, drop_unrepresentable_request_fields=drop_unrepresentable_request_fields)

    def _format_function_call_output(
        self,
        m: dict,
        state: ResponsesConverterState,
        *,
        drop_unrepresentable_request_fields: bool = False,
    ) -> None:
        output = m["output"]
        if isinstance(output, list) and any(part["type"] in {"input_image", "input_video"} for part in output):
            # Chat has no "original" detail; like the shared tool-image path, send it as "high". Only tool
            # images: a user image with "original" stays unsupported.
            output = [
                {**part, "detail": "high"}
                if part["type"] == "input_image" and part.get("detail") == "original"
                else part
                for part in output
            ]
            # The message formatter maps each structured media part and
            # flushes the preceding assistant tool calls. It does not sort calls.
            self._format_message(
                {"role": "tool", "tool_call_id": m["call_id"], "content": output},
                state,
                drop_unrepresentable_request_fields=drop_unrepresentable_request_fields,
            )
            return
        super()._format_function_call_output(
            m, state, drop_unrepresentable_request_fields=drop_unrepresentable_request_fields
        )

    @staticmethod
    def _chat_tool_content_part_to_responses(part: dict[str, Any]) -> dict[str, Any]:
        if part["type"] == "video_url":
            video = part["video_url"]
            return {"type": "input_video", "video_url": video["url"] if isinstance(video, dict) else video}
        return ResponsesConverter._chat_tool_content_part_to_responses(part)


class VideoVLLMModel(VLLMModel):
    """The stock vLLM model server, with ToolMediaConverter as its converter."""

    def get_converter(self) -> ToolMediaConverter:
        return ToolMediaConverter(
            return_token_id_information=self.config.return_token_id_information,
            uses_reasoning_parser=self.config.uses_reasoning_parser,
            custom_tool_strategy=self.config.custom_tool_strategy,
        )


if __name__ == "__main__":
    VideoVLLMModel.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = VideoVLLMModel.run_webserver()
