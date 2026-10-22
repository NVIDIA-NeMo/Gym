# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in tool image/video transport over the stock vLLM model server."""

from typing import Any

from nemo_gym.responses_converter import ResponsesConverter, ResponsesConverterState
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from responses_api_models.vllm_model.app import VLLMModel


class ToolMediaConverter(ResponsesConverter):
    """Keep media in the tool result that returned it, without synthetic user turns.

    Other model servers use the shared converter. This adapter is for a vLLM
    backend explicitly validated for tool-role media.
    """

    def _format_function_call_output(
        self,
        m: dict,
        state: ResponsesConverterState,
        *,
        drop_unrepresentable_request_fields: bool = False,
    ) -> None:
        output = m["output"]
        if isinstance(output, list) and any(part["type"] in {"input_image", "input_video"} for part in output):
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
        )


if __name__ == "__main__":
    VideoVLLMModel.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = VideoVLLMModel.run_webserver()
