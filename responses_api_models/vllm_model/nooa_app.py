# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NOOA provider compatibility layered on Gym's unchanged VLLM implementation."""

from typing import Any

from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from responses_api_models.vllm_model.app import VLLMModel


class NOOAVLLMModel(VLLMModel):
    """Omit NOOA's cache hint and equal reasoning aliases on NVIDIA's gateway."""

    def _apply_sampling_overrides(self, body_dict: dict[str, Any]) -> dict[str, Any]:
        super()._apply_sampling_overrides(body_dict)
        if self.config.base_url and all(
            str(url).rstrip("/") == "https://inference-api.nvidia.com/v1" for url in self.config.base_url
        ):
            body_dict.pop("prompt_cache_key", None)
            if isinstance(body_dict.get("messages"), list):
                messages = list(body_dict["messages"])
                for index, message in enumerate(messages):
                    if (
                        isinstance(message, dict)
                        and "reasoning" in message
                        and "reasoning_content" in message
                        and message["reasoning"] == message["reasoning_content"]
                    ):
                        messages[index] = {key: value for key, value in message.items() if key != "reasoning"}
                body_dict["messages"] = messages
        return body_dict


if __name__ == "__main__":
    NOOAVLLMModel.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = NOOAVLLMModel.run_webserver()
