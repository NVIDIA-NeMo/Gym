# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Visual evaluation against hosted Chat APIs with provider sampling defaults."""

from typing import Any

from fastapi import Request

from responses_api_models.vllm_model.visual_eval import VisualEvalModel


class HostedVisualEvalModel(VisualEvalModel):
    def _preprocess_chat_completion_create_params(self, request: Request, body_dict: dict[str, Any]) -> dict[str, Any]:
        body_dict = super()._preprocess_chat_completion_create_params(request, body_dict)
        # Use hosted-model sampling without forwarding Nemotron
        # sampling/template extensions or OpenCode's sampling defaults.
        for key in (
            "temperature",
            "top_p",
            "top_k",
            "skip_special_tokens",
            "chat_template_kwargs",
            "max_completion_tokens",
        ):
            body_dict.pop(key, None)
        for tool in body_dict.get("tools", []):
            tool.get("function", {}).pop("strict", None)
        return body_dict


if __name__ == "__main__":
    HostedVisualEvalModel.run_webserver()
