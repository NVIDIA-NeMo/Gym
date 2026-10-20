# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in image observations through mini-SWE's existing model extension hook."""

import json

from minisweagent.models.litellm_response_model import LitellmResponseModel


class VisionResponseModel(LitellmResponseModel):
    """Keep native text observations and append images after the tool-result batch."""

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        text_outputs = []
        image_messages = []
        actions = message.get("extra", {}).get("actions", [])
        for index, output in enumerate(outputs):
            text = []
            for line in output.get("output", "").splitlines(keepends=True):
                try:
                    envelope = json.loads(line)
                except ValueError:
                    envelope = None
                if not isinstance(envelope, dict) or envelope.get("_nemo_gym_view_image") != 1:
                    text.append(line)
                    continue
                call_id = actions[index].get("tool_call_id", str(index))
                description = f"Image from bash call {call_id}: {envelope['path']} (sha256: {envelope['sha256']})"
                text.append(description + "\n")
                image_messages.append(
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": description}]
                        + [
                            {
                                "type": "input_image",
                                "image_url": f"data:{part['mimeType']};base64,{part['data']}",
                                "detail": "auto",
                            }
                            for part in envelope["content"]
                        ],
                    }
                )
            text_outputs.append({**output, "output": "".join(text)})
        return super().format_observation_messages(message, text_outputs, template_vars) + image_messages
