# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Image-tool environment: the tool surface and the PivotRL verifier.

``base`` holds the ten image tools, the tool-call parser and the bbox helpers.
``app`` is the PivotRL resources server that scores one emitted tool call
against the action demonstrated in the SFT trajectory.

Only ``base`` is re-exported here. ``app`` is deliberately left out so the
image-tools agent can ``from resources_servers.image_tools import ...`` for the
tool surface without pulling in a verifier it never runs.
"""

from .base import (
    IMAGE_TOOL_NAMES,
    ImageToolsGymToolConfig,
    ImageToolsGymToolLogic,
    ImageToolsGymToolMetadata,
    bbox_iou,
    coerce_bbox,
    execute_image_tool,
    extract_last_assistant_text,
    has_malformed_image_tool_markup,
    has_malformed_image_tool_raw_generation,
    parse_image_tool_calls,
)


__all__ = [
    "IMAGE_TOOL_NAMES",
    "ImageToolsGymToolConfig",
    "ImageToolsGymToolLogic",
    "ImageToolsGymToolMetadata",
    "bbox_iou",
    "coerce_bbox",
    "execute_image_tool",
    "extract_last_assistant_text",
    "has_malformed_image_tool_markup",
    "has_malformed_image_tool_raw_generation",
    "parse_image_tool_calls",
]
