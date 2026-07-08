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
"""Tool JSON-schema cleaning, ported verbatim from EnterpriseOps-Gym.

This is a line-for-line port of ``benchmark/llm_client.py::LLMClient._clean_json_schema``
from https://github.com/ServiceNow/EnterpriseOps-Gym (Apache 2.0). Parity with the
upstream benchmark requires identical cleaning semantics, so please do NOT "improve"
this logic — fix bugs upstream first, then mirror them here.
"""

import logging
from typing import Any, Dict, List


logger = logging.getLogger(__name__)


def clean_json_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Clean a JSON schema to be compatible with LLM APIs (Anthropic, Vertex AI, etc.)

    Removes oneOf, allOf, anyOf at top level and converts to simple object schema.
    Also handles optional parameters with type arrays like ['STRING', 'NULL'].
    """
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}, "required": []}

    # If schema has oneOf/allOf/anyOf at top level, extract the first valid object schema
    if "oneOf" in schema:
        logger.debug("Schema has oneOf at top level, extracting first object schema")
        for option in schema["oneOf"]:
            if isinstance(option, dict) and option.get("type") == "object":
                schema = option
                break
        else:
            # No object schema found, return empty
            return {"type": "object", "properties": {}, "required": []}

    if "allOf" in schema:
        logger.debug("Schema has allOf at top level, merging schemas")
        merged_schema: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}
        for sub_schema in schema["allOf"]:
            if isinstance(sub_schema, dict):
                if "properties" in sub_schema:
                    merged_schema["properties"].update(sub_schema["properties"])
                if "required" in sub_schema:
                    merged_schema["required"].extend(sub_schema["required"])
        schema = merged_schema

    if "anyOf" in schema:
        logger.debug("Schema has anyOf at top level, extracting first object schema")
        for option in schema["anyOf"]:
            if isinstance(option, dict) and option.get("type") == "object":
                schema = option
                break
        else:
            return {"type": "object", "properties": {}, "required": []}

    # Ensure schema has required fields
    if "type" not in schema:
        schema["type"] = "object"

    if schema["type"] == "object" and "properties" not in schema:
        schema["properties"] = {}

    # Clean property schemas recursively to handle optional parameters
    # Vertex AI doesn't support type arrays like ['STRING', 'NULL']
    if "properties" in schema:
        cleaned_properties = {}
        optional_properties: List[str] = []  # Track properties that should not be required

        for prop_name, prop_schema in schema["properties"].items():
            if isinstance(prop_schema, dict):
                cleaned_prop = prop_schema.copy()

                # Handle type arrays like ['STRING', 'NULL'] or ['string', 'null']
                if "type" in cleaned_prop and isinstance(cleaned_prop["type"], list):
                    type_list = cleaned_prop["type"]
                    # Filter out 'null' or 'NULL' and take the first non-null type
                    non_null_types = [t for t in type_list if t.lower() != "null"]
                    if non_null_types:
                        cleaned_prop["type"] = non_null_types[0]
                        # If NULL was in the list, this is an optional parameter
                        if len(non_null_types) < len(type_list):
                            optional_properties.append(prop_name)
                            logger.debug(
                                f"Property '{prop_name}' has optional type {type_list}, "
                                f"converted to {non_null_types[0]}"
                            )
                    else:
                        # All types were null, default to string
                        cleaned_prop["type"] = "string"
                        optional_properties.append(prop_name)

                # Recursively clean nested object schemas
                if cleaned_prop.get("type") == "object" and "properties" in cleaned_prop:
                    cleaned_prop = clean_json_schema(cleaned_prop)

                # Handle arrays with item schemas
                if cleaned_prop.get("type") == "array" and "items" in cleaned_prop:
                    if isinstance(cleaned_prop["items"], dict):
                        cleaned_prop["items"] = clean_json_schema(cleaned_prop["items"])

                cleaned_properties[prop_name] = cleaned_prop
            else:
                cleaned_properties[prop_name] = prop_schema

        schema["properties"] = cleaned_properties

        # Remove optional properties from the required array
        if "required" in schema and optional_properties:
            schema["required"] = [req for req in schema["required"] if req not in optional_properties]
            if optional_properties:
                logger.debug(f"Removed optional properties from required list: {optional_properties}")

    return schema
