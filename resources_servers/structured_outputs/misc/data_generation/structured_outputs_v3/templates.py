# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Shared template banks for structured outputs data generation.

Generalized from the 260309_nano_v3_sdg_*.py scripts to support all formats.
"""

import json
import random
from typing import Any, Dict, List, Optional

import yaml


ALL_FORMATS = ["json", "yaml", "xml", "toml", "csv"]

FORMAT_NAMES = {
    "json": "JSON",
    "yaml": "YAML",
    "xml": "XML",
    "toml": "TOML",
    "csv": "CSV",
}

SCHEMA_INSTRUCTIONS = {
    "json": [
        "Response Formatting Schema (JSON): {schema}",
        "Format your response as an object matching the provided JSON schema: {schema}",
        "Structure your response according to the following JSON schema specification: {schema}. Return only the JSON output.",
        "Response Format (JSON): {schema}",
        "I'd like you to format your response as a JSON object matching the provided schema: {schema}",
        "Structure your response according to the following JSON schema specification: {schema}. Validate that your output conforms to all schema constraints and required properties. Return only the JSON output without styling it in backticks.",
    ],
    "yaml": [
        "Response Formatting Schema (YAML): {schema}",
        "Format your response as valid YAML matching the provided schema: {schema}",
        "Structure your response according to the following schema specification: {schema}. Return only the YAML output.",
        "Response Format (YAML): {schema}",
        "I'd like you to format your response as a YAML document matching the provided schema: {schema}",
        "Structure your response according to the following schema specification: {schema}. Return only the YAML output without styling it in backticks.",
    ],
    "xml": [
        "Response Formatting Schema (XML): {schema}",
        "Format your response as valid XML matching the provided schema: {schema}",
        "Structure your response according to the following schema specification: {schema}. Return only the XML output.",
        "Response Format (XML): {schema}",
        "I'd like you to format your response as an XML document matching the provided schema: {schema}",
        "Structure your response according to the following schema: {schema}. Return only well-formed XML without markdown fencing.",
    ],
    "toml": [
        "Response Formatting Schema (TOML): {schema}",
        "Format your response as valid TOML matching the provided schema: {schema}",
        "Structure your response according to the following schema specification: {schema}. Return only the TOML output.",
        "Response Format (TOML): {schema}",
        "I'd like you to format your response as a TOML document matching the provided schema: {schema}",
        "Structure your response as TOML conforming to the schema: {schema}. Return raw TOML without markdown fencing.",
    ],
    "csv": [
        "Response Formatting Schema (CSV): {schema}",
        "Format your response as CSV with headers matching the provided schema: {schema}",
        "Structure your response as a CSV table according to the following schema: {schema}. Include the header row.",
        "Response Format (CSV): {schema}",
        "Return your response as comma-separated values matching this schema: {schema}. Include headers.",
        "Output CSV data conforming to the schema: {schema}. First row must be headers. No markdown fencing.",
    ],
}

USER_QUERY_INSTRUCTIONS = [
    "Generate output that strictly adheres to the specified schema based on the document provided.",
    "Format the document based on the provided schema.",
    "Fit the document to the given format.",
    "Extract the information from the text and format it according to the schema.",
    "Map the content of this document to the provided data structure.",
    "Parse the document and populate the following data model.",
    "Please provide the answer in a format that conforms to the specified structure.",
    "Convert the unstructured text into the specified structured format.",
    "Ensure your output validates against the given schema.",
    "Restructure the provided information according to the following template.",
]

DOCUMENT_TEMPLATES = [
    "{user_message}\n\nDocument:\n{document}",
    "{user_message}\n\n{document}",
    "# Problem:\n{user_message}\n\n{document}",
    "# Instructions:\n{user_message}\n\n# Document:\n{document}",
    "# Document:\n{document}\n\n# Instructions: {user_message}",
    "# Information\n{document}\n\n# Problem: {user_message}",
    "Given the following text:\n\n{document}\n\n{user_message}",
]

TRANSLATION_TEMPLATES = [
    "Here is data in {source_format}:\n\n{source_output}\n\nSchema: {schema}\n\nConvert this into {target_format} format.",
    "Translate the following {source_format} data into {target_format}.\n\nInput:\n{source_output}\n\nTarget schema: {schema}",
    "Given this {source_format} output:\n\n{source_output}\n\nRe-format it as {target_format} conforming to: {schema}",
    "Convert the structured data below from {source_format} to {target_format}.\n\nData:\n{source_output}\n\nSchema: {schema}\n\nReturn only the {target_format} output.",
]

CORRECTION_TEMPLATES = [
    "The following {fmt} output has errors. Fix it to conform to the schema.\n\nSchema: {schema}\n\nBroken output:\n{corrupted}",
    "This {fmt} data doesn't match the schema. Correct it.\n\nSchema: {schema}\n\nInput:\n{corrupted}",
    "Fix the {fmt} output below so it validates against the given schema.\n\nSchema: {schema}\n\nOutput to fix:\n{corrupted}\n\nReturn only the corrected {fmt}.",
]

SCHEMA_ONLY_TEMPLATES = [
    "Generate a realistic example that conforms to this {fmt} schema:\n\n{schema}",
    "Create sample data matching the following schema. Output as {fmt}.\n\nSchema: {schema}",
    "Produce a valid {fmt} document that conforms to:\n\n{schema}\n\nUse realistic placeholder values.",
    "Given the schema below, generate a plausible {fmt} instance.\n\n{schema}",
]

MULTISTEP_FOLLOWUP_TEMPLATES = [
    "Now convert your response to {target_format} format. Keep the same data but change the output format.",
    "Re-format your previous answer as {target_format}. The schema is: {schema}",
    "Good. Now output the same information in {target_format} format instead.",
    "Convert your answer above to {target_format}. Ensure it validates against: {schema}",
]


def represent_schema(schema_dict: Dict, mode: str, native_schema_str: Optional[str] = None) -> str:
    """Serialize a JSON Schema dict in the given representation mode."""
    if mode == "json":
        return json.dumps(schema_dict)
    elif mode == "yaml":
        return yaml.dump(schema_dict, default_flow_style=False).strip()
    elif mode == "python":
        return repr(schema_dict)
    elif mode == "native" and native_schema_str:
        return native_schema_str
    return json.dumps(schema_dict)


def template_document(user_message: str, document: str, rng: random.Random) -> str:
    return rng.choice(DOCUMENT_TEMPLATES).format(user_message=user_message, document=document)


def template_messages(system_message: str, user_message: str, rng: random.Random) -> List[Dict[str, str]]:
    layouts = [
        [{"role": "user", "content": system_message}, {"role": "user", "content": user_message}],
        [{"role": "user", "content": f"{system_message}\n{user_message}"}],
        [{"role": "user", "content": f"{user_message}\n{system_message}"}],
        [{"role": "user", "content": user_message}, {"role": "user", "content": system_message}],
        [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}],
    ]
    return rng.choice(layouts)


def make_gym_record(
    input_msgs: List[Dict],
    schema_dict: Dict,
    schema_type: str,
    problem_type: str,
    schema_repr: str,
    source_record_id: str,
    num_turns: int = 1,
    source_format: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "responses_create_params": {"input": input_msgs},
        "schema_str": json.dumps(schema_dict, ensure_ascii=False),
        "schema_type": schema_type,
        "problem_type": problem_type,
        "schema_repr": schema_repr,
        "source_format": source_format,
        "num_turns": num_turns,
        "source_record_id": source_record_id,
        "agent_ref": {
            "type": "responses_api_agents",
            "name": "structured_outputs_v3_simple_agent",
        },
    }
