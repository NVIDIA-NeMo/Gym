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
"""Tool-call structured outputs data generation for RL training.

Reads verified structured output records and produces Gym-ready records where
the schema is exposed as an OpenAI Responses API function tool. The prompt text
contains only a document and a short task instruction; it does not include the
schema or a requested text format.
"""

import argparse
import json
import random
import sys
import tomllib
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import xmltodict
import yaml


TOOL_SCHEMA_MODES = ("direct", "extraction_wrapper", "random_wrapper")
DISTRACTOR_STYLES = (
    "separate_tools",
    "numbered_tools",
    "single_tool_multi_key",
    "single_tool_oneof",
    "single_tool_anyof",
    "single_tool_defs_oneof",
    "single_tool_defs_anyof",
)
TOOL_NAME_STYLES = ("semantic", "numbered")
DEFAULT_TOOL_SCHEMA_MODE_WEIGHTS = "direct:10,extraction_wrapper:35,random_wrapper:55"
DEFAULT_WRAPPER_KEY_POOL = "output,result,record,data,answer,summary,extracted_info,structured_response"
DEFAULT_DISTRACTOR_STYLE_WEIGHTS = "separate_tools:35,numbered_tools:35,single_tool_multi_key:30"
DEFAULT_DISTRACTOR_COUNT_WEIGHTS = (
    "0:1,1:1,2:1,3:1,4:1,5:2,6:2,7:3,8:4,9:5,10:7,11:8,12:10,13:11,14:12,15:12,16:10,17:8,18:6,19:4,20:3"
)
DEFAULT_PARALLEL_TOOL_CALLS_TRUE_RATIO = 0.25
DEFAULT_TOOL_NAME_STYLE_WEIGHTS = "semantic:50,numbered:50"
DEFAULT_TOOL_NAME_POOL = (
    "submit_structured_output,extract_record,record_answer,summarize_document,populate_schema,"
    "extraction,extract,summary,structured_output,submit_summary,return_record,document_summary,"
    "extract_structured_data,capture_information,record_extraction"
)
DEFAULT_NUMBERED_TOOL_PREFIX_POOL = "extraction_tool,summary_tool,structured_output_tool,document_tool,response_tool"
DEFAULT_UNION_PAYLOAD_KEY_POOL = (
    "extraction,summary,structured_output,record,data,answer,result,document_info,extracted_info"
)
DEFAULT_SOURCE_FORMATS = "json,yaml,xml,toml,csv"

INSTRUCTION_POOLS = {
    "concise": [
        "Extract the document into the available structured tool.",
        "Return the relevant document facts with the tool.",
        "Call the matching tool with the document's structured data.",
        "Capture the source text as structured tool arguments.",
    ],
    "standard": [
        "Read the document and use the provided function to submit the structured information.",
        "Summarize the source text by filling the available tool arguments.",
        "Capture the key fields from the document in the appropriate function call.",
        "Use the matching tool to return the document information in structured form.",
    ],
    "explicit": [
        "Use the tool call as the final answer; populate its arguments from the document instead of writing prose.",
        "Choose the function that matches the document and fill its arguments only with information supported by the text.",
        "Read the document, identify the requested fields from the available tool, and submit the structured result as function arguments.",
        "Return a function call whose arguments contain the structured document summary; do not include a separate prose answer.",
    ],
}

SYSTEM_INSTRUCTION_POOLS = {
    "tool_required": [
        "The final answer must be a function call.",
        "Use a provided function call for the final answer.",
    ],
    "tool_selection": [
        "Select the provided tool that best matches the document.",
        "Choose the matching function and fill it with facts from the source text.",
    ],
    "no_prose": [
        "Do not write a prose answer; return the structured result through the tool.",
        "Avoid free-form text and answer through the available function.",
    ],
    "final_answer_surface": [
        "Treat the tool arguments as the answer surface.",
        "The structured function arguments are the final response.",
    ],
}

DOCUMENT_HEADERS = [
    "Document:",
    "<document>",
    "# Document",
    "Source text:",
    "Input document:",
    "Reference text:",
]

DESCRIPTION_TEMPLATES = [
    "Submit structured information extracted from the document.",
    "Record the structured answer for this document.",
    "Return a structured summary for the source text.",
    "Capture the document information in a typed payload.",
    "Record the document facts in the required argument shape.",
    "Capture the relevant fields from the document.",
    "Fill this function with information grounded in the source text.",
    "Use this function to return the document-level structured answer.",
]


def parse_schema_to_dict(schema_str: str, fmt: str) -> dict:
    if fmt == "json":
        obj = json.loads(schema_str)
    elif fmt == "yaml":
        obj = yaml.safe_load(schema_str)
    elif fmt == "toml":
        obj = tomllib.loads(schema_str)
    elif fmt == "xml":
        raw = xmltodict.parse(schema_str)
        top = raw.get("schema_definition", raw)
        obj = top if isinstance(top, dict) else raw
    elif fmt == "csv":
        return _csv_schema_to_dict(schema_str)
    else:
        raise ValueError(f"Unknown format: {fmt}")

    if isinstance(obj, dict) and "schema" in obj:
        return obj["schema"]
    return obj


def _csv_schema_to_dict(schema_str: str) -> dict:
    lines = schema_str.strip().split("\n")
    headers = [h.strip() for h in lines[0].split(",")]
    types = [t.strip() for t in lines[1].split(",")] if len(lines) > 1 else ["string"] * len(headers)
    properties = {}
    for name, typ in zip(headers, types):
        properties[name] = {"type": typ if typ in ("string", "integer", "number", "boolean") else "string"}
    return {
        "type": "array",
        "items": {"type": "object", "properties": properties, "required": headers, "additionalProperties": False},
    }


def load_records(path: Path, source_formats: List[str], exclude_substrings: List[str]) -> List[Dict[str, Any]]:
    records = []
    lowered_excludes = [s.lower() for s in exclude_substrings]
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if lowered_excludes and any(s in line.lower() for s in lowered_excludes):
                continue
            record = json.loads(line)
            fmt = record.get("target_output_format", "json")
            if fmt not in source_formats:
                continue
            try:
                schema = parse_schema_to_dict(record.get("structured_schema", ""), fmt)
            except Exception as e:
                print(f"  SKIP line {i}: schema parse failed: {e}")
                continue
            if not isinstance(schema, dict):
                print(f"  SKIP line {i}: parsed schema is not a dict")
                continue
            record["_json_schema"] = schema
            record["_record_id"] = record.get("metadata", {}).get("record_id", f"line_{i}")
            record["_source_index"] = i
            records.append(record)
    return records


def parse_weights(weights_str: str, allowed: Tuple[str, ...], label: str) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for pair in weights_str.split(","):
        k, v = pair.strip().split(":")
        key = k.strip()
        if key not in allowed:
            raise ValueError(f"Unknown {label}: {key}")
        weights[key] = float(v.strip())
    if not weights or sum(weights.values()) <= 0:
        raise ValueError(f"{label} weights must have positive total weight")
    return weights


def parse_int_weights(weights_str: str, label: str) -> Optional[Dict[int, float]]:
    if not weights_str.strip():
        return None

    weights: Dict[int, float] = {}
    for pair in weights_str.split(","):
        k, v = pair.strip().split(":")
        key = int(k.strip())
        if key < 0:
            raise ValueError(f"{label} counts must be non-negative, got {key}")
        weight = float(v.strip())
        if weight < 0:
            raise ValueError(f"{label} weights must be non-negative, got {weight}")
        weights[key] = weight
    if not weights or sum(weights.values()) <= 0:
        raise ValueError(f"{label} weights must have positive total weight")
    return weights


def parse_csv_arg(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in ("1", "true", "yes", "y"):
        return True
    if lowered in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {value}")


def weighted_choice(weights: Dict[str, float], rng: random.Random) -> str:
    total = sum(weights.values())
    threshold = rng.random() * total
    cumulative = 0.0
    for key, weight in weights.items():
        cumulative += weight
        if threshold <= cumulative:
            return key
    return next(reversed(weights))


def is_object_schema(schema: Dict[str, Any]) -> bool:
    return schema.get("type") == "object" or ("properties" in schema and schema.get("type") is None)


def resolve_json_pointer(root: Dict[str, Any], ref: str) -> Any:
    if not ref.startswith("#/"):
        raise ValueError(f"Only local JSON Schema refs are supported in tool schemas, got {ref}")
    current: Any = root
    for raw_part in ref[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Could not resolve JSON Schema ref {ref}")
        current = current[part]
    return current


def is_definition_ref(ref: str) -> bool:
    return ref.startswith("#/definitions/") or ref.startswith("#/$defs/")


def inline_non_definition_refs(schema: Dict[str, Any]) -> Dict[str, Any]:
    root = deepcopy(schema)

    def visit(node: Any, ref_stack: Tuple[str, ...] = ()) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/") and not is_definition_ref(ref):
                if ref in ref_stack:
                    return node
                resolved = visit(deepcopy(resolve_json_pointer(root, ref)), ref_stack + (ref,))
                if not isinstance(resolved, dict):
                    return resolved
                for key, value in node.items():
                    if key == "$ref":
                        continue
                    resolved[key] = visit(value, ref_stack)
                return resolved
            return {key: visit(value, ref_stack) for key, value in node.items()}
        if isinstance(node, list):
            return [visit(item, ref_stack) for item in node]
        return node

    return visit(root)


def strictify_tool_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    strict_schema = inline_non_definition_refs(schema)

    def normalize_enum(value: Any) -> List[Any]:
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            for key in ("item", "items", "value", "values", "enum"):
                if key in value:
                    candidate = value[key]
                    return candidate if isinstance(candidate, list) else [candidate]
            if len(value) == 1:
                candidate = next(iter(value.values()))
                return candidate if isinstance(candidate, list) else [candidate]
            return list(value.values())
        return [value]

    def normalize_nullable(node: Dict[str, Any]) -> None:
        nullable = node.pop("nullable", None)
        if isinstance(nullable, str):
            nullable = nullable.lower() in ("1", "true", "yes", "y")
        if not nullable:
            return

        schema_type = node.get("type")
        if isinstance(schema_type, str):
            if schema_type != "null":
                node["type"] = [schema_type, "null"]
        elif isinstance(schema_type, list):
            if "null" not in schema_type:
                node["type"] = [*schema_type, "null"]

    def normalize_bool_schema_keyword(node: Dict[str, Any], key: str) -> None:
        value = node.get(key)
        if not isinstance(value, str):
            return
        lowered = value.lower()
        if lowered in ("1", "true", "yes", "y"):
            node[key] = True
        elif lowered in ("0", "false", "no", "n"):
            node[key] = False

    def visit(node: Any, *, in_properties_map: bool = False) -> None:
        if isinstance(node, dict):
            if in_properties_map:
                for key, value in list(node.items()):
                    if isinstance(value, (dict, bool)):
                        visit(value)
                    else:
                        node.pop(key, None)
                return

            node.pop("optional", None)
            normalize_nullable(node)
            normalize_bool_schema_keyword(node, "additionalProperties")
            normalize_bool_schema_keyword(node, "additionalItems")
            if "enum" in node:
                enum_values = normalize_enum(node["enum"])
                if enum_values:
                    node["enum"] = enum_values
                else:
                    node.pop("enum", None)
            if isinstance(node.get("format"), str):
                node.pop("format", None)
            if "properties" in node and isinstance(node["properties"], dict):
                node.setdefault("type", "object")
                node["required"] = list(node["properties"])
                node["additionalProperties"] = False
                visit(node["properties"], in_properties_map=True)
            for key, value in node.items():
                if key == "properties":
                    continue
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(strict_schema)
    return strict_schema


def escape_json_pointer_part(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def unescape_json_pointer_part(value: str) -> str:
    return value.replace("~1", "/").replace("~0", "~")


def rewrite_root_definition_refs(node: Any, prefix: str) -> None:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str):
            for base in ("#/definitions/", "#/$defs/"):
                if ref.startswith(base):
                    suffix = ref[len(base) :]
                    first_part, sep, rest = suffix.partition("/")
                    new_first_part = escape_json_pointer_part(prefix + unescape_json_pointer_part(first_part))
                    node["$ref"] = f"#/$defs/{new_first_part}{sep}{rest}"
                    break
        for value in node.values():
            rewrite_root_definition_refs(value, prefix)
    elif isinstance(node, list):
        for item in node:
            rewrite_root_definition_refs(item, prefix)


def hoist_root_definitions(schema: Dict[str, Any], prefix: str = "") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    schema = deepcopy(schema)
    definitions: Dict[str, Any] = {}
    for key in ("definitions", "$defs"):
        raw_definitions = schema.pop(key, None)
        if not isinstance(raw_definitions, dict):
            continue
        for name, definition_schema in raw_definitions.items():
            definitions[prefix + name] = definition_schema

    if definitions:
        rewrite_root_definition_refs(schema, prefix)
        rewrite_root_definition_refs(definitions, prefix)

    return schema, definitions


def attach_definitions(schema: Dict[str, Any], definitions: Dict[str, Any]) -> Dict[str, Any]:
    if definitions:
        schema.setdefault("$defs", {}).update(definitions)
    return schema


def make_verification_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    verification_schema, definitions = hoist_root_definitions(strictify_tool_schema(schema))
    return attach_definitions(verification_schema, definitions)


def relax_schema_for_vllm_tool_grammar(node: Any, *, key: Optional[str] = None) -> Any:
    """Remove JSON Schema constructs that vanilla vLLM tool grammars reject.

    This is intentionally applied only to generated tool `parameters`. The
    verifier still uses the strictified schema from `make_verification_schema`.
    """
    if key in {"enum", "const", "default", "examples"}:
        return deepcopy(node)

    if isinstance(node, bool):
        return {}

    if isinstance(node, list):
        return [relax_schema_for_vllm_tool_grammar(item, key=key) for item in node]

    if not isinstance(node, dict):
        return node

    relaxed = {}
    for child_key, child_value in node.items():
        if child_key in {"additionalProperties", "additionalItems"} and isinstance(child_value, bool):
            continue
        relaxed[child_key] = relax_schema_for_vllm_tool_grammar(child_value, key=child_key)
    return relaxed


def make_vllm_tool_schema(schema: Dict[str, Any], prefix: str = "") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    tool_schema, definitions = hoist_root_definitions(strictify_tool_schema(schema), prefix=prefix)
    return (
        relax_schema_for_vllm_tool_grammar(tool_schema),
        relax_schema_for_vllm_tool_grammar(definitions),
    )


def choose_tool_schema_mode(schema: Dict[str, Any], weights: Dict[str, float], rng: random.Random) -> str:
    mode = weighted_choice(weights, rng)
    if mode == "direct" and not is_object_schema(schema):
        wrapper_weights = {k: v for k, v in weights.items() if k != "direct" and v > 0}
        if not wrapper_weights:
            return "random_wrapper"
        return weighted_choice(wrapper_weights, rng)
    return mode


def choose_wrapper_key(mode: str, wrapper_key_pool: List[str], rng: random.Random) -> Optional[str]:
    if mode == "direct":
        return None
    if mode == "extraction_wrapper":
        return "extraction"
    return rng.choice(wrapper_key_pool)


def make_parameters(schema: Dict[str, Any], mode: str, payload_key: Optional[str]) -> Dict[str, Any]:
    tool_schema, definitions = make_vllm_tool_schema(schema)
    if mode == "direct":
        return attach_definitions(tool_schema, definitions)
    assert payload_key is not None
    parameters = {
        "type": "object",
        "properties": {
            payload_key: tool_schema,
        },
        "required": [payload_key],
    }
    return attach_definitions(parameters, definitions)


def make_union_parameters(
    branch_specs: List[Dict[str, Any]],
    composition_keyword: str,
    use_defs: bool,
) -> Dict[str, Any]:
    branches = []
    definitions: Dict[str, Any] = {}
    for i, spec in enumerate(branch_specs, start=1):
        payload_key = spec["payload_key"]
        branch_schema, branch_definitions = make_vllm_tool_schema(spec["schema"], prefix=f"choice_{i}_")
        definitions.update(branch_definitions)
        branches.append(
            {
                "type": "object",
                "properties": {
                    payload_key: branch_schema,
                },
                "required": [payload_key],
            }
        )

    if use_defs:
        defs = {f"choice_{i}": branch for i, branch in enumerate(branches, start=1)}
        parameters = {
            "type": "object",
            "$defs": defs,
            composition_keyword: [{"$ref": f"#/$defs/choice_{i}"} for i in range(1, len(branches) + 1)],
        }
        return attach_definitions(parameters, definitions)

    parameters = {
        "type": "object",
        composition_keyword: branches,
    }
    return attach_definitions(parameters, definitions)


def make_multi_key_parameters(branch_specs: List[Dict[str, Any]]) -> Dict[str, Any]:
    properties: Dict[str, Any] = {}
    definitions: Dict[str, Any] = {}
    for i, spec in enumerate(branch_specs, start=1):
        payload_key = spec["payload_key"]
        branch_schema, branch_definitions = make_vllm_tool_schema(spec["schema"], prefix=f"choice_{i}_")
        properties[payload_key] = branch_schema
        definitions.update(branch_definitions)
    return attach_definitions({"type": "object", "properties": properties}, definitions)


def unique_tool_name(base_name: str, used_names: set[str]) -> str:
    if base_name not in used_names:
        used_names.add(base_name)
        return base_name

    suffix = 2
    while f"{base_name}_{suffix}" in used_names:
        suffix += 1
    name = f"{base_name}_{suffix}"
    used_names.add(name)
    return name


def unique_payload_key(base_key: str, used_keys: set[str]) -> str:
    if base_key not in used_keys:
        used_keys.add(base_key)
        return base_key

    suffix = 2
    while f"{base_key}_{suffix}" in used_keys:
        suffix += 1
    key = f"{base_key}_{suffix}"
    used_keys.add(key)
    return key


def choose_semantic_tool_name(tool_name_pool: List[str], rng: random.Random) -> str:
    return rng.choice(tool_name_pool)


def choose_numbered_tool_names(prefix_pool: List[str], count: int, rng: random.Random) -> List[str]:
    prefix = rng.choice(prefix_pool)
    return [f"{prefix}_{i}" for i in range(1, count + 1)]


def make_tool(
    *,
    schema: Dict[str, Any],
    mode: str,
    payload_key: Optional[str],
    name: str,
    used_names: set[str],
    strict: bool,
    rng: random.Random,
) -> Dict[str, Any]:
    return {
        "type": "function",
        "name": unique_tool_name(name, used_names),
        "description": rng.choice(DESCRIPTION_TEMPLATES),
        "parameters": make_parameters(schema=schema, mode=mode, payload_key=payload_key),
        "strict": strict,
    }


def make_union_tool(
    *,
    branch_specs: List[Dict[str, Any]],
    name: str,
    composition_keyword: str,
    use_defs: bool,
    strict: bool,
    rng: random.Random,
) -> Dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": rng.choice(DESCRIPTION_TEMPLATES),
        "parameters": make_union_parameters(
            branch_specs=branch_specs,
            composition_keyword=composition_keyword,
            use_defs=use_defs,
        ),
        "strict": strict,
    }


def make_multi_key_tool(
    *,
    branch_specs: List[Dict[str, Any]],
    name: str,
    strict: bool,
    rng: random.Random,
) -> Dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": rng.choice(DESCRIPTION_TEMPLATES),
        "parameters": make_multi_key_parameters(branch_specs=branch_specs),
        "strict": strict,
    }


def sample_num_distractors(
    *,
    rng: random.Random,
    distractor_count_weights: Optional[Dict[int, float]],
    no_distractor_ratio: float,
    geometric_p: float,
    max_distractors: int,
    available_distractors: int,
) -> int:
    max_count = min(max_distractors, available_distractors)
    if max_count <= 0:
        return 0

    if distractor_count_weights:
        valid_weights = {
            count: weight
            for count, weight in distractor_count_weights.items()
            if 0 <= count <= max_count and weight > 0
        }
        if not valid_weights:
            return 0
        return int(weighted_choice({str(k): v for k, v in valid_weights.items()}, rng))

    if rng.random() < no_distractor_ratio:
        return 0

    weights = [(1.0 - geometric_p) ** (k - 1) * geometric_p for k in range(1, max_count + 1)]
    total = sum(weights)
    threshold = rng.random() * total
    cumulative = 0.0
    for k, weight in enumerate(weights, start=1):
        cumulative += weight
        if threshold <= cumulative:
            return k
    return max_count


def make_tool_spec(
    *,
    record: Dict[str, Any],
    is_target: bool,
    rng: random.Random,
    tool_schema_mode_weights: Dict[str, float],
    wrapper_key_pool: List[str],
) -> Dict[str, Any]:
    schema = record["_json_schema"]
    mode = choose_tool_schema_mode(schema, tool_schema_mode_weights, rng)
    return {
        "record": record,
        "schema": schema,
        "mode": mode,
        "payload_key": choose_wrapper_key(mode, wrapper_key_pool, rng),
        "is_target": is_target,
    }


def assign_union_payload_keys(
    tool_specs: List[Dict[str, Any]],
    union_payload_key_pool: List[str],
    rng: random.Random,
) -> None:
    used_keys: set[str] = set()
    for spec in tool_specs:
        spec["mode"] = "union_branch"
        spec["payload_key"] = unique_payload_key(rng.choice(union_payload_key_pool), used_keys)


def assign_multi_key_payload_keys(
    tool_specs: List[Dict[str, Any]],
    union_payload_key_pool: List[str],
    rng: random.Random,
) -> None:
    used_keys: set[str] = set()
    for spec in tool_specs:
        spec["mode"] = "multi_key_object"
        spec["payload_key"] = unique_payload_key(rng.choice(union_payload_key_pool), used_keys)


def choose_distractor_style(num_distractors: int, weights: Dict[str, float], rng: random.Random) -> str:
    if num_distractors <= 0:
        return "none"
    return weighted_choice(weights, rng)


def distractor_style_to_union(style: str) -> Tuple[Optional[str], bool]:
    if style == "single_tool_oneof":
        return "oneOf", False
    if style == "single_tool_anyof":
        return "anyOf", False
    if style == "single_tool_defs_oneof":
        return "oneOf", True
    if style == "single_tool_defs_anyof":
        return "anyOf", True
    return None, False


def format_document(document: str, rng: random.Random) -> str:
    header = rng.choice(DOCUMENT_HEADERS)
    if header == "<document>":
        return f"<document>\n{document}\n</document>"
    return f"{header}\n{document}"


def make_input_messages(document: str, rng: random.Random) -> Tuple[List[Dict[str, str]], str, str, str]:
    instruction_detail_level = rng.choice(list(INSTRUCTION_POOLS))
    instruction = rng.choice(INSTRUCTION_POOLS[instruction_detail_level])
    system_instruction_style = rng.choice(list(SYSTEM_INSTRUCTION_POOLS))
    system_instruction = rng.choice(SYSTEM_INSTRUCTION_POOLS[system_instruction_style])
    doc_block = format_document(document, rng)

    layouts = {
        "system_instruction_user_document": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": doc_block},
        ],
        "user_instruction_before_document": [
            {"role": "user", "content": f"{instruction}\n\n{doc_block}"},
        ],
        "user_instruction_after_document": [
            {"role": "user", "content": f"{doc_block}\n\n{instruction}"},
        ],
        "split_system_user_instruction": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"{instruction}\n\n{doc_block}"},
        ],
        "compact_user_message": [
            {"role": "user", "content": f"{instruction}\n\n{document}"},
        ],
    }
    layout = rng.choice(list(layouts))
    if not any(msg.get("role") == "system" for msg in layouts[layout]):
        system_instruction_style = "none"
    return layouts[layout], layout, instruction_detail_level, system_instruction_style


def make_gym_record(
    *,
    record: Dict[str, Any],
    all_records: List[Dict[str, Any]],
    rng: random.Random,
    tool_schema_mode_weights: Dict[str, float],
    distractor_style_weights: Dict[str, float],
    tool_name_style_weights: Dict[str, float],
    wrapper_key_pool: List[str],
    union_payload_key_pool: List[str],
    tool_name_pool: List[str],
    numbered_tool_prefix_pool: List[str],
    tool_choice: str,
    parallel_tool_calls: bool,
    tool_strict: bool,
    distractor_count_weights: Optional[Dict[int, float]],
    no_distractor_ratio: float,
    distractor_geometric_p: float,
    max_distractors: int,
) -> Dict[str, Any]:
    schema = record["_json_schema"]
    source_schema_type = record.get("target_output_format", "json")
    input_msgs, instruction_layout, instruction_detail_level, system_instruction_style = make_input_messages(
        record.get("document", ""), rng
    )

    distractor_pool = [r for r in all_records if r["_source_index"] != record["_source_index"]]
    num_distractors = sample_num_distractors(
        rng=rng,
        distractor_count_weights=distractor_count_weights,
        no_distractor_ratio=no_distractor_ratio,
        geometric_p=distractor_geometric_p,
        max_distractors=max_distractors,
        available_distractors=len(distractor_pool),
    )
    distractor_records = rng.sample(distractor_pool, num_distractors) if num_distractors else []
    distractor_style = choose_distractor_style(num_distractors, distractor_style_weights, rng)
    tool_name_style = weighted_choice(tool_name_style_weights, rng)

    tool_specs = [
        make_tool_spec(
            record=record,
            is_target=True,
            rng=rng,
            tool_schema_mode_weights=tool_schema_mode_weights,
            wrapper_key_pool=wrapper_key_pool,
        )
    ]
    for distractor in distractor_records:
        tool_specs.append(
            make_tool_spec(
                record=distractor,
                is_target=False,
                rng=rng,
                tool_schema_mode_weights=tool_schema_mode_weights,
                wrapper_key_pool=wrapper_key_pool,
            )
        )

    rng.shuffle(tool_specs)
    composition_keyword, use_defs = distractor_style_to_union(distractor_style)
    tool_union_mode = None
    if composition_keyword is not None:
        assign_union_payload_keys(tool_specs, union_payload_key_pool, rng)
        tool_union_mode = f"{'$defs_' if use_defs else ''}{composition_keyword}"
        if tool_name_style == "numbered":
            tool_name = choose_numbered_tool_names(numbered_tool_prefix_pool, 1, rng)[0]
        else:
            tool_name = choose_semantic_tool_name(tool_name_pool, rng)
        tools = [
            make_union_tool(
                branch_specs=tool_specs,
                name=tool_name,
                composition_keyword=composition_keyword,
                use_defs=use_defs,
                strict=tool_strict,
                rng=rng,
            )
        ]
        target_spec = next(spec for spec in tool_specs if spec["is_target"])
        target_tool_name = tool_name
    elif distractor_style == "single_tool_multi_key":
        assign_multi_key_payload_keys(tool_specs, union_payload_key_pool, rng)
        if tool_name_style == "numbered":
            tool_name = choose_numbered_tool_names(numbered_tool_prefix_pool, 1, rng)[0]
        else:
            tool_name = choose_semantic_tool_name(tool_name_pool, rng)
        tools = [
            make_multi_key_tool(
                branch_specs=tool_specs,
                name=tool_name,
                strict=tool_strict,
                rng=rng,
            )
        ]
        target_spec = next(spec for spec in tool_specs if spec["is_target"])
        target_tool_name = tool_name
    else:
        if distractor_style == "numbered_tools":
            tool_names = choose_numbered_tool_names(numbered_tool_prefix_pool, len(tool_specs), rng)
            tool_name_style = "numbered"
        elif tool_name_style == "numbered":
            tool_names = choose_numbered_tool_names(numbered_tool_prefix_pool, len(tool_specs), rng)
        else:
            tool_names = [choose_semantic_tool_name(tool_name_pool, rng) for _ in tool_specs]

        used_tool_names: set[str] = set()
        tools = []
        target_spec = None
        target_tool_name = None
        for spec, tool_name in zip(tool_specs, tool_names):
            tool = make_tool(
                schema=spec["schema"],
                mode=spec["mode"],
                payload_key=spec["payload_key"],
                name=tool_name,
                used_names=used_tool_names,
                strict=tool_strict,
                rng=rng,
            )
            tools.append(tool)
            if spec["is_target"]:
                target_spec = spec
                target_tool_name = tool["name"]

        assert target_spec is not None
        assert target_tool_name is not None

    num_user_turns = sum(1 for msg in input_msgs if msg.get("role") == "user")
    verification_schema = make_verification_schema(schema)

    return {
        "responses_create_params": {
            "input": input_msgs,
            "tools": tools,
            "tool_choice": tool_choice,
            "parallel_tool_calls": parallel_tool_calls,
        },
        "schema_str": json.dumps(verification_schema, ensure_ascii=False),
        "schema_type": "json",
        "response_mode": "tool_call",
        "problem_type": "direct_tool_call",
        "schema_repr": "tool",
        "source_format": source_schema_type,
        "source_schema_type": source_schema_type,
        "tool_choice": tool_choice,
        "parallel_tool_calls": parallel_tool_calls,
        "tool_name": target_tool_name,
        "tool_schema_mode": target_spec["mode"],
        "tool_payload_key": target_spec["payload_key"],
        "tool_name_style": tool_name_style,
        "distractor_style": distractor_style,
        "tool_union_mode": tool_union_mode,
        "num_turns": num_user_turns,
        "num_tools": len(tools),
        "num_distractors": num_distractors,
        "has_distractors": num_distractors > 0,
        "instruction_layout": instruction_layout,
        "instruction_detail_level": instruction_detail_level,
        "system_instruction_style": system_instruction_style,
        "source_record_id": record.get("_record_id", "unknown"),
        "agent_ref": {
            "type": "responses_api_agents",
            "name": "structured_outputs_v4_simple_agent",
        },
    }


def generate_records(
    *,
    records: List[Dict[str, Any]],
    rng: random.Random,
    samples_per_record: int,
    max_total: Optional[int],
    tool_schema_mode_weights: Dict[str, float],
    distractor_style_weights: Dict[str, float],
    tool_name_style_weights: Dict[str, float],
    wrapper_key_pool: List[str],
    union_payload_key_pool: List[str],
    tool_name_pool: List[str],
    numbered_tool_prefix_pool: List[str],
    tool_choice: str,
    parallel_tool_calls: Optional[bool],
    parallel_tool_calls_true_ratio: float,
    tool_strict: bool,
    distractor_count_weights: Optional[Dict[int, float]],
    no_distractor_ratio: float,
    distractor_geometric_p: float,
    max_distractors: int,
) -> List[Dict[str, Any]]:
    output = []
    shuffled_records = list(records)
    rng.shuffle(shuffled_records)
    for record in shuffled_records:
        for _ in range(samples_per_record):
            if max_total is not None and len(output) >= max_total:
                return output
            sampled_parallel_tool_calls = (
                parallel_tool_calls
                if parallel_tool_calls is not None
                else rng.random() < parallel_tool_calls_true_ratio
            )
            output.append(
                make_gym_record(
                    record=record,
                    all_records=records,
                    rng=rng,
                    tool_schema_mode_weights=tool_schema_mode_weights,
                    distractor_style_weights=distractor_style_weights,
                    tool_name_style_weights=tool_name_style_weights,
                    wrapper_key_pool=wrapper_key_pool,
                    union_payload_key_pool=union_payload_key_pool,
                    tool_name_pool=tool_name_pool,
                    numbered_tool_prefix_pool=numbered_tool_prefix_pool,
                    tool_choice=tool_choice,
                    parallel_tool_calls=sampled_parallel_tool_calls,
                    tool_strict=tool_strict,
                    distractor_count_weights=distractor_count_weights,
                    no_distractor_ratio=no_distractor_ratio,
                    distractor_geometric_p=distractor_geometric_p,
                    max_distractors=max_distractors,
                )
            )
    return output


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.no_distractor_ratio <= 1.0:
        raise ValueError("--no-distractor-ratio must be in [0, 1]")
    if not 0.0 < args.distractor_geometric_p <= 1.0:
        raise ValueError("--distractor-geometric-p must be in (0, 1]")
    if args.max_distractors < 0:
        raise ValueError("--max-distractors must be non-negative")
    if args.samples_per_record <= 0:
        raise ValueError("--samples-per-record must be positive")
    if args.max_total is not None and args.max_total <= 0:
        raise ValueError("--max-total must be positive")
    if not 0.0 <= args.parallel_tool_calls_true_ratio <= 1.0:
        raise ValueError("--parallel-tool-calls-true-ratio must be in [0, 1]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate tool-call structured outputs data")
    parser.add_argument("-i", "--input", required=True, help="Path to verified structured outputs JSONL")
    parser.add_argument("-o", "--output", required=True, help="Path to write Gym-ready JSONL")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-total", type=int, default=None, help="Maximum rows to emit. Omit to use all loaded rows."
    )
    parser.add_argument("--samples-per-record", type=int, default=1)
    parser.add_argument("--source-formats", default=DEFAULT_SOURCE_FORMATS)
    parser.add_argument("--tool-schema-mode-weights", default=DEFAULT_TOOL_SCHEMA_MODE_WEIGHTS)
    parser.add_argument("--distractor-style-weights", default=DEFAULT_DISTRACTOR_STYLE_WEIGHTS)
    parser.add_argument("--tool-name-style-weights", default=DEFAULT_TOOL_NAME_STYLE_WEIGHTS)
    parser.add_argument("--wrapper-key-pool", default=DEFAULT_WRAPPER_KEY_POOL)
    parser.add_argument("--union-payload-key-pool", default=DEFAULT_UNION_PAYLOAD_KEY_POOL)
    parser.add_argument("--tool-name-pool", default=DEFAULT_TOOL_NAME_POOL)
    parser.add_argument("--numbered-tool-prefix-pool", default=DEFAULT_NUMBERED_TOOL_PREFIX_POOL)
    parser.add_argument("--tool-choice", choices=["required", "auto"], default="auto")
    parser.add_argument(
        "--parallel-tool-calls",
        type=parse_bool,
        default=None,
        help="Force parallel_tool_calls for all rows. Omit to sample per row.",
    )
    parser.add_argument(
        "--parallel-tool-calls-true-ratio",
        type=float,
        default=DEFAULT_PARALLEL_TOOL_CALLS_TRUE_RATIO,
        help="Probability of parallel_tool_calls=true when --parallel-tool-calls is omitted.",
    )
    parser.add_argument("--tool-strict", type=parse_bool, default=True)
    parser.add_argument("--max-distractors", type=int, default=20)
    parser.add_argument(
        "--distractor-count-weights",
        default=DEFAULT_DISTRACTOR_COUNT_WEIGHTS,
        help="Comma-separated explicit distractor count weights. Empty string falls back to geometric sampling.",
    )
    parser.add_argument("--no-distractor-ratio", type=float, default=0.30)
    parser.add_argument("--distractor-geometric-p", type=float, default=0.25)
    parser.add_argument("--exclude-substrings", default="", help="Comma-separated case-insensitive row filters")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    rng = random.Random(args.seed)

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found")
        sys.exit(1)

    source_formats = parse_csv_arg(args.source_formats)
    tool_schema_mode_weights = parse_weights(args.tool_schema_mode_weights, TOOL_SCHEMA_MODES, "tool schema mode")
    distractor_style_weights = parse_weights(args.distractor_style_weights, DISTRACTOR_STYLES, "distractor style")
    tool_name_style_weights = parse_weights(args.tool_name_style_weights, TOOL_NAME_STYLES, "tool name style")
    wrapper_key_pool = parse_csv_arg(args.wrapper_key_pool)
    union_payload_key_pool = parse_csv_arg(args.union_payload_key_pool)
    tool_name_pool = parse_csv_arg(args.tool_name_pool)
    numbered_tool_prefix_pool = parse_csv_arg(args.numbered_tool_prefix_pool)
    exclude_substrings = parse_csv_arg(args.exclude_substrings)
    distractor_count_weights = parse_int_weights(args.distractor_count_weights, "distractor count")

    print(f"Loading records from {input_path}...")
    records = load_records(input_path, source_formats=source_formats, exclude_substrings=exclude_substrings)
    print(f"  Loaded {len(records)} records")
    if not records:
        print("No records loaded. Exiting.")
        sys.exit(1)

    print("Generating tool-call direct samples...")
    samples = generate_records(
        records=records,
        rng=rng,
        samples_per_record=args.samples_per_record,
        max_total=args.max_total,
        tool_schema_mode_weights=tool_schema_mode_weights,
        distractor_style_weights=distractor_style_weights,
        tool_name_style_weights=tool_name_style_weights,
        wrapper_key_pool=wrapper_key_pool,
        union_payload_key_pool=union_payload_key_pool,
        tool_name_pool=tool_name_pool,
        numbered_tool_prefix_pool=numbered_tool_prefix_pool,
        tool_choice=args.tool_choice,
        parallel_tool_calls=args.parallel_tool_calls,
        parallel_tool_calls_true_ratio=args.parallel_tool_calls_true_ratio,
        tool_strict=args.tool_strict,
        distractor_count_weights=distractor_count_weights,
        no_distractor_ratio=args.no_distractor_ratio,
        distractor_geometric_p=args.distractor_geometric_p,
        max_distractors=args.max_distractors,
    )
    rng.shuffle(samples)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 60}")
    print("  Structured Outputs v4 Tool-Call SDG Complete")
    print(f"{'=' * 60}")
    print(f"  Source records:  {len(records)}")
    print(f"  Total generated: {len(samples)}")
    print(f"  Output: {output_path}")

    for title, key in [
        ("By tool_schema_mode", "tool_schema_mode"),
        ("By distractor_style", "distractor_style"),
        ("By tool_name_style", "tool_name_style"),
        ("By tool_union_mode", "tool_union_mode"),
        ("By num_distractors", "num_distractors"),
        ("By instruction_layout", "instruction_layout"),
        ("By instruction_detail_level", "instruction_detail_level"),
        ("By system_instruction_style", "system_instruction_style"),
    ]:
        print(f"\n  {title}:")
        counts = Counter(sample.get(key, "?") for sample in samples)
        for value in sorted(counts, key=lambda x: (isinstance(x, str), x)):
            print(f"    {value}: {counts[value]}")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
