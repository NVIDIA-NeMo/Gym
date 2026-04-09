#!/usr/bin/env python3
"""Convert a ds1-style verified jsonl to Gym-ready format for structured_outputs.

Transforms:
  - messages (strip assistant) -> responses_create_params.input
  - structured_schema (any format) -> schema_str (JSON string)
  - target_output_format -> schema_type

Usage:
  python misc/prepare_data.py --input /path/to/ds1_verified.jsonl --output data/ds1_train.jsonl
"""

import argparse
import json
import sys
import tomllib
from pathlib import Path

import xmltodict
import yaml


def parse_schema_to_dict(schema_str: str, fmt: str) -> dict:
    """Parse structured_schema string and return the inner JSON Schema dict."""
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
        return convert_csv_schema(schema_str)
    else:
        raise ValueError(f"Unknown format: {fmt}")

    if isinstance(obj, dict) and "schema" in obj:
        return obj["schema"]
    return obj


def convert_csv_schema(schema_str: str) -> dict:
    """Convert a 2-line CSV schema (headers + types) to a JSON Schema array-of-objects."""
    lines = schema_str.strip().split("\n")
    headers = [h.strip() for h in lines[0].split(",")]
    types = [t.strip() for t in lines[1].split(",")] if len(lines) > 1 else ["string"] * len(headers)

    properties = {}
    for name, typ in zip(headers, types):
        properties[name] = {"type": typ if typ in ("string", "integer", "number", "boolean") else "string"}

    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": properties,
            "required": headers,
            "additionalProperties": False,
        },
    }


def convert_record(record: dict) -> dict:
    fmt = record.get("target_output_format", "json")
    schema_str = record.get("structured_schema", "")

    schema_dict = parse_schema_to_dict(schema_str, fmt)
    json_schema_str = json.dumps(schema_dict, ensure_ascii=False)

    input_msgs = [m for m in record.get("messages", []) if m.get("role") != "assistant"]
    if record.get("system_prompt"):
        has_system = any(m.get("role") == "system" for m in input_msgs)
        if not has_system:
            input_msgs.insert(0, {"role": "system", "content": record["system_prompt"]})

    return {
        "responses_create_params": {"input": input_msgs},
        "schema_str": json_schema_str,
        "schema_type": fmt,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert verified ds1 jsonl to Gym-ready format")
    parser.add_argument("--input", "-i", required=True, help="Path to ds1_verified.jsonl")
    parser.add_argument("--output", "-o", required=True, help="Path to write Gym-ready jsonl")
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)

    if not src.exists():
        print(f"ERROR: {src} not found")
        sys.exit(1)

    dst.parent.mkdir(parents=True, exist_ok=True)

    total, written, skipped = 0, 0, 0
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            record = json.loads(line)
            try:
                gym_record = convert_record(record)
                fout.write(json.dumps(gym_record, ensure_ascii=False) + "\n")
                written += 1
            except Exception as e:
                rid = record.get("metadata", {}).get("record_id", f"line_{total}")
                print(f"  SKIP  {rid}: {e}")
                skipped += 1

    print(f"{written}/{total} written to {dst}  ({skipped} skipped)")


if __name__ == "__main__":
    main()
