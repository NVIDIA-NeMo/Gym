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
import json
import os
import random
from copy import deepcopy
from typing import Dict

import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset


STRUCTURED_OUTPUT_INSTRUCTIONS = [
    """Response Formatting Schema: {schema}""",
    """Format your response as an object matching the provided JSON schema: {schema}""",
    """Structure your response according to the following JSON schema specification: {schema}. Return only the JSON output.""",
    """Your aim is to process the given unstructured input data and return the output based on the Response format schema provided. Provide only the raw output data based on the given response_format. All values for attributes should be in quotes, and never give incomplete responses. Remember, your responses MUST be valid parsable JSON and MUST match the schema specified in response_format. Do not give any introduction in the front.
Response format: {schema}""",
    """Format your response as a JSON object adhering to:
- Schema structure: {schema}
- Validation rules:
  * All data types are verified
  * All strings must be properly escaped
  * There are no unnecessary fields added
  * Must pass JSON schema validation
Ensure compliance with all specifications before responding.""",
    """Create a structured JSON response that:
1. Implements proper data typing
2. Includes all required fields
3. Handles special characters appropriately
4. Is unindented JSON format
5. Validates against schema constraints
6. Provides appropriate array formatting
7. Uses consistent formatting and escaping
8. Maintains proper nesting levels
9. Is grounded in the provided dialog
10. Strictly follows the provided schema: {schema}""",
    """Response Format: {schema}""",
    """I'd like you to format your response as a JSON object matching the provided schema: {schema}""",
    """Structure your response according to the following JSON schema specification: {schema}. Validate that your output conforms to all schema constraints and required properties. Return only the JSON output without styling it in backticks.""",
    """Your aim is to process the given unstructured input data and return the output based on the instructions and the response_format schema provided. Provide only the raw output data in valid JSON format based on the given response_format. All values for JSON attributes should be on quotes and never give incomplete responses. Remember, your responses MUST be valid parsable JSON and MUST match the schema specified in response_format. Do not give any introduction in the front. Your response should ONLY contain the JSON
Response format: {schema}""",
    """Format your response as a JSON object adhering to:
- Schema structure: {schema}
- Validation rules:
  * All strings must be properly escaped
  * All data types are verified
  * There are no unnecessary fields added
  * Must pass JSON schema validation
  * Must not be in Markdown format: i.e. not in ```json``` format.
Ensure compliance with all specifications before responding.""",
    """Create a structured JSON response that:
1. Implements proper data typing
2. Handles special characters appropriately
3. Includes all required fields
4. Maintains proper nesting levels
5. Provides appropriate array formatting
6. Validates against schema constraints
7. Uses consistent formatting and escaping
8. Is unindented JSON format
9. Is grounded in the provided dialog
10. Strictly follows the provided schema: {schema}""",
]

USER_QUERY_INSTRUCTIONS = [
    """Generate a JSON output that strictly adheres to the specified schema based on the document provided.""",
    """Format the document based on the provided schema.""",
    """Fit the document to the given format.""",
    """Extract the information from the text and format it as a JSON object matching this schema.""",
    """Map the content of this document to the provided data structure.""",
    """Parse the document and populate the following data model.""",
    """Please provide the answer in a JSON format that conforms to the specified structure.""",
    """Convert the unstructured text into the specified structured format.""",
    """Ensure your output validates against the given JSON schema.""",
    """Restructure the provided information according to the following template.""",
]


def template_json_schema(input_schema: Dict):
    chance = random.randint(0, 4)
    if chance == 0:
        schema = deepcopy(input_schema)
        schema = {"type": "json_schema", "json_schema": {"name": "scene_description", "schema": schema}}
    elif chance == 1:
        schema = deepcopy(input_schema)
        schema = schema["properties"]
    else:
        schema = input_schema

    chance = random.randint(0, 3)
    if chance == 0:
        schema = schema
    else:
        schema = json.dumps(schema)

    return schema


def template_document(user_message, document):
    chance = random.randint(0, 5)
    if chance == 0:
        content = f"{user_message}\n\nDocument:\n{document}"
    elif chance == 1:
        content = f"{user_message}\n\n{document}"
    elif chance == 2:
        content = f"# Problem:\n{user_message}\n\n{document}"
    elif chance == 3:
        content = f"# Instrutions:\n{user_message}\n\n# Document:\n{document}"
    elif chance == 4:
        content = f"# Document:\n{document}\n\n# Instructions: {user_message}"
    else:
        content = f"# Information\n{document}\n\n# Problem: {user_message}"
    return content


def template_messages(system_message, user_message):
    chance = random.randint(0, 3)
    if chance == 0:
        messages = [{"role": "user", "content": system_message}, {"role": "user", "content": user_message}]
    elif chance == 1:
        messages = [{"role": "user", "content": f"{system_message}\n{user_message}"}]
    elif chance == 2:
        messages = [{"role": "user", "content": f"{user_message}\n{system_message}"}]
    else:
        messages = [{"role": "user", "content": user_message}, {"role": "user", "content": system_message}]
    return messages


def template_sample(schema: Dict, document: str):
    if "$schema" in schema:
        schema.pop("$schema")
    templated_schema = template_json_schema(schema)

    system_message = random.choice(STRUCTURED_OUTPUT_INSTRUCTIONS)
    system_message = system_message.format(schema=templated_schema)

    user_message = random.choice(USER_QUERY_INSTRUCTIONS)
    user_message = template_document(user_message, document)

    messages = template_messages(system_message, user_message)
    return messages


def process_sample(sample):
    try:
        # schema_fields_count = sample["schema_fields_count"]
        json_schema = sample["json_schema"]
        document = sample["document"]

        schema = json.loads(json_schema)
        messages = template_sample(schema, document)

        responses_create_params = {"input": messages}
        sample["responses_create_params"] = responses_create_params
        sample["schema_str"] = json.dumps(schema)
        sample["schema_type"] = "json"
    except Exception:
        sample["responses_create_params"] = None
        sample["schema_str"] = None
        sample["schema_type"] = None

    return sample


def main():
    hf_token = os.environ.get("HF_PAT_NVIDIA")

    ds_1 = load_dataset("nvidia/structured-dataset-nanov3", split="train", token=hf_token)
    ds_2 = load_dataset("nvidia/structured-dataset-nanov3-reasoning", split="train", token=hf_token)
    ds = concatenate_datasets([ds_1, ds_2])

    print(f"Initial ds len={len(ds)}")
    # ds = ds.filter(lambda x: x["output_format"] == "json")
    # print(f"JSON ds len={len(ds)}")
    # ds = ds.filter(lambda x: eval(x["schema_fields_count"]) > 4)
    # print(f"Non-Easy ds len={len(ds)}")

    df = pd.DataFrame(ds)
    df = df.drop_duplicates(subset=["json_schema"])
    # df = df.drop_duplicates(subset=["document"])
    ds = Dataset.from_pandas(df)

    ds = ds.shuffle(1)

    # ds = ds.filter(lambda x: x["output_format"] == "json")
    # ds = ds.filter(lambda x: x["schema_fields_count"] > 4)

    num_processes = os.cpu_count() * 2
    ds = ds.map(process_sample, num_proc=num_processes)
    ds = ds.filter(lambda x: x["responses_create_params"] is not None, num_proc=num_processes)

    ds = ds.select_columns(["responses_create_params", "schema_str", "schema_type", "schema_fields_count"])

    local_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(local_dir, "data"), exist_ok=True)
    ds_train = ds.select(range(64, len(ds)))
    ds_train.to_json(os.path.join(local_dir, "data", "251027_nano_v3_sdg_structured_outputs_json_train.jsonl"))

    ds_val = ds.select(range(64))
    ds_val = concatenate_datasets([ds_val for _ in range(8)])  # average over 8
    ds_val.to_json(os.path.join(local_dir, "data", "251027_nano_v3_sdg_structured_outputs_json_val.jsonl"))

    # ddict = DatasetDict({"train": ds_train, "validation": ds_val})
    # ddict.push_to_hub("", private=True)

    # import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    main()
