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
import yaml
from datasets import Dataset, concatenate_datasets, load_dataset


STRUCTURED_OUTPUT_INSTRUCTIONS = [
    "Response Formatting Schema (YAML): {schema}",
    "Format your response as valid YAML matching the provided schema: {schema}",
    "Structure your response according to the following schema specification: {schema}. Return only the YAML output.",
    """Your aim is to process the given unstructured input data and return the output based on the Response format schema provided. Provide only the raw output data based on the given response_format. All values for attributes should be properly formatted, and never give incomplete responses. Remember, your responses MUST be valid parsable YAML and MUST match the schema specified in response_format. Do not give any introduction in the front.
Response format: {schema}""",
    """Format your response as a YAML document adhering to:
- Schema structure: {schema}
- Validation rules:
  * All data types are verified
  * All strings must be properly quoted where necessary
  * There are no unnecessary fields added
  * Must pass schema validation
Ensure compliance with all specifications before responding.""",
    """Create a structured YAML response that:
1. Implements proper data typing
2. Includes all required fields
3. Handles special characters appropriately
4. Uses proper YAML formatting with correct indentation
5. Validates against schema constraints
6. Provides appropriate list formatting
7. Uses consistent formatting
8. Maintains proper nesting levels
9. Is grounded in the provided dialog
10. Strictly follows the provided schema: {schema}""",
    "Response Format (YAML): {schema}",
    "I'd like you to format your response as a YAML document matching the provided schema: {schema}",
    "Structure your response according to the following schema specification: {schema}. Validate that your output conforms to all schema constraints and required properties. Return only the YAML output without styling it in backticks.",
    """Your aim is to process the given unstructured input data and return the output based on the instructions and the response_format schema provided. Provide only the raw output data in valid YAML format based on the given response_format. Never give incomplete responses. Remember, your responses MUST be valid parsable YAML and MUST match the schema specified in response_format. Do not give any introduction in the front. Your response should ONLY contain the YAML
Response format: {schema}""",
    """Format your response as a YAML document adhering to:
- Schema structure: {schema}
- Validation rules:
  * All strings must be properly quoted where necessary
  * All data types are verified
  * There are no unnecessary fields added
  * Must pass schema validation
  * Must not be in Markdown format: i.e. not in ```yaml``` format.
Ensure compliance with all specifications before responding.""",
    """Create a structured YAML response that:
1. Implements proper data typing
2. Handles special characters appropriately
3. Includes all required fields
4. Maintains proper nesting and indentation levels
5. Provides appropriate list formatting
6. Validates against schema constraints
7. Uses consistent formatting
8. Uses proper YAML syntax
9. Is grounded in the provided dialog
10. Strictly follows the provided schema: {schema}""",
]

USER_QUERY_INSTRUCTIONS = [
    "Generate a YAML output that strictly adheres to the specified schema based on the document provided.",
    "Format the document based on the provided schema.",
    "Fit the document to the given format.",
    "Extract the information from the text and format it as YAML matching this schema.",
    "Map the content of this document to the provided data structure.",
    "Parse the document and populate the following data model.",
    "Please provide the answer in YAML format that conforms to the specified structure.",
    "Convert the unstructured text into the specified structured format.",
    "Ensure your output validates against the given schema.",
    "Restructure the provided information according to the following template.",
    "\U0001f50d Read the document carefully and produce a structured YAML output matching the schema.",
]

DOCUMENT_TEMPLATES = [
    "{user_message}\n\nDocument:\n{document}",
    "{user_message}\n\n{document}",
    "# Problem:\n{user_message}\n\n{document}",
    "# Instructions:\n{user_message}\n\n# Document:\n{document}",
    "# Document:\n{document}\n\n# Instructions: {user_message}",
    "# Information\n{document}\n\n# Problem: {user_message}",
    "\U0001f4c4 Document:\n{document}\n\n\U0001f4dd Task: {user_message}",
    "Given the following text:\n\n{document}\n\n{user_message}",
]


def template_yaml_schema(input_schema: Dict):
    variant = random.randint(0, 3)
    if variant == 0:
        schema = deepcopy(input_schema)["properties"]
    else:
        schema = input_schema

    serialization = random.randint(0, 2)
    if serialization == 0:
        return yaml.dump(schema, default_flow_style=False)
    elif serialization == 1:
        return schema
    else:
        return json.dumps(schema)


def template_document(user_message, document):
    return random.choice(DOCUMENT_TEMPLATES).format(user_message=user_message, document=document)


def template_messages(system_message, user_message):
    layouts = [
        [{"role": "user", "content": system_message}, {"role": "user", "content": user_message}],
        [{"role": "user", "content": f"{system_message}\n{user_message}"}],
        [{"role": "user", "content": f"{user_message}\n{system_message}"}],
        [{"role": "user", "content": user_message}, {"role": "user", "content": system_message}],
        [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}],
    ]
    return random.choice(layouts)


def template_sample(schema: Dict, document: str):
    if "$schema" in schema:
        schema.pop("$schema")
    templated_schema = template_yaml_schema(schema)

    system_message = random.choice(STRUCTURED_OUTPUT_INSTRUCTIONS)
    system_message = system_message.format(schema=templated_schema)

    user_message = random.choice(USER_QUERY_INSTRUCTIONS)
    user_message = template_document(user_message, document)

    return template_messages(system_message, user_message)


def process_sample(sample):
    try:
        schema = json.loads(sample["json_schema"])
        messages = template_sample(schema, sample["document"])

        sample["responses_create_params"] = {"input": messages}
        sample["schema_str"] = json.dumps(schema)
        sample["schema_type"] = "yaml"
    except Exception:
        sample["responses_create_params"] = None
        sample["schema_str"] = None
        sample["schema_type"] = None

    return sample


NUM_VAL = 128


def load_dataset_split():
    hf_token = os.environ.get("HF_PAT_NVIDIA")
    ds_1 = load_dataset("nvidia/structured-dataset-nanov3", split="train", token=hf_token)
    ds_2 = load_dataset("nvidia/structured-dataset-nanov3-reasoning", split="train", token=hf_token)
    ds = concatenate_datasets([ds_1, ds_2])

    df = pd.DataFrame(ds)
    df = df.drop_duplicates(subset=["json_schema"])
    ds = Dataset.from_pandas(df)
    return ds


def _process_split(ds):
    ds = ds.map(process_sample)
    ds = ds.filter(lambda x: x["responses_create_params"] is not None)
    return ds.select_columns(["responses_create_params", "schema_str", "schema_type", "schema_fields_count"])


def main():
    ds = load_dataset_split()
    print(f"Initial ds len={len(ds)}")

    ds_val_raw = ds.select(range(NUM_VAL))
    ds_train_raw = ds.select(range(NUM_VAL, len(ds)))

    ds_val = _process_split(ds_val_raw)
    ds_train = _process_split(ds_train_raw)

    local_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(local_dir, "data"), exist_ok=True)

    ds_train.to_json(os.path.join(local_dir, "data", "260309_nano_v3_sdg_structured_outputs_yaml_train.jsonl"))
    ds_val.to_json(os.path.join(local_dir, "data", "260309_nano_v3_sdg_structured_outputs_yaml_val.jsonl"))


if __name__ == "__main__":
    main()
