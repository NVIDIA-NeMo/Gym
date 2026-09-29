# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# This file is replicated from the CVDP benchmark repository.

import json
import logging
import re


logging.basicConfig(level=logging.INFO)


class ModelHelpers:
    """
    Helper class for model interaction functions.

    This class contains utility functions for creating system prompts,
    determining schemas, and parsing model responses that were previously
    part of the DatasetProcessor class.
    """

    def __init__(self, folders=None, schema=None):
        """
        Initialize the ModelHelpers class.

        Args:
            folders: Base context for system prompts
            schema: Default schema to use for responses
        """
        self.folders = (
            folders
            or """You are a helpful assistance.
Consider that you have a folder structure like the following:\n
    - rtl/*   : Contains files which are RTL code.
    - verif/* : Contains files which are used to verify the correctness of the RTL code.
    - docs/*  : Contains files used to document the project, like Block Guides, RTL Plans and Verification Plans.

When generating files, return the file name in the correct place at the folder structure.
"""
        )

        self.schema = schema or ['{ "code": [{ "<name>" : "<code>"}] }', '{ "response": "<response>" }']

    def create_system_prompt(self, base_context=None, schema=None, category=None):
        """
        Create a system prompt for the model.

        Args:
            base_context: Base context to use (defaults to self.folders)
            schema: Optional JSON schema for structured output
            category: Optional integer indicating the category/problem ID

        Returns:
            The formatted system prompt
        """
        system_prompt = base_context if base_context is not None else self.folders

        # Add category information if provided
        # Define category-specific guidance messages
        self.category_guidance = {
            2: "You are solving an 'RTL Code Completion' problem. To solve this problem correctly, you should only respond with the RTL code generated according to the requirements.",
            3: "You are solving a 'Specification to RTL Translation' problem. To solve this problem correctly, you should only respond with the RTL code translated from the specification.",
            4: "You are solving an 'RTL Code Modification' problem. To solve this problem correctly, you should only respond with the modified RTL code according to the requirements.",
            5: "You are solving a 'Specification to RTL Translation: Module Instantiation and Component Reuse' problem. To solve this problem correctly, you should only respond with the RTL code translated from the specification and with proper module instantiation and component reuse.",
            6: "You are solving an 'RTL Correspondence' problem. To solve this problem correctly, you should only respond with a verbatim quote from the specification (if the context is RTL) that corresponds to the RTL code snippet or verbatim RTL source code (if the context is specification) that corresponds to the specification snippet.",
            7: "You are solving an 'RTL Lint Improvement or Power-Performance Optimization' problem. To solve this problem correctly, you should only respond with improved RTL code to address lint issues or optimize for power/performance.",
            8: "You are solving a 'Testbench Correspondence' problem. To solve this problem correctly, you should only respond with the verbatim testbench code that corresponds to the test plan snippet (if the context is testbench code) or verbatim test plan that corresponds to the testbench code snippet (if the context is testbench code).",
            9: "You are solving a 'Question & Answer on RTL' problem. To solve this problem correctly, you should only respond with a detailed answer to the question about RTL.",
            10: "You are solving a 'Question & Answer on Testbench' problem. To solve this problem correctly, you should only respond with a detailed answer to the question about the testbench.",
            12: "You are solving a 'Test Plan to Testbench Stimulus Generation' problem. To solve this problem correctly, you should only respond with the testbench stimulus code generated based on the test plan specification.",
            13: "You are solving a 'Test Plan to Testbench Checker Generation' problem. To solve this problem correctly, you should only respond with the testbench checker code generated based on the test plan specification.",
            14: "You are solving a 'Test Plan to Assertions Generation' problem. To solve this problem correctly, you should only respond with the assertions for the testbench based on the test plan specification.",
            16: "You are solving an 'RTL Debugging and Bug Fixing' problem. To solve this problem correctly, you should only respond with the RTL code that is debugged and fixed to address the bug.",
        }
        if category is not None and category in self.category_guidance:
            system_prompt += f"\n{self.category_guidance[category]}\n"
        else:
            assert False, f"Category {category} is not a valid category"

        # Add schema instruction if provided
        if schema is not None:
            if isinstance(schema, list):
                system_prompt += "\nProvide the response in one of the following JSON schemas: \n"
                schemas = []
                for sch in schema:
                    schemas.append(f"{sch}")

                system_prompt += "\nor\n".join(schemas)
            else:
                system_prompt += f"\nProvide the response in the following JSON schema: {schema}"
            system_prompt += "\nThe response should be in JSON format, including double-quotes around keys and values, and proper escaping of quotes within values, and escaping of newlines."

        return system_prompt

    def determine_schema(self, files):
        """
        Determine schema based on the number of expected files.

        Args:
            files: List of expected output files

        Returns:
            Tuple of (schema, no_schema flag)
        """
        no_schema = False
        schema_to_use = self.schema

        if len(files) == 1:
            # If only one file is expected, use no schema at all (direct text response)
            no_schema = True
            schema_to_use = None
        elif len(files) > 1:
            # Use default schema for multiple files
            schema_to_use = self.schema
        else:
            # No specific files mentioned, use default schema
            schema_to_use = self.schema

        return schema_to_use, no_schema

    def parse_model_response(self, res, files=None, no_schema=False):
        """
        Parse the model's response based on schema and expected files.

        Args:
            res: Raw response from the model
            files: List of expected output files
            no_schema: Whether schema was used

        Returns:
            Parsed output as a dictionary and success flag
        """

        def extract_code_blocks(text):
            """Helper function to extract code blocks from text."""
            # Match code blocks with optional language identifier
            # Use a more robust pattern that handles nested blocks
            pattern = r"```(?:(\w+)\n)?(.*?)```"
            matches = []
            pos = 0
            while True:
                match = re.search(pattern, text[pos:], re.DOTALL)
                if not match:
                    break
                matches.append(match.group(2))
                pos += match.end()
            return matches

        def process_code_blocks(value):
            """Recursively process values to extract code blocks."""
            if isinstance(value, str):
                code_blocks = extract_code_blocks(value)
                return code_blocks[0] if code_blocks else value
            elif isinstance(value, dict):
                return {k: process_code_blocks(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [process_code_blocks(item) for item in value]
            return value

        try:
            # Single file with direct text mode
            if files and len(files) == 1 and no_schema:
                code_blocks = extract_code_blocks(res)
                if code_blocks:
                    output = {"direct_text": code_blocks[0]}
                else:
                    # For direct text response (no schema), no JSON parsing needed
                    output = {"direct_text": res.strip()}
            else:
                # For JSON responses, parse as before
                response_text = re.sub(r"[\x00-\x1F\x7F]", "", res.strip())
                start = response_text.find("{")
                end = response_text[::-1].find("}")
                response_text = response_text[start : -end if end else None]
                output = json.loads(response_text)

                # Process code blocks in the output
                if "code" in output:
                    output["code"] = process_code_blocks(output["code"])
                if "response" in output:
                    output["response"] = process_code_blocks(output["response"])

            return output, True

        except json.decoder.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response: {e}")
            return {}, False
        except Exception as e:
            logging.error(f"Unexpected error in parse_model_response: {e}")
            return {}, False

    def fix_json_formatting(self, content):
        """
        Fix common JSON formatting issues in model responses.

        Args:
            content: Raw content from the model

        Returns:
            Content with JSON formatting fixed
        """
        try:
            # 1. Add quotes around unquoted keys
            content = re.sub(r"(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1 "\2":', content)
            # 2. Add quotes around unquoted string values
            content = re.sub(r":\s*([a-zA-Z][a-zA-Z0-9_\s]*[a-zA-Z0-9])(\s*[,}])", r': "\1"\2', content)

            try:
                # Validate if it's proper JSON now
                json.loads(content)
            except json.JSONDecodeError:
                # If still invalid JSON, leave as-is
                pass
        except:
            # If regex or other processing fails, return original content
            pass

        return content
