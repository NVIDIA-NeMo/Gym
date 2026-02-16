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

import importlib
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent.parent.parent
NEWTON_BENCH_PATH = REPO_ROOT / "NewtonBench"
if str(NEWTON_BENCH_PATH) not in sys.path:
    sys.path.insert(0, str(NEWTON_BENCH_PATH))


def convert_xml_to_openai(prompt: str, module_name: str, is_code_assisted: bool = False) -> str:
    """
    Cleans the prompt by replacing XML instructions with Tool Use instructions.
    """
    params_match = re.search(r"<run_experiment>\s*\[\s*\{(.*?)\}", prompt, re.DOTALL)
    example_call = f"run_experiment_{module_name}(args...)"

    if params_match:
        content = params_match.group(1)
        keys = re.findall(r'["\'](\w+)["\']\s*:', content)
        if keys:
            args = ", ".join([f"{k}=..." for k in keys])
            example_call = f"run_experiment_{module_name}({args})"

    desc_match = re.search(r"(?:\*\*|\*)System Response:(?:\*\*|\*)\s*(.*?)\s*<experiment_output>", prompt, re.DOTALL)
    return_item = "measurement result"

    if desc_match:
        raw_desc = desc_match.group(1).strip()

        item_match = re.search(r"containing\s+(.*?)(?:\.|\s*\(|:|$)", raw_desc, re.IGNORECASE)
        if not item_match:
            item_match = re.search(r"a list of\s+(.*?)(?:\.|\s*\(|:|$)", raw_desc, re.IGNORECASE)

        if item_match:
            return_item = item_match.group(1).strip()
        else:
            clean_desc = re.sub(r"^The system will return\s+(?:a\s+|the\s+)?", "", raw_desc, flags=re.IGNORECASE)
            clean_desc = re.sub(r"[:\.].*$", "", clean_desc).strip()
            if clean_desc:
                return_item = clean_desc

    return_item = re.sub(
        r"\s+(?:for each experiment|per experiment|for that experiment)\s*$", "", return_item, flags=re.IGNORECASE
    )

    keys_str = ""
    output_match = re.search(r"<experiment_output>\s*\[\s*\{(.*?)\}", prompt, re.DOTALL)
    if output_match:
        content = output_match.group(1)
        keys = re.findall(r'["\'](\w+)["\']\s*:', content)
        if keys:
            if len(keys) == 1:
                keys_str = f", including {keys[0]}"
            elif len(keys) == 2:
                keys_str = f", including {keys[0]} and {keys[1]}"
            else:
                keys_str = f", including {', '.join(keys[:-1])} and {keys[-1]}"

    if return_item.lower().startswith("the "):
        return_desc = f"Each function call returns {return_item} for that experiment{keys_str}."
    else:
        return_desc = f"Each function call returns the {return_item} for that experiment{keys_str}."

    function_format = f"""**How experiments work:**
To run experiments, call the `run_experiment_{module_name}` function. Each function call runs ONE experiment.
To run multiple experiments, make multiple function calls (the system supports parallel calls).

**Example:**
- First call: `{example_call}`
- Second call: `{example_call}`
- etc.

**Response:**
{return_desc}"""

    prompt = re.sub(
        r"use the <run_experiment> tag",
        f"call the **run_experiment_{module_name}** function",
        prompt,
        flags=re.IGNORECASE,
    )

    prompt = re.sub(
        r"use the exact <run_experiment> tag format", "use the exact function call format", prompt, flags=re.IGNORECASE
    )
    prompt = re.sub(
        r"Double-check your JSON syntax before submitting",
        "Double-check your function arguments before calling",
        prompt,
        flags=re.IGNORECASE,
    )
    prompt = re.sub(
        r"If your format is incorrect, the system will ask you to read the initial prompt again",
        "If your function call format is incorrect, the system will return an error message",
        prompt,
        flags=re.IGNORECASE,
    )

    xml_block_pattern = r"(?:\*\*Input/Output Format:\*\*|(?:\*\*|\*)Your Request:(?:\*\*|\*)).*?</experiment_output>"

    if re.search(xml_block_pattern, prompt, re.DOTALL):
        prompt = re.sub(xml_block_pattern, function_format, prompt, flags=re.DOTALL)
    else:
        fallback_pattern = r"<run_experiment>.*?</experiment_output>"
        prompt = re.sub(fallback_pattern, function_format, prompt, flags=re.DOTALL)

    prompt = re.sub(
        r"Provide a JSON array specifying the parameters for one or arbitrarily many experimental sets\.?",
        "Run experiments by calling the function with the appropriate parameters.",
        prompt,
        flags=re.IGNORECASE,
    )

    if is_code_assisted:
        prompt = re.sub(
            r"Format: <python>your_python_code_here</python>",
            'Format: call `execute_python(code="...")`',
            prompt,
            flags=re.IGNORECASE,
        )
        prompt = re.sub(
            r"multiple <python> tags",
            "multiple `execute_python` function calls (the system supports parallel calls)",
            prompt,
            flags=re.IGNORECASE,
        )
        prompt = re.sub(r"Each <python> tag", "Each `execute_python` function call", prompt, flags=re.IGNORECASE)
        prompt = re.sub(r"<python_output>( tags)?", "the function response", prompt, flags=re.IGNORECASE)
        prompt = re.sub(r"</python_output>", "end of the function response", prompt, flags=re.IGNORECASE)
        prompt = re.sub(
            r"with more <python> tags", "with more `execute_python` function calls", prompt, flags=re.IGNORECASE
        )
        prompt = re.sub(r"<python> tags?", "`execute_python` function", prompt, flags=re.IGNORECASE)
        prompt = re.sub(
            r"\*\*CRITICAL: Use EXACTLY these tags:\*\*.*?(?=\*\*Examples:\*\*)",
            "**CRITICAL: Strict Function Call Format**\n- You MUST use the `execute_python` function with a `code` argument.\n- Ensure your code is a valid Python string.\n\n",
            prompt,
            flags=re.DOTALL | re.IGNORECASE,
        )
        prompt = re.sub(r"<python>(.*?)</python>", r'execute_python(code="""\1""")', prompt, flags=re.DOTALL)

    prompt = re.sub(r"\n\n\n+", "\n\n", prompt)

    return prompt.strip()


def get_physics_prompt(module_name: str, system, is_code_assisted: bool = False, noise_level: float = 0.0) -> str:
    """
    Extracts and cleans the physics context from the original NewtonBench prompts.
    """
    try:
        prompts_mod = importlib.import_module(f"modules.{module_name}.prompts")
    except ImportError as e:
        raise ImportError("Could not import prompts module for the specified module_name with error: " + str(e))

    if hasattr(prompts_mod, "get_task_prompt"):
        raw_prompt = prompts_mod.get_task_prompt(
            system=system, is_code_assisted=is_code_assisted, noise_level=noise_level
        )
    else:
        raise ImportError("Module does not have get_task_prompt function")

    cleaned = convert_xml_to_openai(raw_prompt, module_name, is_code_assisted=is_code_assisted)

    return cleaned
