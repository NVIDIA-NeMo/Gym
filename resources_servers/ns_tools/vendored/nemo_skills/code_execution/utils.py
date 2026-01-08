# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import logging
import re
from typing import Tuple

from ..utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


def format_code_output(
    execution_dict,
    code_output_begin: str,
    code_output_end: str,
    code_output_format: str = "llama",
    remaining_code_executions: int | None = None,
):
    """Formatting code output to be displayed as an llm expects it."""
    remaining_ce_string = ""
    if remaining_code_executions is not None:
        if remaining_code_executions > 0:
            remaining_ce_string = (
                f"```system\n"
                f"Remaining code executions: {remaining_code_executions}. "
                f"You will not be able to call code when you run out of executions, so use it wisely. "
                f"Note that you can still continue solving the problem without code after that.\n"
                f"```\n"
            )
        else:
            remaining_ce_string = (
                "```system\n"
                "You have run out of code executions! You can no longer write or execute code. "
                "Now you should continue solving the problem by relying on your mathematical reasoning and analytical skills.\n"
                "```\n"
            )
    if code_output_format == "llama":
        output = execution_dict["process_status"]
        if execution_dict["stdout"]:
            output += f"\n[stdout]\n{execution_dict['stdout']}[/stdout]"
        if execution_dict["stderr"]:
            output += f"\n[stderr]\n{execution_dict['stderr']}[/stderr]"
        output = f"{code_output_begin}\n\n{output}{remaining_ce_string}{code_output_end}\n\n"
    elif code_output_format == "qwen":
        output = ""
        if execution_dict["stdout"]:
            output += f"{execution_dict['stdout']}"
        if execution_dict["stderr"]:
            output += f"{execution_dict['stderr']}"
        output = f"{code_output_begin}{output}{code_output_end}{remaining_ce_string}"
    else:
        raise ValueError(f"Unknown code_output_format: {code_output_format}")

    # wrapping with code output separators
    return output


def _extract_between_separators(generation: str, separators: Tuple[str, str], extract_all: bool = False):
    """Extracting all text between last occurrence of separators[0] and [1].

    If extract_all is True, returning a list with all occurrences of text between separators.
    """
    if extract_all:
        separators = [re.escape(sp) for sp in separators]
        pattern = f"{separators[0]}(.*?){separators[1]}"
        return re.findall(pattern, generation, re.DOTALL)
    return generation.split(separators[0])[-1].split(separators[1])[0]


def extract_code_to_execute(generation: str, code_begin: str, code_end: str, extract_all: bool = False):
    return _extract_between_separators(generation, [code_begin, code_end], extract_all)


def extract_code_output(generation: str, code_output_begin: str, code_output_end: str, extract_all: bool = False):
    return _extract_between_separators(generation, [code_output_begin, code_output_end], extract_all)


def extract_code_block(text: str, languages=None, extract_code_mode: str = "last") -> str:
    if languages is None:
        languages = [""]
    for language in languages:
        matches = re.findall(rf"```{language}\s*\n?(.*?)\n?```", text, re.DOTALL)
        if matches:
            idx = 0 if extract_code_mode == "first" else -1
            return matches[idx].strip()
    return ""


def clean_formal_generation(
    generation: str,
    final_answer_key: str | None = None,
    extract_code_mode: str = "last",
) -> str:
    # Extract part after final_answer_key if present and configured
    if final_answer_key and final_answer_key in generation:
        generation = generation.split(final_answer_key, 1)[1].strip()

    languages = ["lean4", "lean3", "lean", ""]
    extracted_code = extract_code_block(generation, languages, extract_code_mode=extract_code_mode)
    if extracted_code:
        return extracted_code

    # If no explicit code block, remove any surrounding triple backticks
    return re.sub(r"^\s*```(?:lean4|lean3|lean)?\s*|\s*```[\s]*$", "", generation).strip()
