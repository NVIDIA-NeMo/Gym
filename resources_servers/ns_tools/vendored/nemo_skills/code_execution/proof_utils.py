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

"""Shared utilities for proof processing and evaluation."""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict

from .utils import clean_formal_generation
# from .dataset.utils import get_lean4_header  # Not needed for ns_tools (lean4-specific)
from ..utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


# Fallback for get_lean4_header (not vendored from dataset.utils)
def get_lean4_header() -> str:
    """Minimal Lean4 header for proof compilation.
    
    Note: This is a simplified version. For full functionality,
    vendor the complete get_lean4_header from nemo_skills.dataset.utils
    """
    return "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"


@dataclass
class ProofBuildConfig:
    """Configuration for building proofs from generations."""

    final_answer_key: str | None = None
    extract_code_mode: str = "last"  # "first" or "last"
    restate_formal_statement: bool = True
    strip_theorem_from_proof: bool = True


def extract_proof_only(lean_code: str) -> str:
    """Extract only the proof part from a Lean theorem/example.

    This function removes the theorem/example header and returns just the proof body.

    Args:
        lean_code: The full Lean code including theorem statement

    Returns:
        The proof part only (everything after ':=')
    """
    lines = lean_code.strip().splitlines()
    if not lines:
        return ""

    header_start_pattern = re.compile(r"^\s*(theorem|example)\b")
    header_start_idx = None

    # 1. Find where the theorem starts
    for i, line in enumerate(lines):
        if header_start_pattern.match(line):
            header_start_idx = i
            break

    if header_start_idx is None:
        return lean_code.strip()

    # 2. Find where ':=' occurs, starting from the header
    header_end_idx = None
    for i in range(header_start_idx, len(lines)):
        if ":=" in lines[i]:
            header_end_idx = i
            break

    if header_end_idx is None:
        return lean_code.strip()

    # 3. Extract the line after ':='
    header_line, after = lines[header_end_idx].split(":=", 1)
    proof_first_line = after.strip()

    # 4. Collect proof lines
    if proof_first_line:
        proof_lines = [proof_first_line] + lines[header_end_idx + 1 :]
    else:
        proof_lines = lines[header_end_idx + 1 :]

    # 5. Remove leading 'by' (with or without indentation)
    if proof_lines:
        first = proof_lines[0].lstrip()
        if first == "by":
            proof_lines = proof_lines[1:]
        elif first.startswith("by "):
            proof_lines[0] = first[3:]  # Strip 'by '

    return "\n".join(proof_lines).rstrip()


def build_lean4_proof(
    generation: str, data_point: Dict[str, Any], config: ProofBuildConfig, answer_format: str = "lean4-proof"
) -> str:
    """Build a complete Lean4 proof from generation and data point.

    Args:
        generation: The raw generation from the model
        data_point: Dictionary containing header, formal_statement, etc.
        config: Configuration for proof building
        answer_format: Either "lean4-proof" or "lean4-statement"

    Returns:
        Complete Lean4 proof ready for execution
    """
    if answer_format == "lean4-proof":
        # Clean the generation and extract the formal proof
        cleaned_generation = clean_formal_generation(
            generation, final_answer_key=config.final_answer_key, extract_code_mode=config.extract_code_mode
        )

        # Combine header + formal_statement + proof
        header = data_point.get("header", "")
        formal_statement = data_point.get("formal_statement", "") if config.restate_formal_statement else ""

        if config.strip_theorem_from_proof:
            proof_part = extract_proof_only(cleaned_generation)
        else:
            proof_part = cleaned_generation

        predicted_proof = header + formal_statement + proof_part

    elif answer_format == "lean4-statement":
        # For statements, add header and append sorry
        cleaned_generation = clean_formal_generation(generation, extract_code_mode=config.extract_code_mode)
        header = get_lean4_header()
        predicted_proof = header + cleaned_generation + "\n sorry"

    else:
        raise ValueError(f"Unknown answer_format: {answer_format}")

    return predicted_proof


def determine_proof_status(compiler_output: Dict[str, Any]) -> str:
    """Determine proof status from compiler output.

    Args:
        compiler_output: Dictionary containing process_status, stdout, stderr

    Returns:
        Status string: "completed", "timeout", "has_sorry", or other process status
    """
    process_status = compiler_output.get("process_status", "unknown")

    if process_status == "timeout":
        return "timeout"
    elif process_status != "completed":
        return process_status

    # Check stdout and stderr for proof status indicators
    stdout = compiler_output.get("stdout", "").lower()
    stderr = compiler_output.get("stderr", "").lower()
    combined = stdout + "\n" + stderr

    # Check for sorry (incomplete proof)
    if re.search(r"\bsorry\b", combined) is not None:
        return "has_sorry"

    # If process completed without errors, consider it successful
    return "completed"


def prepare_predicted_proof_from_line_dict(
    line_dict: Dict[str, Any],
    config: ProofBuildConfig,
    answer_format: str = "lean4-proof",
    use_predicted_proof_key: bool = False,
) -> str:
    """Prepare predicted_proof from a line dictionary (for batch processing).

    Args:
        line_dict: Dictionary containing generation and other fields
        config: Configuration for proof building
        answer_format: Either "lean4-proof" or "lean4-statement"
        use_predicted_proof_key: If True, use existing predicted_proof key

    Returns:
        Complete Lean4 proof ready for execution

    Raises:
        ValueError: If use_predicted_proof_key is True but key is missing
    """
    if use_predicted_proof_key:
        if "predicted_proof" not in line_dict:
            raise ValueError(
                "predicted_proof key not found in the line_dict. Set use_predicted_proof_key=False to re-combine"
            )
        return line_dict["predicted_proof"]

    return build_lean4_proof(
        generation=line_dict["generation"], data_point=line_dict, config=config, answer_format=answer_format
    )


# ------------------------------------------------------------------------------------------------
# The following code is adapted from https://github.com/Goedel-LM/Goedel-Prover-V2
# Used for multi-turn proof refinement in the lean4_prover workflow
# ------------------------------------------------------------------------------------------------


def remove_comments(text):
    # First remove all /- ... -/ blocks
    text = re.sub(r"/-.*?-/", "", text, flags=re.DOTALL)
    # Then remove -- comments from each line
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        cleaned_line = line.split("--", 1)[0]
        if cleaned_line.strip() == "":
            continue
        cleaned_lines.append(cleaned_line)
    # Join back together and remove excessive empty lines
    cleaned_text = "\n".join(cleaned_lines)
    return cleaned_text.strip()


def move_imports_to_beginning(input_string):
    lines = input_string.split("\n")
    import_lines = [line for line in lines if line.startswith("import")]
    other_lines = [line for line in lines if not line.startswith("import")]
    return "\n".join(import_lines + other_lines)


def return_theorem_to_prove(text):
    # Pattern that matches from 'theorem' or 'lemma' to ':= by sorry' with any content in between
    pattern = r"((?:theorem).*?:=\s*by\s*sorry)"
    match = re.search(pattern, text, re.DOTALL)
    return match.span() if match else None


def return_theorem_to_replace(text):
    # Pattern that matches from 'theorem' or 'lemma' to ':= by sorry' with any content in between
    pattern = r"((?:^|\s)theorem\s+.*?:=\s*by)"
    match = re.search(pattern, text, re.DOTALL)
    return match.span() if match else None


def replace_statement_in_proof(statement, proof):
    if ("apply?" in proof) or ("exact?" in proof):
        return "**Error**, 'apply?' or 'exact?' is used, which is not allowed."
    stats_re = remove_comments(statement)
    stats_span_ = return_theorem_to_prove(stats_re)
    if stats_span_ is None:
        error_app = "\n".join(["\n"] + ["-- " + x for x in statement.split("\n")])
        return f"**Error**, can not find 'theorem' and ':= sorry' in {error_app}"
    proof_str = remove_comments(proof)
    span = return_theorem_to_replace(proof_str)
    if span is None:
        error_app = "\n".join(["\n"] + ["-- " + x for x in proof.split("\n")])
        return f"**Error**, can not find 'theorem' and ':=' in {error_app}"
    return stats_re[: stats_span_[1]].replace("sorry", "") + proof_str[span[1] :]


def refine_by_sorry(text):
    # Define the regular expression pattern
    target_pattern = r":=\s*(?:by\s*)?(?:sorry\s*)?"
    replacement = ":= by sorry"  # The new text we want to insert
    # We construct the pattern with two capturing groups
    # (group 1: the part from 'theorem' to just before our target)
    # (group 2: the target pattern itself)
    combined_pattern = r"(theorem.*?)(" + target_pattern + r")"
    # Find the first match
    match = re.search(combined_pattern, text, re.DOTALL)
    if match:
        # The part of the string BEFORE the target we want to replace
        # We use match.start(2) which is the start of the second group (our target)
        prefix = text[: match.start(2)]
        # Concatenate the prefix with the replacement to get the final, truncated string
        final_text = prefix + replacement
    else:
        final_text = text
    return final_text


def extract_code(inputs):
    import_head = (
        "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"
    )
    pattern = r"```lean4\n(.*?)\n```"
    matches = re.findall(pattern, inputs, re.DOTALL)
    if matches:
        return import_head + matches[-1]
    pattern = r"```lean4\n(.*?)```"
    matches = re.findall(pattern, inputs, re.DOTALL)
    if matches:
        return import_head + matches[-1]
    pattern = r"```lean\n(.*?)```"
    matches = re.findall(pattern, inputs, re.DOTALL)
    if matches:
        return import_head + matches[-1]
    return "None"


def parse_error(log_string):
    """Parse Lean4 compiler error messages from log output."""
    error_pattern = re.compile(
        r"(/lean4/my_project/.*?:\d+:\d+: error:.*?)(?=\n/lean4/my_project|\Z)",
        re.DOTALL,
    )
    errors = error_pattern.findall(log_string)
    pattern = re.compile(r":(\d+):(\d+):")
    error_list = []
    for error in errors:
        match = pattern.search(error)
        error_list.append(
            {
                "pos": {"line": int(match.group(1)), "column": int(match.group(2))},
                "endPos": None,
                "data": error.split("error:")[1],
            }
        )

    return error_list


def get_error_str(code, errors, error_thres=True):
    """Format compiler errors with code context for display."""
    err_str = ""
    code_lines = code.split("\n")
    error_num_thres = 8 if error_thres else len(errors)

    for i, error in enumerate(errors[:error_num_thres]):
        start_line = error["pos"]["line"] - 1
        start_col = error["pos"]["column"]
        if start_line >= len(code_lines):
            LOG.warning(
                "Error line %d exceeds code length %d. Errors: %s, Code: %s", start_line, len(code_lines), errors, code
            )
            continue
        if error["endPos"] is None:
            end_line = start_line
            end_col = len(code_lines[start_line])
        else:
            end_line = error["endPos"]["line"] - 1
            end_col = error["endPos"]["column"]

        err_str += f"\nError {i + 1}:\n"
        err_str += "\nCorresponding Code:\n```lean4\n"
        error_code = ""
        for ii in range(-4, 0):
            if start_line + ii >= 0:
                error_code += f"{code_lines[start_line + ii]}\n"
        if start_line != end_line:
            error_code += code_lines[start_line][:start_col] + "<error>" + code_lines[start_line][start_col:] + "\n"
            if not error_thres:
                for j in range(start_line + 1, end_line):
                    error_code += f"{code_lines[j]}\n"
            else:
                show_line = 6
                for j in range(start_line + 1, min(end_line, start_line + show_line)):
                    error_code += f"{code_lines[j]}\n"
                if end_line > start_line + show_line:
                    leading_spaces = len(code_lines[j]) - len(code_lines[j].lstrip(" "))
                    error_code += "\n" + " " * leading_spaces + "... --[Truncated]-- ...\n"
            error_code += code_lines[end_line][:end_col] + "</error>" + code_lines[end_line][end_col:] + "\n"
        else:
            error_code += (
                code_lines[start_line][:start_col]
                + "<error>"
                + code_lines[start_line][start_col:end_col]
                + "</error>"
                + code_lines[start_line][end_col:]
                + "\n"
            )
        if end_line + 1 < len(code_lines):
            error_code += f"{code_lines[end_line + 1]}\n"
        err_str += error_code
        err_str += "\n```\n"
        err_str += f"\nError Message: {error['data']}\n"
    if len(errors) > error_num_thres:
        err_str += f"\n... [Omitted {len(errors) - error_num_thres} more errors] ...\n"
    return err_str
