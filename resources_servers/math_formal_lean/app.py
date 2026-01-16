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

"""Lean4 formal proof verification resource server with multi-turn self-correction."""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from resources_servers.math_formal_lean.sandbox_client import Lean4SandboxClient


LOG = logging.getLogger(__name__)


@dataclass
class ProofBuildConfig:
    final_answer_key: Optional[str] = None
    extract_code_mode: str = "last"  # "first" or "last"
    restate_formal_statement: bool = True
    strip_theorem_from_proof: bool = True


def extract_code_block(text: str, languages: Optional[list] = None, extract_code_mode: str = "last") -> str:
    """Extract code from markdown code blocks."""
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
    final_answer_key: Optional[str] = None,
    extract_code_mode: str = "last",
) -> str:
    """Clean LLM generation to extract Lean code."""
    if final_answer_key and final_answer_key in generation:
        generation = generation.split(final_answer_key, 1)[1].strip()

    languages = ["lean4", "lean3", "lean", ""]
    extracted_code = extract_code_block(generation, languages, extract_code_mode=extract_code_mode)
    if extracted_code:
        return extracted_code

    return re.sub(r"^\s*```(?:lean4|lean3|lean)?\s*|\s*```[\s]*$", "", generation).strip()


def extract_proof_only(lean_code: str) -> str:
    """Extract only the proof part from a Lean theorem/example."""
    lines = lean_code.strip().splitlines()
    if not lines:
        return ""

    header_start_pattern = re.compile(r"^\s*(theorem|example)\b")
    header_start_idx = None

    for i, line in enumerate(lines):
        if header_start_pattern.match(line):
            header_start_idx = i
            break

    if header_start_idx is None:
        return lean_code.strip()

    header_end_idx = None
    for i in range(header_start_idx, len(lines)):
        if ":=" in lines[i]:
            header_end_idx = i
            break

    if header_end_idx is None:
        return lean_code.strip()

    header_line, after = lines[header_end_idx].split(":=", 1)
    proof_first_line = after.strip()

    if proof_first_line:
        proof_lines = [proof_first_line] + lines[header_end_idx + 1 :]
    else:
        proof_lines = lines[header_end_idx + 1 :]

    if proof_lines:
        first = proof_lines[0].lstrip()
        if first == "by":
            proof_lines = proof_lines[1:]
        elif first.startswith("by "):
            proof_lines[0] = first[3:]

    return "\n".join(proof_lines).rstrip()


def build_lean4_proof(generation: str, data_point: Dict[str, Any], config: ProofBuildConfig) -> str:
    """Build a complete Lean4 proof from generation and data point."""
    cleaned_generation = clean_formal_generation(
        generation, final_answer_key=config.final_answer_key, extract_code_mode=config.extract_code_mode
    )

    header = data_point.get("header", "")
    formal_statement = data_point.get("formal_statement", "") if config.restate_formal_statement else ""

    if config.strip_theorem_from_proof:
        proof_part = extract_proof_only(cleaned_generation)
    else:
        proof_part = cleaned_generation

    return header + formal_statement + proof_part


def determine_proof_status(compiler_output: Dict[str, Any]) -> str:
    """Determine proof status from compiler output."""
    process_status = compiler_output.get("process_status", "unknown")

    if process_status == "timeout":
        return "timeout"
    elif process_status != "completed":
        return process_status

    stdout = compiler_output.get("stdout", "").lower()
    stderr = compiler_output.get("stderr", "").lower()
    combined = stdout + "\n" + stderr

    if re.search(r"\bsorry\b", combined) is not None:
        return "has_sorry"

    return "completed"


# ------------------------------------------------------------------------------------------------
# Error parsing for multi-turn self-correction
# Adapted from NeMo-Skills nemo_skills/code_execution/proof_utils.py
# ------------------------------------------------------------------------------------------------


def parse_error(log_string: str) -> List[Dict[str, Any]]:
    """Parse Lean4 compiler error messages from log output.

    Args:
        log_string: The compiler output containing error messages

    Returns:
        List of error dictionaries with position and error data
    """
    error_pattern = re.compile(
        r"(/lean4/my_project/.*?:\d+:\d+: error:.*?)(?=\n/lean4/my_project|\Z)",
        re.DOTALL,
    )
    errors = error_pattern.findall(log_string)
    pattern = re.compile(r":(\d+):(\d+):")
    error_list = []
    for error in errors:
        match = pattern.search(error)
        if match:
            error_list.append(
                {
                    "pos": {"line": int(match.group(1)), "column": int(match.group(2))},
                    "endPos": None,
                    "data": error.split("error:")[1].strip() if "error:" in error else error,
                }
            )

    return error_list


def get_error_str(code: str, errors: List[Dict[str, Any]], error_thres: int = 8) -> str:
    """Format compiler errors with code context for display.

    Args:
        code: The Lean code that was compiled
        errors: List of parsed error dictionaries
        error_thres: Maximum number of errors to include (default 8)

    Returns:
        Formatted error string with code context
    """
    if not errors:
        return ""

    err_str = ""
    code_lines = code.split("\n")

    for i, error in enumerate(errors[:error_thres]):
        start_line = error["pos"]["line"] - 1
        start_col = error["pos"]["column"]
        if start_line >= len(code_lines) or start_line < 0:
            LOG.warning(
                "Error line %d out of bounds (code has %d lines). Error: %s",
                start_line,
                len(code_lines),
                error,
            )
            continue

        if error["endPos"] is None:
            end_line = start_line
            end_col = len(code_lines[start_line]) if start_line < len(code_lines) else 0
        else:
            end_line = error["endPos"]["line"] - 1
            end_col = error["endPos"]["column"]

        err_str += f"\nError {i + 1}:\n"
        err_str += "\nCorresponding Code:\n```lean4\n"

        # Show context lines before error
        error_code = ""
        for ii in range(-4, 0):
            if 0 <= start_line + ii < len(code_lines):
                error_code += f"{code_lines[start_line + ii]}\n"

        # Show error line(s) with <error> markers
        if start_line < len(code_lines):
            if start_line != end_line:
                error_code += (
                    code_lines[start_line][:start_col] + "<error>" + code_lines[start_line][start_col:] + "\n"
                )
                show_line = 6
                for j in range(start_line + 1, min(end_line, start_line + show_line)):
                    if j < len(code_lines):
                        error_code += f"{code_lines[j]}\n"
                if end_line > start_line + show_line:
                    error_code += "... --[Truncated]-- ...\n"
                if end_line < len(code_lines):
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

        # Show one line after error
        if end_line + 1 < len(code_lines):
            error_code += f"{code_lines[end_line + 1]}\n"

        err_str += error_code
        err_str += "\n```\n"
        err_str += f"\nError Message: {error['data']}\n"

    if len(errors) > error_thres:
        err_str += f"\n... [Omitted {len(errors) - error_thres} more errors] ...\n"

    return err_str


def format_error_feedback(compiler_output: Dict[str, Any], predicted_proof: str) -> str:
    """Format compiler errors into feedback for self-correction.

    Args:
        compiler_output: The compiler output dictionary
        predicted_proof: The proof code that was compiled

    Returns:
        Formatted error message string
    """
    process_status = compiler_output.get("process_status", "unknown")
    stdout = compiler_output.get("stdout", "")
    stderr = compiler_output.get("stderr", "")

    if process_status == "timeout":
        return "The compilation timed out. Please simplify your proof or use more efficient tactics."

    # Parse structured errors from stderr
    errors = parse_error(stderr)
    if errors:
        return get_error_str(predicted_proof, errors)

    # Fallback to raw output if no structured errors found
    combined = stdout + "\n" + stderr
    if combined.strip():
        # Truncate if too long
        if len(combined) > 2000:
            combined = combined[:2000] + "\n... [Truncated]"
        return f"Compilation output:\n{combined}"

    return "The proof failed but no specific error message was captured."


# Nemotron single-turn refinement prompt template
REFINEMENT_PROMPT_TEMPLATE = """Here is a proof attempt for the following theorem in Lean4.

{proof_attempt}

The proof is not correct. Following is the compilation error message:

{error_message}

Your task is to fix this proof. Before producing the Lean 4 code to formally prove the given theorem, do a detailed analysis of the error message. Your final answer must be a single, complete Lean 4 markdown code block containing the completed theorem. Do NOT include any text or explanation before or after the code block. Begin with ```lean4 and end with ```."""


def build_correction_prompt(
    proof_attempt: str,
    error_message: str,
    refinement_template: Optional[str] = None,
) -> str:
    """Build a Nemotron-style single-turn correction prompt.

    Args:
        proof_attempt: The previous proof attempt that failed
        error_message: The formatted error feedback
        refinement_template: Optional custom template (uses default if None)

    Returns:
        The formatted correction prompt
    """
    template = refinement_template or REFINEMENT_PROMPT_TEMPLATE
    return template.format(
        proof_attempt=proof_attempt,
        error_message=error_message,
    )


class MathFormalLeanResourcesServerConfig(BaseResourcesServerConfig):
    sandbox_host: str = "127.0.0.1"
    sandbox_port: int = 6000
    compilation_timeout: float = 30.0
    max_output_characters: int = 1000
    extract_code_mode: str = "last"
    restate_formal_statement: bool = True
    strip_theorem_from_proof: bool = True
    # Multi-turn self-correction settings (error feedback always provided on failure)
    refinement_prompt_template: Optional[str] = None  # Use default if None


class MathFormalLeanRunRequest(BaseRunRequest):
    header: str
    formal_statement: str
    informal_prefix: Optional[str] = None
    name: Optional[str] = None


class MathFormalLeanVerifyRequest(MathFormalLeanRunRequest, BaseVerifyRequest):
    # Multi-turn fields
    turn_index: int = 0  # Current turn number (0 = initial attempt)


class CompilerOutput(BaseModel):
    process_status: str
    stdout: str
    stderr: str


class MathFormalLeanVerifyResponse(BaseVerifyResponse):
    proof_status: str
    predicted_proof: str
    compiler_output: Optional[CompilerOutput] = None
    # Multi-turn fields
    turn_index: int = 0
    needs_correction: bool = False  # True if failed and turns remaining
    error_feedback: Optional[str] = None  # Formatted error for next turn
    correction_prompt: Optional[str] = None  # Full prompt for next attempt


class MathFormalLeanResourcesServer(SimpleResourcesServer):
    config: MathFormalLeanResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._sandbox_client = Lean4SandboxClient(
            host=self.config.sandbox_host,
            port=self.config.sandbox_port,
            max_output_characters=self.config.max_output_characters,
        )
        self._proof_build_config = ProofBuildConfig(
            extract_code_mode=self.config.extract_code_mode,
            restate_formal_statement=self.config.restate_formal_statement,
            strip_theorem_from_proof=self.config.strip_theorem_from_proof,
        )

    async def verify(self, body: MathFormalLeanVerifyRequest) -> MathFormalLeanVerifyResponse:
        """Verify a proof attempt with multi-turn self-correction support.

        The resource server is stateless - it always provides error feedback on failure.
        The agent is responsible for controlling the retry loop and turn counting.

        Returns:
        - If proof succeeds: reward=1.0, needs_correction=False
        - If proof fails: reward=0.0, needs_correction=True, with error_feedback
          and correction_prompt for the agent to use if it wants to retry
        """
        generation = body.response.output_text
        turn_index = body.turn_index

        # Exclude turn_index from body dump to avoid duplicate keyword argument
        body_dict = body.model_dump(exclude={"turn_index"})

        if not generation or not generation.strip():
            LOG.warning("Empty generation received at turn %d", turn_index)
            error_msg = "Empty generation received. Please provide a valid Lean 4 proof."
            return MathFormalLeanVerifyResponse(
                **body_dict,
                reward=0.0,
                proof_status="empty_generation",
                predicted_proof="",
                turn_index=turn_index,
                needs_correction=True,
                error_feedback=error_msg,
                correction_prompt=build_correction_prompt(
                    proof_attempt="(empty)",
                    error_message=error_msg,
                    refinement_template=self.config.refinement_prompt_template,
                ),
            )

        data_point = {
            "header": body.header,
            "formal_statement": body.formal_statement,
        }

        predicted_proof = build_lean4_proof(
            generation=generation,
            data_point=data_point,
            config=self._proof_build_config,
        )

        compiler_output = await self._sandbox_client.execute_lean4(
            code=predicted_proof,
            timeout=self.config.compilation_timeout,
        )

        proof_status = determine_proof_status(compiler_output)

        # Success case: reward 1.0, no correction needed
        if proof_status == "completed":
            return MathFormalLeanVerifyResponse(
                **body_dict,
                reward=1.0,
                proof_status=proof_status,
                predicted_proof=predicted_proof,
                compiler_output=CompilerOutput(**compiler_output),
                turn_index=turn_index,
                needs_correction=False,
            )

        # Failure case: always provide error feedback (agent decides whether to retry)
        error_feedback = format_error_feedback(compiler_output, predicted_proof)
        correction_prompt = build_correction_prompt(
            proof_attempt=generation,  # Use the raw generation as proof attempt
            error_message=error_feedback,
            refinement_template=self.config.refinement_prompt_template,
        )
        LOG.info("Proof failed at turn %d. Error feedback prepared for potential retry.", turn_index)
        return MathFormalLeanVerifyResponse(
            **body_dict,
            reward=0.0,
            proof_status=proof_status,
            predicted_proof=predicted_proof,
            compiler_output=CompilerOutput(**compiler_output),
            turn_index=turn_index,
            needs_correction=True,
            error_feedback=error_feedback,
            correction_prompt=correction_prompt,
        )


if __name__ == "__main__":
    MathFormalLeanResourcesServer.run_webserver()
