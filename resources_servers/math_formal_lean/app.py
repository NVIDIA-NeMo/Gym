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

"""Lean4 formal proof verification resource server."""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

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


class MathFormalLeanResourcesServerConfig(BaseResourcesServerConfig):
    sandbox_host: str = "127.0.0.1"
    sandbox_port: int = 6000
    compilation_timeout: float = 30.0
    max_output_characters: int = 1000
    extract_code_mode: str = "last"
    restate_formal_statement: bool = True
    strip_theorem_from_proof: bool = True


class MathFormalLeanRunRequest(BaseRunRequest):
    header: str
    formal_statement: str
    informal_prefix: Optional[str] = None
    name: Optional[str] = None


class MathFormalLeanVerifyRequest(MathFormalLeanRunRequest, BaseVerifyRequest):
    pass


class CompilerOutput(BaseModel):
    process_status: str
    stdout: str
    stderr: str


class MathFormalLeanVerifyResponse(BaseVerifyResponse):
    proof_status: str
    predicted_proof: str
    compiler_output: Optional[CompilerOutput] = None


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
        generation = body.response.output_text
        if not generation or not generation.strip():
            LOG.warning("Empty generation received")
            return MathFormalLeanVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                proof_status="empty_generation",
                predicted_proof="",
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
        reward = 1.0 if proof_status == "completed" else 0.0

        return MathFormalLeanVerifyResponse(
            **body.model_dump(),
            reward=reward,
            proof_status=proof_status,
            predicted_proof=predicted_proof,
            compiler_output=CompilerOutput(**compiler_output),
        )


if __name__ == "__main__":
    MathFormalLeanResourcesServer.run_webserver()
