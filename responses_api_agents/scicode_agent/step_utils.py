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
# ruff: noqa: E501  (PREFILLED_STEPS_CODE embeds reference code with long lines; keep verbatim)

"""Prompt-context construction and code extraction for the SciCode step loop."""

import re
from pathlib import Path


# Sentinel stored for sub-steps that could not be generated (context window exceeded). Scored as a
# failed step.
OUT_OF_CONTEXT = "_ran_out_of_context_"


def _process_problem_code(sub_steps: list, num_steps: int) -> str:
    header_docstring = sub_steps[num_steps]["function_header"]
    return_str = sub_steps[num_steps]["return_line"]
    return f"{header_docstring}\n\n{return_str}"


def process_problem_steps(sub_steps: list, num_steps: int, previous_llm_code: list, with_background: bool):
    """Build the prompt context: previous-steps block, next-step block, and previous-code string."""
    output_lines = []
    next_step = []
    previous_code = []
    for i in range(num_steps):
        output_lines.append(
            sub_steps[i]["step_description_prompt"] + "\n" + sub_steps[i]["step_background"]
            if with_background
            else sub_steps[i]["step_description_prompt"]
        )
        output_lines.append(previous_llm_code[i])
        previous_code.append(previous_llm_code[i])
        output_lines.append("------")

    next_step.append(
        sub_steps[num_steps]["step_description_prompt"] + "\n" + sub_steps[num_steps]["step_background"]
        if with_background
        else sub_steps[num_steps]["step_description_prompt"]
    )
    next_step.append(_process_problem_code(sub_steps, num_steps))
    output_str = "\n\n".join(output_lines[:-1])  # Remove the last "------"
    next_step_str = "\n\n".join(next_step)
    previous_code_str = "\n".join(previous_code)
    return output_str, next_step_str, previous_code_str


def extract_python_script(response: str) -> str:
    """Extract the Python code block from a model response, stripping import lines."""
    if "```" in response:
        python_script = (
            response.split("```python")[1].split("```")[0]
            if "```python" in response
            else response.split("```")[1].split("```")[0]
        )
    else:
        python_script = response
    python_script = re.sub(r"^\s*(import .*|from .*\s+import\s+.*)", "", python_script, flags=re.MULTILINE)
    return python_script


_CONTEXT_WINDOW_MARKERS = (
    "Requested token count exceeds",
    "exceeds maximum input length",
    "should not exceed max_seq_len",
    "reduce the length of the input messages",
    "'max_completion_tokens' is too large",
    "max_tokens must be at least 1, got -",
)


def is_context_window_error(error: Exception) -> bool:
    """Best-effort detection of a context-window-exceeded error from the model server."""
    msg = str(error)
    return any(marker in msg for marker in _CONTEXT_WINDOW_MARKERS)


def _official_prefilled_step(step_number: str) -> str:
    """The official SciCode reference file for an unscored step, byte for byte (see prefilled_steps/)."""
    return (Path(__file__).parent / "prefilled_steps" / f"{step_number}.txt").read_text(encoding="utf-8")


# A few SciCode sub-steps are prefilled with reference code rather than generated (a quirk of the
# original SciCode protocol). They provide context for later steps but are not themselves scored.
#
# Steps 13.6 and 62.1 define classes (Maxwell; Block and EnlargedBlock) that every later step of their
# problem instantiates. They are the official unscored-step files of the SciCode release, whole. An earlier
# version of this table kept only each class's `__init__` function, so the cumulative program held no class
# and every later step of problems 13 and 62 ended in a NameError, whatever the model wrote.
PREFILLED_STEPS_CODE = {
    ("13", 5): _official_prefilled_step("13.6"),
    ("62", 0): _official_prefilled_step("62.1"),
    (
        "76",
        2,
    ): '''
def generate_dna(N: int, PWM: dict) -> tuple:
    """
    Input:
    N (int): Length of the resultant DNA sequence.
    PWM matrix with keys 'A', 'C', 'G', 'T'

    Output:
    tuple: Insertion location (int), DNA sequence (str), DNA reverse complement (str)
    """
    p = random.randint(0, N - 1)
    nucleotide = 'ACGT'
    uni_weights = [0.25, 0.25, 0.25, 0.25]
    dna_string = ''.join(random.choices(nucleotide, uni_weights, k=N))
    spike_mat = load_motif_from_df(PWM)
    spiked_seq = ''.join((random.choices(nucleotide, weights=[PWM[nuc][i] for nuc in nucleotide], k=1)[0] for i in range(len(PWM['A']))))
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    reversed_seq = dna_string[::-1]
    reverse_complement = ''.join((complement[nuc] for nuc in reversed_seq if nuc in complement))
    new_seq = dna_string[:p] + spiked_seq + dna_string[p:]
    new_seq_rc = reverse_complement[:N - p] + spiked_seq + reverse_complement[N - p:]
    return (p, new_seq, new_seq_rc)
'''.strip(),
}
