# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Source: https://gitlab-master.nvidia.com/pritamb/agentvista
# Revision: 06f6a8f1532766d876415757107a11731b6148ae (Apache-2.0 adapter files).
# Original file: agentvista_harness/scoring.py
"""Original AgentVista judge prompt/parser; provider failures remain unscored."""

import os
import re


def score(question: str, ground_truth: object, responses: list[str]) -> dict:
    if ground_truth is None or ground_truth == "":
        raise ValueError("Missing reference answer")
    if not all(os.environ.get(key) for key in ("VERIFIER_API_KEY", "VERIFIER_END_POINT", "VERIFIER_MODEL_NAME")):
        raise ValueError("Configure VERIFIER_API_KEY, VERIFIER_END_POINT and VERIFIER_MODEL_NAME")
    from utils.general_qa_tool import QUERY_PROMPT, SYSTEM_PROMPT, GPT4VisionClient

    text = " ".join(responses)
    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    prediction = matches[-1].strip() if matches else (responses[-1].strip() if responses else "")
    part, raw = GPT4VisionClient().query(
        QUERY_PROMPT.format(question=question, ground_truth=ground_truth, prediction=prediction),
        system_prompt=SYSTEM_PROMPT,
    )
    if part not in {"0", "1"}:
        raise RuntimeError("Verifier did not return a valid binary score")
    return {
        "status": "PASS" if part == "1" else "FAIL",
        "reward": float(part),
        "prediction": prediction,
        "analysis": raw,
        "reward_kind": "llm_judged",
        "model": os.environ["VERIFIER_MODEL_NAME"],
        "deterministically_verified": False,
    }
