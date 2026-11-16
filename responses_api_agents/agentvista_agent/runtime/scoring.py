# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
    if not part:
        raise RuntimeError("Verifier did not return a valid binary score")
    reward = 1.0 if "1" in part else 0.0
    return {
        "status": "PASS" if reward == 1.0 else "FAIL",
        "reward": reward,
        "prediction": prediction,
        "analysis": raw,
        "reward_kind": "llm_judged",
        "model": os.environ["VERIFIER_MODEL_NAME"],
        "deterministically_verified": False,
    }
