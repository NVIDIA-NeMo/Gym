# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# dependencies = [
#   "litellm",
# ]
# ///

import json
import os
import re
import sys
import time
from pathlib import Path

import litellm


def judge(prompt: str) -> str:
    """Call the judge, retrying a failed call."""
    attempts = int(os.environ["JUDGE_ATTEMPTS"])
    for attempt in range(attempts):
        try:
            response = litellm.completion(
                model=os.getenv("JUDGE_MODEL"),
                api_base=os.getenv("JUDGE_MODEL_API_BASE"),
                api_key=os.getenv("JUDGE_MODEL_API_KEY"),
                messages=[{"role": "user", "content": prompt}],
                timeout=float(os.environ["JUDGE_TIMEOUT"]),
                ## The judge reasons before it answers; the endpoint's default limit cuts long answers.
                max_tokens=int(os.environ["JUDGE_MAX_TOKENS"]),
                max_retries=0,
            )
            return response.choices[0].message.content or ""
        except Exception as error:
            if attempt == attempts - 1:
                raise
            print(f"[judge] attempt {attempt + 1} failed, retrying: {error!r}"[:500], file=sys.stderr)
            time.sleep(10 * (attempt + 1))


TRAJECTORY_PATH = Path("/logs/agent/trajectory.json")
PROMPT_TEMPLATE = Path("/tests/prompt.txt")
JUDGE_VERDICT_PATTERN = re.compile(r"\bVERDICT:\s*(CORRECT|INCORRECT)\b", re.IGNORECASE)
REWARD_PATH = Path("/logs/verifier/reward.txt")


def main():
    trajectory = json.loads(TRAJECTORY_PATH.read_text())
    response_text = trajectory["steps"][-1]["message"]

    prompt = PROMPT_TEMPLATE.read_text().strip().format(response=response_text)

    judge_response_text = judge(prompt)
    matches = JUDGE_VERDICT_PATTERN.findall(judge_response_text)
    if not matches:
        raise ValueError("Judge response did not contain a VERDICT: CORRECT/INCORRECT")

    reward = 1.0 if matches[-1].upper() == "CORRECT" else 0.0
    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REWARD_PATH.write_text(f"{reward}\n")


if __name__ == "__main__":
    main()
