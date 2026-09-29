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
import json
import os
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed

from prompts import SYSTEM_PROMPT_EASY, SYSTEM_PROMPT_HARD
from rich.progress import MofNCompleteColumn, Progress, SpinnerColumn, TimeElapsedColumn
from utils import generate_assistant_response, grade_assistant_response


def generate_rollouts(system_prompt, sample, mode="hard"):
    # Generate rollouts for the sample
    allow_no_json_list = True
    if mode == "hard":
        allow_no_json_list = False
    conversation = [{"role": "system", "content": system_prompt}]
    grade = 1
    for exp_cal_state, user_prompt in zip(sample["expected_calendar_states"], sample["user_prompts"]):
        conversation.append({"role": "user", "content": user_prompt})
        assistant_response, reasoning_content = generate_assistant_response(conversation, args.model)
        conversation.append(
            {"role": "assistant", "content": assistant_response, "reasoning_content": reasoning_content}
        )
        grade, grade_reason = grade_assistant_response(
            assistant_response, exp_cal_state, allow_no_json_list=allow_no_json_list
        )
        if grade in [0, None]:
            break
    return conversation, grade, grade_reason, exp_cal_state


def process_sample(i, sample, min_cal_entries=None, min_time=600, max_time=960, n_retries=4):
    """Process a single sample and return the result with its index."""

    if i % 2 == 0:
        system_prompt = SYSTEM_PROMPT_EASY.format(start_time_str=min_time, end_time_str=max_time)
        mode = "easy"
    else:
        system_prompt = SYSTEM_PROMPT_HARD.format(start_time_str=min_time, end_time_str=max_time)
        mode = "hard"

    for _ in range(n_retries):
        try:
            conversation, grade, grade_reason, exp_cal_state = generate_rollouts(system_prompt, sample, mode=mode)
            if (min_cal_entries is None) or len(exp_cal_state) >= min_cal_entries:
                break
        except Exception:
            conversation = None
            grade = None
            exp_cal_state = None
    return i, {
        "conversation": conversation,
        "grade": grade,
        "grade_reason": grade_reason,
        "exp_cal_state": exp_cal_state,
        "mode": mode,
    }


def main(args):
    # Load the dataset
    with open(args.input, "r") as f:
        dataset = json.load(f)
    if args.offset is not None:
        dataset = dataset[args.offset :]
    if args.n_samples is not None:
        dataset = dataset[: args.n_samples]

    # Use ThreadPoolExecutor to parallelize processing
    ds_out = []
    with Progress(
        SpinnerColumn(), *Progress.get_default_columns(), MofNCompleteColumn(), TimeElapsedColumn()
    ) as progress:
        task = progress.add_task("[cyan]Generating rollouts...", total=len(dataset))

        with ThreadPoolExecutor(max_workers=args.n_workers) as executor:
            futures = {
                executor.submit(process_sample, i, sample, args.min_cal_entries, args.min_time, args.max_time): i
                for i, sample in enumerate(dataset)
            }

            # Collect results as they complete
            for future in as_completed(futures):
                i, result = future.result()

                if result["grade"] is not None:
                    ds_out.append(result)
                progress.update(task, advance=1, description=f"[cyan]Generating rollouts... (completed sample {i})")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    len_conv_avg = 0
    n = 0
    for sample in ds_out:
        if sample["grade"] is not None:
            len_conv_avg += (
                len(sample["conversation"]) - 1
            ) / 2  # -1 for the system prompt, /2 for the user and assistant messages
            n += 1
    len_conv_avg /= n
    print(f"Average number of assistant turns in valid conversations: {len_conv_avg}")
    if args.min_cal_entries is not None and args.min_cal_entries > 0:
        ds_out = [sample for sample in ds_out if len(sample["exp_cal_state"]) >= args.min_cal_entries]
        print(f"Number of samples with at least {args.min_cal_entries} calendar entries: {len(ds_out)}")

    with open(args.output, "w") as f:
        json.dump(ds_out, f, indent=2)
    print(f"Rollouts saved to {args.output}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, help="Input JSONL file", default="./data/train.json")
    parser.add_argument("--output", type=str, help="Output JSONL file", default="./data/rollouts.json")
    parser.add_argument("--model", type=str, help="Model to use", default="Qwen/Qwen3-8B")
    parser.add_argument("--min-time", type=str, help="Minimum time for events", default="10am")
    parser.add_argument("--max-time", type=str, help="Maximum time for events", default="4pm")
    parser.add_argument("--offset", type=int, help="Number of samples to offset", default=None)
    parser.add_argument("--n-samples", type=int, help="Number of samples to generate", default=None)
    parser.add_argument("--n-workers", type=int, help="Number of parallel workers", default=100)
    parser.add_argument(
        "--min-cal-entries",
        type=int,
        help="Minimum number of calendar entries in the expected calendar state",
        default=None,
    )
    args = parser.parse_args()
    main(args)
