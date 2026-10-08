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
# limitations under the License

import argparse
import json
import logging
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import inflect
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from .tokenizer import select_tokenizer


convert = inflect.engine()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
# Basic Configurations
parser.add_argument("--output_folder", type=str)
parser.add_argument("--tokenizer_type", type=str, default="hf", help="[Options] nemo, hf, openai.")
parser.add_argument("--tokenizer_path", type=str, required=True, help="path to the tokenizer model")
parser.add_argument(
    "--max_seq_length",
    type=int,
    required=True,
    help="max sequence length including all input tokens and generated tokens.",
)
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument(
    "--insert_position", type=float, default=-1, help="insert position of the true context in the context."
)
parser.add_argument("--num_samples", type=int, default=None, help="number of samples to generate")
parser.add_argument("--dataset", type=str, default="gsm8k")
parser.add_argument("--fewshot", type=int, default=0)
parser.add_argument("--prompt_type", type=str, default="chat")
parser.add_argument("--num_order", type=int, default=0)
parser.add_argument(
    "--algo_type",
    type=str,
    default="single",
    choices=["single", "attention", "2steps", "3steps", "size_2steps", "size_single"],
)
parser.add_argument("--task_type", type=str, default="retrieve", choices=["retrieve", "solve", "niah"])

args = parser.parse_args()
random.seed(args.random_seed)
np.random.seed(args.random_seed)

# Load Tokenizer
TOKENIZER = select_tokenizer(args.tokenizer_type, args.tokenizer_path)

TOTAL_PROMPT = """{context}\n\n{example}{problem}"""
if args.task_type == "retrieve":
    CONTEXT_PROMPT = "Below are some questions. I will ask you to copy one of them. Please copy and paste the question you find.\n\n{needles}"
    NEEDLE_PROMPT = "Question {i}: {question}"
    if args.algo_type == "single":
        PROBLEM_PROMPT = "Please copy the {order}Question {i} from the context."
    elif args.algo_type == "attention":
        PROBLEM_PROMPT = "Please first pay attention to all the Question {i} from the context and then only copy the {order}Question {i} in your response. Do not output any other questions."
    elif args.algo_type == "2steps":
        # PROBLEM_PROMPT = "Please first find all the Question {i} from the context and then copy the {order}Question {i} at the end."
        PROBLEM_PROMPT = (
            "Please first copy all the Question {i} from the context and then copy the {order}Question {i} at the end."
        )
        # PROBLEM_PROMPT = "Please first copy all instances of Question {i} from the context in the order in which they appear, and then copy the {order}Question {i} (1-indexed) at the end."
    elif args.algo_type == "3steps":
        PROBLEM_PROMPT = "Please first find how many Question {i} from the context, list them in order, and then copy the {order}Question {i} at the end."
    elif args.algo_type == "size_2steps":
        PROBLEM_PROMPT = (
            "Please first find all the"
            + str(args.num_order)
            + " Question {i} from the context and then copy the {order}Question {i} at the end."
        )
    elif args.algo_type == "size_single":
        PROBLEM_PROMPT = (
            "There are "
            + str(args.num_order)
            + " Question {i} in the context. Please copy the {order}Question {i} from the context."
        )

    if args.fewshot > 0:
        EXAMPLE_PROMPT = PROBLEM_PROMPT + "\nQuestion {i}: {question}"

    if args.prompt_type == "base":
        PROBLEM_PROMPT += "\nQuestion {i}:"

elif args.task_type == "niah":
    CONTEXT_PROMPT = "Below are some questions. I will ask you to copy some of them. Please copy and paste the questions you find.\n\n{needles}"
    NEEDLE_PROMPT = "Question {i}: {question}"
    PROBLEM_PROMPT = "Please find and copy all the Question {i} from the context."
    if args.prompt_type == "base":
        PROBLEM_PROMPT += "\nQuestion {i}:"


elif args.task_type == "solve":
    CONTEXT_PROMPT = "Below are some questions. I will ask you to solve one of them. Please solve the question you find and make sure to put the answer (and only answer) inside \\boxed\\{{\\}}.\n\n{needles}"
    NEEDLE_PROMPT = "Question {i}: {question}"
    if args.dataset == "gsm8k" or args.dataset == "math500":
        if args.algo_type == "single":
            PROBLEM_PROMPT = "Please solve the Question {i} from the context step by step."
        elif args.algo_type == "2steps":
            PROBLEM_PROMPT = "Please copy the Question {i} from the context and then solve it step by step."
    elif args.dataset == "mmlu":
        if args.algo_type == "single":
            PROBLEM_PROMPT = "Please solve the Question {i} from the context with an answer from A, B, C, D."
        elif args.algo_type == "2steps":
            PROBLEM_PROMPT = (
                "Please copy the Question {i} from the context and then solve it with an answer from A, B, C, D."
            )
    elif args.dataset == "mbpp":
        if args.algo_type == "single":
            PROBLEM_PROMPT = "Please solve the Question {i} from the context by generating or completing code.\nYour answer should be in the following format:\n```python\n# Your code here\n```"
        elif args.algo_type == "2steps":
            PROBLEM_PROMPT = "Please copy the Question {i} from the context and then solve it by generating or completing code.\nYour answer should be in the following format:\n```python\n# Your code here\n```"

    if args.fewshot > 0:
        if args.algo_type == "single":
            EXAMPLE_PROMPT = PROBLEM_PROMPT + "\nSolution:{solution}"
            if args.prompt_type == "base":
                PROBLEM_PROMPT += "\nSolution:"
        elif args.algo_type == "2steps":
            EXAMPLE_PROMPT = PROBLEM_PROMPT + "\nQuestion {i}: {question}\nSolution:{solution}"
            if args.prompt_type == "base":
                PROBLEM_PROMPT += "\nQuestion {i}:"

examples = []
haystack, needle = [], []
if args.dataset == "gsm8k":
    test_dataset = load_dataset("openai/gsm8k", "main")
    for d in test_dataset["train"]:
        solution, answer = d["answer"].split("#### ")
        haystack.append(
            {
                "Question": d["question"],
                "Solution": " " + solution + f"So the answer is \\boxed{{{answer}}}.",
                "Answer": answer,
            }
        )
    for d in test_dataset["test"]:
        solution, answer = d["answer"].split("#### ")
        needle.append(
            {
                "Question": d["question"],
                "Solution": " " + solution + f"So the answer is \\boxed{{{answer}}}.",
                "Answer": answer,
            }
        )
elif args.dataset == "math500":
    questions = set()
    test_dataset = load_dataset("HuggingFaceH4/MATH-500")
    for d in test_dataset["test"]:
        needle.append(
            {
                "Question": d["problem"],
                "Solution": " " + d["solution"],
                "Answer": d["answer"],
            }
        )
        questions.add(d["problem"])

    from math_verify import parse

    for subject in [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]:
        train_dataset = load_dataset("EleutherAI/hendrycks_math", subject)
        for index, d in enumerate(train_dataset["test"]):
            if d["problem"] not in questions:
                haystack.append(
                    {
                        "Question": d["problem"],
                        "Solution": " " + d["solution"],
                        "Answer": parse(d["solution"])[-1],
                    }
                )


elif args.dataset == "mmlu":
    test_dataset = load_dataset("cais/mmlu", "all")
    options = ["A", "B", "C", "D"]
    haystack = []
    needle = []
    for d in test_dataset["test"]:
        choices = d["choices"]
        item = {
            "Question": d["question"] + f"\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}",
            "Solution": " " + f"\\boxed{{{options[d['answer']]}}}",
            "Answer": options[d["answer"]],
        }
        needle.append(item)

    for d in test_dataset["auxiliary_train"]:
        choices = d["choices"]
        item = {
            "Question": d["question"] + f"\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}",
            "Solution": " " + f"\\boxed{{{options[d['answer']]}}}",
            "Answer": options[d["answer"]],
        }
        haystack.append(item)


elif args.dataset == "mbpp":
    test_dataset = load_dataset("evalplus/mbppplus")
    for d in test_dataset["test"]:
        prompt = d["prompt"].replace("    ", "\t").strip()
        assertion = d["test_list"][0]
        needle.append(
            {
                "task_id": f"Mbpp/{d['task_id']}",
                "Question": f"{prompt}\n{assertion}",
                "Solution": f"\n```python\n{d['code'].strip()}\n```",
                "canonical_solution": f"\n{d['code'].strip()}\n",
                "assertion": "\n".join(d["test_list"]),
            }
        )

    train_dataset = load_dataset("google-research-datasets/mbpp", "full")
    for d in train_dataset["train"]:
        prompt = d["text"].replace("    ", "\t").strip()
        assertion = d["test_list"][0]
        haystack.append(
            {
                "Question": f"{prompt}\n{assertion}",
                "Solution": f"\n```python\n{d['code'].strip()}\n```",
                "canonical_solution": f"\n{d['code'].strip()}\n",
                "assertion": "\n".join(d["test_list"]),
            }
        )
    for d in train_dataset["validation"]:
        prompt = d["text"].replace("    ", "\t").strip()
        assertion = d["test_list"][0]
        haystack.append(
            {
                "Question": f"{prompt}\n{assertion}",
                "Solution": f"\n```python\n{d['code'].strip()}\n```",
                "canonical_solution": f"\n{d['code'].strip()}\n",
                "assertion": "\n".join(d["test_list"]),
            }
        )
    for d in train_dataset["test"]:
        prompt = d["text"].replace("    ", "\t").strip()
        assertion = d["test_list"][0]
        haystack.append(
            {
                "Question": f"{prompt}\n{assertion}",
                "Solution": f"\n```python\n{d['code'].strip()}\n```",
                "canonical_solution": f"\n{d['code'].strip()}\n",
                "assertion": "\n".join(d["test_list"]),
            }
        )
else:
    raise ValueError(f"Dataset {args.dataset} is not supported.")

for item in needle:
    item["Question"] = re.sub(r"\s+", " ", item["Question"])
for item in haystack:
    item["Question"] = re.sub(r"\s+", " ", item["Question"])

logger.info(f"Dataset size: {len(needle)}")


def generate_random_number(num_digits=7):
    lower_bound = 10 ** (num_digits - 1)
    upper_bound = 10**num_digits - 1
    return str(random.randint(lower_bound, upper_bound))


def generate_input_output(index, num_qs):
    if num_qs > len(haystack):
        repeats = (num_qs + len(haystack) - 1) // len(haystack)  # Ceiling division
    else:
        repeats = 1

    curr_context = [dict(item) for item in random.sample([item for item in haystack for _ in range(repeats)], num_qs)]

    if args.num_order > 0:
        random_numbers = [generate_random_number() for _ in range(math.ceil((num_qs + 1) / args.num_order))]
        random_numbers = random_numbers * args.num_order
    else:
        random_numbers = [generate_random_number() for _ in range(num_qs + 1)]

    random.shuffle(random_numbers)
    random_numbers = random_numbers[: num_qs + 1]
    for i, q in enumerate(curr_context):
        q["random_index"] = random_numbers[i]

    random.shuffle(curr_context)
    examples = random.sample(curr_context, args.fewshot)

    true_context = dict(needle[index])
    true_context["random_index"] = random_numbers[-1]
    if args.insert_position < 0:
        insert_position = random.randint(0, len(curr_context))
    else:
        insert_position = int(args.insert_position * len(curr_context))
    curr_context.insert(insert_position, true_context)

    counts = defaultdict(int)
    for i, q in enumerate(curr_context):
        counts[q["random_index"]] += 1
        if args.num_order > 0:
            q["order"] = convert.ordinal(counts[q["random_index"]]) + " (1 indexed) "
        else:
            q["order"] = ""

    needles = "\n\n".join(
        [NEEDLE_PROMPT.format(i=q["random_index"], question=q["Question"]) for i, q in enumerate(curr_context)]
    )
    if args.task_type == "niah":
        problem = PROBLEM_PROMPT.format(i=true_context["random_index"])
    else:
        problem = PROBLEM_PROMPT.format(i=true_context["random_index"], order=true_context["order"])

    if args.fewshot > 0:
        if args.task_type == "retrieve":
            example = "\n\n".join(
                [
                    EXAMPLE_PROMPT.format(i=q["random_index"], question=q["Question"], order=q["order"])
                    for q in examples
                ]
            )
        elif args.task_type == "solve":
            if args.algo_type == "single":
                example = "\n\n".join(
                    [EXAMPLE_PROMPT.format(i=q["random_index"], solution=q["Solution"]) for q in examples]
                )
            elif args.algo_type == "2steps":
                example = "\n\n".join(
                    [
                        EXAMPLE_PROMPT.format(i=q["random_index"], question=q["Question"], solution=q["Solution"])
                        for q in examples
                    ]
                )

        if args.prompt_type == "base":
            example = f"{example}\n\n"
        else:
            example = f"Here are some examples to help you understand the task:\n\n{example}\n\nHere is the actual task you need to solve:\n\n"
    else:
        example = ""

    context = CONTEXT_PROMPT.format(needles=needles)
    input_text = TOTAL_PROMPT.format(
        context=context,
        problem=problem,
        example=example,
    )

    if args.task_type == "retrieve":
        expected_answer = {"expected_answer": [true_context["Question"]]}
    elif args.task_type == "niah":
        expected_answer = {
            "expected_answer": [
                c["Question"] for c in curr_context if c["random_index"] == true_context["random_index"]
            ]
        }
    elif args.task_type == "solve":
        if args.dataset == "mbpp":
            expected_answer = {
                "task_id": true_context["task_id"],
                "assertion": true_context["assertion"],
                "canonical_solution": true_context["canonical_solution"],
            }
        else:
            expected_answer = {"expected_answer": true_context["Answer"]}

    save_dict = {
        "index": index,
        "question": f"{context}\n\n{example}{problem}",
        **expected_answer,
    }
    return input_text, save_dict


def generate_samples(max_seq_length: int, incremental: int = 10):
    write_jsons = []

    # Estimate tokens per question to determine reasonable upper bound
    sample_input_text, _ = generate_input_output(0, incremental)
    sample_tokens = len(TOKENIZER.text_to_tokens(sample_input_text))
    tokens_per_question = sample_tokens / incremental

    # Let's do 3x to allow for some slack since we can get unlucky due to sampling.
    # NOTE: We should test this for really large sequence lengths to make sure it's reasonable.
    estimated_max_questions = int((max_seq_length / tokens_per_question) * 3)

    # Binary search for optimal haystack size
    lower_bound = incremental
    upper_bound = max(estimated_max_questions, incremental * 2)  # Ensure upper_bound is reasonable

    optimal_num_qs = None

    logger.info(f"Estimated {tokens_per_question:.1f} tokens per question")
    logger.info(f"Starting binary search with bounds: {lower_bound} to {upper_bound}")

    while lower_bound <= upper_bound:
        mid = (lower_bound + upper_bound) // 2
        input_text, save_dict = generate_input_output(0, mid)
        total_tokens = len(TOKENIZER.text_to_tokens(input_text))

        logger.info(f"Testing haystack size: {mid}, resulting tokens: {total_tokens}/{max_seq_length}")

        if total_tokens <= max_seq_length:
            # This size works, can we go larger?
            optimal_num_qs = mid
            lower_bound = mid + 1
        else:
            # Too large, need to go smaller
            upper_bound = mid - 1

    num_qs = optimal_num_qs if optimal_num_qs is not None else incremental
    logger.info(f"Final optimal haystack size (number of questions): {num_qs}")

    if args.num_samples is not None:
        needle_sample = random.sample(list(range(len(needle))), min(len(needle), args.num_samples))
    else:
        needle_sample = list(range(len(needle)))

    # Generate samples
    for index in tqdm(needle_sample):
        used_qs = num_qs
        while True:
            try:
                input_text, save_dict = generate_input_output(index, used_qs)
                length = len(TOKENIZER.text_to_tokens(input_text))
                assert length <= max_seq_length, f"{length} exceeds max_seq_length."
                break
            except AssertionError:
                if used_qs > incremental:
                    used_qs -= incremental
                else:
                    raise

        save_dict["length"] = length
        formatted_output = save_dict
        write_jsons.append(formatted_output)

    return write_jsons


def main():
    output_file = Path(args.output_folder) / "test.jsonl"

    write_jsons = generate_samples(max_seq_length=args.max_seq_length, incremental=max(10, args.fewshot))

    with open(output_file, "wt", encoding="utf-8") as fout:
        for entry in write_jsons:
            fout.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
