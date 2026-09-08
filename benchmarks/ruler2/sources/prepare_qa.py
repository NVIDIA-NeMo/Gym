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
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from .tokenizer import select_tokenizer


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
parser.add_argument("--num_samples", type=int, default=500)
parser.add_argument("--dataset", type=str, required=True, help="dataset file")
parser.add_argument("--fewshot", type=int, default=0)
parser.add_argument("--prompt_type", type=str, default="chat")
parser.add_argument("--query_type", type=str, default="id", choices=["id", "doc", "question"])
parser.add_argument("--algo_type", type=str, default="single", choices=["single", "2steps"])
parser.add_argument("--task_type", type=str, default="retrieve", choices=["retrieve", "solve"])

args = parser.parse_args()
random.seed(args.random_seed)
np.random.seed(args.random_seed)

# Load Tokenizer
TOKENIZER = select_tokenizer(args.tokenizer_type, args.tokenizer_path)


TOTAL_PROMPT = """{context}\n\n{example}{problem}"""
NEEDLE_PROMPT = "Document {i}:\n{text}"
if args.task_type == "retrieve":
    if args.query_type == "id":
        CONTEXT_PROMPT = """Below are some documents. I will ask you to copy one of them. Please copy and paste the document you find.\n\n{needles}"""
        PROBLEM_PROMPT = "Please copy the Document {i} from the context."
        if args.fewshot > 0:
            EXAMPLE_PROMPT = PROBLEM_PROMPT + "\nDocument {i}:\n{text}"
        if args.prompt_type == "base":
            PROBLEM_PROMPT += "\nDocument {i}:"

    elif args.query_type == "doc" or args.query_type == "question":
        if args.query_type == "doc":
            CONTEXT_PROMPT = """Below are some documents. I will give you a text at the end. Please find the document index of the text. Only give me the index without any document contents.\n\n{needles}"""
            PROBLEM_PROMPT = "Text: {text}\nMost relevant document index:"
        elif args.query_type == "question":
            # CONTEXT_PROMPT = """Below are some documents. I will give you a text at the end. Please find the document index most relevant to the text. Only give me the index without any document contents.\n\n{needles}"""
            # PROBLEM_PROMPT = "Text: {question}\nMost relevant document index:"
            CONTEXT_PROMPT = """Below are some documents. I will give you a question at the end. Please find the index of the most relevant document that can help answer the question. Only give me the index without any document contents.\n\n{needles}"""
            PROBLEM_PROMPT = (
                "Question: {question}\nIndex of the most relevant document that can help answer the question:"
            )
        if args.fewshot > 0:
            EXAMPLE_PROMPT = PROBLEM_PROMPT + " {i}"
elif args.task_type == "solve":
    CONTEXT_PROMPT = """Below are some documents. I will ask you to answer a question based on the documents. Please answer the question.\n\n{needles}"""
    if args.algo_type == "single":
        PROBLEM_PROMPT = "Please answer the following question based on the documents.\n\nQuestion: {question}"
        # PROBLEM_PROMPT = "Please answer the following question based on the documents. If the question is a yes/no question, please only answer yes or no. If your answer can be extracted from the documents, please directly copy the answer span as short as possible. Please put your answer inside \\boxed{{}}.\n\nQuestion: {question}"
    elif args.algo_type == "2steps":
        PROBLEM_PROMPT = "Please first find and copy paste the documents relevant to the following question and then answer it based on the documents you find.\n\nQuestion: {question}"
    if args.fewshot > 0:
        EXAMPLE_PROMPT = PROBLEM_PROMPT + "\nAnswer: {answer}"
    if args.prompt_type == "base":
        PROBLEM_PROMPT += "\nAnswer:"


# Read SQuAD QA dataset
def read_squad():
    data = load_dataset("squad_v2")["train"]
    haystack = [d["context"] for d in data]
    haystack = list(set(haystack))
    haystack = [{"text": d} for d in haystack]

    data = load_dataset("squad_v2")["validation"]
    title2context = defaultdict(set)
    for d in data:
        title2context[d["title"]].add(d["context"])

    needle = [
        {
            "question": d["question"],
            "answer": d["answers"]["text"],
            "context": [{"text": d["context"]}],
            "distractor": [{"text": t} for t in title2context[d["title"]] if t != d["context"]],
        }
        for d in data
    ]
    needle = [n for n in needle if len(n["answer"]) > 0]

    return haystack, needle


# Read Hotpot QA dataset
def read_hotpotqa():
    data = load_dataset("hotpotqa/hotpot_qa", "distractor")["train"]
    haystack = [f"{t}\n{''.join(s)}" for d in data for t, s in zip(d["context"]["title"], d["context"]["sentences"])]
    haystack = list(set(haystack))
    haystack = [{"text": d} for d in haystack]

    data = load_dataset("hotpotqa/hotpot_qa", "distractor")["validation"]
    needle = [
        {
            "question": d["question"],
            "answer": [d["answer"]],
            "context": [
                {"text": f"{t}\n{''.join(s)}"}
                for t, s in zip(d["context"]["title"], d["context"]["sentences"])
                if t in d["supporting_facts"]["title"]
            ],
            "distractor": [
                {"text": f"{t}\n{''.join(s)}"}
                for t, s in zip(d["context"]["title"], d["context"]["sentences"])
                if t not in d["supporting_facts"]["title"]
            ],
        }
        for d in data
    ]
    needle = [n for n in needle if len(n["answer"]) > 0]

    return haystack, needle


def read_musique():
    data = load_dataset("dgslibisey/MuSiQue")["train"]
    haystack = [f"{p['title']}\n{p['paragraph_text']}" for d in data for p in d["paragraphs"]]
    haystack = list(set(haystack))
    haystack = [{"text": d} for d in haystack]

    data = load_dataset("dgslibisey/MuSiQue")["validation"]
    needle = [
        {
            "question": d["question"],
            "answer": [d["answer"]] + d["answer_aliases"],
            "context": [
                {"text": f"{p['title']}\n{p['paragraph_text']}"} for p in d["paragraphs"] if p["is_supporting"]
            ],
            "distractor": [
                {"text": f"{p['title']}\n{p['paragraph_text']}"} for p in d["paragraphs"] if not p["is_supporting"]
            ],
        }
        for d in data
        if d["answerable"]
    ]
    needle = [n for n in needle if len(n["answer"]) > 0]

    return haystack, needle


# Download dataset
if args.dataset == "squad":
    haystack, needle = read_squad()
elif args.dataset == "hotpotqa":
    haystack, needle = read_hotpotqa()
elif args.dataset == "musique":
    haystack, needle = read_musique()
else:
    raise NotImplementedError(f"{args.dataset} is not implemented.")


def generate_random_number(num_digits=7):
    lower_bound = 10 ** (num_digits - 1)
    upper_bound = 10**num_digits - 1
    return str(random.randint(lower_bound, upper_bound))


def generate_input_output(index, num_docs):
    curr_needle = dict(needle[index])
    curr_needle["context"] = [{**c, "random_index": generate_random_number()} for c in curr_needle["context"]]
    curr_needle["distractor"] = [{**c, "random_index": generate_random_number()} for c in curr_needle["distractor"]]

    if args.fewshot > 0:
        fewshot_examples = random.sample([i for i in range(len(needle)) if i != index], args.fewshot)
        fewshot_examples = [dict(needle[i]) for i in fewshot_examples]
        for e in fewshot_examples:
            e["context"] = [{**c, "random_index": generate_random_number()} for c in e["context"]]
            e["distractor"] = [{**c, "random_index": generate_random_number()} for c in e["distractor"]]
    else:
        fewshot_examples = []

    remaining_haystack_size = len(haystack) - len(
        set(
            [c["text"] for c in (curr_needle["context"] + curr_needle["distractor"])]
            + [f["text"] for e in fewshot_examples for f in (e["context"] + e["distractor"])]
        )
    )
    if remaining_haystack_size <= 0:
        raise ValueError("No remaining haystack documents available after exclusions.")

    if num_docs > remaining_haystack_size:
        repeats = (num_docs + remaining_haystack_size - 1) // remaining_haystack_size  # Ceiling division
    else:
        repeats = 1

    curr_context = random.sample([i for i in range(len(haystack)) for _ in range(repeats)], num_docs)
    curr_context = [{**haystack[i], "random_index": generate_random_number()} for i in curr_context]
    curr_context = (
        curr_context + curr_needle["context"] + [item for example in fewshot_examples for item in example["context"]]
    )
    if num_docs > 0:
        curr_context = (
            curr_context
            + curr_needle["distractor"]
            + [item for example in fewshot_examples for item in example["distractor"]]
        )
    random.shuffle(curr_context)

    needles = "\n\n".join([NEEDLE_PROMPT.format(i=c["random_index"], text=c["text"]) for c in curr_context])
    if args.task_type == "retrieve":
        if args.query_type == "id":
            problem = PROBLEM_PROMPT.format(i=curr_needle["context"][0]["random_index"])
        elif args.query_type == "doc":
            problem = PROBLEM_PROMPT.format(text=curr_needle["context"][0]["text"])
        elif args.query_type == "question":
            problem = PROBLEM_PROMPT.format(question=curr_needle["question"])
    elif args.task_type == "solve":
        problem = PROBLEM_PROMPT.format(question=curr_needle["question"])

    if args.fewshot > 0:
        if args.task_type == "retrieve":
            if args.query_type == "id":
                example = "\n\n".join(
                    [
                        EXAMPLE_PROMPT.format(i=e["context"][0]["random_index"], text=e["context"][0]["text"])
                        for e in fewshot_examples
                    ]
                )
            elif args.query_type == "doc":
                example = "\n\n".join(
                    [
                        EXAMPLE_PROMPT.format(i=e["context"][0]["random_index"], text=e["context"][0]["text"])
                        for e in fewshot_examples
                    ]
                )
            elif args.query_type == "question":
                example = "\n\n".join(
                    [
                        EXAMPLE_PROMPT.format(
                            i=random.sample(e["context"], 1)[0]["random_index"], question=e["question"]
                        )
                        for e in fewshot_examples
                    ]
                )

        elif args.task_type == "solve":
            example = "\n\n".join(
                [
                    EXAMPLE_PROMPT.format(answer=random.sample(e["answer"], 1)[0], question=e["question"])
                    for e in fewshot_examples
                ]
            )

        if args.prompt_type == "base":
            example = f"{example}\n\n"
        else:
            example = f"Here are some examples to help you understand the task:\n\n{example}\n\nHere is the actual task you need to solve:\n\n"

    else:
        example = ""

    context = CONTEXT_PROMPT.format(needles=needles)
    input_text = TOTAL_PROMPT.format(context=context, problem=problem, example=example)
    if args.task_type == "retrieve":
        if args.query_type == "id":
            expected_answer = {"expected_answer": [curr_needle["context"][0]["text"]]}
        elif args.query_type == "doc":
            expected_answer = {"expected_answer": [curr_needle["context"][0]["random_index"]]}
        elif args.query_type == "question":
            expected_answer = {"expected_answer": [c["random_index"] for c in curr_needle["context"]]}
    elif args.task_type == "solve":
        expected_answer = {"expected_answer": curr_needle["answer"]}

    save_dict = {
        "index": index,
        "question": f"{context}\n\n{example}{problem}",
        **expected_answer,
    }
    return input_text, save_dict


def generate_samples(num_samples: int, max_seq_length: int, incremental: int = 5):
    write_jsons = []

    # Estimate tokens per question to determine reasonable upper bound
    sample_input_text, _ = generate_input_output(0, incremental)
    sample_tokens = len(TOKENIZER.text_to_tokens(sample_input_text))
    tokens_per_doc = sample_tokens / incremental

    if max_seq_length > 0:
        # Let's do 3x to allow for some slack since we can get unlucky due to sampling.
        # NOTE: We should test this for really large sequence lengths to make sure it's reasonable.
        estimated_max_docs = int((max_seq_length / tokens_per_doc) * 3)

        # Binary search for optimal haystack size
        lower_bound = incremental
        upper_bound = max(estimated_max_docs, incremental * 2)  # Ensure upper_bound is reasonable

        optimal_num_docs = None

        logger.info(f"Estimated {tokens_per_doc:.1f} tokens per doc")
        logger.info(f"Starting binary search with bounds: {lower_bound} to {upper_bound}")

        while lower_bound <= upper_bound:
            mid = (lower_bound + upper_bound) // 2
            input_text, save_dict = generate_input_output(0, mid)
            total_tokens = len(TOKENIZER.text_to_tokens(input_text))

            logger.info(f"Testing haystack size: {mid}, resulting tokens: {total_tokens}/{max_seq_length}")

            if total_tokens <= max_seq_length:
                # This size works, can we go larger?
                optimal_num_docs = mid
                lower_bound = mid + 1
            else:
                # Too large, need to go smaller
                upper_bound = mid - 1

        num_docs = optimal_num_docs if optimal_num_docs is not None else incremental
        logger.info(f"Final optimal haystack size (number of docs): {num_docs}")
    else:
        num_docs = 0

    # Generate samples
    for index in tqdm(range(num_samples)):
        used_docs = num_docs
        while True:
            try:
                input_text, save_dict = generate_input_output(index, used_docs)
                length = len(TOKENIZER.text_to_tokens(input_text))
                if max_seq_length > 0:
                    assert length <= max_seq_length, f"{length} exceeds max_seq_length."
                break
            except AssertionError:
                if used_docs > incremental:
                    used_docs -= incremental
                else:
                    raise

        save_dict["length"] = length
        formatted_output = save_dict
        write_jsons.append(formatted_output)

    return write_jsons


def main():
    output_file = Path(args.output_folder) / "test.jsonl"

    write_jsons = generate_samples(
        num_samples=args.num_samples,
        max_seq_length=args.max_seq_length,
    )
    with open(output_file, "wt", encoding="utf-8") as fout:
        for entry in write_jsons:
            fout.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
