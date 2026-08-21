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
import math
import random
import uuid
from pathlib import Path

import nltk
import numpy as np
import wonderwords
from tqdm import tqdm

from .tokenizer import select_tokenizer


try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt")
    nltk.download("punkt_tab")
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--output_folder", type=str)
parser.add_argument("--tokenizer_type", type=str, default="hf", help="[Options] nemo, hf, openai.")
parser.add_argument("--tokenizer_path", type=str, required=True, help="path to the tokenizer model")
parser.add_argument(
    "--max_seq_length",
    type=int,
    required=True,
    help="max sequence length including all input tokens and generated tokens.",
)
parser.add_argument("--num_samples", type=int, required=True, help="number of samples to generate")
parser.add_argument("--random_seed", type=int, default=42)

# Complexity Configurations
parser.add_argument("--num_needle_k", type=int, default=1)
parser.add_argument("--num_needle_v", type=int, default=1)
parser.add_argument("--num_needle_q", type=int, default=1)
parser.add_argument("--type_haystack", type=str, default="needle", help="[Options] needle.")
parser.add_argument("--type_needle_k", type=str, default="words", help="[Options] numbers, words, uuids.")
parser.add_argument("--type_needle_v", type=str, default="numbers", help="[Options] numbers, words, uuids.")
parser.add_argument("--num_digits_k", type=int, default=7)
parser.add_argument("--num_digits_v", type=int, default=7)

args = parser.parse_args()
random.seed(args.random_seed)
np.random.seed(args.random_seed)
args.num_needle_k = max(args.num_needle_k, args.num_needle_q)

# Load Tokenizer
TOKENIZER = select_tokenizer(args.tokenizer_type, args.tokenizer_path)

TEMPLATE_SINGLE = """A special magic {type_needle_v} is hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat is the special magic {type_needle_v} for {query} mentioned in the provided text? The special magic {type_needle_v} for {query} mentioned in the provided text is"""
TEMPLATE_MULTIPLE = """Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize them. I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat are all the special magic {type_needle_v} for {query} mentioned in the provided text? The special magic {type_needle_v} for {query} mentioned in the provided text are"""

# Define Needle/Haystack Format
needle = "One of the special magic {type_needle_v} for {key} is: {value}."
if args.type_haystack == "needle":
    haystack = needle
else:
    raise NotImplementedError(f"{args.type_haystack} is not implemented.")

# Words
nouns = wonderwords.random_word._get_words_from_text_file("nounlist.txt")
adjs = wonderwords.random_word._get_words_from_text_file("adjectivelist.txt")
words = [f"{adj}-{noun}" for adj in adjs for noun in nouns]
words = sorted(list(set(words)))

# Positions
DEPTHS = list(np.round(np.linspace(0, 100, num=40, endpoint=True)).astype(int))


def generate_random_number(num_digits=7):
    lower_bound = 10 ** (num_digits - 1)
    upper_bound = 10**num_digits - 1
    return str(random.randint(lower_bound, upper_bound))


def generate_random_word():
    word = random.choice(words)
    return word


def generate_random_uuid():
    return str(uuid.UUID(int=random.getrandbits(128), version=4))


def generate_random(type_needle: str, digits: int | None = None):
    if type_needle == "numbers":
        if digits is None:
            raise ValueError("digits must be provided when type_needle='numbers'")
        return generate_random_number(digits)
    elif type_needle == "words":
        return generate_random_word()
    elif type_needle == "uuids":
        return generate_random_uuid()
    else:
        raise NotImplementedError(f"{type_needle} is not implemented.")


def generate_input_output(num_haystack):
    keys, values, needles = [], [], []
    for _ in range(args.num_needle_k):
        keys.append(generate_random(args.type_needle_k, args.num_digits_k))
        value = []
        for _ in range(args.num_needle_v):
            value.append(generate_random(args.type_needle_v, args.num_digits_v))
            needles.append(
                needle.format(
                    type_needle_v=args.type_needle_v,
                    key=keys[-1],
                    value=value[-1],
                )
            )
        values.append(value)

    random.shuffle(needles)

    # Context
    if args.num_needle_v == 1:
        sentences = [
            haystack.format(
                type_needle_v=args.type_needle_v,
                key=generate_random(args.type_needle_k, args.num_digits_k),
                value=generate_random(args.type_needle_v, args.num_digits_v),
            )
            for _ in range(num_haystack)
        ]
    else:
        haystack_values = [generate_random(args.type_needle_v, args.num_digits_v) for _ in range(num_haystack)]
        haystack_keys = (
            [
                generate_random(args.type_needle_k, args.num_digits_k)
                for _ in range(math.ceil(num_haystack / args.num_needle_v))
            ]
            * args.num_needle_v
        )[:num_haystack]
        sentences = [
            haystack.format(
                type_needle_v=args.type_needle_v,
                key=haystack_keys[i],
                value=haystack_values[i],
            )
            for i in range(num_haystack)
        ]
        random.shuffle(sentences)

    indexes = sorted(random.sample(range(num_haystack), len(needles)), reverse=True)
    for index, element in zip(indexes, needles):
        sentences.insert(index, element)
    context = "\n".join(sentences)

    ## Query and Answer
    indices = random.sample(range(args.num_needle_k), args.num_needle_q)
    queries = [keys[i] for i in indices]
    answers = [a for i in indices for a in values[i]]
    query = ", ".join(queries[:-1]) + ", and " + queries[-1] if len(queries) > 1 else queries[0]

    if args.num_needle_q * args.num_needle_v == 1:
        template = TEMPLATE_SINGLE
        type_needle_v = args.type_needle_v[:-1]  # remove "s"
    else:
        template = TEMPLATE_MULTIPLE
        type_needle_v = args.type_needle_v

    input_text = template.format(
        type_needle_v=type_needle_v,
        context=context,
        query=query,
    )

    return input_text, answers


def generate_samples(num_samples: int, max_seq_length: int, incremental: int = 500):
    write_jsons = []

    if args.type_haystack == "needle":
        incremental = max(5, args.num_needle_v * args.num_needle_k)

    if args.max_seq_length < 4096:
        incremental = 5

    # Estimate tokens per question to determine reasonable upper bound
    sample_input_text, _ = generate_input_output(incremental)
    sample_tokens = len(TOKENIZER.text_to_tokens(sample_input_text))
    tokens_per_haystack = sample_tokens / incremental

    # Let's do 3x to allow for some slack since we can get unlucky due to sampling.
    # NOTE: We should test this for really large sequence lengths to make sure it's reasonable.
    estimated_max_questions = int((max_seq_length / tokens_per_haystack) * 3)

    # Binary search for optimal haystack size
    lower_bound = incremental
    upper_bound = max(estimated_max_questions, incremental * 2)  # Ensure upper_bound is reasonable

    optimal_num_haystack = None

    logger.info(f"Estimated {tokens_per_haystack:.1f} tokens per haystack")
    logger.info(f"Starting binary search with bounds: {lower_bound} to {upper_bound}")
    while lower_bound <= upper_bound:
        mid = (lower_bound + upper_bound) // 2
        input_text, _answers = generate_input_output(mid)
        total_tokens = len(TOKENIZER.text_to_tokens(input_text))

        logger.info(f"Testing haystack size: {mid}, resulting tokens: {total_tokens}/{max_seq_length}")

        if total_tokens <= max_seq_length:
            # This size works, can we go larger?
            optimal_num_haystack = mid
            lower_bound = mid + 1
        else:
            # Too large, need to go smaller
            upper_bound = mid - 1

    num_haystack = optimal_num_haystack if optimal_num_haystack is not None else incremental
    logger.info(f"Final optimal haystack size (number of haystack): {num_haystack}")

    # Generate samples
    for index in tqdm(range(num_samples)):
        used_haystack = num_haystack
        while True:
            try:
                input_text, answer = generate_input_output(used_haystack)
                length = len(TOKENIZER.text_to_tokens(input_text))
                assert length <= max_seq_length, f"{length} exceeds max_seq_length."
                break
            except AssertionError:
                if used_haystack > incremental:
                    used_haystack -= incremental
                else:
                    raise

        formatted_output = {
            "index": index,
            "question": input_text,
            "expected_answer": answer,
            "length": length,
        }
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
