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
"""Prepare BEAM (Beyond a Million Tokens) data for NeMo Gym.

Source: https://huggingface.co/datasets/Mohammadta/BEAM (splits ``100K``,
``500K``, ``1M``; CC BY-SA 4.0). Paper: https://arxiv.org/abs/2510.27246.

Prompt formats
--------------
``single_turn`` (default): the whole chat is rendered as a ``User: ... /
Assistant: ...`` transcript inside one user message, followed by the probing
question -- a direct, single-turn recall task (the setup described for the
Muse Glimmer BEAM-128K report).

``multi_turn``: upstream's ``long-context`` mode
(``src/answer_probing_questions/long_term_memory_methods.py``): the chat is
replayed as alternating user / assistant messages and the question is a final
user turn.

``answer_only_note=True`` (``--answer_only_note``) prefixes the question with
upstream's instruction::

    "\\n            NOTE: Only provide the answer without any explanations. \\n            Question: {question}"

It is off by default: it suppresses the explanations many rubrics check for.
Upstream reproduction is ``--prompt_format multi_turn --answer_only_note``.

One output row per probing question (about 20 per conversation, two per memory
ability).

Length filter
-------------
When ``max_context_tokens`` is set, a question is dropped if its full prompt
(whole chat plus question) is longer than ``max_context_tokens``; nothing is
truncated. Serve the policy with a window of at least ``max_context_tokens`` plus
the max output tokens.

With a HuggingFace tokenizer that has a chat template, the prompt length matches
``apply_chat_template(..., add_generation_prompt=True)`` exactly, including any
default system prompt the template injects: per-role wrapper costs are
calibrated against the template and checked with one full render per
conversation (``--verify_all`` renders every row). With a tiktoken encoding it
uses upstream's estimate (``2 + sum(4 + len(content))``).

Invocation
----------
``gym eval prepare`` calls ``prepare()`` with no arguments (``cl100k_base``, no
length filter, all three splits). To build a custom variant, run the script
directly::

    python benchmarks/beam/prepare.py \\
        --tokenizer_name meta-models/Muse-Glimmer-30B \\
        --max_context_tokens 131072 \\
        --output_fpath benchmarks/beam/data/beam_muse_128k_benchmark.jsonl
"""

import argparse
import ast
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from datasets import load_dataset
from tqdm import tqdm


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
DEFAULT_OUTPUT_FPATH = DATA_DIR / "beam_benchmark.jsonl"

HF_DATASET = "Mohammadta/BEAM"
ALL_SPLITS = ("100K", "500K", "1M")
DEFAULT_TOKENIZER_NAME = "cl100k_base"  # upstream's counter for non-GPT / non-Gemini models

# Verbatim from upstream ``probing_question_evaluation`` (long-context branch).
QUESTION_TEMPLATE = (
    "\n            NOTE: Only provide the answer without any explanations. \n            Question: {question}"
)
# Without upstream's answer-only instruction: the probing question as-is.
PLAIN_QUESTION_TEMPLATE = "{question}"

PROMPT_FORMATS = ("single_turn", "multi_turn")
# single_turn: everything before the question; ends with ":\n" so the question's
# tokens add to the prefix's (the newline closes a pre-token).
SINGLE_TURN_PREFIX = (
    "Below is the full history of my earlier conversation with you.\n\n"
    "<conversation_history>\n{history}\n</conversation_history>\n\n"
    "Based on our conversation history above, answer my question:\n"
)

# Field holding the reference answer differs per ability.
_REFERENCE_FIELDS = ("answer", "ideal_answer", "ideal_response", "ideal_summary", "expected_compliance")

Message = Dict[str, str]


class PromptCounter:
    """Counts prompt tokens for a message list the way the policy server will see it.

    With a chat template, a message costs a fixed per-role wrapper plus its
    content tokens (the template's special tokens split BPE merges), so prompt
    length is additive and calibrated once against ``apply_chat_template``.
    ``exact_count`` renders the template for spot checks.
    """

    def __init__(self, tokenizer_name: str) -> None:
        self.tokenizer_name = tokenizer_name
        self._hf = None
        try:
            import tiktoken

            enc = tiktoken.get_encoding(tokenizer_name)
            self._encode: Callable[[str], int] = lambda text: len(enc.encode(text, disallowed_special=()))
            # Upstream ``total_tokens``: 4 per message + 2 for priming.
            self.role_overhead = {"user": 4, "assistant": 4}
            self.fixed_overhead = 2
            return
        except (ImportError, ValueError):
            pass

        from transformers import AutoTokenizer

        self._hf = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        self._encode = lambda text: len(self._hf.encode(text, add_special_tokens=False))
        if not self.has_chat_template:
            self.role_overhead = {"user": 4, "assistant": 4}
            self.fixed_overhead = 2
            return
        x = self._encode("x")
        user = [{"role": "user", "content": "x"}]
        pair = user + [{"role": "assistant", "content": "x"}]
        one, two, three = self.exact_count(user), self.exact_count(pair), self.exact_count(pair + user)
        self.role_overhead = {"user": three - two - x, "assistant": two - one - x}
        self.fixed_overhead = one - x - self.role_overhead["user"]

    @property
    def has_chat_template(self) -> bool:
        return self._hf is not None and bool(self._hf.chat_template)

    def content_tokens(self, text: str) -> int:
        return self._encode(text)

    def message_cost(self, message: Message) -> int:
        return self.role_overhead[message["role"]] + self._encode(message["content"])

    def exact_count(self, messages: List[Message]) -> int:
        """Prompt length via ``apply_chat_template``; falls back to the additive estimate without one."""
        if not self.has_chat_template:
            return self.fixed_overhead + sum(self.message_cost(m) for m in messages)
        ids = self._hf.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        return len(ids)


def flatten_chat(chat: Sequence[Sequence[dict]]) -> List[Message]:
    """Upstream's long-context replay: every message of every batch, in order, role + content only."""
    return [{"role": m["role"], "content": m["content"]} for batch in chat for m in batch]


def render_transcript(history: List[Message]) -> str:
    """``User: ...`` / ``Assistant: ...`` turns separated by blank lines."""
    return "\n\n".join(("User" if m["role"] == "user" else "Assistant") + ": " + m["content"] for m in history)


def reference_answer(question: dict) -> str:
    for field in _REFERENCE_FIELDS:
        value = question.get(field)
        if value:
            return str(value)
    return ""


def prepare(
    tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
    max_context_tokens: Optional[int] = None,
    splits: Sequence[str] = ALL_SPLITS,
    output_fpath: Path = DEFAULT_OUTPUT_FPATH,
    verify_all: bool = False,
    answer_only_note: bool = False,
    prompt_format: str = "single_turn",
) -> Path:
    """Download BEAM, build one Gym row per probing question that fits, return the output path."""
    output_fpath = Path(output_fpath)
    output_fpath.parent.mkdir(parents=True, exist_ok=True)

    counter = PromptCounter(tokenizer_name)
    if prompt_format not in PROMPT_FORMATS:
        raise ValueError(f"prompt_format must be one of {PROMPT_FORMATS}, got {prompt_format!r}")
    template = QUESTION_TEMPLATE if answer_only_note else PLAIN_QUESTION_TEMPLATE
    print(f"Tokenizer {tokenizer_name} (chat template: {counter.has_chat_template}); max tokens: {max_context_tokens}")

    n_rows = n_dropped = 0
    with output_fpath.open("w", encoding="utf-8") as fout:
        for split in splits:
            dataset = load_dataset(HF_DATASET, split=split)
            for conv in tqdm(dataset, desc=f"BEAM {split}"):
                history = flatten_chat(conv["chat"])
                if prompt_format == "multi_turn":
                    base_tokens = counter.fixed_overhead + sum(counter.message_cost(m) for m in history)
                else:
                    prefix = SINGLE_TURN_PREFIX.format(history=render_transcript(history))
                    base_tokens = counter.fixed_overhead + counter.message_cost({"role": "user", "content": prefix})
                probing = ast.literal_eval(conv["probing_questions"])
                exact = verify_all
                first = True
                for ability, questions in probing.items():
                    for q_index, question in enumerate(questions):
                        question_text = template.format(question=question["question"])
                        if prompt_format == "multi_turn":
                            messages = history + [{"role": "user", "content": question_text}]
                            n_tokens = base_tokens + counter.message_cost(messages[-1])
                        else:
                            messages = [{"role": "user", "content": prefix + question_text}]
                            # The prefix already paid the user-turn wrapper.
                            n_tokens = base_tokens + counter.content_tokens(question_text)
                        # One exact chat-template render per conversation keeps the additive count
                        # honest; on a mismatch, every row of that conversation is counted exactly.
                        if first or exact:
                            n_exact = counter.exact_count(messages)
                            if n_exact != n_tokens and not exact:
                                print(f"WARNING: estimated {n_tokens} tokens but chat template gives {n_exact}")
                                exact = True
                            n_tokens = n_exact
                        first = False
                        if max_context_tokens is not None and n_tokens > max_context_tokens:
                            n_dropped += 1
                            continue
                        row = {
                            "responses_create_params": {"input": messages},
                            "verifier_metadata": {
                                "conversation_id": str(conv["conversation_id"]),
                                "chat_size": split,
                                "ability": ability,
                                "question_index": q_index,
                                "question": question["question"],
                                "rubric": question.get("rubric") or [],
                                "reference_answer": reference_answer(question),
                                "difficulty": question.get("difficulty"),
                            },
                            "prompt_tokens": n_tokens,
                        }
                        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                        n_rows += 1

    print(
        f"Wrote {n_rows} rows to {output_fpath} (splits={list(splits)}, tokenizer={tokenizer_name}, "
        f"max tokens={max_context_tokens}; dropped {n_dropped} over the limit)"
    )
    return output_fpath


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--tokenizer_name",
        default=DEFAULT_TOKENIZER_NAME,
        help="tiktoken encoding name or HuggingFace model id / local path (chat template used if present).",
    )
    parser.add_argument(
        "--max_context_tokens",
        type=int,
        default=None,
        help="Drop questions whose full prompt is longer than this many tokens. Default: no filter.",
    )
    parser.add_argument("--splits", nargs="+", default=list(ALL_SPLITS), choices=ALL_SPLITS)
    parser.add_argument("--output_fpath", type=Path, default=DEFAULT_OUTPUT_FPATH)
    parser.add_argument(
        "--prompt_format",
        choices=PROMPT_FORMATS,
        default="single_turn",
        help="single_turn: chat as a transcript in one user message (default); multi_turn: upstream replay.",
    )
    parser.add_argument(
        "--answer_only_note",
        action="store_true",
        help="Prefix the question with upstream's 'NOTE: Only provide the answer without any explanations.'",
    )
    parser.add_argument(
        "--verify_all", action="store_true", help="Count every row with the chat template (slow on long chats)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    prepare(
        tokenizer_name=args.tokenizer_name,
        max_context_tokens=args.max_context_tokens,
        splits=args.splits,
        output_fpath=args.output_fpath,
        verify_all=args.verify_all,
        answer_only_note=args.answer_only_note,
        prompt_format=args.prompt_format,
    )
