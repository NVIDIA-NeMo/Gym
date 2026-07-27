# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime preflight checks for Tau3 banking_knowledge retrieval modes."""

from __future__ import annotations

import argparse
import os
import shutil
import sys


SANDBOX_TOOLS = ("srt", "rg", "bwrap", "socat")
OPENAI_CONFIGS = {
    "alltools",
    "openai_embeddings",
    "openai_embeddings_grep",
    "openai_embeddings_reranker",
    "openai_embeddings_reranker_grep",
}
OPENROUTER_CONFIGS = {
    "alltools-qwen",
    "qwen_embeddings",
    "qwen_embeddings_grep",
    "qwen_embeddings_reranker",
    "qwen_embeddings_reranker_grep",
}
RERANKER_CONFIGS = {
    "bm25_reranker",
    "bm25_reranker_grep",
    "openai_embeddings_reranker",
    "openai_embeddings_reranker_grep",
    "qwen_embeddings_reranker",
    "qwen_embeddings_reranker_grep",
}
SANDBOX_CONFIGS = {
    "terminal_use",
    "terminal_use_write",
    "alltools",
    "alltools-qwen",
}
OFFLINE_CONFIGS = {
    "no_knowledge",
    "full_kb",
    "golden_retrieval",
    "grep_only",
    "bm25",
    "bm25_grep",
}
SUPPORTED_CONFIGS = sorted(OPENAI_CONFIGS | OPENROUTER_CONFIGS | RERANKER_CONFIGS | SANDBOX_CONFIGS | OFFLINE_CONFIGS)


def check_retrieval_config(retrieval_config: str) -> list[str]:
    failures = []

    if retrieval_config in SANDBOX_CONFIGS:
        missing = [tool for tool in SANDBOX_TOOLS if shutil.which(tool) is None]
        if missing:
            failures.append(f"missing required sandbox tools: {', '.join(missing)}")

    if retrieval_config in OPENAI_CONFIGS and not os.environ.get("OPENAI_API_KEY"):
        failures.append("missing OPENAI_API_KEY for OpenAI embeddings")

    if retrieval_config in OPENROUTER_CONFIGS and not os.environ.get("OPENROUTER_API_KEY"):
        failures.append("missing OPENROUTER_API_KEY for OpenRouter/Qwen embeddings")

    if retrieval_config in RERANKER_CONFIGS and not os.environ.get("OPENAI_API_KEY"):
        failures.append("missing OPENAI_API_KEY for pointwise LLM reranker")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("retrieval_config", choices=SUPPORTED_CONFIGS)
    args = parser.parse_args()

    failures = check_retrieval_config(args.retrieval_config)
    if failures:
        print(f"banking_knowledge/{args.retrieval_config} runtime preflight failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(f"banking_knowledge/{args.retrieval_config} runtime preflight passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
