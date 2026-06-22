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

# Decontamination of training data against evaluation/test sets.
#
# This is a port of the two-stage LLM decontamination pipeline from NeMo-Skills
# (https://github.com/NVIDIA-NeMo/Skills), which itself follows the lmsys
# methodology (https://lmsys.org/blog/2023-11-14-llm-decontaminator/):
#
#   1. Semantic retrieval: for every prepared training problem, find the top-k
#      most similar problems in the provided test sets using sentence-transformer
#      embeddings + cosine similarity.
#   2. LLM verification: ask an LLM judge whether each (train, test) candidate
#      pair is in fact the same problem.
#   3. Filtering: drop training rows confirmed to overlap a test problem.
#
# The heavy dependencies (torch, transformers) are imported lazily inside
# `_embed_texts` so that importing this module - and running `ng_prepare_data`
# without decontamination enabled - incurs zero additional cost.

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from tqdm.auto import tqdm

from nemo_gym.openai_utils import NeMoGymAsyncOpenAI


LOG = logging.getLogger(__name__)


# Adapted from https://github.com/lm-sys/llm-decontaminator/blob/main/detect_instruct.py
DEFAULT_CONTAMINATION_PROMPT = """Help me determine if the following two problems are the same.

First problem: {problem1}
Second problem: {problem2}

Disregard the names and minor changes in word order that appear within.
If the two problems are very similar and if they produce the same answer, we consider them to be the same problem.
Respond with only "True" (problems are the same) or "False" (problems are different). Do not respond with anything else."""


class DecontaminationConfig(BaseModel):
    """Configuration for decontaminating prepared splits against test sets.

    When this config is omitted from `ng_prepare_data`, decontamination is skipped
    entirely (no embedding model is loaded and no LLM is queried).
    """

    test_set_jsonls: List[str] = Field(
        description="JSONL file paths of the evaluation/test sets to decontaminate against. "
        "Rows may be in NeMo Gym responses format or carry a flat `problem_text_key`."
    )
    decontaminate_types: List[str] = Field(
        default_factory=lambda: ["train"],
        description="Which collated splits (by dataset type) to decontaminate, e.g. ['train', 'validation']. "
        "Each maps to `<output_dirpath>/<type>.jsonl`.",
    )
    problem_text_key: str = Field(
        default="problem",
        description="Fallback flat key to read the problem text from when a row is not in responses format.",
    )

    # Embedding-based retrieval.
    embedding_model: str = Field(
        default="sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        description="HF transformers model id used to embed problems (mean-pooled) for similarity retrieval.",
    )
    top_k: int = Field(default=5, description="Number of nearest test problems retrieved per training problem.")
    batch_size: int = Field(default=2048, description="Batch size for computing embeddings.")
    chunk_size: int = Field(default=10000, description="Chunk size for the pairwise similarity computation.")

    # LLM judge (uses NeMo Gym's OpenAI-compatible client).
    judge_model: str = Field(default="gpt-4o-mini", description="Model name for the LLM contamination judge.")
    judge_base_url: Optional[str] = Field(
        default=None,
        description="OpenAI-compatible base URL for the judge (default: https://api.openai.com/v1).",
    )
    judge_api_key: Optional[str] = Field(
        default=None,
        description="API key for the judge. If unset, read from global config under `judge_api_key_name`.",
    )
    judge_api_key_name: str = Field(
        default="openai_api_key",
        description="Global-config key to read the judge API key from when not set explicitly.",
    )
    judge_temperature: float = Field(default=0.0, description="Sampling temperature for the judge.")
    judge_max_tokens: int = Field(default=16, description="Max tokens for the judge response ('True'/'False').")
    judge_max_concurrency: int = Field(default=32, description="Maximum number of concurrent judge requests.")
    check_both_ways: bool = Field(
        default=False,
        description="If true, query (train, test) and (test, train) orderings and flag contamination if either is True.",
    )
    judge_prompt_template: str = Field(
        default=DEFAULT_CONTAMINATION_PROMPT,
        description="Prompt template with `{problem1}` and `{problem2}` placeholders.",
    )

    # Output behavior.
    report_dirpath: Optional[str] = Field(
        default=None,
        description="Directory for the per-split contamination report JSONL (default: the prepare output dir).",
    )
    dry_run: bool = Field(
        default=False,
        description="If true, write the contamination report but do not filter the prepared splits.",
    )


def _content_to_text(content: Any) -> Optional[str]:
    """Flatten responses-API message content (str or list of content parts) to text."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        joined = "\n".join(parts).strip()
        return joined or None
    return None


def extract_problem_text(row: Dict[str, Any], problem_text_key: str = "problem") -> Optional[str]:
    """Extract the problem text from a data row.

    Handles NeMo Gym responses format (`responses_create_params.input` as a string or a
    list of messages - the first user message is used) and falls back to a flat key.
    """
    rcp = row.get("responses_create_params")
    if isinstance(rcp, dict):
        inputs = rcp.get("input")
        if isinstance(inputs, str):
            text = inputs.strip()
            if text:
                return text
        elif isinstance(inputs, list):
            # Prefer the first user message; otherwise the first message with content.
            for msg in inputs:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    text = _content_to_text(msg.get("content"))
                    if text:
                        return text
            for msg in inputs:
                if isinstance(msg, dict) and "content" in msg:
                    text = _content_to_text(msg.get("content"))
                    if text:
                        return text

    value = row.get(problem_text_key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _read_problem_texts(file_paths: List[str], problem_text_key: str) -> List[str]:
    """Read unique, order-preserving problem texts from the given JSONL files."""
    seen: Dict[str, None] = {}
    for file_path in file_paths:
        with open(file_path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                text = extract_problem_text(json.loads(line), problem_text_key)
                if text:
                    seen.setdefault(text, None)
    return list(seen.keys())


def _embed_texts(texts: List[str], config: DecontaminationConfig) -> "Any":  # pragma: no cover
    """Embed texts with a transformers model using mean pooling + L2 normalization.

    This mirrors how sentence-transformers computes embeddings for MiniLM-style models,
    but depends only on `transformers` + `torch` (not sentence-transformers, which pulls
    in scikit-learn/scipy - excluded from this repo to keep the base install slim).
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.embedding_model)
    model = AutoModel.from_pretrained(config.embedding_model)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    all_embeddings = []
    for start in tqdm(range(0, len(texts), config.batch_size), desc="Embedding"):
        batch = texts[start : start + config.batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**encoded)
        # Mean pooling over tokens, weighted by the attention mask.
        mask = encoded["attention_mask"].unsqueeze(-1).to(output.last_hidden_state.dtype)
        summed = (output.last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        all_embeddings.append(pooled.cpu())
    return torch.cat(all_embeddings, dim=0)


def _retrieve_similar(  # pragma: no cover
    train_texts: List[str], test_texts: List[str], config: DecontaminationConfig
) -> List[Dict[str, Any]]:
    """For each training problem, retrieve the top-k most similar test problems.

    Heavy deps (torch, transformers) are imported here (via `_embed_texts`) so this is
    the only place that pays for them - and only when decontamination is actually run.
    """
    import torch

    LOG.info("Embedding %d test problems...", len(test_texts))
    test_emb = _embed_texts(test_texts, config)
    LOG.info("Embedding %d training problems...", len(train_texts))
    train_emb = _embed_texts(train_texts, config)

    k = min(config.top_k, len(test_texts))
    candidates: List[Dict[str, Any]] = []
    n = train_emb.shape[0]
    for start in tqdm(range(0, n, config.chunk_size), desc="Computing similarity"):
        end = min(start + config.chunk_size, n)
        # Embeddings are L2-normalized, so a dot product is the cosine similarity.
        sim_chunk = train_emb[start:end] @ test_emb.T  # (chunk, num_test)
        topk = torch.topk(sim_chunk, k=k, dim=1)
        for row_i in range(end - start):
            idxs = topk.indices[row_i].tolist()
            scores = topk.values[row_i].tolist()
            candidates.append(
                {
                    "problem": train_texts[start + row_i],
                    "similar_items": [test_texts[j] for j in idxs],
                    "similarity_scores": scores,
                }
            )
    return candidates


async def _judge_pair(
    client: NeMoGymAsyncOpenAI,
    config: DecontaminationConfig,
    problem1: str,
    problem2: str,
    semaphore: asyncio.Semaphore,
) -> str:
    """Query the LLM judge for a single (problem1, problem2) pair."""
    prompt = config.judge_prompt_template.format(problem1=problem1, problem2=problem2)
    async with semaphore:
        result = await client.create_chat_completion(
            model=config.judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.judge_temperature,
            max_tokens=config.judge_max_tokens,
        )
    return result["choices"][0]["message"]["content"]


async def _check_one_candidate(
    client: NeMoGymAsyncOpenAI,
    config: DecontaminationConfig,
    candidate: Dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    """Determine whether a training problem is contaminated by any of its similar test items."""
    problem = candidate["problem"]
    similar_items = candidate["similar_items"]

    # Exact-match short-circuit (mirrors NeMo-Skills' prefill_generation) - no LLM call needed.
    for similar in similar_items:
        if problem.strip().lower() == similar.strip().lower():
            return {**candidate, "contaminated": True, "generations": ["True (exact match)"]}

    pairs: List[Tuple[str, str]] = []
    for similar in similar_items:
        pairs.append((problem, similar))
        if config.check_both_ways:
            pairs.append((similar, problem))

    generations = await asyncio.gather(*[_judge_pair(client, config, p1, p2, semaphore) for p1, p2 in pairs])
    contaminated = any(g.strip().startswith("True") for g in generations)
    return {**candidate, "contaminated": contaminated, "generations": list(generations)}


async def _check_contamination_async(
    candidates: List[Dict[str, Any]], config: DecontaminationConfig, api_key: str
) -> List[Dict[str, Any]]:
    client = NeMoGymAsyncOpenAI(
        base_url=config.judge_base_url or "https://api.openai.com/v1",
        api_key=api_key,
    )
    semaphore = asyncio.Semaphore(config.judge_max_concurrency)
    return await asyncio.gather(
        *[_check_one_candidate(client, config, candidate, semaphore) for candidate in candidates]
    )


def _filter_split(data_path: Path, contaminated_keys: set, problem_text_key: str) -> Tuple[int, int]:
    """Rewrite `data_path` in place, dropping rows whose problem text is contaminated.

    Returns (kept, removed).
    """
    tmp_path = data_path.with_name(f"{data_path.stem}_decontaminated.jsonl")
    kept, removed = 0, 0
    with open(data_path, "rt", encoding="utf-8") as fin, open(tmp_path, "wt", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = extract_problem_text(row, problem_text_key)
            if text is not None and text.strip().lower() in contaminated_keys:
                removed += 1
                continue
            fout.write(f"{json.dumps(row)}\n")
            kept += 1
    tmp_path.replace(data_path)
    return kept, removed


def run_decontamination(config: "Any", output_dir: Path) -> None:
    """Decontaminate the collated splits produced by ng_prepare_data.

    `config` is a TrainDataProcessorConfig whose `.decontamination` field holds a
    DecontaminationConfig (or a dict). Called only when that field is set.
    """
    dconf = config.decontamination
    if dconf is None:
        return
    if not isinstance(dconf, DecontaminationConfig):
        dconf = DecontaminationConfig.model_validate(dconf)

    # Resolve the judge API key from the global config if not provided explicitly.
    api_key = dconf.judge_api_key
    if not api_key:
        from nemo_gym.global_config import get_global_config_dict

        global_config = get_global_config_dict()
        api_key = global_config.get(dconf.judge_api_key_name, "")
    if not api_key:
        LOG.warning(
            "No judge API key found (set `decontamination.judge_api_key` or `%s` in env). "
            "Proceeding - this will only work for endpoints that do not require auth.",
            dconf.judge_api_key_name,
        )

    test_texts = _read_problem_texts(dconf.test_set_jsonls, dconf.problem_text_key)
    LOG.info("Loaded %d unique problems from %d test set file(s).", len(test_texts), len(dconf.test_set_jsonls))
    if not test_texts:
        LOG.warning("No test-set problems found; skipping decontamination.")
        return

    report_dir = Path(dconf.report_dirpath) if dconf.report_dirpath else Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    for dtype in dconf.decontaminate_types:
        data_path = Path(output_dir) / f"{dtype}.jsonl"
        if not data_path.exists():
            LOG.warning("Collated split '%s' not found at %s; skipping.", dtype, data_path)
            continue

        train_texts = _read_problem_texts([str(data_path)], dconf.problem_text_key)
        if not train_texts:
            LOG.warning("No problems extracted from %s; skipping.", data_path)
            continue

        candidates = _retrieve_similar(train_texts, test_texts, dconf)
        decisions = asyncio.run(_check_contamination_async(candidates, dconf, api_key))

        report_path = report_dir / f"{dtype}_contamination.jsonl"
        with open(report_path, "wt", encoding="utf-8") as f:
            for decision in decisions:
                f.write(f"{json.dumps(decision)}\n")

        contaminated_keys = {d["problem"].strip().lower() for d in decisions if d["contaminated"]}
        LOG.info(
            "[%s] %d/%d unique problems flagged as contaminated (%.2f%%). Report: %s",
            dtype,
            len(contaminated_keys),
            len(train_texts),
            100 * len(contaminated_keys) / max(1, len(train_texts)),
            report_path,
        )

        if dconf.dry_run:
            LOG.info("[%s] dry_run=True; leaving %s unmodified.", dtype, data_path)
            continue

        kept, removed = _filter_split(data_path, contaminated_keys, dconf.problem_text_key)
        LOG.info("[%s] Removed %d contaminated rows, kept %d. Updated %s", dtype, removed, kept, data_path)
