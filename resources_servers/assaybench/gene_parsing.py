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
#
# `parse_genes_from_output` and `extract_genes_from_raw_response` (with their regex and
# blocklist) are copied from Genentech's AssayBench reference harness,
# benchmarking/predictions_generation/collect_llm_predictions.py at commit 640c68b
# (https://github.com/Genentech/AssayBench), and remain under its original MIT terms
# (reproduced below). The copies are unmodified apart from formatting and type hints; they are
# vendored because that script is not part of the `assaybench` PyPI package, and the parsing
# it does is part of what the paper's numbers were produced with. Modifications Copyright (c)
# 2026 NVIDIA CORPORATION & AFFILIATES and contributors, licensed under the Apache License 2.0
# (SPDX header above).
#
# MIT License
#
# Copyright (c) 2026 Genentech, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Reading a ranked gene list back out of a model response, the way the reference harness did.

The reference harness ran every LLM through DSPy's ``ChainOfThought`` and read the ``answer``
field out of the ``[[ ## answer ## ]]`` section of the reply; ``parse_dspy_completion`` does that
split. The two gene parsers are the harness's own, vendored verbatim (see the header): the strict
comma-split with an HGNC-shaped regex for the ``answer`` field, and the multi-strategy scan of a raw
reply it fell back to when DSPy could not parse the completion at all.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional


# ──────────────────────────────────────────────────────────
# DSPy ChatAdapter completion format
# ──────────────────────────────────────────────────────────

# `dspy.adapters.chat_adapter.field_header_pattern` (dspy 3.3.1).
FIELD_HEADER_PATTERN = re.compile(r"\[\[ ## (\w+) ## \]\]")

# `ChainOfThought(RankingSignature)` output fields, in order. DSPy's parser demands every one
# of them; a reply with an `answer` section but no `reasoning` section is a parse failure.
DSPY_OUTPUT_FIELDS = ("reasoning", "answer")


def parse_dspy_completion(completion: str, output_fields=DSPY_OUTPUT_FIELDS) -> Optional[Dict[str, str]]:
    """Split a completion into DSPy output fields; ``None`` if DSPy's parser would have raised.

    Mirrors ``dspy.adapters.chat_adapter.ChatAdapter.parse`` for ``str`` fields: lines are
    grouped under the ``[[ ## name ## ]]`` header that precedes them, the first section for a field
    wins, text before any header is ignored, and every output field must be present. The
    ``[[ ## completed ## ]]`` marker is just a header that closes the last section, so a reply that
    omits it still parses.
    """
    sections: List[tuple[Optional[str], List[str]]] = [(None, [])]
    for line in completion.splitlines():
        match = FIELD_HEADER_PATTERN.match(line.strip())
        if match:
            remaining = line[match.end() :].strip()
            sections.append((match.group(1), [remaining] if remaining else []))
        else:
            sections[-1][1].append(line)

    fields: Dict[str, str] = {}
    for name, lines in sections:
        if name in output_fields and name not in fields:
            fields[name] = "\n".join(lines).strip()

    if set(fields) != set(output_fields):
        return None
    return fields


# ──────────────────────────────────────────────────────────
# Vendored from collect_llm_predictions.py (see header)
# ──────────────────────────────────────────────────────────

# Compiled pattern for validating HGNC-like gene symbols:
# Starts with uppercase letter, followed by uppercase letters/digits/hyphens, 2-15 chars total.
# e.g., TP53, BRCA1, HLA-A, CDKN2A, C1orf43
_GENE_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9][-A-Z0-9]{0,13}$")

# Blocklist of common uppercase tokens that pass the gene regex but are NOT genes.
# These are database column names, metadata labels, file format tokens, etc. that
# can appear in agentic model output (e.g., DepMap column headers, format keywords).
_NON_GENE_BLOCKLIST = frozenset(
    {
        # DepMap / database column name fragments
        "ID", "RRID", "CCLE", "CCLEN", "COSMICID", "BROAD", "WTSIM",
        "TCGACODE", "DEPMAP", "SANGER", "ENTREZ",
        # Common abbreviations / format tokens
        "CSV", "TSV", "JSON", "HTML", "HTTP", "HTTPS", "URL", "URI",
        "PDF", "PNG", "JPG", "GIF", "SVG",
        "NULL", "NONE", "TRUE", "FALSE", "NAN", "INF",
        "AND", "NOT", "THE", "FOR", "WITH", "FROM", "INTO",
        "GENE", "GENES", "SYMBOL", "NAME", "TYPE", "CODE",
        "FDR", "PVALUE", "LOG", "MEAN", "STD", "VAR",
        "RNA", "DNA", "MRNA", "CDNA",  # biological terms, not gene symbols
        "CRISPR", "CRISPRI", "CRISPRN", "KNOCKOUT", "KO",
        "MODEL", "DATA", "INDEX", "ROW", "COL", "COLUMN",
        "FILE", "PATH", "DIR",
    }
)  # fmt: skip


def parse_genes_from_output(output_text: Optional[str]) -> List[str]:
    """Parse gene list from model output.

    Splits by comma and validates each token looks like an HGNC gene symbol.
    Tokens that are too long, contain lowercase words, or otherwise don't
    match the gene pattern are filtered out.  Also filters out known
    non-gene tokens (database column names, format keywords, etc.).
    """
    # Handle None or empty output (e.g., from truncated responses)
    if output_text is None or not isinstance(output_text, str):
        return []

    genes = []
    for token in output_text.split(","):
        token = token.strip()
        if not token:
            continue
        # If the token is a clean gene symbol, keep it directly
        if _GENE_SYMBOL_RE.match(token):
            if token not in _NON_GENE_BLOCKLIST:
                genes.append(token)
        else:
            # Try to extract a gene symbol from within the token
            # (handles cases like "PAFAH1B1\end{solution}..." or "1. TP53")
            m = _GENE_SYMBOL_RE.search(token) if len(token) < 50 else None
            if m and m.group() not in _NON_GENE_BLOCKLIST:
                genes.append(m.group())
            # Otherwise skip -- not a gene symbol
    return genes


def extract_genes_from_raw_response(raw_text: Optional[str]) -> List[str]:
    """
    Extract gene list from raw LM response text, handling agentic model output
    that includes markdown, code blocks, tool output, etc.

    Strategy order (highest to lowest confidence):
      0. Parse the "answer" field from a \\begin{solution} JSON block (biomni-specific)
      1. Find the longest comma-separated gene list line in the response
      2. Find numbered lists (1. TP53, 2. BRCA1, ...)
      3. Collect all gene-like tokens from the entire response

    Args:
        raw_text: Raw response text from the LM

    Returns:
        List of gene symbols, or empty list if none found
    """
    if not raw_text or not isinstance(raw_text, str):
        return []

    gene_re = re.compile(r"[A-Z][A-Z0-9][-A-Z0-9]{0,13}")

    def _extract_gene(token: str) -> Optional[str]:
        """Extract a gene symbol from a token, filtering out blocklisted terms."""
        m = gene_re.search(token)
        if m and m.group() not in _NON_GENE_BLOCKLIST:
            return m.group()
        return None

    # Strategy 0: Extract the "answer" field from \begin{solution}...\end{solution}
    # Biomni wraps its final answer in this block as JSON with "answer" key.
    # This is the most authoritative source -- use it if available.
    solution_match = re.search(r"\\begin\{solution\}\s*(.*?)\s*\\end\{solution\}", raw_text, re.DOTALL)
    if solution_match:
        solution_text = solution_match.group(1).strip()
        # Try to parse as JSON and extract the "answer" field
        try:
            solution_json = json.loads(solution_text)
            if isinstance(solution_json, dict) and "answer" in solution_json:
                answer_text = solution_json["answer"]
                tokens = [t.strip() for t in answer_text.split(",")]
                gene_tokens = [g for t in tokens for g in [_extract_gene(t)] if g]
                if len(gene_tokens) >= 5:
                    return gene_tokens
        except (json.JSONDecodeError, ValueError, TypeError):
            # JSON parse failed (e.g., truncated answer). Try regex extraction
            # from the answer field value directly.
            answer_match = re.search(r'"answer"\s*:\s*"([^"]*)', solution_text)
            if answer_match:
                answer_text = answer_match.group(1)
                tokens = [t.strip() for t in answer_text.split(",")]
                gene_tokens = [g for t in tokens for g in [_extract_gene(t)] if g]
                if len(gene_tokens) >= 5:
                    return gene_tokens

    # Strategy 1: Find lines that look like comma-separated gene lists
    # (at least 5 comma-separated gene-like tokens on one line)
    # Uses search within each token to handle leading/trailing noise
    best_genes: List[str] = []
    for line in raw_text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("```"):
            continue
        tokens = [t.strip() for t in line.split(",")]
        gene_tokens = [g for t in tokens for g in [_extract_gene(t)] if g]
        if len(gene_tokens) >= 5 and len(gene_tokens) > len(best_genes):
            best_genes = gene_tokens

    if best_genes:
        return best_genes

    # Strategy 2: Find numbered lists like "1. TP53\n2. BRCA1\n..."
    numbered_raw = re.findall(r"^\s*\d+[\.\)]\s*([A-Z][A-Z0-9][-A-Z0-9]{0,13})\b", raw_text, re.MULTILINE)
    numbered = [g for g in numbered_raw if g not in _NON_GENE_BLOCKLIST]
    if len(numbered) >= 5:
        return numbered

    # Strategy 3: Collect all gene-like tokens from the entire response
    gene_pattern = r"\b[A-Z][A-Z0-9][-A-Z0-9]{0,13}\b"
    all_genes = re.findall(gene_pattern, raw_text)
    seen = set()
    unique_genes = []
    for g in all_genes:
        if g not in seen and len(g) >= 2 and g not in _NON_GENE_BLOCKLIST:
            seen.add(g)
            unique_genes.append(g)
    if len(unique_genes) >= 5:
        return unique_genes

    return []
