# Description

Question answering over the LangChain, LangSmith and LangGraph documentation. The agent
searches with `search_docs`, then answers with a JSON object:

```json
{"answer": "<letter A/B/C/D>", "cited_pages": ["<page path used>"]}
```

Each task carries the gold option letter, the gold answer text and the `gold_page` that
supports it. `prepare.py` writes multiple-choice rows when the question set has options
and free-form rows otherwise, matching `reward_mode`. Retrieval is a
local BM25 index or the live `docs.langchain.com/mcp`, both returning Title, Link, Page
and Content blocks. Reward is either exact option-letter match or an LLM equivalence
judge. Server code and the BM25 and MCP variants live in
`resources_servers/langchain_docs_qa`.

Multiple choice over public docs does not by itself require retrieval. A model that
already knows the docs answers without searching, and RL then drops the tool.

# Example usage

## Running servers

```bash
gym env start \
    --model-type vllm_model \
    --environment langchain_docs_qa
```

## Collecting rollouts

```bash
gym eval run --no-serve \
    --agent langchain_docs_qa_simple_agent \
    --input environments/langchain_docs_qa/data/example.jsonl \
    --output results/langchain_docs_qa_rollouts.jsonl \
    --limit 5
```

## Preparing data

```bash
python environments/langchain_docs_qa/prepare.py \
    --download \
    --raw-data-dir /path/to/data/langchain-docs \
    --questions /path/to/questions.jsonl
```

`prepare.py` chunks the docs into `data/chunks.jsonl` for BM25 and converts a JSONL of
`question`, `gold_answer` and `gold_page` records into task rows. Add `options` and
`gold_letter` to get multiple-choice rows. MCP retrieval needs no corpus.

# Licensing information

Code: Apache 2.0

Data:
- LangChain documentation: MIT

Dependencies:
- nemo_gym: Apache 2.0
