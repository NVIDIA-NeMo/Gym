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
and Content blocks.

Multiple choice over public docs does not by itself require retrieval. A model that
already knows the docs answers without searching, and RL then drops the tool.

# Configs

| config | retrieval | reward |
|---|---|---|
| `langchain_docs_qa.yaml` | BM25 | option letter |
| `langchain_docs_qa_bm25_judge.yaml` | BM25 | judge |
| `langchain_docs_qa_mcp_mcqa.yaml` | MCP | option letter |
| `langchain_docs_qa_mcp_judge.yaml` | MCP | judge |

Judge configs read `judge_cred_file`, or `OPENAI_BASE_URL` and `OPENAI_API_KEY`.

# Example usage

## Running servers

```bash
gym env start \
    --model-type vllm_model \
    --resources-server langchain_docs_qa
```

## Collecting rollouts

```bash
gym eval run --no-serve \
    --agent langchain_docs_qa_agent \
    --input environments/langchain_docs_qa/data/example.jsonl \
    --output results/langchain_docs_qa_rollouts.jsonl \
    --limit 5
```

## Preparing data

See `environments/langchain_docs_qa`, which holds `prepare.py` and the example data.

# Licensing information

Code: Apache 2.0

Data:
- LangChain documentation: MIT

Dependencies:
- nemo_gym: Apache 2.0
