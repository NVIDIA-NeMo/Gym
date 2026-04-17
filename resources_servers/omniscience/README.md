# Omniscience Resources Server

Evaluates factual knowledge using an LLM judge on the [AA-Omniscience](https://huggingface.co/datasets/ArtificialAnalysis/AA-Omniscience-Public) benchmark.

## Verification

The server sends the model's response and the gold answer to an LLM judge, which grades on a four-tier scale:

| Grade | Meaning | Reward |
|-------|---------|--------|
| A: CORRECT | Answer matches or is equivalent to gold target | 1.0 |
| B: INCORRECT | Answer contradicts gold target | 0.0 |
| C: PARTIAL_ANSWER | Accurate but insufficient detail | 0.0 |
| D: NOT_ATTEMPTED | Model refused or stated it doesn't know | 0.0 |

## Metrics

- **reward**: Binary (1.0 = CORRECT, 0.0 = anything else)
- **omniscience_index**: `is_correct - is_incorrect` (per sample: +1, 0, or -1)
- **is_hallucination**: 1.0 when INCORRECT (confident wrong answer)

## Data Format

Each JSONL row must have:
```json
{
  "question": "...",
  "expected_answer": "...",
  "domain": "...",
  "topic": "..."
}
```
