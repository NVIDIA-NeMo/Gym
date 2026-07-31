# lc_niah

Long-context needle-in-a-haystack (NIAH) rule-based resources server for
graphwalks-style tasks (BFS / parent-finding over a large directed graph). It grades a
response on two signals at once:

1. **Answer correctness** (`answer_score`) — the final answer (the assistant's
   `output_text`) is parsed for a `Final Answer: [..]` node list and scored as F1
   against `expected_answer` (a JSON list of nodes).
2. **Reasoning/input overlap** — the model's reasoning (the `reasoning` item's
   summary text) should have *small* overlap with the input message, i.e. it should
   not just copy the prompt back into its chain of thought. Three independent
   overlap signals are computed (each in `[0, 1]`, higher = more copying):
   - `overlap_seq_match` — `difflib.SequenceMatcher` ratio (global similarity)
   - `overlap_ngram16` — fraction of the reasoning's 16-grams that appear in the input
   - `overlap_lcs` — longest common substring length / reasoning length

## Reward

Two config knobs decide the reward:

- **`overlap_metric_rule`** — which overlap signal becomes the single `reasoning_overlap`
  penalty: `seq_match`, `ngram16`, or `lcs` (default `lcs`).
- **`overlap_grading_rule`** — how `answer_score` and `reasoning_overlap` combine:

  | rule | reward |
  |------|--------|
  | `base` | `answer_score` |
  | `multiply` (default) | `answer_score * (1 - reasoning_overlap)` |
  | `minus` | `answer_score - reasoning_overlap` |

Under the default `multiply`, the overlap only matters once the answer is correct (a
wrong answer scores `0`), so verbatim copying of the prompt into the reasoning is
penalized even when the answer is right. `base` ignores the overlap entirely and
rewards answer correctness alone.

The verify response always exposes `answer_score`, the three `overlap_*` signals, and
the combined `reasoning_overlap` for inspection, regardless of which rules are active.

## Config

See `configs/lc_niah.yaml`.

## Dataset

Each JSONL row needs `responses_create_params.input` (the graphwalks prompt — graph
edges plus the BFS/parents operation, ending with a `Final Answer: [..]` instruction)
and an `expected_answer` (a JSON list of node ids, passed to the verifier). See
`data/example.jsonl` for small synthetic smoke-test rows.

Train / validation datasets are not committed to git — they are fetched from the
GitLab dataset registry via a `gitlab_identifier` in the config.

# Licensing information
Code: Apache-2.0
Data: example data is synthetic / illustrative.

Dependencies
- nemo_gym: Apache 2.0
