<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Indic MBPP paired smoke evidence

Run date: 2026-10-11. This is a six-example integration check, not a full
benchmark result or an estimate of multilingual model quality.
The initial run and inspected outputs are preserved below; a separate
[post-rebase check](#post-rebase-validation) records the fresh follow-up run.

## Configuration and provenance

Used [smoke.yaml](smoke.yaml) through `gym eval run`, the shared English MBPP
prompt, `simple_agent`, the vLLM model adapter, and the real SciCodePile verifier.
Inputs were freshly built with the updated preparation code for task IDs 11 and
367 in `en`, `hi`, and `kn`; no cached model responses or reference solutions
were used for generation. Task 11 exercises string manipulation; task 367
exercises candidate-defined classes used by the original test setup.

- Model: `google/gemma-4-31B-it`.
- Model snapshot: `842da3794eaa0b77d5f08bae87a17459d91ff475`.
- Source: `ai4bharat/indic-mbpp`, `test.parquet`.
- Recorded source revision: `64e7f7cecd2a6b66a0bcbc8c4c02200d40ae520c`.
- Source SHA-256: `d945fec4af126e0d00d31c9dff015ec93054ee9ba2f5f5b44b7f9ee267ba5ecb`.
- Prepared six-row JSONL SHA-256: `c7746ca6654c92e17271be0d38200dad23b2183c597435faca589a243221a6db`.
- Generation: thinking on; temperature 1.0; top-p 0.95; top-k 64; seed 0;
  maximum 8,192 output tokens, including reasoning; one attempt; concurrency 3.
- Runtime: vLLM 0.21.0, two H100 GPUs, BF16 weights, FP8 KV cache, 65,536-token
  context, tensor parallelism 2, eager execution, prefix caching off.
- Verification: all three original assertions, unchanged 120-second subprocess
  timeout; challenge assertions excluded. Model reasoning is not executed.

The online source refresh returned a repository/access error with the available
credentials. This run used a previously downloaded local snapshot, checking its
SHA-256 against its saved download provenance before preparing new inputs. It
does not claim to validate current upstream HEAD. The revision above records
this run only; preparation still downloads the default revision without a pin.
Authorized dataset access is required to reproduce the source download.

Equivalent model-server launch (use an authorized copy of the model snapshot):

```bash
vllm serve google/gemma-4-31B-it \
  --revision 842da3794eaa0b77d5f08bae87a17459d91ff475 \
  --served-model-name google/gemma-4-31B-it --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 2 --dtype bfloat16 --kv-cache-dtype fp8 \
  --max-model-len 65536 --gpu-memory-utilization 0.85 --max-num-seqs 32 \
  --enable-chunked-prefill --max-num-batched-tokens 2048 \
  --enforce-eager --no-enable-prefix-caching --reasoning-parser gemma4 \
  --limit-mm-per-prompt '{"image":0,"video":0,"audio":0}' \
  --generation-config vllm --override-generation-config '{"eos_token_id":[1,106,50]}' \
  --trust-remote-code
```

See the [README commands](README.md#paired-real-model-smoke) for input preparation
and rollout collection. The live Gym configuration was inspected to confirm the
sampling overrides, thinking flag, and output limit survived composition.

## Saved results

| Returned task ID | Reward | Verifier status | Output tokens |
| --- | ---: | --- | ---: |
| `Mbpp/en/11` | 1 | pass | 1,725 |
| `Mbpp/en/367` | 1 | pass | 2,737 |
| `Mbpp/hi/11` | 1 | pass | 1,361 |
| `Mbpp/hi/367` | 1 | pass | 1,568 |
| `Mbpp/kn/11` | 1 | pass | 1,068 |
| `Mbpp/kn/367` | 1 | pass | 2,028 |

Coverage: 6/6 returned results, 2/2 per language. Every response had status
`completed`, a separate `reasoning` item and final `message`, a null
`failure_reason`, and reward 1. No response reached the output-token cap;
the harness-failure metric was 0 and the failed-request artifact was empty.
The saved `task_id` preserves the language without an input-file join.
Reasoning-token counts were not separately reported by this endpoint; output
tokens above include the full generated response.

## Inspected final outputs

The following code was extracted from the model's final answers, not its
reasoning. Blank-line whitespace is normalized for display. All six final
answers were inspected; these three illustrate the English/Indic comparison
and the candidate-defined-class setup case.

### English, task 11: reward 1

Problem: remove the first and last occurrence of a character from a string.
The prompt's example is `assert remove_Occ("hello","l") == "heo"`.
The model returns `"heo"` for that example and passes both withheld assertions.

```python
def remove_Occ(s, char):
    first = s.find(char)
    if first == -1:
        return s
    last = s.rfind(char)
    if first == last:
        return s[:first] + s[first+1:]
    return s[:first] + s[first+1:last] + s[last+1:]
```

### Hindi, task 11: reward 1

The actual description in the saved prompt was:

> स्ट्रिंग से दिए गए character के पहले और अंतिम occurrence को हटाने के लिए एक python फ़ंक्शन लिखें।

The wrapper and example assertion were identical to the English task. This
answer also produces `"heo"` and passes the same three assertions.

```python
def remove_Occ(s, char):
    first = s.find(char)
    last = s.rfind(char)

    if first == -1:
        return s
    if first == last:
        # Only one occurrence exists
        return s[:first] + s[first + 1:]

    # Remove first and last occurrences
    return s[:first] + s[first + 1:last] + s[last + 1:]
```

### Kannada, task 367: reward 1

The actual description in the saved prompt was:

> ಒಂದು ಬೈನರಿ ಟ್ರೀ ಬ್ಯಾಲೆನ್ಸ್ಡ್ ಆಗಿದೆಯೇ ಅಥವಾ ಇಲ್ಲವೇ ಎಂದು ಪರಿಶೀಲಿಸಲು ಒಂದು ಫಂಕ್ಷನ್ ಅನ್ನು ಬರೆಯಿರಿ.

The prompt also included the original tree setup and first assertion,
`assert is_tree_balanced(root) == False`. The generated `Node` class was available
when the verifier ran that setup. All three original assertions passed.

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

def is_tree_balanced(root):
    def check_height(node):
        # Base case: an empty node has height 0
        if node is None:
            return 0

        # Recursively find the height of the left subtree
        left_height = check_height(node.left)
        if left_height == -1:
            return -1  # Left subtree is not balanced

        # Recursively find the height of the right subtree
        right_height = check_height(node.right)
        if right_height == -1:
            return -1  # Right subtree is not balanced

        # If the difference between heights is greater than 1, it's unbalanced
        if abs(left_height - right_height) > 1:
            return -1

        # Return the height of the current node
        return max(left_height, right_height) + 1

    # If check_height returns -1, the tree is unbalanced
    return check_height(root) != -1
```

## Additional verifier checks

All 500 reference solutions from the recorded source snapshot passed through
the updated prepared English rows and the real verifier. An intentionally
incorrect task-11 solution (`def remove_Occ(s, ch): return s`) received reward 0
with `AssertionError`, not a harness failure. Preparing all 15 languages yielded
7,500 unique qualified task IDs with unchanged tests and entry points.

The CI-collected preparation tests cover provenance and config composition.
The SciCodePile response-roundtrip tests cover both reward 1 and reward 0 for
English/Hindi IDs; restoring the old language-less IDs makes those tests fail.

## Post-rebase validation

Repeated on 2026-10-11 after rebasing onto upstream `main` at
`4a9236e48c82f0dc23365a4c0088b1e54d00d3a8` and adding the explicit
`indic_mbpp_environment_server` relay, matching the current benchmark layout.
Used the same six-row input, model snapshot, and generation settings recorded
above, with fresh model requests rather than reused responses. The live Gym
configuration confirmed the sampling overrides, thinking flag, 8,192-token
limit, and relay reference to `indic_mbpp_simple_agent`, with
`error_on_agent_without_environment_server: true`.

| Returned task ID | Reward | Verifier status | Output tokens |
| --- | ---: | --- | ---: |
| `Mbpp/en/11` | 1 | pass | 1,454 |
| `Mbpp/en/367` | 1 | pass | 3,104 |
| `Mbpp/hi/11` | 1 | pass | 1,393 |
| `Mbpp/hi/367` | 1 | pass | 2,859 |
| `Mbpp/kn/11` | 1 | pass | 1,070 |
| `Mbpp/kn/367` | 1 | pass | 2,975 |

All six responses completed below the token cap, retained their language-qualified
IDs, and had null `failure_reason` values. All final answers were inspected:
the task-11 answers remove the first/last matching character with string slices;
the task-367 answers define `Node` and use recursive subtree-height checks.
All original assertions passed. The aggregate harness-failure metric was 0,
the failed-request artifact was empty, and the evaluation process exited 0.
The optional telemetry-based rollout-health report classified all six records
as `unobserved` because no captures were recorded; this is not a rollout-health
check pass.

Also reran the 500 reference solutions and incorrect-solution control described
above, all 114 preparation/SciCodePile tests, and all five `legacy_agent` tests.
Both config-composition tests now check the named relay and require strict
environment wiring. All-files pre-commit checks passed. The full repository CI
suite was not run locally.
