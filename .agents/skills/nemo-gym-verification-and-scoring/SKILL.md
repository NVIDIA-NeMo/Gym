---
name: nemo-gym-verification-and-scoring
description: >-
  Designs, implements, and validates NeMo Gym verification and scoring, including
  deterministic checks, execution/state verification, LLM-as-judge, and reward
  models. Use when the user asks to "write verifier", "implement verify", "reward
  function", "scoring logic", "LLM as judge", "reward model", "verification
  strategy", "how to verify", "design verifier", or "reward signal".
---

# Verification and Scoring

Assume the user knows the behavior they want to reward, but needs help turning it into a reliable Gym verifier. Define correctness before writing code.

## Choose the strongest available evidence

Prefer verifier patterns in this order when they genuinely measure the task:

1. exact/equivalence checks for canonical answers;
2. execution or final-state checks for code, tools, databases, and interactive tasks;
3. constrained rubric-based LLM judging for semantic outcomes;
4. a trained reward model when deterministic evidence and a judge rubric are insufficient.

Combine signals only when each component has a clear meaning. Do not use an LLM judge merely because parsing is inconvenient.

Read [Build verifiers](../../../fern/versions/latest/pages/build-verifiers/index.mdx), [Verification patterns](../../../fern/versions/latest/pages/build-verifiers/verification-patterns/index.mdx), and [LLM as judge](../../../fern/versions/latest/pages/build-verifiers/verification-patterns/llm-as-judge.mdx).

## Specify the reward contract

Write down:

- what observable output or final state constitutes success;
- what privileged task metadata the verifier may use;
- score range and whether rewards are binary, graded, or multi-component;
- treatment of malformed, empty, timed-out, and partially correct attempts;
- invariances such as whitespace, ordering, equivalent representations, or harmless extra text;
- anti-shortcut and contamination checks.

Keep privileged metadata outside `responses_create_params.input`.

## Implement in Gym

- Put reusable environment scoring in the resources server's `verify()` method.
- Validate the request with an explicit Pydantic model when task metadata has structure.
- Return a valid `BaseVerifyResponse` and preserve required request fields.
- Read final session state when success depends on actions, not the last message.
- Treat model-caused malformed output as a scored failure with useful metadata, not a server crash.
- Surface infrastructure/judge outages separately; do not silently turn them into reward zero.
- Bound external execution, judge calls, and subprocesses with explicit timeout and concurrency behavior.
- For multi-reward output, follow [Multi-reward verification](../../../fern/versions/latest/pages/build-verifiers/multi-reward-verification.mdx).

## LLM-as-judge requirements

- Give the judge the task, candidate result, ground truth/rubric, and an explicit output schema.
- Avoid exposing irrelevant identity or ordering signals.
- Parse judge output defensively.
- Separate judge transport failure from a negative judgment.
- Calibrate on labeled passing, failing, borderline, adversarial, and malformed examples.
- Measure agreement and inspect disagreement; a sophisticated prompt is not validation.

## Validate quality

Test:

- clear pass and clear fail;
- empty and malformed output;
- equivalent valid forms;
- boundary/partial-credit cases;
- adversarial shortcuts and reward hacking;
- state isolation across concurrent sessions;
- judge or execution timeout/failure.

Then profile real rollouts:

- inspect false positives and false negatives;
- compare against human labels or executable ground truth where available;
- ensure a capable model produces a meaningful, non-saturated distribution;
- verify the reward ordering matches the intended behavior;
- document known blind spots.

Do not mark the environment verified or use the reward for training until verifier errors can be distinguished from agent failures.
