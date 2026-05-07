# skill_judge

LLM-as-judge resources server that grades a list of behavioral assertions in a
single LLM call and returns per-assertion binary grades plus an aggregate
reward (fraction satisfied).

Built for the skill-evaluation methodology described in [agentskills.io](https://agentskills.io/specification)
and the NeMo Gym skill-eval harness, but works on any `(prompt, response,
tool_calls, assertions)` tuple.

## Endpoints

- `POST /judge` — direct grading. Body: `JudgeRequest`. Response: `JudgeResponse` (grades, reward, usage, parse_error).
- `POST /verify` — standard Gym flow. Extracts `assertions`, `tool_calls`, and
  `expected_output` from `verifier_metadata`. Response: `AssertionGradeResponse`
  (subclass of `BaseVerifyResponse` with `grades`, `judge_usage`, `parse_error`).

## Robustness

- Judge output is parsed as a JSON array. Markdown code fences are stripped.
  If the model emits prose around the array, the outermost `[...]` is extracted.
- Missing grades for expected assertion ids default to `satisfied=false` with a
  note in `evidence`.
- Unparseable output yields `reward=0.0`, `parse_error` set, and all assertions
  marked unsatisfied.
- Evidence is truncated to 200 characters per assertion.

# Licensing information
Code: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0
