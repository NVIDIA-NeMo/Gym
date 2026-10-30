# SWE Bench

Adds multiple SWE benchmarks to Gym. These are useable across agents/harnesses.

## Details

- Data source: `princeton-nlp/SWE-bench_Verified`

## Example usage

```bash
# Prepare benchmark data
gym eval prepare --config benchmarks/swebench/verified/opencode.yaml

# Running servers
gym env start \
    --model-type vllm_model \
    --config benchmarks/swebench/verified/opencode.yaml

# Collecting rollouts
gym eval run --no-serve \
    --agent mini_swe_agent_2_swebench_verified \
    --input benchmarks/swebench/data/swebench_verified_benchmark.jsonl \
    --output results/swebench_verified/rollouts.jsonl
```

## SWE-bench Pro multi-turn

`benchmarks/swebench/pro_multiturn` splits SWE-bench Pro tasks into a sequence of user requests. The first request is
the prompt; each later one is sent to the agent only after it finishes the previous request. The final repository
state is graded with the task's original fail-to-pass and pass-to-pass tests by the unchanged `swebench_pro`
resources server.

`decompositions.json` maps an instance ID to its turns. Each turn has a short `request` plus 1-based, inclusive
line ranges of the task's original `requirements` and `interface` text, which are appended verbatim. `prepare.py`
rejects a task unless every requirement and interface line lands in exactly one turn, so the turns together carry the
full original spec. Later turns are stored in the row's `followup_user_messages`, which `opencode_sandboxed_agent`
replays into the same OpenCode session with `opencode run --session`.

The agent never sees the original problem statement. Each `request` is a short ask in the user's own words, derived from
the task, that says what the turn is for and leaves the details to its spec lines. It must not lead the policy toward
reward hacking or hand it redundant information. `prepare.py` rejects a request that names an identifier, path or value
absent from the problem statement and the spec lines assigned up to that turn, that mentions tests or grading, that
comments on earlier work, that previews later requests, or that shares a run of eight or more words with the problem
statement or with the spec lines of that turn or an earlier one. Rows whose text is stored as a JSON-quoted string with
escaped newlines are decoded first, so their bullets can be split by line.

The first 10 tasks were split by hand, and two of them are the writer's examples. `generate_decompositions.py` writes
the rest with any OpenAI-compatible model: the writer sees only the problem statement, requirements and interface
(never the tests or the gold patch), every candidate must pass the checks above plus a bullet-integrity check and a
majority of judge samples, and failed attempts are retried with the reasons. `--merge` adds accepted splits to
`decompositions.json`, optionally limited to tasks whose golden patch verifies. `--rewrite` writes new requests for
existing splits with the turn boundaries fixed, under the same checks, and `--merge --replace` overwrites the old
splits.

```bash
python -m benchmarks.swebench.pro_multiturn.generate_decompositions \
    --base-url http://<vllm-host>:8000/v1 --model <served-model> --output generated.jsonl
python -m benchmarks.swebench.pro_multiturn.generate_decompositions \
    --merge --output generated.jsonl --instance-ids golden_pass_ids.txt
python -m benchmarks.swebench.pro_multiturn.generate_decompositions \
    --base-url http://<vllm-host>:8000/v1 --model <served-model> --output rewritten.jsonl --rewrite
python -m benchmarks.swebench.pro_multiturn.generate_decompositions --merge --replace --output rewritten.jsonl
gym eval prepare --config benchmarks/swebench/pro_multiturn/opencode.yaml
```
