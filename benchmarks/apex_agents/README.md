# APEX-Agents

Long-horizon professional-services tasks from [APEX-Agents](https://huggingface.co/datasets/mercor/apex-agents):
an agent works inside a simulated workplace (files, mail, chat, calendar, spreadsheets, documents, presentations,
PDFs, code execution) and delivers a console answer or an edited or new file. Each task is graded against a held-out
rubric.

The harness is part of what is measured: the benchmark runs `responses_api_agents/apex_agent` (a Stirrup model/tool
loop inside a pinned Archipelago world image) and scores with `resources_servers/apex_agents`. Their READMEs hold
the component details; this file records the measurement contract.

## Provenance

| | |
| --- | --- |
| Source | `mercor/apex-agents` on Hugging Face (gated; accept the terms with the account that runs preparation) |
| License | CC-BY-4.0; the dataset owner restricts it to evaluation use |
| Tasks | 452 of the 480 published tasks. Two worlds are excluded because they need external EDGAR/FMP data that the public offline world artifacts do not include (`EXCLUDED_WORLD_IDS` in `prepare.py`) |
| World runtime | Archipelago at revision `0cb5c476c219a9df637e0bd37fb86b2361f4ab89`, built into the sandbox image |
| Grading prompt | Archipelago's criterion grader at commit `b581241ed67deed520f8d7854d20e39569e259f5` |

The dataset revision is not pinned by `prepare.py`; record the revision you prepared from next to any reported
number. The port was checked against revision `92c86856cf1b11f9833a8a076b3a45a63afa3929`.

## Task contract

- **Model-visible input:** the task prompt only. The rubric, expected output type and gold response stay under
  `verifier_metadata` and are never written into the task sandbox.
- **State:** one world ZIP per task, restored into the sandbox before the rollout. Preparation downloads every world
  ZIP into `data/world_cache`; rollouts read that cache offline.
- **Tools:** Stirrup's `list_tools`, `inspect_tool`, `add_tool`, `remove_tool`, `todo_write` and `finish`; the
  workplace MCP tools become callable once the model adds them.
- **Submission:** `finish` is the only way to submit. A rollout that ends without a completed `finish` (turn limit,
  timeout, `status="incomplete"`) is not graded and scores 0.
- **Artifacts the verifier reads:** the final answer plus the initial and final snapshots of `filesystem/` and
  `.apps_data/`.

## Scoring and run conditions

- **Reward:** 1.0 when every rubric criterion is judged met, otherwise 0.0 (`all_pass`). Higher is better. The
  aggregate is the mean reward over tasks and repeats.
- **Judge:** one LLM call per criterion over the changed deliverables (extracted text, plus rendered pages when
  LibreOffice is available). The judge model is configuration (`judge_model_name`, default `gemini-3-flash`);
  scores from different judges are not comparable.
- **Defaults:** 200 model turns, 32,768 output tokens per turn, temperature 1.0, top_p 1.0, agent timeout 12,600 s.
- **Wall-clock sensitivity:** the timeout is wall-clock, so the pass rate depends on model-serving throughput and
  tool latency. Keep concurrency, serving capacity and `apex_agent_mcp_session_affinity` fixed across the models
  you compare.
- **Repeats:** the dataset declares one repeat. Rollouts are stochastic; use several repeats for a comparison.

## Data preparation

Put a Hugging Face read token for an account that accepted the dataset terms in the root `env.yaml` as `hf_token`,
then:

```bash
gym eval prepare --benchmark apex_agents
```

This writes `data/apex_agents_benchmark.jsonl` and fills `data/world_cache`. Both are gitignored.

## Running

```bash
gym eval run --benchmark apex_agents --model-type vllm_model \
    --split benchmark --limit 1 --num-repeats 1 \
    --output results/apex_agents_smoke.jsonl
```

The agent needs Apptainer on the host and the Archipelago image; `responses_api_agents/apex_agent/README.md`
describes the image build and the sandbox options.
