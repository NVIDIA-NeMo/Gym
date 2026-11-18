# BIRD-Interact

[BIRD-Interact](https://bird-interact.github.io/) ([paper](https://arxiv.org/abs/2510.05318)) evaluates interactive
text-to-SQL: the request is ambiguous, part of the knowledge base is hidden, and the agent must ask a simulated user
before it writes PostgreSQL. This recipe runs its Lite and Full releases on the
[`bird_interact` Environment Server](../../environment_servers/bird_interact/README.md), which describes the two
modes, the user simulator, the rewards and the grading. Its c-Interact follows the BIRD-Interact ADK: its protocol,
its simulator and its grading, as [auto-ontology-eval](https://github.com/NVIDIA/auto-ontology-eval) runs it. The server's own five-task example runs without this
recipe.

| Config | Release | Mode | Tasks |
|---|---|---|---|
| `config.yaml` | Lite: 18 databases | c-Interact: the agent asks and submits with two tools, and the server opens each turn | 300 |
| `a_interact.yaml` | Lite | a-Interact: the agent calls tools, `ask` among them | 300 |
| `full.yaml` | Full: 22 databases | c-Interact | 600 |
| `full_a_interact.yaml` | Full | a-Interact | 600 |

Every task has a follow-up question, its phase 2. Both kinds of task run. As in BIRD-Interact, a query (195 in Lite,
410 in Full) is graded by comparing its result with the solution's, even when its ground truth has test cases. A
management task (105 in Lite, 190 in Full) is graded by its own test cases, in the episode's sandbox. auto-ontology-eval
runs only the query tasks, so compare its scores with the query tasks' alone.

## 1. Build the database image

Each episode starts its own sandbox from this image. Its build restores every database of the release, from
BIRD-Interact's official image, so an episode starts PostgreSQL instead of restoring dumps, and a write in one episode
never reaches another. The build fails if a table fails to import. Lite builds in under a minute once the base image
is pulled, and is 1.1 GB.

```bash
docker build --platform linux/amd64 --tag benchmarks/bird_interact:lite benchmarks/bird_interact
docker build --platform linux/amd64 --tag benchmarks/bird_interact:full \
  --build-arg BIRD_INTERACT_IMAGE=docker.io/shawnxxh/bird-interact-postgresql-full@sha256:de4b88f6f3211238318a3dc265914f4e38122ec67ece642bbf3df0c3b781040c \
  benchmarks/bird_interact
```

The configs use these tags (`bird_interact_image`), which the Docker sandbox provider finds locally. For a remote
provider, push the image to a registry it pulls from and pass `+bird_interact_image=<registry>/<image>:<tag>`.

## 2. Get the ground truth

The public releases withhold the solution SQL and the test cases. Email
[bird.bench25@gmail.com](mailto:bird.bench25@gmail.com) with `[bird-interact-lite GT&Test Cases]` or
`[bird-interact-full GT&Test Cases]` in the subject, as the
[BIRD-Interact repository](https://github.com/bird-bench/BIRD-Interact#dataset-uses) asks, and the file is sent
automatically.

Put the file in `benchmarks/bird_interact/data/`, where `prepare.py` reads it by default:
`bird_interact_gt_kg_testcases_1008.jsonl` for Lite and `bird_interact_full_gt_kg_testcases_08022.jsonl` for Full.
`data/.gitignore` keeps it out of the repository: the authors withhold the ground truth, so never publish it.

## 3. Prepare

`env.yaml` needs the policy model (`policy_base_url`, `policy_api_key`, `policy_model_name`) and the simulator
(`user_model_base_url`, `user_model_api_key`, `user_model_name`); preparing reads the config too, so set them first.

```bash
gym eval prepare --benchmark bird_interact
```

To read the ground truth from another place, add `+prepare_script_args.ground_truth_jsonl=<file>`.

`prepare.py` downloads the release from Hugging Face (`birdsql/bird-interact-lite` or `-full`): its rows, and for each
database the schema, column meanings and knowledge base, into `data/<release>/databases/`, the servers'
`database_dir`. It merges the ground truth into the rows by `instance_id`, for the task and its follow-up, and writes
`data/<release>/<mode>.jsonl` with the tasks the server can grade. It prints how many it kept and why it left the
others out. For another config, pass `--config benchmarks/bird_interact/<config>` in place of `--benchmark`. It
downloads the dataset revision the ground truth files belong to (`RELEASE_REVISIONS`).

## 4. Run

```bash
gym eval run --benchmark bird_interact --model-type vllm_model --split benchmark \
  --config nemo_gym/sandbox/providers/docker/configs/docker.yaml \
  --output results/bird_interact_lite_c.jsonl
```

- **Simulator.** The server runs the ADK's simulator. For scores comparable with auto-ontology-eval's, set
  `user_model_name` to the ADK's default simulator model, `anthropic/claude-haiku-4-5-20251001` (BIRD-Interact's own
  harness uses `gemini-2.0-flash-001`). An endpoint that returns a reasoning field with the reply, as some serve
  Claude Haiku 5.5, also needs `user_model_uses_reasoning_parser=true`: with the parser off, the model server refuses
  the reply. The simulator reads only the reply's text, and its token limits (500 and 1024) were set for a model that
  does not think, so turn thinking off where the endpoint allows it, for example
  `'++user_model.responses_api_models.vllm_model.extra_body={thinking:{type:disabled}}'`. Run the policy at
  temperature 0 with at most 4096 output tokens, as the ADK runs its agent; the server's README says where to set them.
- **Harnesses.** c-Interact needs an agent that resumes its session across turns and takes tool grants, such as
  `simple_agent`; a-Interact an agent that takes tool grants. The server's README lists which harnesses run each mode,
  and how to point the a-Interact server at an MCP-only harness such as Codex.
- **Outputs hold the ground truth.** Each row the run writes, such as `results/*_materialized_inputs.jsonl`, carries
  the task's solution SQL and test cases. Keep the run's outputs private, as the ground truth itself.

## Results

None published yet.

## Known issues

In 24 Lite tasks, a solution SQL does not run when a submission is graded. The server grades that submission as wrong,
as the ADK's `ex_base` returns 0 when the solution fails, and lists the error in the rollout's `solution_errors`. No
agent can pass that phase, so leave these tasks out when you compare against a score that excludes them.

| Cause | Tasks (phase) |
|---|---|
| The ADK's `DISTINCT` removal, which Gym copies, turns `DISTINCT ON (...)` into `ON (...)` and `IS DISTINCT FROM` into `IS FROM` | `crypto_10` (2), `disaster_1` (2), `fake_5` (1, 2), `insider_M_4` (1, 2), `insider_M_6` (1, 2), `mental_4` (2), `mental_5` (1, 2), `mental_6` (1, 2), `mental_11` (1, 2), `museum_3` (2), `museum_5` (1, 2) |
| The solution is not valid SQL: a comma before `FROM` (`archeology_7`, `fake_7`), a stray word (`alien_8`), a row where a number is needed (`cross_db_M_5`), a table that does not exist (`museum_7`) | `alien_8` (2), `archeology_7` (1), `cross_db_M_5` (2), `fake_7` (2), `museum_7` (2) |
| Phase 2's solution uses a view, table or function that phase 1's solution creates; it fails when the agent's phase 1 SQL passed but created it with another name, columns or types | `archeology_M_3`, `crypto_M_2`, `disaster_M_2`, `disaster_M_3`, `gaming_M_2`, `mental_M_2`, `polar_M_4`, `polar_M_5` (2) |

The first two causes fail every submission of that phase; the third only after some phase 1 submissions. Gym keeps
the ADK's normalization, so its scores stay comparable with the ADK's and auto-ontology-eval's.

## Licensing

BIRD-Interact's data and databases: CC BY-SA 4.0. The image's base and its dumps come from BIRD-Interact
(`docker.io/shawnxxh/bird-interact-postgresql`, `-full`).
