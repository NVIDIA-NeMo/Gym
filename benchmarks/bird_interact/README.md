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
| `auto_ontology_agent.yaml`, `full_auto_ontology_agent.yaml` | Lite, Full | c-Interact with Auto Ontology's ReAct agent; see [Auto Ontology](#auto-ontology) | 300, 600 |

Every task has a follow-up question, its phase 2. Both kinds of task run: query tasks (195 in Lite, 410 in Full) and
management tasks (105 in Lite, 190 in Full). As in BIRD-Interact, each phase is graded by its own category: a query
phase by comparing its result with the solution's, a management phase by its own test cases, in the episode's
sandbox. Many management tasks have a query follow-up. auto-ontology-eval runs only the query tasks, so compare its
scores with the query tasks' alone.

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
  harness uses `gemini-2.0-flash-001`). An endpoint that returns a reasoning field with the reply also needs
  `user_model_uses_reasoning_parser=true`: with the parser off, the model server refuses
  the reply. The simulator reads only the reply's text, and its token limits (500 and 1024) were set for a model that
  does not think, so turn thinking off where the endpoint allows it, for example
  `'++user_model.responses_api_models.vllm_model.extra_body={thinking:{type:disabled}}'`. Run the policy at
  temperature 0 with at most 4096 output tokens, as the ADK runs its agent; the server's README says where to set them.
- **Harnesses.** c-Interact needs an agent that resumes its session across turns and takes tool grants, such as
  `simple_agent`; a-Interact an agent that takes tool grants. The server's README lists which harnesses run each mode,
  and how to point the a-Interact server at an MCP-only harness such as Codex.
- **Outputs hold the ground truth.** Each row the run writes, such as `results/*_materialized_inputs.jsonl`, carries
  the task's solution SQL and test cases. Keep the run's outputs private, as the ground truth itself.

## Auto Ontology

`auto_ontology_agent.yaml` (Lite) and `full_auto_ontology_agent.yaml` (Full) run c-Interact with
[Auto Ontology](https://github.com/NVIDIA/auto-ontology)'s ReAct agent in place of `simple_agent`, on the
[`auto_ontology_agent`](../../responses_api_agents/auto_ontology_agent/README.md) Agent Server. For each episode, the
agent starts a sandbox of its own from the catalog image: the release's databases and Auto Ontology's catalog of them.
Each turn runs one Auto Ontology episode there, with Auto Ontology's prompts and tools (`retrieve`, `run_sql`,
`submit`) and the server's `ask_user`. The query that Auto Ontology accepts goes to the server's `submit_sql`, which
grades it as it grades the query of any agent.

1. Build the release's database image (step 1), then the catalog image:

   ```bash
   benchmarks/bird_interact/auto_ontology/build_image.sh lite <models env>
   ```

   `<models env>` is a `KEY=VALUE` file with Auto Ontology's model keys: `DEFAULT_MODELS_API_KEY`, and `EMBED_*` or
   `*_MODEL` settings where they differ from Auto Ontology's defaults. The build embeds and compiles each database of
   the release with these models. The keys reach the build as a BuildKit secret, so no image layer holds them. Use
   `full` for Full.
2. Prepare and run:

   ```bash
   gym eval prepare --config benchmarks/bird_interact/auto_ontology_agent.yaml
   gym eval run --config benchmarks/bird_interact/auto_ontology_agent.yaml --model-type vllm_model --split benchmark \
     --config nemo_gym/sandbox/providers/docker/configs/docker.yaml \
     --output results/bird_interact_lite_c_auto_ontology.jsonl ++auto_ontology_api_key=<embedding key>
   ```

   `auto_ontology_api_key` is the key of the embedding endpoint that the catalog was built with. Retrieval calls it
   for each search.

What differs from `simple_agent`:

- **The catalog holds the schema only**, as c-Interact shows it to an agent. It holds no column meanings, which
  c-Interact does not show, and no knowledge base, part of which each task hides. The agent gets the task's visible
  knowledge from the first request, as evidence. auto-ontology-eval builds its BIRD-Interact catalog the same way.
- **Prompts.** The agent reads the ADK's texts (the database, the visible knowledge, the request, the debug message
  and the follow-up question) and gives them to Auto Ontology's own prompts. The ADK's instruction does not reach the
  model. A section on asking the user, with the questions that are left, is added to Auto Ontology's system prompt.
- **Turns.** Each turn is a new Auto Ontology episode. Its evidence carries the earlier answers of the user and the
  earlier queries, with the server's reply to the last one.
- **Management tasks.** Auto Ontology runs read-only queries only, so it cannot solve a management task. Compare its
  scores on the query tasks, as auto-ontology-eval runs only those.
- **The pin.** The image builds the Auto Ontology that `benchmarks/birdbench/auto_ontology/build_image.sh` pins, the
  one Gym runs. It holds NVIDIA/auto-ontology#559, the hook that lets a caller add tools to a ReAct episode, which
  `ask_user` needs.

## Results

None published yet.

## Known issues

In 24 Lite tasks, a solution SQL does not run when a submission is graded. The server grades that submission as wrong,
as the ADK's `ex_base` returns 0 when the solution fails, and lists the error in the rollout's `solution_errors`. No
agent can pass that phase, so leave these tasks out when you compare against a score that excludes them. The
aggregate metrics count the samples that hold one (`solution_errors`), and `mask_solution_errors: true` masks them,
for training, where a constant 0 is not a learning signal.

| Cause | Tasks (phase) |
|---|---|
| The ADK's `DISTINCT` removal, which Gym copies, turns `DISTINCT ON (...)` into `ON (...)` and `IS DISTINCT FROM` into `IS FROM` | `crypto_10` (2), `disaster_1` (2), `fake_5` (1, 2), `insider_M_4` (2), `insider_M_6` (2), `mental_4` (2), `mental_5` (1, 2), `mental_6` (1, 2), `mental_11` (1, 2), `museum_3` (2), `museum_5` (1, 2) |
| The solution is not valid SQL: a comma before `FROM` (`archeology_7`, `fake_7`), a stray word (`alien_8`), a row where a number is needed (`cross_db_M_5`), a table that does not exist (`museum_7`) | `alien_8` (2), `archeology_7` (1), `cross_db_M_5` (2), `fake_7` (2), `museum_7` (2) |
| Phase 2's solution uses a view, table or function that phase 1's solution creates; it fails when the agent's phase 1 SQL passed but created it with another name, columns or types | `archeology_M_3`, `crypto_M_2`, `disaster_M_2`, `disaster_M_3`, `gaming_M_2`, `mental_M_2`, `polar_M_4`, `polar_M_5` (2) |

The first two causes fail every submission of that phase; the third only after some phase 1 submissions. Gym keeps
the ADK's normalization, so its scores stay comparable with the ADK's and auto-ontology-eval's.

Three management tasks need a superuser where the ADK has one: Lite's `gaming_M_4` (1) and `virtual_M_4` (1) set
`session_replication_role`, and Full's `virtual_idol_M_2` (2) creates a role. Gym runs agent SQL as a role that is not a
superuser (see the server's README), so a submission that does the same fails there. None of them is a query task.

## Licensing

BIRD-Interact's data and databases: CC BY-SA 4.0. The image's base and its dumps come from BIRD-Interact
(`docker.io/shawnxxh/bird-interact-postgresql`, `-full`).
