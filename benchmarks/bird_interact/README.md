# BIRD-Interact

[BIRD-Interact](https://bird-interact.github.io/) ([paper](https://arxiv.org/abs/2510.05318)) evaluates interactive
text-to-SQL: the request is ambiguous, part of the knowledge base is hidden, and the agent must ask a simulated user
before it writes PostgreSQL. This recipe runs its Lite and Full releases on the
[`bird_interact` Environment Server](../../environment_servers/bird_interact/README.md), which describes the two
modes, the user simulator, the rewards and the grading. Its c-Interact follows the BIRD-Interact ADK: its protocol,
its simulator and its grading, as auto-ontology-eval runs it. The server's own five-task example runs without this
recipe.

| Config | Release | Mode | Tasks |
|---|---|---|---|
| `config.yaml` | Lite: 18 databases | c-Interact: the agent asks and submits with two tools, and the server opens each turn | Up to 195 of 300 |
| `a_interact.yaml` | Lite | a-Interact: the agent calls tools, `ask` among them | Up to 195 of 300 |
| `full.yaml` | Full: 22 databases | c-Interact | Up to 410 of 600 |
| `full_a_interact.yaml` | Full | a-Interact | Up to 410 of 600 |

Every task has a follow-up question, its phase 2. Only query tasks run, the ones auto-ontology-eval runs. As in
BIRD-Interact, a query is graded by comparing its result with the solution's, even when its ground truth has test
cases; a management task (105 in Lite, 190 in Full) is graded by its custom test cases, which the server does not run
yet, so `prepare.py` leaves out each task that has such a phase. Every query task's follow-up is a query too.

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

## 3. Prepare

```bash
gym eval prepare --benchmark bird_interact +prepare_script_args.ground_truth_jsonl=<ground truth file>
```

`prepare.py` downloads the release from Hugging Face (`birdsql/bird-interact-lite` or `-full`): its rows, and for each
database the schema, column meanings and knowledge base, into `data/<release>/databases/`, the servers'
`database_dir`. It merges the ground truth into the rows by `instance_id`, for the task and its follow-up, and writes
`data/<release>/<mode>.jsonl` with the tasks the server can grade. It prints how many it kept and why it left the
others out. For another config, pass `--config benchmarks/bird_interact/<config>` in place of `--benchmark`.

## 4. Run

`env.yaml` needs the policy model (`policy_base_url`, `policy_api_key`, `policy_model_name`) and the simulator
(`user_model_base_url`, `user_model_api_key`, `user_model_name`).

```bash
gym eval run --benchmark bird_interact --model-type vllm_model --split benchmark \
  --config nemo_gym/sandbox/providers/docker/configs/docker.yaml \
  --output results/bird_interact_lite_c.jsonl
```

- **Simulator.** The server runs the ADK's simulator. For scores comparable with auto-ontology-eval's, set
  `user_model_name` to the ADK's default simulator model, `anthropic/claude-haiku-4-5-20251001` (BIRD-Interact's own
  harness uses `gemini-2.0-flash-001`).
- **Harnesses.** c-Interact needs an agent that resumes its session across turns and takes tool grants, such as
  `simple_agent`; a-Interact an agent that takes tool grants. The server's README lists which harnesses run each mode,
  and how to point the a-Interact server at an MCP-only harness such as Codex.

## Results

None yet: Lite and Full run once the ground truth is in hand (GYMINT-592).

## Licensing

BIRD-Interact's data and databases: CC BY-SA 4.0. The image's base and its dumps come from BIRD-Interact
(`docker.io/shawnxxh/bird-interact-postgresql`, `-full`).
