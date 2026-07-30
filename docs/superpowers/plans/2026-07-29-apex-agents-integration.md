# Apex Agents (Mercor) Benchmark Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate Mercor's **Apex Agents** benchmark (`mercor/apex-agents`, 480 long-horizon cross-application professional-services tasks) into NeMo Gym **as a benchmark eval**, faithful to the paper's reported numbers, structured so the environment + dataset investment carries over to a future training integration.

**Decision (team-endorsed, 2026-07-29):** Add Apex **eval-first by wrapping Archipelago's canonical implementation** — its agent (`react_toolbelt`) *and* grader — as an **External-loop** integration (NVIDIA environment-onboarding RFC, UC22: "adopt the canonical implementation rather than re-implement it, record provenance"). The heavy, irreducible cost is standing up Mercor's **environment** (the multi-app MCP sandbox) under **Apptainer**, and we pay that either way. The agent and grader are the cheap parts; wrapping them is fastest and reproduces the paper's numbers. If we later commit to training, the **environment + dataset work carries over unchanged**, and we would then own the agent (a ReAct loop; the only throwaway piece — a wrapped ReAct runner yields no token-ids/logprobs) and optionally the verifier (gdpval-style). Nothing done here is wasted.

**Architecture:** Three Gym components mirror the `tau2` / `mini_swe_agent` external-wrap pattern, `gdpval`'s file layout, and the **cvdp sandbox pattern** (PR #2076):
- **Environment** — Archipelago's `environment/` + `mcp_servers/` (+ the `agents/` runner) built into a container **image** that Gym's sandbox API runs. The backend is provider-neutral (`sandbox_provider` config), **default Apptainer**.
- **Agent server** (`responses_api_agents/apex_agents_agent/`) — `/run` opens a Gym `AsyncSandbox` from the env image, uploads world/task assets, and `exec`s a guest **`sandbox_entrypoint.py`** that (inside the box, over localhost) boots the env service, populates it, configures `/apps`, runs Archipelago's `react_toolbelt` agent against `/mcp/` (policy = Gym Model Server via env var), and snapshots the result. The host `download`s the snapshot + trajectory. Wraps `examples/hugging_face_task/main.py`'s flow, run inside the sandbox.
- **Resources server** (`resources_servers/apex_agents/`) — `verify()` shells out to Archipelago's `grading/runner` (host subprocess; needs only the snapshot files + judge endpoint, no sandbox) on (snapshots, trajectory, rubric→verifiers) and maps `final_score` → reward.
- **Benchmark entry** (`benchmarks/apex_agents/`) — `prepare.py` (HF → Gym JSONL) + `config.yaml` (composition), à la `benchmarks/gdpval/`.

**Tech Stack:** Python 3.12+, async FastAPI (NeMo Gym `SimpleResponsesAPIAgent` / `SimpleResourcesServer`), Ray, Apptainer, `huggingface_hub`, Archipelago (`agents` + `grading` + `environment`, Apache-2.0, **pinned to an NVIDIA fork at a ref**), NeMo Gym Model Server (vLLM/OpenAI) as the policy endpoint.

## Global Constraints

- Line length **119**; Ruff lint + format (double quotes, isort); Python **3.12+**, async-first.
- Test coverage **>= 95%** for new Gym server code (the wrapped Archipelago subprocesses are covered by the Phase 0/4 gates, not unit coverage).
- The **policy** model must be reachable via NeMo Gym's Model Server (so any vLLM/OpenAI model can be evaluated). Archipelago's agent + grader use **LiteLLM internally** — acceptable, they are the wrapped external framework; do not add LiteLLM to Gym code we own.
- Configuration flows through Gym Hydra YAML, not env vars (except established secrets/paths: `HF_TOKEN`, judge/policy keys, container path).
- `/run` and `verify()` must be **async**. Bound concurrent sandboxes and Archipelago subprocesses with `asyncio.Semaphore` (each rollout = one heavy container).
- **Sandboxing: use Gym's provider-neutral sandbox API (`nemo_gym.sandbox.AsyncSandbox` + `SandboxSpec`), NOT raw `apptainer`/`docker` calls.** Follow the cvdp pattern (PR #2076, `responses_api_agents/cvdp_agent/app.py`): the agent server config carries a single-key `sandbox_provider` map defaulting to `{"apptainer": {}}` (swappable to `{"docker": {}}`, `{"opensandbox": {}}`, etc. — backend is config, not code) plus a `sandbox_spec` for extra fields. Orchestration runs **inside** the sandbox via a uploaded guest entrypoint + `AsyncSandbox.exec()`; inputs/outputs move via `spec.files` / `upload()` / `download()` (or provider bind mounts in `sandbox_spec.provider_options`). The Archipelago env image is the sandbox `image`.
- Decode subprocess output with `errors="replace"`. A failed rollout / missing output returns `reward=0.0` with an error status — never crash the server.
- All commits: DCO sign-off (`-s`) **and** cryptographic signature (`-S`).
- Runs on **Linux** (eval target); macOS dev-only.
- **Provenance (UC22):** pin Archipelago to an NVIDIA fork at a specific commit (`archipelago @ git+https://github.com/<org>/<fork>@<ref>`, pattern per `responses_api_agents/tau2/requirements.txt`). Record upstream URL + ref in the READMEs.
- **Reference score / tolerance:** the paper reports frontier models at **<25% pass@1**, ~**40% pass@8**. Because we wrap the canonical grader+agent, Phase 4 should **reproduce these** (parity is the acceptance bar, not "own baseline").
- Dataset license **CC-BY-4.0** (on train/validation entries); Archipelago code **Apache-2.0**.

## Reference Files (read before implementing)

In-repo patterns:
- `responses_api_agents/tau2/requirements.txt`, `responses_api_agents/mini_swe_agent/requirements.txt` — external-benchmark git-pin (+ Docker/heavy-dep precedent).
- `benchmarks/gdpval/prepare.py`, `benchmarks/gdpval/config.yaml`, `benchmarks/gdpval/README.md` — benchmark entry-point + composition pattern.
- `resources_servers/gdpval/app.py` — resources server + `/aggregate_metrics` shape.
- **Sandbox API — `nemo_gym/sandbox/api.py` (`AsyncSandbox`), `nemo_gym/sandbox/providers/base.py` (`SandboxSpec`, `SandboxProvider`), `nemo_gym/sandbox/providers/apptainer/provider.py`** — the provider-neutral sandbox surface we build on.
- **cvdp (PR #2076) — `responses_api_agents/cvdp_agent/app.py` + `sandbox_entrypoint.py`** — the reference for: `sandbox_provider={"apptainer":{}}` config, building a `SandboxSpec` (image, env, `provider_options.binds`), `async with AsyncSandbox(provider, spec) as box:`, uploading a guest entrypoint, `box.exec(...)`, and downloading outputs. **Copy this pattern.**

Upstream Archipelago files to read **at the pinned ref** (confirm exact schemas/flags — do not guess):
- `examples/hugging_face_task/main.py` + `*.json` configs — the canonical end-to-end flow this integration ports (env boot → populate → apps → agent `runner.main` → snapshot → grading `runner.main`).
- `environment/Dockerfile`, `environment/docker-compose.yml`, `environment/README.md` — env image + REST API (`/health`, `/data/populate`, `/apps`, `/data/snapshot`, `/mcp/`).
- `agents/runner/main.py`, `agents/runner/agents/models.py` — agent CLI flags (`--mcp-gateway-url`, `--initial-messages`, `--agent-config`, `--orchestrator-model`, `--output`) + `AgentTrajectoryOutput` schema.
- `grading/runner/main.py`, `grading/runner/evals/registry.py` — grading CLI flags + `output_llm` verifier + `GradingSettings` / `ScoringMethodResult` / `final_score` output shape.

---

## Phase 0 — De-risk (spikes; gates before building)

Resolve the three highest-risk unknowns. **Do not start Phase 1+ until all gates pass.** Deliverable = a verified finding + a committed note, not production code.

### Task 0.1: Reproduce-first — vanilla Archipelago ground truth + provenance

**Files:**
- Create: `docs/superpowers/notes/apex-repro-baseline.md`

**Interfaces:**
- Produces: `{task_id → final_score}` for ≥3 tasks (one per domain: investment banking, consulting, legal) with a named orchestrator + judge model, plus the Archipelago commit ref. Consumed by Task 4.2 (parity target) and provenance pins.

- [ ] **Step 1: Clone Archipelago at a chosen ref and record it**

```bash
git clone https://github.com/Mercor-Intelligence/archipelago /tmp/archipelago
cd /tmp/archipelago && git rev-parse HEAD   # record as the provenance ref
```

- [ ] **Step 2: Run the HF example on 3 tasks (Docker)**

Follow `examples/hugging_face_task/README.md`; set an LLM key in `agents/.env`. Run:

```bash
cd examples/hugging_face_task
./run.sh task_9ba58a6197114140877a1df1754d2993   # IB (default)
./run.sh <a_consulting_task_id>
./run.sh <a_legal_task_id>
```

Expected: each writes `output/<task_id>/grades.json` with a `final_score`.

- [ ] **Step 3: Record scores + exact config**

Write to `apex-repro-baseline.md`: the commit ref, `orchestrator_config.json` model, `grading_settings.json` `llm_judge_model`, `agent_config.json`, and the `{task_id → final_score}` table. This is the provenance + parity target.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/notes/apex-repro-baseline.md
git commit -sS -m "docs(apex): record Archipelago reproduce-first ground truth"
```

**GATE:** grading produced sensible per-criterion scores + a `final_score` for all three. If broken at this ref, pick a different ref.

### Task 0.2: Run the Archipelago env image under Gym's sandbox API (the real risk)

**Files:**
- Create: `docs/superpowers/notes/apex-sandbox-spike.md`
- Create: `responses_api_agents/apex_agents_agent/containers/apex_env.def` (draft, if a `.sif` build is needed)

**Interfaces:**
- Produces: a working recipe to run the Archipelago env+agents **image** via `nemo_gym.sandbox.AsyncSandbox` with `provider={"apptainer": {}}`, and — inside the box, over localhost — pass `/health`, `/data/populate`, `/apps`, `/mcp/`, a `code` tool call, and `/data/snapshot`. Confirms the guest-entrypoint approach works. Consumed by Tasks 3.1–3.3.

- [ ] **Step 1: Build the env+agents image and make it Apptainer-runnable**

Build Archipelago's `environment` image (and ensure the `agents/` runner is available inside — bake it in or plan to `upload`/bind it). Confirm how Gym's apptainer provider consumes it (a local `.sif`, or `docker://`/`docker-daemon://` that the provider pulls — see cvdp `app.py` `_resolve_image`/`apptainer pull`).

- [ ] **Step 2: Drive it through `AsyncSandbox` with the apptainer provider**

Minimal script using `nemo_gym.sandbox`:

```python
import asyncio
from pathlib import Path
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec

async def main():
    spec = SandboxSpec(image="<archipelago-env-image>", workdir="/app",
                       env={}, provider_options={"binds": []})
    async with AsyncSandbox({"apptainer": {}}, spec) as box:
        await box.start()
        await box.upload(Path("world.zip"), "/tmp/world.zip")
        r = await box.exec("bash -lc 'python /app/probe_entrypoint.py'", timeout_s=1800)
        print(r.return_code, (r.stdout or "")[-2000:])
        await box.download("/output/final_snapshot.zip", Path("final.zip"))

asyncio.run(main())
```

Where `probe_entrypoint.py` (uploaded) starts the env service in the background, waits for localhost `/health`, populates the world, `POST /apps`, lists tools + invokes one `code` tool over `/mcp/` (**exercises `sandbox_fs.so` under Apptainer's user namespace — the riskiest item**), and `POST /data/snapshot` to `/output/final_snapshot.zip`.

- [ ] **Step 3: Write the spike note; commit**

Document: image build/pull recipe, the `sandbox_provider`/`SandboxSpec` shape that worked (binds for writable state, env vars), whether the env service binds a fixed localhost port inside the box, and **the code-sandbox go/no-go**.

```bash
git add docs/superpowers/notes/apex-sandbox-spike.md responses_api_agents/apex_agents_agent/containers/apex_env.def
git commit -sS -m "spike(apex): run archipelago env under gym AsyncSandbox (apptainer)"
```

**GATE (go/no-go):** the full in-box flow (health + populate + apps + one code-tool call + snapshot) succeeds through `AsyncSandbox`. If the `code` LD_PRELOAD sandbox can't run under the apptainer provider, escalate (try `{"docker": {}}` provider, disable `code` for tasks that don't need it) — do **not** proceed assuming it works.

### Task 0.3: Confirm Archipelago's agent + grader accept a Gym-served model endpoint

**Files:**
- Modify: `docs/superpowers/notes/apex-apptainer-spike.md` (append)

**Interfaces:**
- Produces: confirmation that (a) `agents/runner --orchestrator-model` can be pointed at the Gym Model Server (`openai/<served_model>` + `OPENAI_API_BASE`/base_url), and (b) `grading/runner`'s `llm_judge_model` can point at a Gym judge model server (like gdpval's `gdpval_judge_model` proxy). Consumed by Tasks 2.x and 3.x.

- [ ] **Step 1: Re-run the Task 0.1 example with orchestrator + judge pointed at a stub OpenAI-compatible base_url**

Confirm both routes reach the custom endpoint. Record the exact model-string + base_url + api_key mechanism for each.

- [ ] **Step 2: Commit the findings** (`-sS`).

**GATE:** both policy and judge can be pointed at Gym-served endpoints. If either hardcodes a provider, record the minimal fork patch (part of provenance).

---

## Phase 1 — Data preparation

### Task 1.1: Benchmark scaffold + `prepare.py`

**Files:**
- Create: `benchmarks/apex_agents/__init__.py`, `benchmarks/apex_agents/prepare.py`, `benchmarks/apex_agents/data/.gitignore`
- Test: `benchmarks/apex_agents/tests/test_prepare.py`

**Interfaces:**
- Produces: `convert_task(task, worlds) -> dict` and `prepare() -> Path` writing `data/apex_agents.jsonl`, one row per task:
  `{"responses_create_params": {"input": []}, "task_id", "world_id", "world_name", "domain", "task_name", "prompt", "rubric": [...], "has_task_input_files": bool}`.
  Field names from HF `tasks_and_rubrics.json` (`task_id`, `world_id`, `domain`, `task_name`, `prompt`, `rubric` with per-criterion `verifier_id`/`criteria`) and `world_descriptions.json` (`world_id`, `world_name`). The agent builds the prompt from `prompt`; the verifier reads `rubric` (rubric→verifiers) from `verifier_metadata`.

- [ ] **Step 1: Write the failing test**

```python
# benchmarks/apex_agents/tests/test_prepare.py
from benchmarks.apex_agents.prepare import convert_task

def test_convert_task_builds_gym_row():
    task = {"task_id": "task_abc", "world_id": "world_1", "domain": "investment_banking",
            "task_name": "Accretion/Dilution", "prompt": "Build the model.",
            "rubric": [{"verifier_id": "v1", "criteria": "Model is correct"}],
            "task_input_files": ["/filesystem/in.xlsx"]}
    worlds = {"world_1": {"world_id": "world_1", "world_name": "World 1"}}
    row = convert_task(task, worlds)
    assert row["responses_create_params"] == {"input": []}
    assert row["task_id"] == "task_abc"
    assert row["world_id"] == "world_1"
    assert row["world_name"] == "World 1"
    assert row["domain"] == "investment_banking"
    assert row["prompt"] == "Build the model."
    assert row["rubric"] == [{"verifier_id": "v1", "criteria": "Model is correct"}]
    assert row["has_task_input_files"] is True
```

- [ ] **Step 2: Run → FAIL** (`pytest benchmarks/apex_agents/tests/test_prepare.py -v`).

- [ ] **Step 3: Implement `prepare.py`**

```python
# benchmarks/apex_agents/prepare.py  (Apache-2.0 header like benchmarks/gdpval/prepare.py)
from __future__ import annotations
import json, os
from pathlib import Path

BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "apex_agents.jsonl"
HF_DATASET = "mercor/apex-agents"

def convert_task(task: dict, worlds: dict) -> dict:
    world = worlds.get(task["world_id"], {})
    return {
        "responses_create_params": {"input": []},
        "task_id": task["task_id"],
        "world_id": task["world_id"],
        "world_name": world.get("world_name", ""),
        "domain": task.get("domain", ""),
        "task_name": task.get("task_name", ""),
        "prompt": task["prompt"],
        "rubric": task.get("rubric", []),
        "has_task_input_files": bool(task.get("task_input_files")),
    }

def prepare() -> Path:
    from huggingface_hub import hf_hub_download
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    tasks_path = hf_hub_download(HF_DATASET, "tasks_and_rubrics.json", repo_type="dataset", token=token)
    worlds_path = hf_hub_download(HF_DATASET, "world_descriptions.json", repo_type="dataset", token=token)
    tasks = json.loads(Path(tasks_path).read_text())
    worlds = {w["world_id"]: w for w in json.loads(Path(worlds_path).read_text())}
    with OUTPUT_FPATH.open("w") as f:
        for task in tasks:
            f.write(json.dumps(convert_task(task, worlds)) + "\n")
    print(f"Wrote {len(tasks)} tasks to {OUTPUT_FPATH}")
    return OUTPUT_FPATH

if __name__ == "__main__":
    prepare()
```

- [ ] **Step 4: Run → PASS.**

- [ ] **Step 5: Add `data/.gitignore`**

```
*apex_agents.jsonl
*train.jsonl
*validation.jsonl
```

- [ ] **Step 6: Commit**

```bash
git add benchmarks/apex_agents/
git commit -sS -m "feat(apex): benchmark scaffold and prepare.py"
```

### Task 1.2: `example.jsonl` + registry upload

**Files:**
- Create: `benchmarks/apex_agents/data/example.jsonl` (committed, 5 rows)

- [ ] **Step 1: Run `prepare` and slice 5 rows spanning the 3 domains**

```bash
HF_TOKEN=<token> python -m benchmarks.apex_agents.prepare
head -n 5 benchmarks/apex_agents/data/apex_agents.jsonl > benchmarks/apex_agents/data/example.jsonl
```

- [ ] **Step 2: Upload the full set to the GitLab registry**

```bash
ng_upload_dataset_to_gitlab +dataset_name=apex_agents +version=0.0.1 \
  +input_jsonl_fpath=benchmarks/apex_agents/data/apex_agents.jsonl
```

- [ ] **Step 3: Commit the example only**

```bash
git add benchmarks/apex_agents/data/example.jsonl
git commit -sS -m "feat(apex): committed 5-row example dataset"
```

**GATE:** 5 valid example rows; full set in registry; `apex_agents.jsonl` gitignored.

---

## Phase 2 — Resources server (wraps Archipelago's grader)

### Task 2.1: Scaffold + pin the grader

**Files:**
- Create: `resources_servers/apex_agents/` via `ng_init_resources_server +entrypoint=resources_servers/apex_agents`
- Modify: `resources_servers/apex_agents/requirements.txt` (Archipelago grading pin)
- Create: `resources_servers/apex_agents/README.md` (provenance from Task 0.1)

- [ ] **Step 1: Scaffold + pin**

```bash
ng_init_resources_server +entrypoint=resources_servers/apex_agents
```

`requirements.txt` (use the fork+ref from Task 0.1):

```
-e nemo-gym[dev] @ ../../
archipelago-grading @ git+https://github.com/<org>/<archipelago-fork>@<ref>#subdirectory=grading
```

- [ ] **Step 2: Commit**

```bash
git add resources_servers/apex_agents/
git commit -sS -m "feat(apex): scaffold resources server + grading pin"
```

### Task 2.2: Rubric → verifiers builder (pure, TDD)

**Files:**
- Create: `resources_servers/apex_agents/verifiers.py`
- Test: `resources_servers/apex_agents/tests/test_verifiers.py`

**Interfaces:**
- Produces: `build_verifiers(rubric, world_id, task_id) -> list[dict]`, mirroring `examples/hugging_face_task/main.py`: each criterion → `{verifier_id, verifier_version: 1, world_id, task_id, eval_config_id: "ec_output_llm", verifier_values: {criteria, is_primary_objective}, verifier_index, verifier_dependencies: None}`, first criterion `is_primary_objective=True`. Consumed by Task 2.3.

- [ ] **Step 1: Write the failing test**

```python
from resources_servers.apex_agents.verifiers import build_verifiers

def test_build_verifiers_marks_first_primary():
    rubric = [{"verifier_id": "a", "criteria": "C1"}, {"verifier_id": "b", "criteria": "C2"}]
    out = build_verifiers(rubric, world_id="w", task_id="t")
    assert len(out) == 2
    assert out[0]["eval_config_id"] == "ec_output_llm"
    assert out[0]["verifier_values"] == {"criteria": "C1", "is_primary_objective": True}
    assert out[1]["verifier_values"]["is_primary_objective"] is False
    assert out[0]["world_id"] == "w" and out[0]["task_id"] == "t"
    assert out[1]["verifier_index"] == 1

def test_build_verifiers_empty_rubric():
    assert build_verifiers([], "w", "t") == []
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement**

```python
# resources_servers/apex_agents/verifiers.py
from __future__ import annotations

def build_verifiers(rubric: list[dict], world_id: str, task_id: str) -> list[dict]:
    return [
        {"verifier_id": c.get("verifier_id", f"v{i}"), "verifier_version": 1,
         "world_id": world_id, "task_id": task_id, "eval_config_id": "ec_output_llm",
         "verifier_values": {"criteria": c["criteria"], "is_primary_objective": i == 0},
         "verifier_index": i, "verifier_dependencies": None}
        for i, c in enumerate(rubric)
    ]
```

- [ ] **Step 4: Run → PASS. Step 5: Commit**

```bash
git add resources_servers/apex_agents/verifiers.py resources_servers/apex_agents/tests/test_verifiers.py
git commit -sS -m "feat(apex): rubric->verifiers builder"
```

### Task 2.3: `verify()` — run Archipelago grader, map `final_score` → reward

**Files:**
- Modify: `resources_servers/apex_agents/app.py`
- Create: `resources_servers/apex_agents/grading_runner.py`
- Test: `resources_servers/apex_agents/tests/test_app.py`

**Interfaces:**
- Consumes: `build_verifiers` (2.2). The verify request carries in `verifier_metadata`: `world_id`, `task_id`, `rubric`, and artifact paths from the agent (Task 3.3): `initial_snapshot_path`, `final_snapshot_path`, `trajectory_path`.
- Produces: `parse_final_score(grades: dict) -> float` (pure) and `run_grader(...) -> dict` (async subprocess). `ApexVerifyResponse(reward, per_criterion, status)`. **Field names `grades["scoring_method_result"]["final_score"]` / `grades["verifier_results"]` must be confirmed against `grading/runner` output at the pinned ref (Task 0.1's `grades.json`); adjust extractor + test together if different.** Consumed by Task 2.4 + Task 3.3 (agent calls `/verify`).

- [ ] **Step 1: Write the failing test for `parse_final_score`**

```python
from resources_servers.apex_agents.grading_runner import parse_final_score

def test_parse_final_score_reads_scoring_result():
    grades = {"scoring_method_result": {"final_score": 0.75},
              "verifier_results": [{"verifier_id": "a", "score": 1.0}]}
    assert parse_final_score(grades) == 0.75

def test_parse_final_score_missing_defaults_zero():
    assert parse_final_score({}) == 0.0
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement `grading_runner.py` + `verify()`**

```python
# resources_servers/apex_agents/grading_runner.py
from __future__ import annotations
import asyncio, json, tempfile, uuid
from pathlib import Path

def parse_final_score(grades: dict) -> float:
    return float((grades.get("scoring_method_result") or {}).get("final_score", 0.0))

async def run_grader(*, grading_dir, initial_snapshot, final_snapshot, trajectory,
                     verifiers, grading_settings, eval_configs, scoring_config, timeout) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        (t / "verifiers.json").write_text(json.dumps(verifiers))
        (t / "grading_settings.json").write_text(json.dumps(grading_settings))
        (t / "eval_configs.json").write_text(json.dumps(eval_configs))
        (t / "scoring_config.json").write_text(json.dumps(scoring_config))
        out = t / "grades.json"
        cmd = ["uv", "run", "python", "-m", "runner.main",
               "--grading-run-id", f"gr_{uuid.uuid4().hex[:8]}",
               "--initial-snapshot", initial_snapshot, "--final-snapshot", final_snapshot,
               "--trajectory", trajectory, "--verifiers", str(t / "verifiers.json"),
               "--grading-settings", str(t / "grading_settings.json"),
               "--eval-configs", str(t / "eval_configs.json"),
               "--scoring-config", str(t / "scoring_config.json"), "--output", str(out)]
        proc = await asyncio.create_subprocess_exec(*cmd, cwd=grading_dir,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        try:
            await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill(); return {}
        if proc.returncode != 0 or not out.exists():
            return {}
        return json.loads(out.read_text())
```

`app.py`: `ApexAgentsConfig(grading_dir: str, grading_timeout_secs: int = 1800, num_graders: int = 8, llm_judge_model: str, eval_configs: list, scoring_config: dict, grading_settings_overrides: dict = {})`; `model_post_init` → `Semaphore(num_graders)`. `verify()`:
- read `world_id/task_id/rubric` + the three paths; any path missing → `reward=0.0, status="no_deliverable"`;
- `verifiers = build_verifiers(rubric, world_id, task_id)`;
- `grading_settings = {"llm_judge_model": self.config.llm_judge_model, **overrides}` (judge → Gym judge model server per Task 0.3);
- `async with self._sema: grades = await run_grader(...)`;
- `reward = parse_final_score(grades)`; return `ApexVerifyResponse(**body.model_dump(), reward=reward, per_criterion=grades.get("verifier_results", []), status="ok" if grades else "grader_error")`.

- [ ] **Step 4: Add a `verify()` test with `run_grader` patched** (canned grades → reward maps; missing path → `0.0`/`no_deliverable`). `@pytest.mark.skipif(shutil.which("uv") is None, ...)` only on any test that truly shells out.

- [ ] **Step 5: Run → PASS. Step 6: Commit**

```bash
git add resources_servers/apex_agents/grading_runner.py resources_servers/apex_agents/app.py resources_servers/apex_agents/tests/test_app.py
git commit -sS -m "feat(apex): verify() wraps archipelago grader, maps final_score to reward"
```

### Task 2.4: `/aggregate_metrics` — pass@1 / pass@k

**Files:**
- Modify: `resources_servers/apex_agents/app.py`
- Test: `resources_servers/apex_agents/tests/test_app.py`

**Interfaces:**
- Produces: `compute_aggregate(rows) -> dict` and `aggregate_metrics()` emitting `mean/reward` (pass@1 ≈ mean reward), `apex/pass_at_1` (fraction of rollouts with `reward == 1.0`), `apex/pass_at_k` (mean over tasks of per-task max reward). Mirrors `resources_servers/gdpval/app.py`.

- [ ] **Step 1: Write the failing test**

```python
import pytest

def test_aggregate_pass_at_1_and_k():
    from resources_servers.apex_agents.app import compute_aggregate
    rows = [{"task_id": "t1", "reward": 1.0}, {"task_id": "t1", "reward": 0.0},
            {"task_id": "t2", "reward": 0.5}]
    m = compute_aggregate(rows)
    assert m["mean/reward"] == pytest.approx(0.5)
    assert m["apex/pass_at_1"] == pytest.approx(1/3)
    assert m["apex/pass_at_k"] == pytest.approx((1.0 + 0.5) / 2)
```

- [ ] **Step 2: Run → FAIL. Step 3: Implement `compute_aggregate` (pure), call it from `aggregate_metrics()`. Step 4: Run → PASS. Step 5: Commit**

```bash
git add resources_servers/apex_agents/app.py resources_servers/apex_agents/tests/test_app.py
git commit -sS -m "feat(apex): aggregate metrics pass@1/pass@k"
```

---

## Phase 3 — Agent server (Gym sandbox + Archipelago's `react_toolbelt` agent inside)

### Task 3.1: Scaffold + guest entrypoint (orchestrates the env + agent inside the box)

**Files:**
- Create: `responses_api_agents/apex_agents_agent/app.py`, `configs/apex_agents.yaml`, `requirements.txt`
- Create: `responses_api_agents/apex_agents_agent/sandbox_entrypoint.py` (guest script; runs **inside** the sandbox)
- Test: `responses_api_agents/apex_agents_agent/tests/test_app.py`

**Interfaces:**
- Produces: `ApexAgentsAgent(SimpleResponsesAPIAgent)` + `ApexAgentsAgentConfig(sandbox_provider: dict = {"apptainer": {}}, sandbox_spec: dict = {}, image: str, orchestrator_model: str, agent_config: dict, resources_server, model_server, concurrency: int = 8, ...)`; `model_post_init` → `Semaphore(concurrency)`. `_read_entrypoint_source() -> str` (reads `sandbox_entrypoint.py` verbatim, à la cvdp `_RUNNER_SOURCE_PATH`). `build_spec(task_info, model_url) -> SandboxSpec` (pure — image, env incl. `NV_MODEL_URL`/orchestrator model + task metadata, `provider_options.binds`). The **guest entrypoint** (documented contract, tested in Task 0.2): reads uploaded world/task assets + `initial_messages.json`, boots the env service on localhost, waits `/health`, populates + `POST /apps`, runs Archipelago's `react_toolbelt` agent against localhost `/mcp/` with the orchestrator model, writes `/output/trajectory.json` + `/output/final_snapshot.zip` (+ copies the seeded world as `/output/initial_snapshot.zip`). Consumed by Task 3.2.

- [ ] **Step 1: Write the failing test for `build_spec` (pure)**

```python
from responses_api_agents.apex_agents_agent.app import build_spec

def test_build_spec_sets_image_env_and_model_url():
    spec = build_spec({"task_id": "t", "world_id": "w", "prompt": "do it"},
                      model_url="http://gym-model:8000/v1")
    assert spec.image  # non-empty image
    assert spec.env["NV_MODEL_URL"] == "http://gym-model:8000/v1"
    assert spec.env["APEX_TASK_ID"] == "t"
```

- [ ] **Step 2: Run → FAIL. Step 3: Implement `app.py` scaffold + `build_spec` + `_read_entrypoint_source`, and write `sandbox_entrypoint.py`** (port `examples/hugging_face_task/main.py`'s populate → apps → run agent → snapshot into a single in-box script; the model is reached via `NV_MODEL_URL` like cvdp). **Step 4: Run → PASS.**

- [ ] **Step 5: `requirements.txt`**

```
-e nemo-gym[dev] @ ../../
huggingface_hub
```

(The Archipelago `agents` + env code live in the sandbox **image**, not as a Gym pip dep — pinned via the image build in Task 0.2.)

- [ ] **Step 6: Commit**

```bash
git add responses_api_agents/apex_agents_agent/
git commit -sS -m "feat(apex): agent scaffold + sandbox spec + guest entrypoint"
```

### Task 3.2: Fetch assets + `/run` via `AsyncSandbox`

**Files:**
- Modify: `responses_api_agents/apex_agents_agent/app.py`
- Create: `responses_api_agents/apex_agents_agent/assets.py`
- Test: `responses_api_agents/apex_agents_agent/tests/test_app.py`

**Interfaces:**
- Consumes: `build_spec`/`_read_entrypoint_source` (3.1), `AsyncSandbox`/`SandboxSpec` from `nemo_gym.sandbox`, resources `/verify` (2.3).
- Produces: `fetch_world_assets(world_id, task_id) -> (world_zip_path, task_dir|None)` (HF `world_files_zipped/<world_id>.zip` + `task_files/<task_id>/**`). `/run`:
  1. resolve the policy model URL from the Gym Model Server ref; `spec = build_spec(task_info, model_url)` + upload the guest entrypoint via `spec.files`.
  2. `async with self._sema, AsyncSandbox(self.config.sandbox_provider, spec) as box: await box.start()`.
  3. `world_zip, task_dir = fetch_world_assets(...)`; `await box.upload(world_zip, "/inputs/world.zip")` (+ task files); write + upload `initial_messages.json`.
  4. `res = await box.exec("bash -lc 'python /app/sandbox_entrypoint.py'", timeout_s=cfg.agent_timeout_s)`.
  5. `await box.download("/output/final_snapshot.zip", final_path)`, `download("/output/initial_snapshot.zip", init_path)`, `download("/output/trajectory.json", traj_path)`.
  6. read `trajectory.json`; if `status != "completed"` → return `reward=0.0` (no grading).
  7. else POST `/verify` with `verifier_metadata` += `{initial_snapshot_path, final_snapshot_path, trajectory_path, world_id, task_id, rubric}`; return its result.
  Whole body wrapped → any sandbox/exec failure returns an error verify-response with `reward=0.0`. Guarded by `Semaphore` (one sandbox per rollout).

- [ ] **Step 1: Write the async `/run` happy-path test** — patch `AsyncSandbox` (a fake whose `exec` returns rc 0 and whose `download` writes a canned `trajectory.json` with `status="completed"`), `fetch_world_assets`, and `server_client.post("/verify")` → `reward=0.7`. Assert returned reward `0.7`, that the entrypoint was `exec`'d, and that `/verify`'s `verifier_metadata` includes the three artifact paths + `world_id/task_id/rubric`. Add a test where the fake trajectory `status="failed"` → `reward=0.0` and `/verify` is **not** called.
- [ ] **Step 2: Run → FAIL. Step 3: Implement `fetch_world_assets` + `run()`. Step 4: Run → PASS.**
- [ ] **Step 5: Commit**

```bash
git add responses_api_agents/apex_agents_agent/assets.py responses_api_agents/apex_agents_agent/app.py responses_api_agents/apex_agents_agent/tests/test_app.py
git commit -sS -m "feat(apex): /run drives env+agent via AsyncSandbox, then verifies"
```

### Task 3.3: Compose the config

**Files:**
- Create: `benchmarks/apex_agents/config.yaml`
- Modify: `responses_api_agents/apex_agents_agent/configs/apex_agents.yaml`

**Interfaces:**
- Produces: `benchmarks/apex_agents/config.yaml` composing (a) `apex_agents_agent` (`sandbox_provider: {apptainer: {}}`, `sandbox_spec: {}`, `image: <archipelago-env-image>`, `orchestrator_model`, `agent_config={agent_config_id: react_toolbelt_agent, agent_config_values:{max_steps, timeout}}`, `resources_server`, `model_server: policy_model`, `concurrency`), (b) `apex_agents` resources server (`grading_dir`, `llm_judge_model`, `eval_configs=[{eval_config_id: ec_output_llm, eval_defn_id: output_llm}]`, `scoring_config`), (c) an `apex_judge_model` proxy (like `gdpval_judge_model`), (d) the dataset entry (`jsonl_fpath`, `gitlab_identifier` apex_agents/0.0.1, `license: CC-BY-4.0`, `prepare_script: benchmarks/apex_agents/prepare.py`). Model on `benchmarks/gdpval/config.yaml`. Note `sandbox_provider` is swappable (`{docker: {}}`, `{opensandbox: {}}`) — backend is config, not code.

- [ ] **Step 1: Author `config.yaml`** (`verified: false`, auto-added by pre-commit).
- [ ] **Step 2: Resolve-check**

```bash
gym env resolve "+config_paths=[benchmarks/apex_agents/config.yaml]"
```

- [ ] **Step 3: Commit**

```bash
git add benchmarks/apex_agents/config.yaml responses_api_agents/apex_agents_agent/configs/apex_agents.yaml
git commit -sS -m "feat(apex): compose agent + resources + judge + dataset config"
```

---

## Phase 4 — End-to-end + baseline (parity target)

### Task 4.1: Server tests + example smoke test

- [ ] **Step 1: Isolated server tests**

```bash
ng_test +entrypoint=resources_servers/apex_agents
ng_test +entrypoint=responses_api_agents/apex_agents_agent
pytest tests/unit_tests/ -x
```

Expected: pass; ≥95% coverage on new Gym code.

- [ ] **Step 2: End-to-end smoke on `example.jsonl`**

```bash
gym eval prepare --benchmark apex_agents
gym eval run --model-type <policy> --benchmark apex_agents --split example --output results/apex_example.jsonl
gym env viewer +jsonl_fpath=results/apex_example.jsonl
```

Expected: a task boots under Apptainer, the Archipelago agent runs, snapshot + trajectory produced, the grader returns a `final_score`, reward populated.

**GATE:** a full task runs end-to-end under Apptainer with a graded reward.

### Task 4.2: Reward profiling + **parity vs. Task 0.1 ground truth**

**Files:**
- Create: `docs/superpowers/notes/apex-baseline.md`

- [ ] **Step 1: Collect rollouts across a model suite** (policy of interest; ≥1 OSS instruct; ≥1 OSS thinking; ≥1 closed-source), `num_repeats` ≥ 3–5:

```bash
gym eval run --model-type <m> --benchmark apex_agents --split benchmark --output results/apex_<m>.jsonl
python scripts/print_aggregate_results.py +jsonl_fpath=results/apex_<m>_metrics.json
```

- [ ] **Step 2: Parity check** — on the shared tasks, Gym scores should **match Task 0.1's Archipelago ground truth** within tolerance (we wrap the same agent + grader). Divergence = an integration bug; inspect with `gym env viewer`.
- [ ] **Step 3: Sanity vs. paper** — frontier ≈ **<25% pass@1 / ~40% pass@8**; closed ≥ open. Record numbers + W&B links in `apex-baseline.md`.
- [ ] **Step 4: Raise `num_repeats` until run-to-run variance < 1%; commit.**

```bash
git add docs/superpowers/notes/apex-baseline.md
git commit -sS -m "docs(apex): reward-profiling baseline + parity check"
```

**GATE:** Gym scores match Task 0.1 within tolerance **and** sit in the paper's range.

---

## Phase 5 — Polish, docs, PR

### Task 5.1: READMEs, provenance, customer-evals, pre-commit, PR

**Files:**
- Modify: `resources_servers/apex_agents/README.md`, `responses_api_agents/apex_agents_agent/README.md`, `benchmarks/apex_agents/README.md`
- Modify: `scripts/run_customer_evals.py`
- Modify: `benchmarks/apex_agents/config.yaml` (`verified: true` after baselining)

- [ ] **Step 1: Write the READMEs** — run recipe (`gym eval prepare/run`, Apptainer `.sif` build from `containers/apex_env.def`, required policy + judge endpoints), provenance (Archipelago upstream URL + pinned ref; agents + grading subdir pins), dataset license CC-BY-4.0, and the parity/reference numbers.
- [ ] **Step 2: Add the `CustomerEval` entry** in `scripts/run_customer_evals.py` (`eval_name="apex_agents"`, config path, `RolloutCollectionConfig` with agent name + `data/apex_agents/validation.jsonl`).
- [ ] **Step 3: Pre-commit (scoped), fix, restage**

```bash
pre-commit run --files resources_servers/apex_agents/**/* responses_api_agents/apex_agents_agent/**/* benchmarks/apex_agents/**/*
git checkout -- <any unrelated server dirs the hooks touched>
```

- [ ] **Step 4: `verified: true`** (only after Phase 4 gates), commit.
- [ ] **Step 5: Open the PR** — architecture (External-loop wrap of Archipelago agent+grader; env as Apptainer `.sif`), provenance ref, parity table + W&B links, Apptainer build recipe.

```bash
git commit -sS -am "docs(apex): readmes, provenance, customer-evals; verify=true"
```

**GATE:** pre-commit clean; PR shows parity + provenance + Apptainer recipe.

---

## Self-Review

**Spec coverage:** Eval-first wrap of Archipelago agent + grader ✓ (Phase 2 grader, Phase 3 agent) · environment via Gym's provider-neutral sandbox API, default Apptainer (cvdp/PR #2076 pattern) ✓ (0.2, 3.1–3.2) · data prep ✓ (1.1–1.2) · composition ✓ (3.3) · reproduce-first + parity baseline ✓ (0.1, 4.2) · provenance + reference score + license ✓ (Global Constraints, 2.1, 3.1, 5.1) · RFC External-loop/UC22 alignment ✓ · policy via Gym Model Server ✓ (0.3, 3.2).

**Training carry-over (out of scope; note in PR):** if training is greenlit, the **environment `.sif` + dataset + config skeleton carry over unchanged**; we would then replace the wrapped Archipelago agent with a Gym-native ReAct-over-MCP loop (policy through the Model Server for token-ids/logprobs) and optionally own the verifier (compose `resources_servers/gdpval/` judge). The wrapped agent is the only throwaway piece — cheap.

**Placeholder scan:** infra/spike steps (0.1–0.3, 4.1) use commands + explicit GATE criteria (no host unit test asserts "Apptainer serves the env"); pure functions (`convert_task`, `build_verifiers`, `parse_final_score`, `compute_aggregate`, `build_spec`) have real test code, and `/run` is tested against a faked `AsyncSandbox`. **Confirm-against-pinned-ref items** (inherent to wrapping): the agent CLI flags + `AgentTrajectoryOutput.status`, the grading CLI flags + `grades["scoring_method_result"]["final_score"]`/`verifier_results` shape — each flagged in-task, test+impl adjusted together after reading the ref (Task 0.1's real `grades.json`/`trajectory.json` are the source of truth).

**Type consistency:** `build_verifiers` (2.2) output feeds `run_grader`'s `verifiers.json` (2.3); the artifact paths `/run` writes into `verifier_metadata` (3.2: `initial_snapshot_path, final_snapshot_path, trajectory_path, world_id, task_id, rubric`) are exactly those `verify()` reads (2.3); `reward` is the single score field across 2.3 → 2.4 → 3.2; the guest `sandbox_entrypoint.py` output contract (`/output/{final_snapshot,initial_snapshot}.zip`, `/output/trajectory.json`) is consumed by `/run`'s `download` calls (3.2); `build_spec`/`sandbox_provider` naming consistent across 3.1–3.3.
