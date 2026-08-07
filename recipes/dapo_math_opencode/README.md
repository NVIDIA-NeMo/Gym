# DAPO math × OpenCode-in-sandbox (reference recipe)

RL-trains a model to solve math **through the OpenCode CLI harness** running inside an
OpenSandbox box, with token IDs captured for training via run-token buffering
(`tokenid-core`).

This is the configuration that produced the reference learning curve
(**train reward 0.17 → 0.97 over ~171 steps**). It is the combination that matters:

| | easy math (DAPO) | hard math |
|---|---|---|
| **simple agent** (model already competent) | flat — nothing to learn | flat |
| **opencode harness** (model *not* competent) | **learns** ← this recipe | learns slowly |

The reward measures **harness competence**, not math difficulty. You need a harness the
model cannot yet drive, on math it *can* solve once it can drive it. Break either
condition and the curve is flat.

---

## What's in this recipe

| file | purpose |
|---|---|
| `config.yaml` | the environment: `sandbox_agent` + OpenCode + OpenSandbox |
| `launch.sh` | sanitized reference launcher (Slurm + NeMo-RL GRPO) |

Requires from this repo: `responses_api_agents/sandbox_agent/` and the `tokenid-core`
token-ID buffering in `nemo_gym/base_responses_api_{agent,model}.py`.

**No NeMo-RL changes are needed.** NeMo-RL is used as-is; everything here is Gym config
plus launcher.

---

## 1. Prepare the data

One JSONL row per task, in Responses API format. Every row's `agent_ref.name` **must
equal the top-level server key** in `config.yaml` (`sandbox_opencode_math`), or Hydra
fails with `ConfigKeyError`.

```jsonc
{
  "agent_ref": {"type": "responses_api_agents", "name": "sandbox_opencode_math"},
  "question": "...",
  "expected_answer": "...",
  "responses_create_params": {"input": [{"role": "user", "content": "Solve ..."}]}
}
```

Retag an existing math set:

```bash
python3 - <<'PY'
import json
src, dst = "math_train.jsonl", "dapo_opencode.jsonl"
with open(src) as f, open(dst, "w") as out:
    for line in f:
        row = json.loads(line)
        row["agent_ref"] = {"type": "responses_api_agents", "name": "sandbox_opencode_math"}
        out.write(json.dumps(row) + "\n")
PY
```

## 2. Prerequisites

- **OpenSandbox** reachable from the compute nodes; export `OPENSANDBOX_DOMAIN` and
  `OPENSANDBOX_API_KEY` (the config reads them via `${oc.env:...}`).
- **A model** in HF format, plus a NeMo-RL checkout and container.
- **`WANDB_API_KEY`** if logging (the launcher will refuse to submit without it).

## 3. Choose the model transport

- `model_transport: direct` — boxes can reach the compute nodes. Simplest; use it if
  it works.
- `model_transport: endpoint_bridge` — boxes **cannot** reach compute (RFC1918, no
  inbound route). A tiny in-box HTTP bridge receives model calls and the outer agent
  polls and forwards them. Requires `mode: agent_only_runner`, and the harness base URL
  must be the literal `__SANDBOX_MODEL_URL__` placeholder, which the in-box runner
  substitutes. Do **not** also set `model_server` inside `agent_config`.

Verify before assuming: run a listener on a compute node and have a box dial it. A TCP
timeout means you need the bridge.

## 4. Launch

```bash
export OPENSANDBOX_DOMAIN=... OPENSANDBOX_API_KEY=...
export WANDB_API_KEY=...

JOB_NAME=math-opencode-$(date +%Y%m%d) \
MODEL=/path/to/model-hf \
TRAIN=/path/to/dapo_opencode.jsonl \
NRL_DIR=/path/to/nemo-rl \
GYM_DIR=/path/to/Gym \
CONTAINER=/path/to/container.sqsh \
NODES=16 ./launch.sh
```

Chain chunks against a wall-clock limit by submitting the same `JOB_NAME` N times —
`--dependency=singleton` serializes them and each resumes from the last checkpoint.

## 5. Verify it's healthy

| signal | healthy | meaning |
|---|---|---|
| `train/token_mult_prob_error` | ~1.0 | token IDs round-trip; >1 order of magnitude means retokenization mismatch |
| `train/num_masked_seqs_by_logprob_error` | ~0 | sequences dropped for logprob disagreement |
| rollouts collected | 256/256 | a stuck tail means sandbox-layer trouble, not the CLI |
| reward | climbing from ~0.02 | if flat at a high value, the harness is already mastered — no headroom |

## Gotchas that cost real debugging time

- **`gbs == pps × gpp`** — asserted; and keep agent `concurrency` equal to `gbs`.
- **`max_num_steps`** — pipeclean configs ship with a tiny value (e.g. 10). Runs then
  *finish* early rather than fail, which looks identical to success. It is baked into
  the optimizer state, so **changing it breaks checkpoint resume** — start a new run.
- **`CPUS_PER_WORKER` ≤ node CPU count** — too high and the Ray head srun blocks forever
  with no log.
- **Sequence length** — some CLIs request very large `max_output_tokens`; if
  `max_model_len` is below that, *every* request 400s and the harness silently scores 0.
- **A single wedged rollout can kill the whole job.** `rollout_timeout` bounds only the
  in-box exec, so bound the outer `/run` too and return a masked sample rather than
  raising — an exception there 500s and takes down every node in the job.
- **Dense vs MoE models** — MoE-specific settings (expert parallel, MTP) must be
  disabled for dense models. MTP is gated on `is not None`, so `0` does **not** disable
  it; use `null`.
