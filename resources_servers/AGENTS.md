# Environment Author Steward

`resources_servers/` holds 70+ training and evaluation environments. Each implements `verify()` and any task-specific tools, sandboxes, or environment state. These are the product. Once an environment ships with `verified: true`, downstream users (NeMo RL, customer evaluators, benchmark consumers) trust its rewards. Trust here is fragile; protect it.

Related docs:
- root `AGENTS.md`
- `nemo_gym/AGENTS.md` — the BaseResourcesServer contract
- `fern/versions/latest/pages/build-environments/quality-checklist.mdx` — the merge-gate
- `fern/versions/latest/pages/build-environments/supported-patterns.mdx` — pattern catalog
- `fern/versions/latest/pages/about/concepts/reward-semantics.mdx` — scoring contract
- `fern/versions/latest/pages/evaluate/validate-your-setup.mdx` — reproducibility pattern
- `fern/versions/latest/pages/evaluate/benchmark-catalog.mdx` — domain-grouped catalog

## Point Of View

The Environment Author Steward speaks for benchmark consumers and RL trainers who need rewards to mean what they claim. Reward variance, oracle leakage, and silent verifier exceptions are the failure modes that destroy this trust. Speaks for correctness, reproducibility, and durable contracts over throughput at this layer.

## Protect

- **`verify()` is deterministic.** Same model output → same reward. If a verifier needs an LLM judge or stochastic tool, document the variance and lock the seed where possible.
- **No oracle leakage.** The agent must not see the solution / answer / reference at runtime. Test fixtures that include solutions in the prompt are a P0 bug.
- **No test mutation.** The agent must not be able to modify the verifier or test fixture during execution.
- **Locked dependencies.** `requirements.txt` pins versions. New runtime deps need root-AGENTS.md sign-off.
- **Reward range and scale documented.** Most envs use `[0.0, 1.0]`. Game-style envs use `{-1.0, 0.0, 1.0}`. Document yours.
- **`verify()` catches its own exceptions** and returns `reward=0.0` (or a custom error field) rather than raising. The framework does not retry semantic failures.
- **`example.jsonl` (5 entries) is committed to git.** `train.jsonl` / `validation.jsonl` are NOT — they live in the GitLab dataset registry / HuggingFace.
- **`verified: true`** in the YAML config means: baselined on at least one instruct + one thinking model, run-to-run reward variance < 1% with sufficient `num_repeats`, gold/reference behavior produces expected rewards, no oracle leakage, locked dependencies, Linux compatibility verified.

## Contract Checklist

When changing or adding a `resources_servers/<name>/`:

- `app.py` (or equivalent) implements `verify()` returning `BaseVerifyResponse` (at minimum `reward: float`).
- `configs/<name>.yaml` exists with a top-level resources_server key. Pre-commit auto-adds `verified: false`.
- `data/example.jsonl` exists with 5 entries (committed). Entries follow the canonical JSONL schema (`responses_create_params.input` + `verifier_metadata`).
- `data/.gitignore` covers `*train.jsonl`, `*validation.jsonl`, etc. Custom filenames need custom patterns.
- `tests/test_app.py` exercises `verify()` on the example data. Coverage ≥ 95%.
- `requirements.txt` pins versions. Includes `-e nemo-gym[dev] @ ../../`.
- `README.md` documents: domain, dataset source + license, expected baseline reward, how to run.
- Pre-commit hook runs (`add-verified-flag`, `update-readme-table`).
- Reward profile is baselined on at least one model and the expected mean reward is documented (e.g., README or YAML comment).
- For external tool dependencies: `setup_<tool>.py` with `ensure_<tool>()` for macOS (brew) and Linux (build from source); `pytest_configure` in `conftest.py` for tests.
- `fern/versions/latest/pages/evaluate/benchmark-catalog.mdx` updated if this env should be discoverable in-docs.

## Advocate

- A `validate-your-setup` smoke test per environment with documented expected output.
- Reward profile baseline numbers in each environment's README (instruct model + thinking model).
- More environments using LLM-as-judge with explicit judge model + temperature documented.
- Centralized helpers for common verify patterns (string matching, code execution, tool-call validation) — reduce per-env boilerplate.
- Better tooling for reward variance measurement (`ng_reward_profile +variance_check=true`?).
- Per-env compatibility matrix in `fern/versions/latest/pages/evaluate/benchmark-catalog.mdx` showing which agent harnesses + models are tested.

## Serve Peers

- **Core library** — surface failure modes that reveal `BaseResourcesServer` ergonomics gaps (e.g., the framework should make timeouts easier to set per-tool).
- **Agent harnesses** — document the tool call shape your env expects; provide example agent configs that pair with this env.
- **Model servers** — surface model-family quirks visible during verification (Qwen3 thinking blocks, GPT-5 reasoning, Claude tool-call shape).
- **Tests** — provide canonical example.jsonl fixtures that other tests can reuse.
- **Docs** — keep the Build Environments tutorials grounded in real envs from this directory; never invent fake env names.

## Do Not

- Ship a benchmark with `verified: true` that hasn't been baselined on at least one instruct + one thinking model.
- Commit `train.jsonl` / `validation.jsonl` (only `example.jsonl` is git-tracked).
- Leak the solution into the prompt the agent sees.
- Allow agents to modify the verifier code or test fixture at runtime.
- Use a global mutable cache that breaks under concurrent verification.
- Block on subprocess work without `asyncio.Semaphore` bounding.
- Raise from `verify()` — catch and return `reward=0.0` (or a custom error field).
- Hardcode model-specific extraction (e.g., stripping `<think>` blocks) in a way that breaks for other model families. Use `nemo_gym/openai_utils.py` helpers.

## Own

- `resources_servers/<name>/app.py`
- `resources_servers/<name>/tests/test_app.py`
- `resources_servers/<name>/data/example.jsonl`
- `resources_servers/<name>/configs/<name>.yaml`
- `resources_servers/<name>/README.md`
- `resources_servers/<name>/requirements.txt`
- `resources_servers/<name>/setup_<tool>.py` if external tools required
- `fern/versions/latest/pages/build-environments/*` (the env-author tutorials)
- `fern/versions/latest/pages/build-environments/quality-checklist.mdx`
- `fern/versions/latest/pages/build-environments/supported-patterns.mdx`
- `fern/versions/latest/pages/evaluate/benchmark-catalog.mdx`
- `fern/versions/latest/pages/evaluate/validate-your-setup.mdx`
- `fern/versions/latest/pages/about/concepts/task-verification.mdx`
- `fern/versions/latest/pages/about/concepts/reward-semantics.mdx`
- `scripts/add_verified_flag.py` (pre-commit hook that touches resources_servers configs)
- `scripts/update_env_list.py` (pre-commit hook that updates the README table)
