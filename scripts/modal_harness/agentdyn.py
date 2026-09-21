# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch the AgentDyn defense matrix on Modal -- one container per (model, defense) cell.

    modal run --detach -m scripts.modal_harness.agentdyn --git-ref <branch>
    modal run --detach -m scripts.modal_harness.agentdyn --models ultra,kimi --defenses camel,progent

AgentDyn's agent server serializes rollouts behind a semaphore of 1, so processes are the
only parallelism available and the grid fans out one per cell. On a single machine that
capped the run at five concurrent stacks; here each cell is its own container with its own
ports, Ray cluster and memory, so the ceiling is the Modal concurrency limit instead.

Two things differ from a typical campaign and both are handled below rather than by the
generic runner:

* **No judge.** ASR and utility are deterministic checks against suite state, so the stack
  is two servers and the YAML defines no judge. `judge_server_name=None` suppresses the
  block entirely -- emitting one would fail the merged-config check on a dangling reference.
* **The treatment is server config, not row data.** Every cell reads the same byte-identical
  selector file and selects its defense with `default_defense` on the agent server, so the
  cell identity goes into the generated env YAML rather than into the input.
"""

from __future__ import annotations

from scripts.modal_harness.campaign import app, campaign_state, run_campaign, skip_reason


#: key -> (slug fragment, base_url, model id, token env var)
MODELS: dict[str, tuple[str, str, str, str]] = {
    "ultra": (
        "ultra",
        "https://snorkelai-fdr--ep-nvidia-nemotron-3-ultra-550b-a55b-nvfp-63eebc.us-west.modal.direct/v1",
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
        "MODAL_PROXY_TOKEN",
    ),
    # Dedicated deployment for this grid. Same model id as the shared endpoint
    # (`moonshotai/Kimi-K3`, verified via /v1/models), so this is a deployment change
    # rather than a model change and kimi rows stay poolable across it.
    "kimi": (
        "kimi",
        "https://snorkelai-fdr--ep-kimi-k3-tb3-a-server.us-west.modal.direct/v1",
        "moonshotai/Kimi-K3",
        "MODAL_PROXY_TOKEN",
    ),
    "qwen": (
        "qwen",
        "https://snorkelai-fdr--ep-qwen3-5-122b-a10b-fp8-server.us-west.modal.direct/v1",
        "Qwen/Qwen3.5-122B-A10B-FP8",
        "MODAL_PROXY_TOKEN",
    ),
    "supervl": (
        "supervl",
        "https://snorkelai-fdr--nemotron-3-5-super-vl-ea-nemotronvision.us-east.modal.direct/v1",
        "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
        "SUPER_VL_MODAL_TOKEN",
    ),
}

DEFENSES = (
    "prompt_guard_2_detector",
    "piguard_detector",
    "transformers_pi_detector",
    "spotlighting_with_delimiting",
    "repeat_user_prompt",
    "tool_filter",
    "camel",
    "progent",
    "drift",
)

#: These load a HuggingFace classifier, one of them from a gated repo. They need HF_TOKEN in
#: the campaign secret and benefit most from the shared HF cache volume, since otherwise
#: every cold container re-downloads ~1GB.
DETECTOR_DEFENSES = frozenset({"prompt_guard_2_detector", "piguard_detector", "transformers_pi_detector"})

CONFIG_PATHS = ["benchmarks/agentdyn/config.yaml"]
AGENT = "agentdyn_benchmark"
INPUT_PATH = "benchmarks/agentdyn/data/agentdyn_v1_2_2.jsonl"
PREPARE_MODULE = "benchmarks.agentdyn.prepare"
EXPECTED_ROWS = 620

#: AgentDyn's stack is the agent and the policy model; there is no judge.
EXPECTED_SERVERS = 2


def cell_config(model_key: str, defense: str) -> dict[str, object]:
    """Per-cell server overrides written into the generated env YAML.

    `default_defense` is the treatment. `model_system_role` is a per-model quirk: the Qwen
    deployment rejects the `developer` role with "Unexpected message role", so it has to be
    told to use `system` instead.
    """
    agent: dict[str, object] = {"default_defense": defense}
    if model_key == "qwen":
        agent["model_system_role"] = "system"
    return {"agentdyn_benchmark": {"responses_api_agents": {"agentdyn_agent": agent}}}


@app.local_entrypoint()
def main(
    git_ref: str = "claude-rundgren/agentdyn-defense-matrix-84bfb9",
    models: str = ",".join(MODELS),
    defenses: str = ",".join(DEFENSES),
    concurrency: int = 64,
) -> None:
    model_keys = [key.strip() for key in models.split(",") if key.strip()]
    defense_keys = [key.strip() for key in defenses.split(",") if key.strip()]
    unknown = [key for key in model_keys if key not in MODELS]
    if unknown:
        raise SystemExit(f"unknown model keys: {', '.join(unknown)}")
    unknown = [key for key in defense_keys if key not in DEFENSES]
    if unknown:
        raise SystemExit(f"unknown defenses: {', '.join(unknown)}")

    # Skip cells that are already complete or already running. A preempted launcher takes
    # its un-spawned cells with it, and the obvious recovery -- rerun the same command --
    # would otherwise spawn a second container for every cell that did start.
    state = campaign_state.remote("agentdyn")

    handles = []
    skipped = []
    for model_key in model_keys:
        fragment, base_url, model_id, token_var = MODELS[model_key]
        for defense in defense_keys:
            slug = f"{fragment}-{defense}"
            reason = skip_reason(state, slug, EXPECTED_ROWS)
            if reason:
                skipped.append(f"{slug}: {reason}")
                continue
            handles.append(
                (
                    slug,
                    run_campaign.spawn(
                        slug=slug,
                        namespace="agentdyn",
                        config_paths=CONFIG_PATHS,
                        agent=AGENT,
                        input_path=INPUT_PATH,
                        policy_base_url=base_url,
                        policy_model=model_id,
                        policy_token_var=token_var,
                        judge_server_name=None,
                        expected_rows=EXPECTED_ROWS,
                        expected_servers=EXPECTED_SERVERS,
                        concurrency=concurrency,
                        prepare_module=PREPARE_MODULE,
                        git_ref=git_ref,
                        extra_config=cell_config(model_key, defense),
                    ),
                )
            )

    for line in skipped:
        print(f"  skip {line}")
    total = len(handles) * EXPECTED_ROWS
    print(f"\n{len(handles)} cells x {EXPECTED_ROWS} = {total:,} rollouts, running in parallel\n")
    for slug, handle in handles:
        result = handle.get()
        short = f" ({result['missing']} missing)" if result["missing"] else ""
        print(f"  {slug:36s} {result['landed']}/{result['expected']}{short}")
