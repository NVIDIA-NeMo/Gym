# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare MR69's 16-node SWE recipe for an existing single-agent-turn harness.

This command only writes configuration and retags dataset routing references. It does
not install dependencies, start services, submit jobs, or run training. Use the
printed command with the original pipeline launcher on the target cluster.

The external recipe is read, not vendored: NVIDIA post-training/pipeline MR69,
revision 36e2364f9f79e9071a4f8e76634f73d3c204e6f9. Model, optimizer, sampling,
batch sizes, parallelism, async scheduling, and checkpointing come from that
recipe and its launcher. The existing flat-row bridge subclasses
SingleAgentTurnEnvironmentServer, so MR69's original RL checkout stays unchanged.
"""

import argparse
import hashlib
import json
import shlex
import subprocess
from collections import Counter
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


PIPELINE_REVISION = "36e2364f9f79e9071a4f8e76634f73d3c204e6f9"
RL_REVISION = "8acd27a3cade77b9e87ab606f7179b450296b957"
SANDBOX_DOMAIN = "k8s-cellrout-cellrout-101e0e7b17-b0ecd469dafdeefb.elb.us-east-2.amazonaws.com"
RECIPE_DIR = Path("RLVR/nemotron-3.5-nano-swe")
RECIPE_CONFIG = "swe_sc_8n_masked_nocompact_lag3_sbx4800_gb300.yaml"
HARNESSES = {
    "opencode": "opencode_agent",
    "claude_code": "claude_code_agent",
    "hermes": "hermes_agent",
    "pi": "pi_agent",
}
SOURCES = ("swe_rebench", "scale_swe", "swemer_v1", "swemer_v2", "swe_next")
MODEL = "geshen-honest-dolphin-v10-iter6000-mopd-step_110-hf"
TRAIN_DATA = "mixed_3_7_tasks_combined_shuffled_seed20260927.jsonl"


def _load_config(path: Path) -> DictConfig:
    config = OmegaConf.load(path)
    if not isinstance(config, DictConfig):
        raise ValueError(f"Expected a mapping in {path}")
    parent = config.pop("defaults", None)
    if parent is None:
        return config
    if not isinstance(parent, str):
        raise ValueError(f"Expected MR69's single-file defaults in {path}")
    return OmegaConf.merge(_load_config(path.parent / parent), config)


def _check_rl(rl_root: Path) -> None:
    revision = subprocess.check_output(["git", "-C", str(rl_root), "rev-parse", "HEAD"], text=True).strip()
    if revision != RL_REVISION:
        raise ValueError(f"Use MR69's RL checkout at {RL_REVISION}, got {revision}")
    gym_root = rl_root / "3rdparty/Gym-workspace/Gym"
    if gym_root.resolve() != Path(__file__).resolve().parents[1]:
        raise ValueError(f"Run this script from the shareable Gym tip checked out at {gym_root}")


def _retag_rows(source: Path, destination: Path, *, harness: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    with source.open() as src, destination.open("x") as dst:
        for index, line in enumerate(src):
            row = json.loads(line)
            name = (row.get("agent_ref") or {}).get("name", "")
            taskset = name.removesuffix("_opencode_sandboxed_agent")
            if taskset not in SOURCES:
                raise ValueError(
                    f"{source}:{index + 1}: unsupported source {name!r}. No rows are filtered; "
                    "DeNovo is loaded by MR69 but is outside its five-source default blend and these session adapters."
                )
            if not isinstance(row.get("responses_create_params"), dict):
                raise ValueError(f"{source}:{index + 1}: missing responses_create_params")
            # Preserve every task field, prompt, image, verifier setting and row order.
            # MR69's RL data processor continues to receive the same flat rows.
            alias = f"{taskset}_{HARNESSES[harness]}"
            row["agent_ref"] = {**row["agent_ref"], "name": alias}
            if "task_source" in row:
                row["task_source"] = alias
            dst.write(json.dumps(row) + "\n")
            counts[taskset] += 1
    if not counts:
        raise ValueError(f"Dataset is empty: {source}")
    return dict(counts)


def _compose(config: DictConfig, *, harness: str) -> DictConfig:
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    gym = config.env.nemo_gym
    # Keep all resource, proxy, capture and OpenSandbox settings. Replace only
    # the flat harness definitions and the config files that instantiate them.
    for source in (*SOURCES, "denovo_swe"):
        del gym[f"{source}_opencode_sandboxed_agent"]
    gym.config_paths = [
        "responses_api_models/vllm_model/configs/vllm_model_for_training.yaml",
        f"responses_api_agents/{HARNESSES[harness]}/configs/{HARNESSES[harness]}.yaml",
        *(f"resources_servers/{source}/configs/{source}.yaml" for source in (*SOURCES, "denovo_swe")),
        "nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml",
    ]
    agent = HARNESSES[harness]
    pairing = OmegaConf.load(
        Path(__file__).resolve().parents[1] / f"environment_servers/single_agent_turn/configs/swe_{harness}.yaml"
    )
    settings = OmegaConf.merge(
        pairing[agent].responses_api_agents[agent],
        {
            "token_id_capture": True,
            "datasets": [],
            "num_workers": 1,
        },
    )
    # Gym resolves its serving alias after RL injects the model endpoint.
    # Keep that Gym-only interpolation out of the outer RL configuration.
    if "model" in settings:
        del settings["model"]
    if harness == "hermes":
        settings.update(max_tokens=196608, sandbox_runner_timeout_seconds=4800)
    else:
        settings.update(context_window=196608, max_output_tokens=196608, timeout=4800)
    for source in SOURCES:
        alias = f"{source}_{agent}"
        gym[alias] = {"_inherit_from": agent, "responses_api_agents": {agent: settings}}
        gym[f"{source}_resources_server"].resources_servers[source].allowed_agents = [agent]
        gym[f"{source}_environment"] = {
            "environment_servers": {
                "single_agent_turn_legacy": {
                    "entrypoint": "app.py",
                    "resources_server": {"type": "resources_servers", "name": f"{source}_resources_server"},
                    "agent_server": {"type": "responses_api_agents", "name": alias},
                    "default_episode_timeout_seconds": 8400,
                    "cleanup_timeout_seconds": 180,
                    "resources_tool_transports": [],
                }
            }
        }
    gym.sandbox.opensandbox.connection.domain = SANDBOX_DOMAIN
    gym.sandbox.opensandbox.connection.protocol = "https"
    # The original rule tests process completion, not whether the model stopped
    # naturally. A length-limited model result can still have a completed process.
    for rule in config.env.mask_sample_rules:
        if rule.field == "opencode_finished":
            rule.field = "agent_execution_finished"
    return config


def main() -> None:
    """Write one prepared recipe and its provenance; print the original launch command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--rl-root", type=Path, required=True)
    parser.add_argument("--harness", choices=HARNESSES, required=True)
    parser.add_argument(
        "--assets-root", type=Path, required=True, help="MR69 SWE_ASSETS directory on the target cluster"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    pipeline = args.pipeline_root.resolve()
    revision = subprocess.check_output(["git", "-C", str(pipeline), "rev-parse", "HEAD"], text=True).strip()
    if revision != PIPELINE_REVISION:
        raise ValueError(f"Expected pipeline {PIPELINE_REVISION}, got {revision}")
    recipe = pipeline / RECIPE_DIR
    if subprocess.check_output(
        ["git", "-C", str(pipeline), "status", "--porcelain", "--", str(RECIPE_DIR)], text=True
    ).strip():
        raise ValueError("MR69 recipe has local changes; use the pinned recipe without modifications")
    _check_rl(args.rl_root)
    model = args.assets_root.resolve() / "models" / MODEL
    train = args.assets_root.resolve() / "data" / TRAIN_DATA
    if not model.is_dir():
        raise ValueError(f"MR69 model directory is missing: {model}")
    output = args.output_dir.resolve()
    # A fresh directory prevents overwriting any source dataset or prior recipe.
    output.mkdir(parents=True, exist_ok=False)
    counts = _retag_rows(train, output / "train.jsonl", harness=args.harness)
    config = _compose(_load_config(recipe / "configs" / RECIPE_CONFIG), harness=args.harness)
    # These are the original 8n11 launcher's effective overrides, not smoke settings.
    config.cluster.num_nodes = 16
    config.cluster.segment_size = 4
    config.policy.generation.colocated.resources.num_nodes = 12
    config.grpo.max_num_steps = 50
    config.policy.model_name = str(model)
    config.data.train.data_path = str(output / "train.jsonl")
    config.data.validation.data_path = str(output / "train.jsonl")
    for key in ("model_save_format", "save_consolidated"):
        del config.checkpointing[key]
    config.checkpointing.update(save_period=5, ft_save_period=5, keep_top_k=10)
    OmegaConf.save(config, output / "config.yaml", resolve=False)
    manifest = {
        "pipeline_revision": revision,
        "rl_revision": subprocess.check_output(
            ["git", "-C", str(args.rl_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "gym_revision": subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"], text=True
        ).strip(),
        "harness": args.harness,
        "input": str(train),
        "counts": counts,
        "validation": "configuration preparation only; no services, model rollouts or training run",
    }
    with train.open("rb") as stream:
        manifest["input_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    (output / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    command = [
        "env",
        f"NEMO_RL_ROOT={args.rl_root.resolve()}",
        f"CONFIG_PATH_HOST={output / 'config.yaml'}",
        f"MODEL_PATH={model}",
        f"TRAIN_PATH={output / 'train.jsonl'}",
        f"VAL_PATH={output / 'train.jsonl'}",
        "MOUNT_GYM_CODE=1",
        f"OPENSANDBOX_DOMAIN={SANDBOX_DOMAIN}",
        "OPENSANDBOX_PROTOCOL=https",
        "bash",
        str(recipe / "scripts/launch_nano35_swe_8n11.sh"),
        f"env.nemo_gym.sandbox.opensandbox.connection.domain={SANDBOX_DOMAIN}",
        "env.nemo_gym.sandbox.opensandbox.connection.protocol=https",
    ]
    print(f"Prepared {sum(counts.values())} rows with unchanged task content. Launch explicitly when authorized:")
    print(shlex.join(command))


if __name__ == "__main__":
    main()
