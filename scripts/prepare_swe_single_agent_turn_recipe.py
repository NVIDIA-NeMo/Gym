# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare matched SWE training inputs using RL's existing controller and Gym task loaders.

No model calls, sandboxes or training jobs are started. Harbor preparation may read
image metadata from registries; checkpoint preparation reads Hub file hashes. The output is a new directory for one comparison arm.
"""

import argparse
import copy
import importlib.util
import json
import math
import subprocess
from pathlib import Path

from huggingface_hub import HfApi
from omegaconf import DictConfig, OmegaConf
from scripts.prepare_swe_verifier_controls import ROOT, clean_revision, digest, select_rows
from scripts.swe_rows_to_harbor.convert import convert, server_config, task_folder_name

from nemo_gym.data import materialize_task
from nemo_gym.data.tasks.harbor import materialize_task as materialize_harbor_task
from nemo_gym.data.tasks.harbor.materialize import taskset_mapping
from nemo_gym.data.tasks.harbor.prepare import PreparedTaskset, prepare_target


PARSER_REVISION = "c71902a8cf8d2b725f63d51f199f4d3e56f68d2d"
MODELS = {
    "4b": ("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f", 1),
    "30b": ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "bf77c3174f68ad409e1c2aa60daeb46e32d1c606", 2),
}
HARNESSES = {
    "opencode": {
        "opencode_version": "1.17.11",
        "concurrency": 8,
        "context_window": 32768,
        "max_output_tokens": 1024,
        "timeout": 480,
        "sandbox_install_timeout_seconds": 480,
        "report_agent_sandbox_oom": True,
        "opencode_config": {"compaction": {"auto": False, "prune": False}},
    },
    "claude_code": {
        "claude_code_version": "2.1.108",
        "model": "claude-sonnet-4-6",
        "model_server": {"type": "responses_api_models", "name": "policy_model"},
        "anthropic_api_key": "",
        "bare": True,
        "thinking": "enabled",
        "max_turns": 30,
        "timeout": 480,
    },
    "hermes": {
        "context_length": 32768,
        "max_tokens": 1024,
        "sandbox_runner_timeout_seconds": 480,
        "compression_enabled": False,
        "chat_template_kwargs_enabled": True,
        "max_turns": 30,
        "terminal_timeout": 60,
    },
    "pi": {
        "pi_version": "0.80.2",
        "context_window": 32768,
        "max_output_tokens": 1024,
        "output_token_policy": "fixed",
        "auto_compaction": False,
        "timeout": 480,
    },
    "codex": {
        "codex_version": "0.144.4",
        "model": "gym-policy-model",
        "model_context_window": None,
        "model_auto_compact_token_limit": None,
        "features": {"multi_agent": False, "code_mode": False, "token_budget": False},
        "timeout": 480,
    },
    "nooa": {
        "entrypoint": "app.py",
        "model_server": {"type": "responses_api_models", "name": "policy_model"},
        "nooa": {
            "agent_class": "nooa_atom.gym:AtomHarness",
            "invocation_adapter": "responses_api_agents.nooa_agent.atom_adapter:invoke_atom",
            "execution_mode": "sandboxed",
            "init_kwargs": {"turn_method": "handle_batch"},
        },
        "context_window": 32768,
        "sandbox_runner_timeout_seconds": 480,
        "max_policy_calls": None,
    },
}
OLD_AGENTS = {
    "opencode": "opencode_sandboxed_agent",
    "claude_code": "claude_code_agent",
    "hermes": "hermes_sandboxed_agent",
    "pi": "pi_sandboxed_agent",
    "codex": "codex_agent",
}
CODEX_EXPORT = {
    "source_url": "https://github.com/openai/codex",
    "source_revision": "8c68d4c87dc54d38861f5114e920c3de2efa5876",
    "codex_version": "0.144.4",
    "model": "gym-policy-model",
    "status": "exporter_checks_passed",
    "exporter_sha256": "f3408007c4289c1a4908ed93298803d669d6d24c172c6a44fa1426ab4f902852",
}


def validate_harnesses(harnesses: list[str], *, route: str | None = None) -> None:
    """Require the fixed mixture order and an actual implementation of the selected route."""
    if not harnesses or any(name not in HARNESSES for name in harnesses):
        raise ValueError("Select supported harnesses: " + " ".join(HARNESSES))
    if len(set(harnesses)) != len(harnesses):
        raise ValueError("Each harness must occur once in the requested blend")
    if harnesses != [name for name in HARNESSES if name in harnesses]:
        raise ValueError("Use task-major harness order: " + " ".join(HARNESSES))
    if route == "old-run" and "nooa" in harnesses:
        raise ValueError("Atom (nooa) has no old-run route; select single-agent-turn or harbor")


def harness_assets(harnesses: list[str], *, codex_catalog: Path | None) -> dict[str, bytes]:
    """Validate prepared runtime inputs before creating an output directory or contacting the Hub."""
    validate_harnesses(harnesses)
    assets = {}
    if "nooa" in harnesses:
        assets["atom-settings.yaml"] = b"atom:\n  summarization:\n    policy: none\n  context_error_policy: raise\n"
    if "claude_code" in harnesses:
        settings = {
            "env": {
                "DISABLE_COMPACT": "1",
                "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "32768",
                "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "1024",
            }
        }
        assets["claude-settings.json"] = (json.dumps(settings, indent=2) + "\n").encode()
    if "codex" not in harnesses:
        if codex_catalog is not None:
            raise ValueError("--codex-catalog requires selecting codex")
        return assets
    if codex_catalog is None or codex_catalog.name != "models.json":
        raise ValueError("Codex requires --codex-catalog models.json from the pinned exporter")
    for name in ("models.json", "fallback.json", "provenance.json"):
        assets[f"codex/{name}"] = (codex_catalog.parent / name).read_bytes()
    provenance = json.loads(assets["codex/provenance.json"])
    for key, expected in CODEX_EXPORT.items():
        if provenance.get(key) != expected:
            raise ValueError(f"Codex export provenance mismatch: {key}")
    for name in ("models.json", "fallback.json"):
        if digest(assets[f"codex/{name}"]) != provenance.get(f"{name}_sha256"):
            raise ValueError(f"Codex export bytes changed: {name}")
    catalog = json.loads(assets["codex/models.json"])["models"]
    slugs = [entry["slug"] for entry in catalog]
    alias = HARNESSES["codex"]["model"]
    if len(set(slugs)) != len(slugs) or alias not in slugs:
        raise ValueError("Codex catalog requires unique slugs and the exact gym-policy-model alias")
    descriptor = catalog[slugs.index(alias)]
    fallback = json.loads(assets["codex/fallback.json"])
    for field in ("context_window", "max_context_window", "auto_compact_token_limit"):
        if descriptor.pop(field, None) is not None:
            raise ValueError(f"Codex automatic context control must be absent: {field}")
        fallback.pop(field, None)
    if descriptor != fallback or not isinstance(descriptor.get("base_instructions"), str):
        raise ValueError("Codex catalog must preserve the complete resolved fallback except context fields")
    return assets


def load_rl_config(rl_root: Path) -> DictConfig:
    """Use the selected RL revision's own inheritance rules without importing training workers."""
    spec = importlib.util.spec_from_file_location("swe_rl_config", rl_root / "nemo_rl/utils/config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recipe = OmegaConf.load(ROOT / "examples/training/swe_single_agent_turn.yaml")
    parent = recipe.pop("defaults")
    return module.merge_with_override(module.load_config(rl_root / "examples/configs" / parent), recipe)


def configure_gym(
    config: DictConfig,
    *,
    harnesses: list[str],
    route: str,
    output: Path,
    parser_cache: Path,
    prepared: PreparedTaskset | None = None,
) -> None:
    """Bind each harness to its own resources or Harbor environment instance."""
    validate_harnesses(harnesses, route=route)
    gym = config.env.nemo_gym
    # These recipes gather pairing evidence; they do not extend the benchmark's qualified harness list.
    gym.allow_unsupported_pairing = True
    gym.config_paths = [
        "responses_api_models/vllm_model/configs/vllm_model_for_training.yaml",
        "nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml",
    ]
    gym.environment_server_routes = {}
    gym.environment_routing_mode = "agent" if route == "old-run" else "taskset"
    if route != "harbor":
        resources = server_config("swe_rebench")
        resources.update(
            evaluation_timeout=180,
            inconclusive_verification_retries=1,
            log_parser_cache_dir=str(parser_cache),
            verification_log_dir=str(output / "verifier"),
        )
        resources["sandbox_config"].update(ttl_s=1800, ready_timeout_s=120, env={"OPENCODE_VERSION": "1.17.11"})
        resources["sandbox_config"]["provider_options"]["platform"] = {"os": "linux", "arch": "amd64"}
        gym.swe_resources = {"resources_servers": {"swe_rebench": resources}}
    for harness in harnesses:
        agent = OLD_AGENTS[harness] if route == "old-run" else f"{harness}_agent"
        # Pi has no generic sandboxed YAML; NOOA's template also starts an unbound environment.
        if agent not in ("pi_sandboxed_agent", "nooa_agent"):
            gym.config_paths.append(f"responses_api_agents/{agent}/configs/{agent}.yaml")
        settings = copy.deepcopy(HARNESSES[harness])
        if route == "old-run" and harness == "opencode":
            settings = {
                "opencode_version": "1.17.11",
                "concurrency": 8,
                "opencode_max_context_window": 32768,
                "opencode_query_via_stdin": True,
                "sandbox_timeout": 480,
                "report_agent_sandbox_oom": True,
                "sandbox_provider": "sandbox",
                "sandbox_config": {},
                "opencode_env": {"OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX": "1024"},
                "opencode_config": {
                    "compaction": {"auto": False, "prune": False},
                    "provider": {
                        "nemo_gym": {
                            "models": {
                                "dummy_model": {
                                    "limit": {"output": 1024},
                                    "interleaved": {"field": "reasoning_content"},
                                }
                            }
                        }
                    },
                },
            }
        elif route == "old-run" and harness == "hermes":
            settings.pop("sandbox_runner_timeout_seconds")
            settings.pop("chat_template_kwargs_enabled")
            settings.update(
                runtime_python=None,
                model="${policy_model_name}",
                concurrency=32,
                sandbox_timeout=480,
                chat_template_kwargs={"truncate_history_thinking": False},
            )
        elif agent == "pi_sandboxed_agent":
            settings.update(
                entrypoint="app.py",
                model_server={"type": "responses_api_models", "name": "policy_model"},
                model="${policy_model_name}",
                sandbox_provider="sandbox",
                sandbox_config={},
                network_access="inherit",
                execution_failure_reward_zero=False,
            )
        if harness == "claude_code":
            settings["settings"] = str(output / "claude-settings.json")
        elif harness == "codex":
            settings["model_catalog_json"] = str(output / "codex/models.json")
        elif harness == "nooa":
            settings["settings_file"] = str(output / "atom-settings.yaml")
        settings.update(token_id_capture=True, num_workers=1)
        if harness != "nooa":
            settings["datasets"] = []
        if route == "old-run":
            settings["sandbox_provider"] = "sandbox"
        settings["resources_server"] = (
            {"type": "resources_servers", "name": "swe_resources"} if route == "old-run" else None
        )
        gym[agent] = {"responses_api_agents": {agent: settings}}
        taskset = f"swe_rebench_{harness}"
        environment = f"{taskset}_environment"
        impl = {"old-run": "legacy_agent", "single-agent-turn": "single_agent_turn", "harbor": "harbor"}[route]
        options = {
            "entrypoint": "app.py",
            "agent_server": {"type": "responses_api_agents", "name": agent},
            "default_episode_timeout_seconds": 1200,
        }
        if route != "old-run":
            options.update(cleanup_timeout_seconds=180, max_concurrent_episodes=None, queue_timeout_seconds=None)
            gym.environment_server_routes[taskset] = environment
        if route == "single-agent-turn":
            options["resources_server"] = {"type": "resources_servers", "name": "swe_resources"}
            options["resources_tool_transports"] = []
        elif route == "harbor":
            options.update(
                sandbox_provider="sandbox",
                sandbox_ready_timeout_s=120,
                sandbox_ttl_slack_s=420,
                enforce_task_timeouts=False,
                verifier_completion_key="evaluation_completed",
                inconclusive_verification_retries=1,
                sandbox_resources_override={"cpu": 8, "memory_mib": 30720, "disk_gib": 30},
                sandbox_provider_options={
                    "platform": {"os": "linux", "arch": "amd64"},
                    "resource_requests": {"cpu": 1, "memory_mib": 4096, "disk_gib": 30},
                },
                artifacts_dir=str(output / "verifier" / harness),
                tasksets={taskset: taskset_mapping(prepared.tasks, prepared.folder)},
            )
        gym[environment] = {"environment_servers": {impl: options}}


def materialize_training_rows(
    rows: list[dict], *, harnesses: list[str], route: str, prepared: PreparedTaskset | None = None
) -> list[dict]:
    """Keep task-major order across harnesses and preserve each route's actual request contract."""
    validate_harnesses(harnesses, route=route)
    by_id = {task.task_id: task for task in prepared.tasks} if prepared else {}
    materialized = []
    for row in rows:
        for harness in harnesses:
            taskset = f"swe_rebench_{harness}"
            if route == "old-run":
                record = row | {
                    "agent_ref": {"type": "responses_api_agents", "name": OLD_AGENTS[harness]},
                    "task_source": taskset,
                }
            elif prepared:
                record = materialize_harbor_task(
                    by_id[task_folder_name("swe_rebench", row["instance_id"])], taskset
                ).model_dump(mode="json")
            else:
                record = materialize_task(row, taskset=taskset)
                payload = record["task_input"]
                params = payload.pop("responses_create_params")
                record["task_input"] = {"responses_create_params": params, "task_data": payload}
            materialized.append(record)
    return materialized


def model_manifest(repo: str, revision: str, path: Path) -> dict:
    """Read immutable Hub file hashes without downloading checkpoint weights."""
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise ValueError("Hub metadata did not resolve to the requested model commit")
    files = {}
    for entry in info.siblings:
        if entry.rfilename.endswith(".md") or entry.rfilename == ".gitattributes":
            continue
        algorithm, value = ("sha256", entry.lfs.sha256) if entry.lfs else ("git_blob_sha1", entry.blob_id)
        if not value:
            raise ValueError(f"Hub metadata lacks a content hash: {entry.rfilename}")
        files[entry.rfilename] = {"algorithm": algorithm, "digest": value}
    return {
        "repo": repo,
        "revision": revision,
        "path": str(path),
        "files": files,
        "status": "Expected Hub content hashes; local snapshot is checked before launch",
    }


def prepare(args: argparse.Namespace) -> dict:
    """Write a fixed cohort/task/harness order and its complete configuration provenance."""
    rl_root, output = args.rl_root.resolve(), args.output_dir.resolve()
    validate_harnesses(args.harnesses, route=args.route)
    assets = harness_assets(args.harnesses, codex_catalog=args.codex_catalog)
    if clean_revision(rl_root) != args.rl_revision:
        raise ValueError(f"RL HEAD must match the explicit reviewed revision {args.rl_revision}")
    parser_cache = args.parser_cache.resolve()
    if clean_revision(parser_cache / "SWE-rebench-V2") != PARSER_REVISION:
        raise ValueError(f"Use the matched verifier parser revision {PARSER_REVISION}")
    gym_revision = clean_revision(ROOT)
    gym_pin = subprocess.check_output(
        ["git", "-C", str(rl_root), "rev-parse", "HEAD:3rdparty/Gym-workspace/Gym"], text=True
    ).strip()
    if gym_pin != gym_revision:
        raise ValueError("The reviewed RL commit must pin this Gym revision in its submodule and dependency lock")
    if args.steps in (1, 5) and (args.count != 8 or args.harnesses != ["opencode"]):
        raise ValueError("The one/five-update qualification uses OpenCode and the first eight tasks")
    rows, tasks = select_rows(args.source, args.roster, args.image_pins, count=args.count)
    model, revision, ep = MODELS[args.model]
    model_path = args.model_path.resolve()
    if model_path.name != revision:
        raise ValueError(f"Use the pinned Hugging Face snapshot directory ending in {revision}")
    model_provenance = model_manifest(model, revision, model_path)
    output.mkdir(parents=True, exist_ok=False)
    for name, data in assets.items():
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    source = output / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    prepared = None
    if args.route == "harbor":
        convert(
            source,
            output / "harbor",
            dataset_name="swe-training",
            parser_cache_dir=args.parser_cache,
            agent_timeout_sec=480,
            evaluation_timeout_sec=180,
            default_source="swe_rebench",
        )
        prepared = prepare_target(
            str(output / "harbor"), output_root=output / "prepared", sandbox_provider="opensandbox"
        )
        if prepared.skipped or len(prepared.tasks) != len(rows):
            raise ValueError(f"Harbor preparation must preserve every selected task: {prepared.skipped}")
    materialized = materialize_training_rows(rows, harnesses=args.harnesses, route=args.route, prepared=prepared)
    train = output / "train.jsonl"
    train.write_text("".join(json.dumps(row) + "\n" for row in materialized))
    config = load_rl_config(rl_root)
    configure_gym(
        config, harnesses=args.harnesses, route=args.route, output=output, parser_cache=parser_cache, prepared=prepared
    )
    config.policy.model_name = str(model_path)
    config.policy.megatron_cfg.expert_model_parallel_size = ep
    config.grpo.num_prompts_per_step = 8 * len(args.harnesses)
    config.policy.train_global_batch_size = 4 * config.grpo.num_prompts_per_step
    config.grpo.max_num_steps = args.steps
    config.grpo.max_num_epochs = math.ceil(args.steps * 8 / args.count)
    config.async_rl.max_inflight_prompts = 4 if args.steps in (1, 5) else 8 if len(args.harnesses) == 1 else 16
    config.data.train.data_path = str(train)
    config.checkpointing.checkpoint_dir = str(output / "checkpoints")
    if args.steps == 1:
        config.checkpointing.save_period = 1
    config.logger.log_dir = str(output / "logs")
    config.logger.wandb_enabled = args.wandb_project is not None
    if args.wandb_project:
        config.logger.wandb.project = args.wandb_project
    config.logger.wandb.name = output.name
    OmegaConf.save(config, output / "config.yaml", resolve=False)
    manifest = {
        "gym_commit": gym_revision,
        "rl_commit": args.rl_revision,
        "gym_lock_sha256": digest((ROOT / "uv.lock").read_bytes()),
        "rl_lock_sha256": digest((rl_root / "uv.lock").read_bytes()),
        "source_pool_sha256": json.loads(args.roster.read_text())["source_uncompressed_sha256"],
        "roster_sha256": digest(args.roster.read_bytes()),
        "image_pins_sha256": digest(args.image_pins.read_bytes()),
        "parser": {"path": str(parser_cache / "SWE-rebench-V2"), "commit": PARSER_REVISION},
        "model": model_provenance,
        "route": args.route,
        "harnesses": args.harnesses,
        "tasks": tasks,
        "steps": args.steps,
        "episodes": args.steps * 8 * len(args.harnesses) * 4,
        "ordering": "cohort of eight tasks, task, harness; no shuffle; repeat full epochs",
        "runtime_files_sha256": {},
        "files_sha256": {
            name: digest((output / name).read_bytes())
            for name in ("source.jsonl", "train.jsonl", "config.yaml", *assets)
        },
        "qualification": "Preparation only. No model, sandbox, optimizer update or container qualification.",
    }
    for label, root in (("gym", ROOT), ("rl", rl_root)):
        names = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"], text=True).split("\0")
        for name in names:
            path = root / name
            if (
                path.is_file()
                and path.suffix in {".py", ".yaml", ".yml", ".toml", ".lock", ".sh", ".js", ".mjs", ".cjs"}
                and "tests" not in path.parts
            ):
                manifest["runtime_files_sha256"][f"{label}/{name}"] = digest(path.read_bytes())
    (output / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rl-root", "source", "parser-cache", "model-path", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--rl-revision", required=True, help="Exact reviewed RL commit used by the container build")
    for name in ("roster", "image-pins"):
        parser.add_argument(f"--{name}", type=Path, default=ROOT / "examples/training/swe_selection.json")
    parser.add_argument("--route", choices=("old-run", "single-agent-turn", "harbor"), default="single-agent-turn")
    parser.add_argument("--harnesses", choices=HARNESSES, nargs="+", default=["opencode"])
    parser.add_argument(
        "--codex-catalog", type=Path, help="Pinned export models.json with adjacent fallback.json and provenance.json"
    )
    parser.add_argument("--model", choices=MODELS, default="4b")
    parser.add_argument("--count", choices=(8, 16, 32, 64), type=int, default=32)
    parser.add_argument("--steps", choices=(1, 5, 20), type=int, default=20)
    parser.add_argument("--wandb-project")
    manifest = prepare(parser.parse_args())
    print(
        json.dumps(
            {
                "prepared_rows": len(manifest["tasks"]) * len(manifest["harnesses"]),
                "planned_episodes": manifest["episodes"],
            }
        )
    )


if __name__ == "__main__":
    main()
