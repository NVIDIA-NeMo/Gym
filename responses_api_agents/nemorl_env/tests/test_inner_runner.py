# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf


TASK = Path(__file__).parents[1] / "task_environment"


@pytest.fixture
def base_config():
    return OmegaConf.create(
        {
            "grpo": {},
            "policy": {
                "tokenizer": {},
                "dtensor_cfg": {},
                "generation": {
                    "vllm_cfg": {},
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": -1,
                },
            },
            "data": {
                "train": {},
                "validation": None,
                "default": {
                    "prompt_file": "examples/prompts/cot.txt",
                    "system_prompt_file": None,
                    "processor": "math_hf_data_processor",
                    "env_name": "math",
                },
            },
            "env": {"math": {"math_verify_impl": "hf_math_verify"}},
            "loss_fn": {},
            "checkpointing": {},
            "cluster": {},
            "logger": {"wandb": {}},
        }
    )


@pytest.mark.parametrize(
    ("budget", "checkpoint_deadline"),
    [(None, "00:01:00:00"), (3480, "00:00:58:00"), (1800, "00:00:30:00"), (60, "00:00:01:00")],
)
def test_training_has_no_benchmark_data_or_validation(base_config, monkeypatch, budget, checkpoint_deadline):
    if budget is None:
        monkeypatch.delenv("INNER_TRAIN_SECONDS", raising=False)
    else:
        monkeypatch.setenv("INNER_TRAIN_SECONDS", str(budget))
    authored = OmegaConf.load(TASK / "recipe.yaml")
    authored.data.default.system_prompt_file = "/root/aime25.jsonl"
    authored.env = {"math": {"math_verify_impl": "wrong_verifier"}}
    tree = ast.parse((TASK / "launch_inner.py").read_text())
    stop = next(
        i
        for i, node in enumerate(tree.body)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "train_log"
    )
    nodes = [node for node in tree.body[:stop] if not isinstance(node, (ast.Import, ast.ImportFrom))]
    namespace = {"OmegaConf": OmegaConf, "Path": Path, "os": os}
    with patch.object(OmegaConf, "load", side_effect=[base_config, authored]), patch.object(OmegaConf, "save"):
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "launch_inner.py", "exec"), namespace)
    cfg = namespace["cfg"]
    assert cfg.grpo.num_prompts_per_step == cfg.grpo.num_generations_per_prompt == 8
    assert cfg.policy.train_global_batch_size == 64
    assert namespace["seconds"] == (3600 if budget is None else budget)
    assert cfg.checkpointing.checkpoint_must_save_by == checkpoint_deadline
    assert cfg.data.train.data_path == "/testbed/train_math.jsonl"
    assert cfg.data.train.split_validation_size == 0 and cfg.data.validation is None
    assert cfg.grpo.val_period == 0 and not cfg.grpo.val_at_start and not cfg.grpo.val_at_end
    assert cfg.data.default.system_prompt_file is None
    assert cfg.env.math.math_verify_impl == "hf_math_verify"
    assert cfg.checkpointing.model_save_format is None


@pytest.mark.parametrize("stage", ["train", "eval"])
@pytest.mark.parametrize("mode", ["setup", "execute", None])
def test_bootstrap_separates_pristine_setup_from_authored_execution(tmp_path, monkeypatch, stage, mode):
    monkeypatch.delenv("INNER_TRAIN_STARTED_AT", raising=False)
    work = tmp_path / "testbed"
    repo = work / "NeMo-RL"
    (repo / ".venv/bin").mkdir(parents=True)
    stubs = tmp_path / "bin"
    stubs.mkdir()
    command_log = tmp_path / "commands.log"
    stub = stubs / "command"
    stub.write_text('#!/bin/bash\nprintf "%s|%s\\n" "${INNER_TRAIN_STARTED_AT-unset}" "$0 $*" >> "$COMMAND_LOG"\n')
    stub.chmod(0o755)
    for name in ("git", "apt-get", "sed", "python", "uv"):
        (stubs / name).symlink_to(stub)
    (repo / ".venv/bin/python").symlink_to(stub)
    script = (TASK / "run.sh").read_text().replace("/testbed", str(work)).replace("/opt/conda/bin", str(stubs))
    subprocess.run(
        ["bash", "-s", "--", stage, *([mode] if mode else [])],
        input=script,
        text=True,
        env={**os.environ, "PATH": f"{stubs}:{os.environ['PATH']}", "COMMAND_LOG": str(command_log)},
        check=True,
        timeout=10,
    )
    commands = command_log.read_text()
    starts = {line.split("|", 1)[0] for line in commands.splitlines()}
    assert len(starts) == 1
    if mode == "execute":
        assert next(iter(starts)).isdigit()
    else:
        assert starts == {"unset"}
    assert ("git clone " in commands) is (mode != "execute")
    assert ("checkout 1cee83587d0f0d2ba82e7cdeced9772641fddbe3" in commands) is (mode != "execute")
    assert ("apt-get " in commands) is (mode != "execute")
    assert ("sed -i " in commands) is (mode != "setup")
    assert "uv sync --frozen" in commands
    applies_patch = stage == "train" and mode != "setup"
    assert ("apply --check /root/change.diff" in commands) is applies_patch
    if mode == "setup":
        assert "launch_inner.py" not in commands and "evaluate.py" not in commands
    else:
        runner = "launch_inner.py" if stage == "train" else "evaluate.py"
        assert runner in commands
        assert commands.index("uv sync --frozen") < commands.index(runner)
        if applies_patch:
            assert commands.index("git apply /root/change.diff") < commands.index("uv sync --frozen")


@pytest.mark.parametrize("anchored", [True, False])
def test_real_timeout_checker_counts_model_initialization(tmp_path, monkeypatch, anchored):
    timer = pytest.importorskip("nemo_rl.utils.timer")
    if anchored:
        monkeypatch.setenv("INNER_TRAIN_STARTED_AT", "1000")
    else:
        monkeypatch.delenv("INNER_TRAIN_STARTED_AT", raising=False)
    monkeypatch.setenv("PYTHONPATH", str(Path(timer.__file__).parents[2]))
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples/run_grpo.py").write_text(
        "import json, sys\n"
        "from unittest.mock import patch\n"
        "from nemo_rl.utils.timer import TimeoutChecker\n"
        "with patch('nemo_rl.utils.timer.time.time', return_value=1180):\n"
        "    timer = TimeoutChecker('00:00:05:00', fit_last_save_time=True)\n"
        "    timer.start_iterations()\n"
        "with patch('nemo_rl.utils.timer.time.time', return_value=1300):\n"
        "    due = timer.check_save()\n"
        "print(json.dumps({'start': timer.start_time, 'due': due, 'argv': sys.argv}))\n"
    )
    tree = ast.parse((TASK / "launch_inner.py").read_text())
    start = next(
        i
        for i, node in enumerate(tree.body)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "train_command"
    )
    stop = next(i for i in range(start, len(tree.body)) if isinstance(tree.body[i], ast.With))
    namespace = {"os": os, "sys": sys, "repo": tmp_path, "work": tmp_path}
    exec(compile(ast.Module(body=tree.body[start:stop], type_ignores=[]), "launch_inner.py", "exec"), namespace)
    result = subprocess.run(namespace["train_command"], text=True, capture_output=True, check=True, timeout=30)
    payload = json.loads(result.stdout)
    assert payload["start"] == (1000 if anchored else 1180)
    assert payload["due"] is anchored
    assert payload["argv"] == [
        str(tmp_path / "examples/run_grpo.py"),
        "--config",
        str(tmp_path / "resolved_grpo.yaml"),
    ]


def test_evaluation_config_is_pristine_and_has_no_resume(base_config):
    tree = ast.parse((TASK / "evaluate.py").read_text())
    start = next(
        i
        for i, node in enumerate(tree.body)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "cfg"
    )
    stop = next(i for i in range(start, len(tree.body)) if isinstance(tree.body[i], ast.For))
    namespace = {
        "OmegaConf": OmegaConf,
        "repo": Path("/pristine/NeMo-RL"),
        "model_dir": Path("/testbed/eval_model"),
        "os": os,
    }
    with patch.object(OmegaConf, "load", return_value=base_config) as load:
        exec(compile(ast.Module(body=tree.body[start:stop], type_ignores=[]), "evaluate.py", "exec"), namespace)
    load.assert_called_once_with(Path("/pristine/NeMo-RL/examples/configs/grpo_math_1B.yaml"))
    cfg = namespace["cfg"]
    assert [data.data_path for data in cfg.data.validation] == ["/root/math_eval.jsonl", "/root/aime25.jsonl"]
    assert all(data.split_validation_size == 0 for data in cfg.data.validation)
    assert [data.prompt_file for data in cfg.data.validation] == [
        "examples/prompts/cot.txt",
        "/testbed/cot_prompt.txt",
    ]
    assert cfg.data.train.split_validation_size == 0
    assert cfg.data.default.system_prompt_file is None
    assert cfg.grpo.val_at_start and cfg.grpo.stop_at_validation_threshold == 0.0
    assert cfg.grpo.max_val_samples == 62 and cfg.grpo.val_batch_size == 2
    assert not cfg.checkpointing.enabled and cfg.checkpointing.checkpoint_must_save_by is None
    assert cfg.policy.model_name == cfg.policy.tokenizer.name == "/testbed/eval_model"
    generation = cfg.policy.generation
    assert generation.max_new_tokens == generation.vllm_cfg.max_model_len == 32768
    assert generation.temperature == generation.val_temperature == 0.0
    assert generation.top_p == generation.val_top_p == 1.0
    assert generation.top_k == generation.val_top_k == -1


def test_legacy_checkpoint_exports_only_bfloat16_safetensors(tmp_path, capsys):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from safetensors.torch import load_file, save_file
    from torch.distributed.checkpoint import save
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    model = torch.nn.Linear(3, 2)
    checkpoint = tmp_path / "step_3"
    save({"model": model.state_dict()}, checkpoint_id=str(checkpoint / "policy/weights"))
    tree = ast.parse((TASK / "launch_inner.py").read_text())
    start = next(
        i
        for i, node in enumerate(tree.body)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "weights_path"
    )
    exec(
        compile(ast.Module(body=tree.body[start:], type_ignores=[]), "launch_inner.py", "exec"),
        {
            "torch": torch,
            "save_file": save_file,
            "dcp_to_torch_save": dcp_to_torch_save,
            "work": tmp_path,
            "checkpoint": checkpoint,
            "steps": 3,
            "train_reward": 0.125,
            "json": json,
        },
    )
    restored = load_file(str(tmp_path / "model.safetensors"))
    assert all(tensor.dtype == torch.bfloat16 for tensor in restored.values())
    assert all(torch.equal(restored[name], tensor.to(torch.bfloat16)) for name, tensor in model.state_dict().items())
    assert not (tmp_path / "export.pt").exists()
    output = capsys.readouterr().out
    assert 'NEMORL_ENV_TRAIN_RESULT={"train_reward": 0.125, "inner_steps": 3}' in output
    assert "NEMORL_ENV_RESULT=" not in output


@pytest.mark.parametrize("corruption", [None, "tied_omitted", "missing", "extra", "shape", "dtype", "nan"])
def test_real_safetensors_boundary_with_tiny_qwen(tmp_path, corruption):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    from safetensors.torch import load_file, save_file

    config = transformers.Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        tie_word_embeddings=True,
    )
    original = transformers.AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    state = {name: tensor.to(torch.bfloat16).clone() for name, tensor in original.state_dict().items()}
    key = "model.layers.0.self_attn.q_proj.weight"
    if corruption == "tied_omitted":
        del state["lm_head.weight"]
    elif corruption == "missing":
        del state[key]
    elif corruption == "extra":
        state["unexpected.weight"] = torch.ones(1)
    elif corruption == "shape":
        state[key] = torch.ones(1)
    elif corruption == "dtype":
        state[key] = state[key].to(torch.int64)
    elif corruption == "nan":
        state[key][0, 0] = float("nan")
    save_file(state, str(tmp_path / "model.safetensors"))

    tree = ast.parse((TASK / "evaluate.py").read_text())
    start = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.With))
    stop = next(
        i
        for i in range(start, len(tree.body))
        if isinstance(tree.body[i], ast.Assign)
        and isinstance(tree.body[i].targets[0], ast.Name)
        and tree.body[i].targets[0].id == "cfg"
    )

    nodes = [node for node in tree.body[start:stop] if not isinstance(node, ast.Delete)]
    namespace = {
        "torch": torch,
        "AutoModelForCausalLM": transformers.AutoModelForCausalLM,
        "AutoTokenizer": transformers.AutoTokenizer,
        "model_config": config,
        "model_name": "test-only",
        "model_revision": "test-only",
        "work": tmp_path,
        "load_file": load_file,
        "save_file": save_file,
    }
    with patch.object(transformers.AutoTokenizer, "from_pretrained"):
        if corruption not in (None, "tied_omitted"):
            with pytest.raises((ValueError, RuntimeError)):
                exec(compile(ast.Module(body=nodes, type_ignores=[]), "evaluate.py", "exec"), namespace)
            return
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "evaluate.py", "exec"), namespace)
    restored = load_file(str(tmp_path / "eval_model/model.safetensors"))
    assert set(restored) == set(original.state_dict())
    assert torch.equal(restored[key], state[key])
    assert torch.equal(restored["lm_head.weight"], restored["model.embed_tokens.weight"])
