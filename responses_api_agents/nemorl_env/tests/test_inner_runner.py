# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from nemo_gym.config_types import DatasetConfig
from nemo_gym.global_config import GlobalConfigDictParser


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
    authored.pop("defaults")
    authored.data.pop("_override_")
    authored.env.pop("_override_")
    base_config.data = authored.data
    base_config.env = authored.env
    authored.policy.optimizer.kwargs.lr = 1e-5
    authored.data.train.data_path = "/testbed/custom_gym_train.jsonl"
    tree = ast.parse((TASK / "launch_inner.py").read_text())
    stop = next(
        i
        for i, node in enumerate(tree.body)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "train_log"
    )
    nodes = [node for node in tree.body[:stop] if not isinstance(node, (ast.Import, ast.ImportFrom))]
    namespace = {
        "OmegaConf": OmegaConf,
        "Path": Path,
        "os": os,
        "load_config": MagicMock(return_value=OmegaConf.merge(base_config, authored)),
    }
    with patch.object(OmegaConf, "save"):
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "launch_inner.py", "exec"), namespace)
    cfg = namespace["cfg"]
    assert cfg.grpo.num_prompts_per_step == cfg.grpo.num_generations_per_prompt == 8
    assert cfg.policy.train_global_batch_size == 64
    assert namespace["seconds"] == (3600 if budget is None else budget)
    assert cfg.checkpointing.checkpoint_must_save_by == checkpoint_deadline
    namespace["load_config"].assert_called_once_with(Path("/testbed/recipe.yaml"))
    assert cfg.policy.optimizer.kwargs.lr == 1e-5
    assert cfg.data.train.data_path == "/testbed/custom_gym_train.jsonl"
    assert cfg.data.train.split_validation_size == 0 and cfg.data.validation is None
    assert cfg.grpo.val_period == 0 and not cfg.grpo.val_at_start and not cfg.grpo.val_at_end
    assert cfg.data.default.system_prompt_file is None
    assert cfg.env.should_use_nemo_gym
    assert cfg.data.default.dataset_name == "NemoGymDataset"
    assert cfg.data.default.processor == "nemo_gym_data_processor"
    assert cfg.policy.generation.vllm_cfg.async_engine
    assert cfg.policy.generation.vllm_cfg.expose_http_server
    assert not cfg.policy.generation.vllm_cfg.skip_tokenizer_init
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
    assert "--extra vllm --extra nemo_gym" in commands
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


def test_training_rows_use_native_gym_schema(tmp_path):
    tree = ast.parse((TASK / "launch_inner.py").read_text())
    start = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.With))
    originals = [{"input": "Compute 1 + 1", "output": "2"}, {"input": "Compute 2 + 3", "output": 5}]
    (tmp_path / "train_math.jsonl").write_text("".join(json.dumps(row) + "\n" for row in originals))
    exec(
        compile(ast.Module(body=[tree.body[start]], type_ignores=[]), "launch_inner.py", "exec"),
        {"work": tmp_path, "json": json},
    )
    rows = [json.loads(line) for line in (tmp_path / "gym_train.jsonl").read_text().splitlines()]
    assert len(rows) == len(originals)
    for row, original in zip(rows, originals, strict=True):
        assert row["agent_ref"] == {"type": "responses_api_agents", "name": "math_with_judge_simple_agent"}
        assert row["question"] == original["input"]
        assert row["expected_answer"] == str(original["output"])
        assert row["responses_create_params"]["input"][0]["content"].endswith(original["input"])
        assert "\\boxed{}" in row["responses_create_params"]["input"][0]["content"]


@pytest.mark.parametrize("available", [511, 513])
def test_prepare_generates_disjoint_splits_and_author_tasks(tmp_path, monkeypatch, available):
    from responses_api_agents.nemorl_env import prepare

    load = MagicMock(return_value=iter({"problem": f"Problem {i}", "expected_answer": i} for i in range(available)))
    monkeypatch.setattr(prepare, "load_dataset", load)
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    if available < 512:
        with pytest.raises(ValueError, match="Expected at least 512"):
            prepare.prepare()
        assert not list(tmp_path.rglob("*.jsonl"))
        return
    prepare.prepare()
    train = [json.loads(line) for line in (tmp_path / "task_environment/train_math.jsonl").read_text().splitlines()]
    evaluation = [json.loads(line) for line in (tmp_path / "data/math_eval.jsonl").read_text().splitlines()]
    authors = [json.loads(line) for line in (tmp_path / "data/train.jsonl").read_text().splitlines()]
    assert len(train) == 480 and len(evaluation) == 32
    assert {row["input"] for row in train}.isdisjoint(row["input"] for row in evaluation)
    assert sorted(train + evaluation, key=lambda row: int(row["output"])) == [
        {"input": f"Problem {i}", "output": str(i)} for i in range(512)
    ]
    assert [row["output"] for row in evaluation[:3]] == ["176", "51", "152"]
    assert len(authors) == 8
    for row in authors:
        assert row["agent_ref"] == {"type": "responses_api_agents", "name": "nemorl_env"}
        assert row["responses_create_params"]["input"] == []
        assert row["responses_create_params"]["metadata"]["problem_statement"]
    assert load.call_args.kwargs["streaming"] is True
    assert load.call_args.kwargs["revision"]


@pytest.mark.parametrize("anchored", [True, False])
def test_real_timeout_checker_counts_model_initialization(tmp_path, monkeypatch, anchored):
    timer = pytest.importorskip("nemo_rl.utils.timer")
    if anchored:
        monkeypatch.setenv("INNER_TRAIN_STARTED_AT", "1000")
    else:
        monkeypatch.delenv("INNER_TRAIN_STARTED_AT", raising=False)
    monkeypatch.setenv("PYTHONPATH", str(Path(timer.__file__).parents[2]))
    (tmp_path / "examples/nemo_gym").mkdir(parents=True)
    (tmp_path / "examples/nemo_gym/run_grpo_nemo_gym.py").write_text(
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
        str(tmp_path / "examples/nemo_gym/run_grpo_nemo_gym.py"),
        "--config",
        str(tmp_path / "resolved_grpo.yaml"),
    ]


@pytest.mark.parametrize("failure", [None, "missing", "duplicate", "reward", "sidecar", "subprocess"])
def test_gym_evaluation_scores_complete_rollouts_and_cleans_up(tmp_path, monkeypatch, capsys, failure):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    for benchmark, count, question_key, answer_key in (
        ("math_eval", 32, "input", "output"),
        ("aime25", 30, "question", "expected_answer"),
    ):
        (tmp_path / f"{benchmark}.jsonl").write_text(
            "".join(json.dumps({question_key: f"{benchmark}-{i}", answer_key: "42"}) + "\n" for i in range(count))
        )
    tree = ast.parse((TASK / "evaluate.py").read_text().replace("/root/", str(tmp_path) + "/"))
    start = next(
        i
        for i, node in enumerate(tree.body)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "cfg"
    )
    server = MagicMock()
    server.poll.return_value = None
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = [1] * 100

    def run_gym(command, **kwargs):
        assert command[:3] == [str(tmp_path / ".venv/bin/gym"), "eval", "run"]
        assert kwargs["cwd"] == tmp_path / "3rdparty/Gym-workspace/Gym"
        assert kwargs["check"]
        assert command[command.index("--num-repeats") + 1] == "1"
        assert command[command.index("--split") + 1] == "validation"
        assert "--model-type" not in command
        assert "--max-output-tokens" not in command
        if failure == "subprocess":
            raise subprocess.CalledProcessError(1, command)
        rows = [json.loads(line) for line in (tmp_path / "eval_inputs.jsonl").read_text().splitlines()]
        assert len(rows) == 62
        assert all(row["responses_create_params"]["max_output_tokens"] == 32668 for row in rows)
        assert all("\\boxed{}" in row["responses_create_params"]["input"][0]["content"] for row in rows)
        for row in rows:
            row["reward"] = float(row.pop("question").startswith("aime25"))
            row["response"] = {"output": []}
        rows.reverse()
        if failure == "missing":
            rows.pop()
        elif failure == "duplicate":
            rows[-1] = rows[0]
        elif failure == "reward":
            rows[0]["reward"] = 0.5
        elif failure == "sidecar":
            (tmp_path / "eval_rollouts_failures.jsonl").write_text('{"error":"timeout"}\n')
        (tmp_path / "eval_rollouts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    namespace = {
        "OmegaConf": OmegaConf,
        "repo": tmp_path,
        "work": tmp_path,
        "model_dir": tmp_path / "eval_model",
        "AutoTokenizer": SimpleNamespace(from_pretrained=MagicMock(return_value=tokenizer)),
        "Path": Path,
        "json": json,
        "os": os,
        "sys": sys,
        "time": time,
        "urllib": urllib,
        "subprocess": subprocess,
    }
    with (
        patch.object(subprocess, "Popen", return_value=server) as popen,
        patch.object(subprocess, "run", side_effect=run_gym),
        patch.object(urllib.request, "urlopen", return_value=MagicMock()),
    ):
        code = compile(ast.Module(body=tree.body[start:], type_ignores=[]), "evaluate.py", "exec")
        if failure:
            expected = subprocess.CalledProcessError if failure == "subprocess" else RuntimeError
            with pytest.raises(expected):
                exec(code, namespace)
        else:
            exec(code, namespace)
    server.terminate.assert_called_once()
    server.wait.assert_called_once_with(timeout=30)
    assert popen.call_args.args[0][:3] == [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
    _, configs = GlobalConfigDictParser().load_extra_config_paths([str(tmp_path / "gym_eval.yaml")])
    cfg = OmegaConf.merge(*configs)
    assert not cfg.policy_model.responses_api_models.vllm_model.uses_reasoning_parser
    assert cfg.policy_model.responses_api_models.vllm_model.entrypoint == "app.py"
    assert not cfg.math_with_judge.resources_servers.math_with_judge.should_use_judge
    datasets = cfg.math_with_judge_simple_agent.responses_api_agents.simple_agent.datasets
    assert len(datasets) == 1
    dataset = DatasetConfig.model_validate(OmegaConf.to_container(datasets[0]))
    assert dataset.type == "validation" and dataset.num_repeats == 1
    assert dataset.jsonl_fpath == str(tmp_path / "eval_inputs.jsonl")
    output = capsys.readouterr().out
    if failure:
        assert "NEMORL_ENV_RESULT=" not in output
    else:
        result = json.loads(output.split("NEMORL_ENV_RESULT=")[1])
        assert result == {"reward": 0.5, "math_eval_exact": 0.0, "aime25_exact": 1.0, "completed": 1, "wandb_url": ""}


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
