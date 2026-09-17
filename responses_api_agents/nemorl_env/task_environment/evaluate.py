import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


repo = Path(os.environ.get("NEMORL_ROOT", "/workspace/NeMo-RL"))
work = Path("/testbed")
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model_revision = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
model_config = AutoConfig.from_pretrained(model_name, revision=model_revision, trust_remote_code=False)
with torch.device("meta"):
    reference = AutoModelForCausalLM.from_config(model_config, trust_remote_code=False, attn_implementation="eager")

state = load_file(work / "model.safetensors", device="cpu")
if any(not tensor.is_floating_point() or not tensor.isfinite().all().item() for tensor in state.values()):
    raise ValueError("checkpoint must contain finite floating-point weights")
if model_config.tie_word_embeddings and "lm_head.weight" not in state:
    state["lm_head.weight"] = state["model.embed_tokens.weight"].clone()
reference.load_state_dict(state, strict=True, assign=True)
model_dir = work / "eval_model"
model_dir.mkdir()
save_file(state, str(model_dir / "model.safetensors"), metadata={"format": "pt"})
model_config.save_pretrained(model_dir)
AutoTokenizer.from_pretrained(model_name, revision=model_revision, trust_remote_code=False).save_pretrained(model_dir)
del state, reference


cfg = {
    "config_paths": [
        "responses_api_models/vllm_model/configs/vllm_model.yaml",
        "resources_servers/math_with_judge/configs/math_with_judge.yaml",
    ],
    "math_with_judge_simple_agent": {
        "responses_api_agents": {
            "simple_agent": {
                "datasets": [
                    {
                        "name": "inner_eval",
                        "type": "validation",
                        "license": "TBD",
                        "jsonl_fpath": str(work / "eval_inputs.jsonl"),
                        "num_repeats": 1,
                    }
                ]
            }
        }
    },
    "policy_model": {"responses_api_models": {"vllm_model": {"uses_reasoning_parser": False}}},
}
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)
prompt = "Solve the following math problem. Work step by step and put your final answer inside \\boxed{}.\n\n"
questions = set()
inputs = []
rows = [json.loads(line) for line in Path("/root/aime25.jsonl").read_text().splitlines()]
if len(rows) != 30:
    raise ValueError("AIME25 must contain exactly 30 rows")
for row in rows:
    question = row["question"]
    if question in questions:
        raise ValueError("evaluation questions must be unique")
    questions.add(question)
    messages = [{"role": "user", "content": prompt + question}]
    output_tokens = 32768 - len(tokenizer.apply_chat_template(messages, add_generation_prompt=True))
    if output_tokens <= 0:
        raise ValueError("evaluation prompt exceeds model context")
    inputs.append(
        {
            "question": question,
            "expected_answer": str(row["expected_answer"]),
            "responses_create_params": {"input": messages, "max_output_tokens": output_tokens},
        }
    )
(work / "eval_inputs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in inputs))
OmegaConf.save(OmegaConf.create(cfg), work / "gym_eval.yaml")
rollout_file = work / "eval_rollouts.jsonl"
concurrency = int(os.environ.get("NEMORL_ENV_EVAL_CONCURRENCY", "30"))
if concurrency < 1:
    raise ValueError("Evaluation concurrency must be positive")
with (work / "aime25.log").open("w") as stream:
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(model_dir),
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--max-model-len",
            "32768",
            "--max-num-seqs",
            str(concurrency),
            "--gpu-memory-utilization",
            "0.85",
            "--enforce-eager",
        ],
        stdout=stream,
        stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.monotonic() + 600
        while True:
            if server.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError("evaluation vLLM failed to become ready; see aime25.log")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5):
                    break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(2)
        subprocess.run(
            [
                str(repo / ".venv/bin/gym"),
                "eval",
                "run",
                "--config",
                str(work / "gym_eval.yaml"),
                "--model",
                str(model_dir),
                "--model-url",
                "http://127.0.0.1:8000/v1",
                "--model-api-key",
                "EMPTY",
                "--split",
                "validation",
                "--num-repeats",
                "8",
                "--concurrency",
                str(concurrency),
                "--temperature",
                "0.7",
                "--top-p",
                "1",
                "--output",
                str(rollout_file),
            ],
            cwd=repo / "3rdparty/Gym-workspace/Gym",
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
rows = [json.loads(line) for line in rollout_file.read_text().splitlines()]
for row in rows:
    row["question"] = row["responses_create_params"]["input"][0]["content"].removeprefix(prompt)
failures = rollout_file.with_name("eval_rollouts_failures.jsonl")
if failures.exists() and failures.read_text().strip():
    raise RuntimeError("Gym evaluation reported failed rollouts")
if Counter(row["question"] for row in rows) != Counter({question: 8 for question in questions}):
    raise RuntimeError("Gym evaluation must return exactly eight answers for each of the 30 AIME25 questions")
if any(row["reward"] not in (0.0, 1.0) for row in rows):
    raise RuntimeError("Gym evaluation must return binary math rewards")
aime_accuracy = sum(row["reward"] for row in rows) / 240
wandb_url = ""
if "WANDB_API_KEY" in os.environ:
    import wandb

    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY"),
        project=os.environ.get("WANDB_PROJECT", "cmunley-rlenv"),
        name=f"{os.environ.get('WANDB_NAME', 'nemorl-env-inner-eval')}-rollouts",
        job_type="evaluation",
    )
    run.log(
        {
            "aime25/rollouts": wandb.Table(
                columns=["question", "reward", "response"],
                data=[[row["question"], row["reward"], json.dumps(row["response"])] for row in rows],
            ),
            "aime25/accuracy": aime_accuracy,
            "aime25/avg@8": aime_accuracy,
        }
    )
    wandb_url = run.url
    run.finish()
print(
    "NEMORL_ENV_RESULT="
    + json.dumps(
        {
            "reward": aime_accuracy,
            "aime25_exact": aime_accuracy,
            "aime25_avg_at_8": aime_accuracy,
            "completed": 1,
            "wandb_url": wandb_url,
        }
    )
)
