import json
from itertools import islice
from pathlib import Path

from datasets import load_dataset


ROOT = Path(__file__).parent
PROMPT = (
    "Improve recipe.yaml or NeMo-RL/ (including its Gym sources) to maximize the final "
    "AIME25 avg@8 accuracy of Qwen2.5-1.5B-Instruct after one-GPU GRPO training. Choose and "
    "implement your best justified change, including core loss, optimization, generation, "
    "or data-processing changes. Wire the change into training and test it. "
    "The inner batch is fixed at 8 prompts x 8 responses. After pristine setup, authored "
    "dependency builds, model initialization, training, and checkpoint export share a "
    "hard 60-minute budget. A fresh unpatched evaluator scores only AIME25. "
    "Do not obtain or train on evaluation examples."
)


def prepare() -> None:
    dataset = load_dataset(
        "nvidia/OpenMathInstruct-2",
        revision="469216e3f46f4dacf476b382e192485ea51a143e",
        split="train",
        streaming=True,
    )
    rows = [{"input": row["problem"], "output": str(row["expected_answer"])} for row in islice(dataset, 512)]
    if len(rows) != 512:
        raise ValueError("Expected at least 512 OpenMathInstruct-2 rows")
    authors = [
        {
            "responses_create_params": {
                "input": [],
                "metadata": {
                    "instance_id": f"nemorl-env-aime25-{i}",
                    "dataset_name": "nemorl-env",
                    "problem_statement": PROMPT,
                    "instance_dict": "{}",
                    "image": "nemo-gpu-researcher:dev",
                },
            },
            "agent_ref": {"type": "responses_api_agents", "name": "nemorl_env"},
        }
        for i in range(5)
    ]
    for path, records in (
        (ROOT / "task_environment/train_math.jsonl", rows),
        (ROOT / "data/train.jsonl", authors[:1]),
        (ROOT / "data/example.jsonl", authors),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in records))
        print(f"Wrote {len(records)} rows to {path}")


if __name__ == "__main__":
    prepare()
