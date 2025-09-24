import json
from argparse import ArgumentParser
from pathlib import Path
from random import Random

from kestrel.env import DatasetConfig, HypothesisDataset


HERE = Path(__file__).parent


def write_jsonl(idcs: list[int], out_path: Path) -> None:
    lines = [json.dumps({"task_idx": i, "responses_create_params": {"input": []}}) for i in idcs]
    out_path.write_text("\n".join(lines))
    print(f"Wrote {len(lines)} lines to {out_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=1405)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()

    # Note that this will also download & cache capsule data
    num_problems = len(HypothesisDataset(DatasetConfig()))

    random = Random(args.seed)
    val_indices = random.sample(range(num_problems), int(num_problems * args.val_fraction))
    train_indices = [i for i in range(num_problems) if i not in val_indices]

    data_path = HERE / "data"
    data_path.mkdir(parents=True, exist_ok=True)
    write_jsonl(train_indices, data_path / "bbh_train.jsonl")
    write_jsonl(val_indices, data_path / "bbh_validation.jsonl")
    write_jsonl(train_indices[:5], data_path / "bbh_example.jsonl")
