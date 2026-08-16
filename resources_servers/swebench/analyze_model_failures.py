from argparse import ArgumentParser
from collections import Counter

import orjson
from tqdm.auto import tqdm


parser = ArgumentParser()
parser.add_argument("--rollout-jsonl", type=str, required=True)
args = parser.parse_args()

rewards = Counter()
with open(args.rollout_jsonl) as f:
    for line in tqdm(f):
        row = orjson.loads(line)

        instance_id = row["instance_id"]
        reward = row["reward"]

        rewards[instance_id] += reward


# Assume 3 repeats
for instance_id, total_reward in rewards.items():
    if total_reward == 3:
        continue

    print(f"{instance_id}: {total_reward} / 3")
