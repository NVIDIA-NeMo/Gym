import json
from collections import Counter

from tqdm.auto import tqdm


c = Counter()
counting_interval = 20
with open("resources_servers/comp_coding/data/opencodereasoning_filtered_25k_train.jsonl") as f:
    for row in tqdm(f, desc="Processing"):
        row = json.loads(row)
        num_inputs = len(row["verifier_metadata"]["unit_tests"]["inputs"])
        railed_num_inputs = counting_interval * (num_inputs // counting_interval)
        c[railed_num_inputs] += 1

print(f"Number of test cases, bucketed every {counting_interval}")
for key, value in sorted(c.items(), key=lambda p: p[0]):
    print(f"{key:>3}: {value}")
