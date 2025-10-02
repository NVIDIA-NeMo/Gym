import json
from collections import Counter

from tqdm.auto import tqdm


c = Counter()
counting_interval = 20
total_count = 0
with open("resources_servers/comp_coding/data/opencodereasoning_filtered_25k_train.jsonl") as f:
    for row in tqdm(f, desc="Processing"):
        row = json.loads(row)
        num_inputs = len(row["verifier_metadata"]["unit_tests"]["inputs"])
        railed_num_inputs = counting_interval * (num_inputs // counting_interval)
        c[railed_num_inputs] += 1
        total_count += 1

print(f"Number of test cases, bucketed every {counting_interval}")
cumulative_count = 0
for key, value in sorted(c.items(), key=lambda p: p[0]):
    cumulative_count += value
    cumulative_pct = 100 * cumulative_count / total_count
    print(f"{key:>3}: {value:>6} (cumulative {cumulative_pct:.0f}%)")
