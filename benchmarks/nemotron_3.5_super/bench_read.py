from itertools import islice
from time import time

import orjson
import ray
from tqdm.auto import tqdm


ray.init()


@ray.remote
def decode_batch(batch):
    results = []

    for line_no, (row_idx, row_str) in batch:
        results.append((line_no, row_idx, orjson.loads(row_str)))

    return results


def batched(iterable, size):
    iterator = iter(iterable)

    while batch := list(islice(iterator, size)):
        yield batch


start_time = time()
print("Starting read")
with open(
    "/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/jiaqiz/data/gym/all_super_env/tau_pivot/super_row_54_1000_synthetic_tau_all_rollouts_leq_60_passrate.jsonl",
    "rb",
    buffering=1024 * 1024,
) as file:
    rows = ((line_no, (line_no, line)) for line_no, line in enumerate(tqdm(file, desc="Reading file")))
    batches = list(batched(rows, 10_000))

print("Starting json load")
refs = [decode_batch.remote(batch) for batch in batches]

records = []
for batch_result in tqdm(ray.util.as_completed(refs), desc="Loading batches"):
    records.extend(ray.get(batch_result))
print(f"Finished loading {len(records)} rows in {time() - start_time:.2f}s")
