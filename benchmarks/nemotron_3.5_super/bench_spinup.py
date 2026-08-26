"""Time Gym's real spin-up to first rollout dispatch, Lustre vs /raid.

Mirrors RolloutCollectionHelper.run_from_config's pre-dispatch work exactly:
  fresh  : _preprocess_rows_from_config -> write materialized inputs
  resume : _load_from_cache (skips preprocessing entirely)
Preprocessing does not touch the output filesystem, so it is run once and shared.
"""

from os import makedirs
from os.path import getsize
from pathlib import Path
from shutil import rmtree
from time import time

import orjson

from nemo_gym.global_config import get_global_config_dict
from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper


global_config = get_global_config_dict()
input_jsonl_fpath = global_config["input_jsonl_fpath"]
output_jsonl_fpath = global_config["output_jsonl_fpath"]
num_repeats = global_config["num_repeats"]

print(f"src={getsize(input_jsonl_fpath) / 2**30:.2f} GB num_repeats={num_repeats}\n")
makedirs(Path(output_jsonl_fpath).parent, exist_ok=True)

rc_config = RolloutCollectionConfig.model_validate(
    {
        "input_jsonl_fpath": input_jsonl_fpath,
        "output_jsonl_fpath": output_jsonl_fpath,
        "num_repeats": num_repeats,
        "resume_from_cache": True,
    }
)

rc_helper = RolloutCollectionHelper()

print("Starting _preprocess_rows_from_config...")
start_time = time()
rows = rc_helper._preprocess_rows_from_config(rc_config)
print(f"_preprocess_rows_from_config {len(rows):,} materialized rows in {time() - start_time:.2f}s\n")

print("Starting to write rows...")
start_time = time()
with rc_config.materialized_jsonl_fpath.open("wb") as f:
    for row in rows:
        f.write(orjson.dumps(row) + b"\n")
print(f"Writing rows took {time() - start_time:.2f}s")

rmtree(output_jsonl_fpath, ignore_errors=True)
