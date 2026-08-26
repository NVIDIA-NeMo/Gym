"""Time Gym's real spin-up to first rollout dispatch, Lustre vs /raid.

Mirrors RolloutCollectionHelper.run_from_config's pre-dispatch work exactly:
  fresh  : _preprocess_rows_from_config -> write materialized inputs
  resume : _load_from_cache (skips preprocessing entirely)
Preprocessing does not touch the output filesystem, so it is run once and shared.
"""

import os
import shutil
import time

import orjson

from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper


SRC = (
    "/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/jiaqiz/data/gym/"
    "all_super_env/tau_pivot/super_row_54_1000_synthetic_tau_all_rollouts_leq_60_passrate.jsonl"
)
LUS = (
    "/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/jkyi/"
    "research-scratch/260824_gym_reward_profiling_update/bench_scratch"
)
RAID = f"/raid/scratch/spinup_{os.environ.get('SLURM_JOB_ID', 'x')}"
REPEATS = 8

print(f"node={os.uname().nodename}  src={os.path.getsize(SRC) / 2**30:.2f} GB  num_repeats={REPEATS}\n")
os.makedirs(LUS, exist_ok=True)
os.makedirs(RAID, exist_ok=True)
helper = RolloutCollectionHelper()


def cfg_for(d):
    return RolloutCollectionConfig.model_validate(
        {
            "input_jsonl_fpath": SRC,
            "output_jsonl_fpath": os.path.join(d, "rollouts.jsonl"),
            "num_repeats": REPEATS,
            "resume_from_cache": True,
        }
    )


try:
    print("=== [shared] read + parse + expand (filesystem-independent) ===")
    t0 = time.time()
    rows = helper._preprocess_rows_from_config(cfg_for(LUS))
    t_pre = time.time() - t0
    print(f"  _preprocess_rows_from_config       {len(rows):,} materialized rows in {t_pre:7.1f}s\n")

    res = {}
    for label, d in (("LUSTRE", LUS), ("/raid ", RAID)):
        c = cfg_for(d)
        t0 = time.time()
        with c.materialized_jsonl_fpath.open("wb") as f:
            for row in rows:
                f.write(orjson.dumps(row) + b"\n")
        os.sync()
        t_w = time.time() - t0
        sz = c.materialized_jsonl_fpath.stat().st_size
        # an empty rollouts file is what makes the resume gate fire
        open(c.output_jsonl_fpath, "w").close()
        t0 = time.time()
        helper._load_from_cache(c)
        t_r = time.time() - t0
        res[label] = (t_w, t_r, sz)
        print(f"=== {label} ===")
        print(
            f"  write materialized inputs          {sz / 2**30:6.2f} GB in {t_w:7.1f}s -> {sz / 2**20 / t_w:8.1f} MB/s"
        )
        print(f"  resume read (_load_from_cache)             in {t_r:7.1f}s -> {sz / 2**20 / t_r:8.1f} MB/s\n")

    print("=== TIME TO FINISH LOADING (tau_pivot, 170,320 src rows) ===")
    print("    NOTE: no servers started, no rollouts dispatched. Loading only.")
    print(f"  {'':10} {'fresh (preprocess+write)':>26} {'resume (read only)':>22}")
    for label in ("LUSTRE", "/raid "):
        t_w, t_r, _ = res[label]
        print(f"  {label:10} {t_pre + t_w:24.1f}s {t_r:20.1f}s")

    scale = 623_006 / 170_320
    print(f"\n=== EXTRAPOLATED to the full 26-entry sweep (x{scale:.2f}) ===")
    print(f"  {'':10} {'fresh':>14} {'resume':>14}")
    for label in ("LUSTRE", "/raid "):
        t_w, t_r, _ = res[label]
        print(f"  {label:10} {(t_pre + t_w) * scale / 60:12.1f}m {t_r * scale / 60:12.1f}m")
finally:
    shutil.rmtree(RAID, ignore_errors=True)
    shutil.rmtree(LUS, ignore_errors=True)
