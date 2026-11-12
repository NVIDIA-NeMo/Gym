# Description

Data links: ?

# Quickstart
## Apply golden patches
### Start resources server
```bash
gym env start \
    --config resources_servers/swebench/configs/swebench.yaml \
    --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
    +swebench_resources_server.resources_servers.swebench.is_verifying_golden_patch=true
```

### One golden patch smoke test
In a separate terminal:
```bash
# If you haven't already, prepare SWE Bench benchmark data
gym eval prepare --config benchmarks/swebench/verified/opencode.yaml

python resources_servers/swebench/client.py \
    +benchmark_jsonl=benchmarks/swebench/data/swebench_verified_benchmark.jsonl
```

### Full SWE Verified golden patch smoke test
In a separate terminal:
```bash
python resources_servers/swebench/apply_golden_patch.py \
    +benchmark_jsonl=benchmarks/swebench/data/swebench_verified_benchmark.jsonl \
    +limit=...  # No limit for full samples
```


## Expected golden patch resolve rates
SWE Bench Verified
```bash
Resolved: 492 / 500 (98.40%): ... [19:29, 258.00s/it]
```

SWE Bench Multilingual
```bash
Resolved: 300 / 300 (100.00%): [26:31<00:00,  5.30s/it]
```

SWE Bench Pro


## SWE Bench Multimodal grading
- **Test-patch binary files.** Some tasks' tests compare against reference images that a text diff cannot carry
  (`image_assets["test_patch"]`; 51 files in 11 Chart.js dev tasks). SWE-bench's evaluation script downloads them with
  `curl`. `benchmarks/swebench/multimodal/prepare.py` stores their bytes in the row (`content_base64`), the server
  uploads them into the verification sandbox only (never the agent's), and the evaluation script copies them into
  place instead of downloading, so the verification sandbox needs no internet access. A row whose test-patch files
  lack their bytes is rejected.
- **Failure-only repositories.** SWE-bench grades Chart.js, marked and p5.js from failures alone: a test missing from
  the log counts as passing. A resolved verdict for these repositories therefore also needs the test runner's own
  summary showing that its suite ran to completion (`failure_only_grading.py`): Chrome's Karma run executed every
  planned test, Jasmine ran its specs without an "Incomplete" report, or every p5.js browser suite printed passing
  tests. Without it the evaluation is reported as not completed, and the sample is masked as a verifier error.


# Licensing information
Code: ?
Data: ?

Dependencies
- nemo_gym: Apache 2.0
?
