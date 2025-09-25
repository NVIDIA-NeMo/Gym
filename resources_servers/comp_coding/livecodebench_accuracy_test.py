# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
We use the livecodebench verification logic directly so we don't need to re-implement all the code parsing, test case run, etc ourselves.
The train data we use is fundamentally different from livecodebench however.

Reproduce the accuracy test setting used to test the accuracy of our integration:
```bash
git clone https://github.com/LiveCodeBench/LiveCodeBench
cd LiveCodeBench
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
# Downgrade datasets to match the poetry.lock version.
uv pip install datasets==2.18.0

HF_HOME=.cache \
OPENAI_KEY={your OpenAI API key} \
python -m lcb_runner.runner.main \
    --model gpt-4o-2024-05-13 \
    --scenario codegeneration \
    --evaluate \
    --continue_existing_with_eval \
    --num_process_evaluate 4 \
    --release_version release_v5 \
    --start_date 2024-07-01 \
    --end_date 2025-02-01
```

This is the expected output:
```bash
Downloading builder script: 5.01kB [00:00, 5.57MB/s]
Downloading readme: 3.39kB [00:00, 23.3MB/s]
Downloading data: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.25G/1.25G [00:11<00:00, 107MB/s]
Downloading data: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 713M/713M [00:06<00:00, 107MB/s]
Downloading data: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 623M/623M [00:05<00:00, 107MB/s]
Downloading data: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.20G/1.20G [00:11<00:00, 105MB/s]
Downloading data: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 558M/558M [00:05<00:00, 107MB/s]
Downloading data: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 134M/134M [00:01<00:00, 107MB/s]
Generating test split: 880 examples [00:10, 86.65 examples/s]
Loaded 322 problems
 15%|██████████████████████████████▉                                                                                                                                                                            | 49/322 [04:52<23:04,  5.07s/it]
```
"""
